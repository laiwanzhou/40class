from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from thermal_data import ThermalClipDataset
from thermal_model import ThermalMobileNet, fp32_size_mb, parameter_count
from thermal_tsm_model import ThermalResNetTSM


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Thermal 轻量动作分类模型")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="临时覆盖配置中的 epochs")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loader(
    dataset: ThermalClipDataset,
    batch_size: int,
    num_workers: int,
    balanced_sampling: bool,
    seed: int,
    train: bool,
) -> DataLoader:
    sampler = None
    shuffle = train
    if train and balanced_sampling:
        counts = Counter(sample.class_id for sample in dataset.samples)
        weights = torch.tensor([1.0 / counts[sample.class_id] for sample in dataset.samples], dtype=torch.double)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        drop_last=train,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    max_grad_norm: float | None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []

    for clips, labels, _sample_ids in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        losses.append(float(loss.detach().cpu()))
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "macro_f1": float(f1_score(labels_all, predictions_all, average="macro", zero_division=0)),
        "labels": labels_all,
        "predictions": predictions_all,
    }


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    seed = int(config["seed"])
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["use_amp"] and device.type == "cuda")

    train_dataset = ThermalClipDataset(
        args.manifest,
        split="train",
        num_frames=int(config["num_frames"]),
        image_size=int(config["image_size"]),
        augment=True,
    )
    val_dataset = ThermalClipDataset(
        args.manifest,
        split="val",
        num_frames=int(config["num_frames"]),
        image_size=int(config["image_size"]),
        augment=False,
    )
    train_loader = create_loader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        balanced_sampling=bool(config["balanced_sampling"]),
        seed=seed,
        train=True,
    )
    val_loader = create_loader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        balanced_sampling=False,
        seed=seed,
        train=False,
    )

    model_name = str(config.get("model", "mobilenet_v3_small"))
    if model_name == "mobilenet_v3_small":
        model = ThermalMobileNet(num_classes=40, dropout=float(config["dropout"]))
    elif model_name == "resnet18_tsm_tcn":
        model = ThermalResNetTSM(num_classes=40, dropout=float(config["dropout"]))
    else:
        raise ValueError(f"未知模型：{model_name}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    params = parameter_count(model)
    model_mb = fp32_size_mb(model)
    print(f"设备：{device}，模型：{model_name}")
    print(f"训练/验证 trial：{len(train_dataset)}/{len(val_dataset)}")
    print(f"模型参数：{params:,}，FP32 参数大小约 {model_mb:.2f} MB")
    print(
        f"输入：{config['num_frames']} 帧 × {config['image_size']}×{config['image_size']}，"
        f"batch={config['batch_size']}，epochs={config['epochs']}"
    )

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_labels: list[int] = []
    best_predictions: list[int] = []
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.time()
        max_grad_norm = config.get("max_grad_norm")
        max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp, max_grad_norm
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, None)
        scheduler.step()

        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "seconds": round(time.time() - epoch_started, 2),
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {row['train_loss']:.4f}, acc {row['train_accuracy']:.4f}, F1 {row['train_macro_f1']:.4f} | "
            f"val loss {row['val_loss']:.4f}, acc {row['val_accuracy']:.4f}, F1 {row['val_macro_f1']:.4f} | "
            f"{row['seconds']:.1f}s"
        )

        current_f1 = float(val_metrics["macro_f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            best_labels = list(val_metrics["labels"])
            best_predictions = list(val_metrics["predictions"])
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "val_accuracy": float(val_metrics["accuracy"]),
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            print(f"连续 {epochs_without_improvement} 轮未提升，提前停止。")
            break

    matrix = confusion_matrix(best_labels, best_predictions, labels=list(range(40)))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    best_row = history[best_epoch - 1]
    summary = {
        "device": str(device),
        "model": model_name,
        "train_trials": len(train_dataset),
        "val_trials": len(val_dataset),
        "parameters": params,
        "fp32_parameter_size_mb": model_mb,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_row["val_accuracy"],
        "best_val_macro_f1": best_row["val_macro_f1"],
        "total_seconds": round(time.time() - started, 2),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成。最佳 epoch={best_epoch}，val acc={best_row['val_accuracy']:.4f}，val F1={best_f1:.4f}")
    print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()
