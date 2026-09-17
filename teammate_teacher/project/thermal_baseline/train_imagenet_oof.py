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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from thermal_model import fp32_size_mb, parameter_count
from thermal_oof_data import ThermalOOFDataset
from thermal_tsm_model import ThermalResNetTSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one ImageNet Thermal subject-disjoint fold"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
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
    dataset: ThermalOOFDataset, config: dict, train: bool
) -> DataLoader:
    sampler = None
    shuffle = train
    if train and bool(config["balanced_sampling"]):
        counts = Counter(sample.class_id for sample in dataset.samples)
        weights = torch.tensor(
            [1.0 / counts[sample.class_id] for sample in dataset.samples],
            dtype=torch.double,
        )
        generator = torch.Generator().manual_seed(int(config["seed"]))
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=generator
        )
        shuffle = False
    workers = int(config["num_workers"] if train else config.get("val_num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
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
    sample_ids_all: list[str] = []
    logits_all: list[torch.Tensor] = []
    for batch in loader:
        clips = batch["clip"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
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
        predictions_all.extend(logits.argmax(1).detach().cpu().tolist())
        sample_ids_all.extend(batch["sample_id"])
        logits_all.append(logits.detach().float().cpu())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_all, predictions_all)
        ),
        "macro_f1": float(
            f1_score(labels_all, predictions_all, average="macro", zero_division=0)
        ),
        "labels": np.asarray(labels_all, dtype=np.int64),
        "predictions": np.asarray(predictions_all, dtype=np.int64),
        "sample_ids": np.asarray(sample_ids_all),
        "logits": torch.cat(logits_all).numpy(),
    }


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: Path, model: nn.Module, config: dict, epoch: int, metrics: dict
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_accuracy": float(metrics["accuracy"]),
            "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
            "val_macro_f1": float(metrics["macro_f1"]),
        },
        path,
    )


def save_best_outputs(output_dir: Path, metrics: dict) -> None:
    with (output_dir / "val_predictions_best_accuracy.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prediction"])
        writer.writerows(
            zip(
                metrics["sample_ids"].tolist(),
                metrics["labels"].tolist(),
                metrics["predictions"].tolist(),
            )
        )
    np.savez_compressed(
        output_dir / "val_logits_best_accuracy.npz",
        sample_ids=metrics["sample_ids"],
        labels=metrics["labels"],
        logits=np.asarray(metrics["logits"], dtype=np.float32),
    )
    np.savetxt(
        output_dir / "confusion_matrix_best_accuracy.csv",
        confusion_matrix(
            metrics["labels"], metrics["predictions"], labels=np.arange(40)
        ),
        delimiter=",",
        fmt="%d",
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
    config["manifest"] = str(args.manifest.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    seed_everything(int(config["seed"]))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["use_amp"] and device.type == "cuda")
    common = {
        "manifest_path": args.manifest,
        "num_frames": int(config["num_frames"]),
        "image_height": int(config["image_height"]),
        "image_width": int(config["image_width"]),
        "normalization": str(config.get("normalization", "legacy")),
    }
    train_dataset = ThermalOOFDataset(split="train", augment=True, **common)
    val_dataset = ThermalOOFDataset(split="val", augment=False, **common)
    train_loader = create_loader(train_dataset, config, True)
    val_loader = create_loader(val_dataset, config, False)

    model = ThermalResNetTSM(
        num_classes=40,
        dropout=float(config["dropout"]),
        imagenet_pretrained=bool(config["imagenet_pretrained"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_grad_norm = config.get("max_grad_norm")
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

    print(
        f"device={device}, train/val={len(train_dataset)}/{len(val_dataset)}, "
        f"frames={config['num_frames']}, size={config['image_width']}x"
        f"{config['image_height']}, batch={config['batch_size']}",
        flush=True,
    )
    print(
        f"parameters={parameter_count(model):,}, "
        f"fp32={fp32_size_mb(model):.2f} MiB, "
        f"imagenet={config['imagenet_pretrained']}",
        flush=True,
    )

    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_f1 = -1.0
    best_accuracy_epoch = 0
    best_f1_epoch = 0
    best_accuracy_metrics: dict[str, object] | None = None
    started = time.time()
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            max_grad_norm,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, None, scaler, use_amp, None
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "seconds": round(time.time() - epoch_started, 2),
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        print(
            f"Epoch {epoch:02d} | train acc {row['train_accuracy']:.4f}, "
            f"F1 {row['train_macro_f1']:.4f} | val acc "
            f"{row['val_accuracy']:.4f}, bal {row['val_balanced_accuracy']:.4f}, "
            f"F1 {row['val_macro_f1']:.4f}, loss {row['val_loss']:.4f} | "
            f"{row['seconds']:.1f}s",
            flush=True,
        )
        save_checkpoint(output_dir / "last.pt", model, config, epoch, val_metrics)
        if float(val_metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            best_accuracy_epoch = epoch
            best_accuracy_metrics = val_metrics
            save_checkpoint(
                output_dir / "best_accuracy.pt", model, config, epoch, val_metrics
            )
            save_best_outputs(output_dir, val_metrics)
        if float(val_metrics["macro_f1"]) > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            best_f1_epoch = epoch
            save_checkpoint(
                output_dir / "best_macro_f1.pt", model, config, epoch, val_metrics
            )

    assert best_accuracy_metrics is not None
    best_row = history[best_accuracy_epoch - 1]
    summary = {
        "device": str(device),
        "train_trials": len(train_dataset),
        "val_trials": len(val_dataset),
        "parameters": parameter_count(model),
        "fp32_parameter_size_mb": fp32_size_mb(model),
        "imagenet_pretrained": bool(config["imagenet_pretrained"]),
        "normalization": str(config.get("normalization", "legacy")),
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_val_accuracy": best_row["val_accuracy"],
        "accuracy_checkpoint_balanced_accuracy": best_row["val_balanced_accuracy"],
        "accuracy_checkpoint_macro_f1": best_row["val_macro_f1"],
        "best_macro_f1_epoch": best_f1_epoch,
        "best_val_macro_f1": best_f1,
        "total_seconds": round(time.time() - started, 2),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
