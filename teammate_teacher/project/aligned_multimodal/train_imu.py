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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

from imu_data import (
    CHANNEL_GROUPS,
    DEVICES,
    IMUDataset,
    compute_channel_normalizer,
    read_index,
)
from imu_model import IMUTemporalModel, fp32_size_mb, parameter_count


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fixed-epoch subject-disjoint IMU TCN")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "cache" / "imu_32")
    parser.add_argument("--fold-summary", type=Path, default=PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel-group", choices=tuple(CHANNEL_GROUPS), default="accgyro")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device-dropout", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


@torch.inference_mode()
def evaluate(
    model: IMUTemporalModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    labels: list[int] = []
    sample_ids: list[str] = []
    logits: list[np.ndarray] = []
    for batch in loader:
        output = model(
            batch["imu"].to(device, non_blocking=True),
            batch["time_mask"].to(device, non_blocking=True),
            batch["device_mask"].to(device, non_blocking=True),
        )
        logits.append(output.float().cpu().numpy())
        labels.extend(batch["label"].numpy().tolist())
        sample_ids.extend(batch["sample_id"])
    logits_array = np.concatenate(logits)
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = logits_array.argmax(axis=1)
    return {
        "sample_ids": sample_ids,
        "labels": labels_array,
        "logits": logits_array,
        "predictions": predictions,
        "metrics": metrics(labels_array, predictions),
    }


def make_loader(dataset: IMUDataset, batch_size: int, workers: int, training: bool) -> DataLoader:
    sampler = None
    shuffle = False
    if training:
        labels = [row.class_id for row in dataset.rows]
        counts = Counter(labels)
        weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = args.seed + args.fold
    seed_everything(seed)
    cache_dir = args.cache_dir.resolve()
    rows = [row for row in read_index(cache_dir / "index.csv") if row.split == "train" and row.usable]
    fold_summary = json.loads(args.fold_summary.resolve().read_text(encoding="utf-8"))
    fold_info = fold_summary["folds"][args.fold]
    train_users = set(fold_info["train_users"])
    val_users = set(fold_info["val_users"])
    train_rows = [row for row in rows if row.user_id in train_users]
    val_rows = [row for row in rows if row.user_id in val_users]
    if set(row.user_id for row in train_rows) & set(row.user_id for row in val_rows):
        raise RuntimeError("Train/validation subjects overlap")

    values = np.load(cache_dir / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    time_mask = np.load(cache_dir / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    train_indices = np.asarray([row.cache_index for row in train_rows], dtype=np.int64)
    channel_indices = CHANNEL_GROUPS[args.channel_group]
    mean, std = compute_channel_normalizer(values, time_mask, train_indices, channel_indices)

    train_dataset = IMUDataset(
        cache_dir, train_rows, args.channel_group, mean, std, True,
        device_dropout=args.device_dropout, noise_std=args.noise_std,
    )
    val_dataset = IMUDataset(cache_dir, val_rows, args.channel_group, mean, std, False)
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, True)
    val_loader = make_loader(val_dataset, args.batch_size * 2, 0, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMUTemporalModel(
        input_channels=len(channel_indices),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            imu = batch["imu"].to(device, non_blocking=True)
            batch_time_mask = batch["time_mask"].to(device, non_blocking=True)
            batch_device_mask = batch["device_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with autocast(device.type, enabled=device.type == "cuda"):
                logits = model(imu, batch_time_mask, batch_device_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            count += len(labels)
        scheduler.step()
        val_result = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            "train_accuracy": correct / max(count, 1),
            "val_accuracy_monitor_only": val_result["metrics"]["accuracy"],
            "val_balanced_accuracy_monitor_only": val_result["metrics"]["balanced_accuracy"],
            "val_macro_f1_monitor_only": val_result["metrics"]["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"fold={args.fold} epoch={epoch:02d}/{args.epochs} "
            f"train={row['train_accuracy']:.4f} val={row['val_accuracy_monitor_only']:.4f}",
            flush=True,
        )

    final_result = evaluate(model, val_loader, device)
    drop_metrics = {}
    for device_index, device_name in enumerate(DEVICES):
        drop_dataset = IMUDataset(
            cache_dir, val_rows, args.channel_group, mean, std, False,
            forced_drop_device=device_index,
        )
        drop_result = evaluate(
            model, make_loader(drop_dataset, args.batch_size * 2, 0, False), device
        )
        drop_metrics[device_name] = drop_result["metrics"]

    config = vars(args).copy()
    config["cache_dir"] = str(cache_dir)
    config["fold_summary"] = str(args.fold_summary.resolve())
    config["output_dir"] = str(output)
    config["seed_effective"] = seed
    config["train_users"] = sorted(train_users)
    config["val_users"] = sorted(val_users)
    config["selection_protocol"] = "fixed final epoch; validation trajectory is monitor-only"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "normalizer_mean": mean,
        "normalizer_std": std,
        "epoch": args.epochs,
        "final_fixed_epoch": True,
    }
    torch.save(checkpoint, output / "final.pt")
    np.savez_compressed(
        output / "val_logits.npz",
        sample_ids=np.asarray(final_result["sample_ids"]),
        labels=final_result["labels"],
        logits=final_result["logits"],
        predictions=final_result["predictions"],
        fold=np.full(len(final_result["labels"]), args.fold, dtype=np.int64),
    )
    with (output / "val_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prediction", "fold"])
        writer.writerows(
            zip(
                final_result["sample_ids"],
                final_result["labels"].tolist(),
                final_result["predictions"].tolist(),
                [args.fold] * len(final_result["labels"]),
            )
        )
    with (output / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "fold": args.fold,
        "channel_group": args.channel_group,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "parameters": parameter_count(model),
        "fp32_parameter_size_mb": fp32_size_mb(model),
        "fixed_epoch": args.epochs,
        "metrics": final_result["metrics"],
        "drop_one_device_metrics": drop_metrics,
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "elapsed_seconds": round(time.time() - started, 2),
        "selection_protocol": config["selection_protocol"],
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
