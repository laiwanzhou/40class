from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from aligned_data import AlignedMultimodalDataset
from p27_temporal_skeleton_model import P27TemporalSkeleton
from probe_p27r3_incremental_information import metric_bundle, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    PROJECT_DIR / "configs" / "p27_temporal_skeleton_inner_fold0.json"
)
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "runs" / "p27_strong_inner" / "temporal_skeleton" / "fold_0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P27 full-sequence graph-temporal Skeleton")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: AlignedMultimodalDataset,
    config: dict[str, Any],
    train: bool,
) -> DataLoader:
    sampler = None
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
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def augment_skeleton(source: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    source = source.clone()
    batch_size = len(source)
    yaw_limit = math.radians(float(config["yaw_degrees"]))
    angles = source.new_empty(batch_size).uniform_(-yaw_limit, yaw_limit)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    for start in (0, 4, 7):
        x = source[..., start].clone()
        z = source[..., start + 2].clone()
        source[..., start] = cosine[:, None, None] * x + sine[:, None, None] * z
        source[..., start + 2] = -sine[:, None, None] * x + cosine[:, None, None] * z
    noise_std = float(config["coordinate_noise_std"])
    if noise_std > 0:
        noise = torch.randn_like(source) * noise_std
        noise[..., 3] = 0.0
        source = source + noise
    joint_dropout = float(config["joint_dropout"])
    if joint_dropout > 0:
        keep = (
            torch.rand(
                source.shape[0],
                source.shape[1],
                source.shape[2],
                1,
                device=source.device,
            )
            >= joint_dropout
        )
        source = source * keep
    return source


def standard_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def run_epoch(
    model: P27TemporalSkeleton,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    use_amp = bool(config["use_amp"] and device.type == "cuda")
    for batch in loader:
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            skeleton = augment_skeleton(skeleton, config)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(skeleton)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["max_grad_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach()) * len(labels)
        total_samples += len(labels)
        labels_all.append(labels.detach().cpu().numpy())
        predictions_all.append(logits.detach().argmax(1).cpu().numpy())
        logits_all.append(logits.detach().float().cpu().numpy())
    labels_array = np.concatenate(labels_all)
    predictions_array = np.concatenate(predictions_all)
    return (
        {
            "loss": total_loss / max(total_samples, 1),
            **standard_metrics(labels_array, predictions_array),
        },
        labels_array,
        predictions_array,
        np.concatenate(logits_all),
    )


def save_checkpoint(
    path: Path,
    model: P27TemporalSkeleton,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "protocol": config["protocol"],
            "epoch": epoch,
            "config": config,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    seed_everything(int(config["seed"]))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {
        "manifest_path": args.manifest.resolve(),
        "modalities": ["skeleton"],
        "num_frames": int(config["num_frames"]),
        "image_height": 144,
        "image_width": 192,
        "cache_dir": "cache/aligned_192x144",
        "skeleton_representation": config["skeleton_representation"],
        "skeleton_raw_cache_dir": "cache/skeleton_raw",
    }
    train_dataset = AlignedMultimodalDataset(
        split="train", augment=True, **common
    )
    held_dataset = AlignedMultimodalDataset(
        split="val", augment=False, **common
    )
    train_loader = make_loader(train_dataset, config, True)
    held_loader = make_loader(held_dataset, config, False)
    model = P27TemporalSkeleton(
        input_dim=int(config["skeleton_input_dim"]),
        num_frames=int(config["num_frames"]),
        graph_width=int(config["graph_width"]),
        temporal_width=int(config["temporal_width"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        dropout=float(config["dropout"]),
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
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(config["use_amp"] and device.type == "cuda")
    )
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_accuracy_epoch = 0
    best_macro_f1_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_metrics, _, _, _ = run_epoch(
            model, train_loader, criterion, device, config, optimizer, scaler
        )
        held_metrics, held_labels, held_predictions, held_logits = run_epoch(
            model, held_loader, criterion, device, config, None, scaler
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"held_{key}": value for key, value in held_metrics.items()},
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(output / "history.csv", history)
        save_checkpoint(output / "last.pt", model, config, epoch, held_metrics)
        if held_metrics["accuracy"] > best_accuracy:
            best_accuracy = held_metrics["accuracy"]
            best_accuracy_epoch = epoch
            save_checkpoint(
                output / "best_accuracy.pt", model, config, epoch, held_metrics
            )
            np.savez_compressed(
                output / "best_accuracy_held_logits.npz",
                labels=held_labels,
                predictions=held_predictions,
                logits=held_logits,
                sample_ids=np.asarray(
                    [sample.sample_id for sample in held_dataset.samples]
                ),
                subjects=np.asarray(
                    [sample.user_id for sample in held_dataset.samples]
                ),
                outer_held_predictions_generated=np.asarray(False),
            )
        if held_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = held_metrics["macro_f1"]
            best_macro_f1_epoch = epoch
            save_checkpoint(
                output / "best_macro_f1.pt", model, config, epoch, held_metrics
            )
        print(
            f"epoch={epoch:02d} train={train_metrics['accuracy']:.4f} "
            f"held={held_metrics['accuracy']:.4f} "
            f"bal={held_metrics['balanced_accuracy']:.4f} "
            f"f1={held_metrics['macro_f1']:.4f}",
            flush=True,
        )
    archive = np.load(output / "best_accuracy_held_logits.npz", allow_pickle=False)
    summary = {
        "protocol": config["protocol"],
        "outer_fold": int(config["outer_fold"]),
        "inner_fold": int(config["inner_fold"]),
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "train_samples": len(train_dataset),
        "held_samples": len(held_dataset),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "fp16_parameter_mib": float(
            sum(parameter.numel() for parameter in model.parameters()) * 2 / 1024**2
        ),
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_macro_f1_epoch": best_macro_f1_epoch,
        "best_accuracy_metrics": metric_bundle(
            archive["labels"], archive["predictions"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
