from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from p32_fused_data import (
    BalancedLengthBucketBatchSampler,
    LengthBucketBatchSampler,
    P32FusedTrialDataset,
    collate_p32_trials,
)
from p32_part_fusion_temporal_model import (
    P32PartFusionTemporalModel,
    model_size_mib,
    parameter_count,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VISUAL = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_MOTION = PROJECT_DIR / "runs" / "p31_skeleton_imu_full"
DEFAULT_FOLD = PROJECT_DIR / "data" / "subject_folds" / "fold_0.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p32_step15_fold0"
DEFAULT_P12_SUMMARY = PROJECT_DIR / "runs" / "p12_complete_oof" / "summary.json"
DEFAULT_HARD_PROTOCOL = PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"

SMALL_IDS = np.asarray(
    [1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39],
    dtype=np.int64,
)
DEFAULT_HARD_IDS = np.asarray(
    [7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 35, 37, 38, 39],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P32 Step 15: formal fold-0 full-sequence 40-class training"
    )
    parser.add_argument("--visual-run", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--motion-run", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--fold-csv", type=Path, default=DEFAULT_FOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def source_id(row: dict[str, str]) -> str:
    return f"{row['class_name']}/{row['user_id']}/{row['trial_id']}"


def fold_sample_ids(path: Path) -> dict[str, set[str]]:
    selected = {"train": set(), "val": set()}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split in selected:
                selected[split].add(source_id(row))
    return selected


def make_loaders(
    train_dataset: P32FusedTrialDataset,
    val_dataset: P32FusedTrialDataset,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, BalancedLengthBucketBatchSampler, DataLoader, LengthBucketBatchSampler]:
    train_sampler = BalancedLengthBucketBatchSampler(
        train_dataset.frame_lengths,
        [int(row["class_id"]) for row in train_dataset.rows],
        batch_size=batch_size,
        seed=seed,
    )
    val_sampler = LengthBucketBatchSampler(
        val_dataset.frame_lengths,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "collate_fn": collate_p32_trials,
    }
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **common)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **common)
    return train_loader, train_sampler, val_loader, val_sampler


def learning_rate_for_epoch(
    epoch: int,
    max_epochs: int,
    warmup_epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_learning_rate * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_learning_rate + (base_learning_rate - minimum_learning_rate) * cosine


def grouped_metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    if len(labels) == 0:
        return {"samples": 0, "accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def metric_bundle(
    labels: np.ndarray,
    predictions: np.ndarray,
    users: list[str],
    hard_ids: np.ndarray,
) -> dict[str, Any]:
    small_mask = np.isin(labels, SMALL_IDS)
    hard_mask = np.isin(labels, hard_ids)
    per_user: dict[str, Any] = {}
    user_array = np.asarray(users)
    for user in sorted(set(users)):
        mask = user_array == user
        per_user[user] = grouped_metric(labels[mask], predictions[mask])
    per_class = []
    for class_id in range(40):
        true_mask = labels == class_id
        predicted_mask = predictions == class_id
        true_positive = int((true_mask & predicted_mask).sum())
        support = int(true_mask.sum())
        predicted = int(predicted_mask.sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "support": support,
                "predicted": predicted,
                "correct": true_positive,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "is_small": bool(class_id in set(SMALL_IDS.tolist())),
                "is_hard": bool(class_id in set(hard_ids.tolist())),
            }
        )
    return {
        "overall": grouped_metric(labels, predictions),
        "small": grouped_metric(labels[small_mask], predictions[small_mask]),
        "hard": grouped_metric(labels[hard_mask], predictions[hard_mask]),
        "per_user": per_user,
        "per_class": per_class,
    }


def train_epoch(
    model: P32PartFusionTemporalModel,
    head: nn.Module,
    loader: DataLoader,
    sampler: BalancedLengthBucketBatchSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    label_smoothing: float,
    log_every: int,
) -> dict[str, float | int]:
    sampler.set_epoch(epoch)
    model.train()
    head.train()
    started = time.perf_counter()
    previous_end = started
    loader_wait = 0.0
    total_loss = 0.0
    total_correct = 0
    trials = 0
    real_frames = 0
    padded_frames = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader, 1):
        arrived = time.perf_counter()
        loader_wait += arrived - previous_end
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type in {"cuda", "cpu"},
        ):
            logits = head(model(batch)["trial_embedding"])
            loss = nn.functional.cross_entropy(
                logits, batch["label"], label_smoothing=label_smoothing
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
        scaler.step(optimizer)
        scaler.update()
        batch_trials = len(batch["label"])
        total_loss += float(loss.detach()) * batch_trials
        total_correct += int((logits.argmax(dim=1) == batch["label"]).sum().item())
        trials += batch_trials
        real_frames += int(batch["frame_mask"].sum().item())
        padded_frames += int(batch["frame_mask"].numel())
        previous_end = time.perf_counter()
        if log_every > 0 and batch_index % log_every == 0:
            print(
                json.dumps(
                    {
                        "stage": "train_batch",
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": len(loader),
                        "elapsed_seconds": round(previous_end - started, 2),
                        "last_batch_T": int(batch["frame_mask"].shape[1]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "loss": total_loss / max(trials, 1),
        "sampled_accuracy": total_correct / max(trials, 1),
        "trials": trials,
        "real_frames": real_frames,
        "padding_efficiency": real_frames / max(padded_frames, 1),
        "seconds": seconds,
        "loader_wait_seconds": loader_wait,
        "trials_per_second": trials / max(seconds, 1e-6),
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
    }


@torch.inference_mode()
def evaluate(
    model: P32PartFusionTemporalModel,
    head: nn.Module,
    loader: DataLoader,
    sampler: LengthBucketBatchSampler,
    device: torch.device,
    hard_ids: np.ndarray,
) -> dict[str, Any]:
    sampler.set_epoch(0)
    model.eval()
    head.eval()
    started = time.perf_counter()
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    users: list[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type in {"cuda", "cpu"},
        ):
            logits = head(model(batch)["trial_embedding"])
            loss = nn.functional.cross_entropy(logits, batch["label"])
        losses.extend([float(loss)] * len(batch["label"]))
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        users.extend(batch["user_id"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(np.int64)
    predictions = logits.argmax(axis=1).astype(np.int64)
    return {
        "loss": float(np.mean(losses)),
        "seconds": time.perf_counter() - started,
        "metrics": metric_bundle(labels, predictions, users, hard_ids),
        "logits": logits,
        "labels": labels,
        "predictions": predictions,
        "sample_ids": sample_ids,
        "users": users,
    }


def checkpoint_payload(
    model: P32PartFusionTemporalModel,
    head: nn.Module,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "P32_Step15_inference_only_v1",
        "fold": 0,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
    }


def write_predictions(output: Path, prefix: str, evaluation: dict[str, Any]) -> None:
    csv_path = output / f"{prefix}_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "user_id", "label", "prediction", "correct"),
        )
        writer.writeheader()
        for sample_id, user, label, prediction in zip(
            evaluation["sample_ids"],
            evaluation["users"],
            evaluation["labels"].tolist(),
            evaluation["predictions"].tolist(),
        ):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "user_id": user,
                    "label": label,
                    "prediction": prediction,
                    "correct": int(label == prediction),
                }
            )
    np.savez_compressed(
        output / f"{prefix}_logits.npz",
        sample_ids=np.asarray(evaluation["sample_ids"]),
        users=np.asarray(evaluation["users"]),
        labels=evaluation["labels"],
        predictions=evaluation["predictions"],
        logits=evaluation["logits"].astype(np.float16),
    )
    atomic_json(output / f"{prefix}_metrics.json", evaluation["metrics"])


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "epoch",
            "learning_rate",
            "train_loss",
            "train_sampled_accuracy",
            "train_seconds",
            "val_loss",
            "val_accuracy",
            "val_balanced_accuracy",
            "val_macro_f1",
            "val_small_accuracy",
            "val_hard_accuracy",
            "val_seconds",
            "best_accuracy_epoch",
            "best_macro_f1_epoch",
            "stale_epochs",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row[key] for key in fieldnames})


def load_reference(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    fold = value["metrics"]["thermal_routed_final"]["per_fold"]["0"]
    return {
        "accuracy": float(fold["accuracy"]),
        "balanced_accuracy": float(fold["balanced_accuracy"]),
        "macro_f1": float(fold["macro_f1"]),
        "oof_accuracy": float(value["metrics"]["thermal_routed_final"]["all"]["accuracy"]),
    }


def main() -> None:
    args = parse_args()
    if args.min_epochs > args.max_epochs:
        raise ValueError("min-epochs cannot exceed max-epochs")
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Sequence lengths vary from batch to batch. cuDNN benchmarking would
        # re-run algorithm search for many distinct T values during epoch 1.
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = fold_sample_ids(args.fold_csv.resolve())
    train_dataset = P32FusedTrialDataset(
        args.visual_run.resolve(), args.motion_run.resolve(), sample_ids=selected["train"]
    )
    val_dataset = P32FusedTrialDataset(
        args.visual_run.resolve(), args.motion_run.resolve(), sample_ids=selected["val"]
    )
    if len(train_dataset) != len(selected["train"]) or len(val_dataset) != len(selected["val"]):
        raise RuntimeError(
            f"fold/cache mismatch: train {len(train_dataset)}/{len(selected['train'])}, "
            f"val {len(val_dataset)}/{len(selected['val'])}"
        )
    if set(row["user_id"] for row in train_dataset.rows) & set(
        row["user_id"] for row in val_dataset.rows
    ):
        raise RuntimeError("fold0 is not subject-disjoint")

    hard_ids = DEFAULT_HARD_IDS
    if DEFAULT_HARD_PROTOCOL.exists():
        hard_protocol = json.loads(DEFAULT_HARD_PROTOCOL.read_text(encoding="utf-8"))
        hard_ids = np.asarray(hard_protocol["hard_class_ids"], dtype=np.int64)
    reference = load_reference(DEFAULT_P12_SUMMARY)
    train_loader, train_sampler, val_loader, val_sampler = make_loaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.eval_batch_size,
        args.workers,
        args.seed,
    )
    device = torch.device(args.device)
    model = P32PartFusionTemporalModel().to(device)
    head = nn.Sequential(nn.LayerNorm(384), nn.Dropout(0.15), nn.Linear(384, 40)).to(device)
    parameters = list(model.parameters()) + list(head.parameters())
    decay = [parameter for parameter in parameters if parameter.requires_grad and parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters if parameter.requires_grad and parameter.ndim < 2]
    optimizer_kwargs: dict[str, Any] = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        # The automatic CUDA foreach path stalled indefinitely once during the
        # formal Windows run.  The scalar-tensor update path is mathematically
        # equivalent and avoids depending on that backend optimisation.
        "foreach": False,
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        **optimizer_kwargs,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    class_counts = np.bincount(
        np.asarray([int(row["class_id"]) for row in train_dataset.rows]), minlength=40
    )
    config = {
        "stage": "P32_Step15_fold0_only",
        "fold": 0,
        "fold_csv": str(args.fold_csv.resolve()),
        "train_subjects": sorted({row["user_id"] for row in train_dataset.rows}),
        "val_subjects": sorted({row["user_id"] for row in val_dataset.rows}),
        "train_trials": len(train_dataset),
        "val_trials": len(val_dataset),
        "train_real_frames": int(sum(train_dataset.frame_lengths)),
        "val_real_frames": int(sum(val_dataset.frame_lengths)),
        "train_class_counts": class_counts.tolist(),
        "class_balancing": "uniform class then uniform trial with replacement; 1941 samples/epoch",
        "all_original_frames_preserved_within_every_sampled_trial": True,
        "imu_interval_pooling": "vectorized exact Bx5xT segmented reduction",
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "early_stopping": "stop only when neither accuracy nor macro-F1 improves for patience epochs",
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "optimizer": "AdamW foreach=False; no weight decay for bias/norm vectors",
        "mixed_precision": device.type == "cuda",
        "seed": args.seed,
        "model_parameters": parameter_count(model) + parameter_count(head),
        "model_fp32_mib": model_size_mib(model, 4) + model_size_mib(head, 4),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "reference_p12_fold0": reference,
        "small_class_ids": SMALL_IDS.tolist(),
        "hard_class_ids": hard_ids.tolist(),
    }
    atomic_json(output / "frozen_config.json", config)
    print(json.dumps({"stage": "start", **config}, ensure_ascii=False), flush=True)

    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_accuracy_epoch = 0
    best_macro_f1_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    stopped_early = False
    run_started = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        learning_rate = learning_rate_for_epoch(
            epoch,
            args.max_epochs,
            args.warmup_epochs,
            args.learning_rate,
            args.minimum_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train = train_epoch(
            model,
            head,
            train_loader,
            train_sampler,
            optimizer,
            scaler,
            device,
            epoch,
            args.label_smoothing,
            args.log_every,
        )
        evaluation = evaluate(model, head, val_loader, val_sampler, device, hard_ids)
        metrics = evaluation["metrics"]
        accuracy = float(metrics["overall"]["accuracy"])
        macro_f1 = float(metrics["overall"]["macro_f1"])
        accuracy_improved = accuracy > best_accuracy + args.min_delta
        macro_improved = macro_f1 > best_macro_f1 + args.min_delta

        if accuracy_improved:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch
            atomic_checkpoint(
                output / "best_accuracy.pt",
                checkpoint_payload(model, head, epoch, metrics, config),
            )
            write_predictions(output, "best_accuracy", evaluation)
        if macro_improved:
            best_macro_f1 = macro_f1
            best_macro_f1_epoch = epoch
            atomic_checkpoint(
                output / "best_macro_f1.pt",
                checkpoint_payload(model, head, epoch, metrics, config),
            )
            write_predictions(output, "best_macro_f1", evaluation)

        stale_epochs = 0 if (accuracy_improved or macro_improved) else stale_epochs + 1
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train["loss"],
            "train_sampled_accuracy": train["sampled_accuracy"],
            "train_seconds": train["seconds"],
            "val_loss": evaluation["loss"],
            "val_accuracy": accuracy,
            "val_balanced_accuracy": metrics["overall"]["balanced_accuracy"],
            "val_macro_f1": macro_f1,
            "val_small_accuracy": metrics["small"]["accuracy"],
            "val_hard_accuracy": metrics["hard"]["accuracy"],
            "val_seconds": evaluation["seconds"],
            "best_accuracy_epoch": best_accuracy_epoch,
            "best_macro_f1_epoch": best_macro_f1_epoch,
            "stale_epochs": stale_epochs,
        }
        history.append(row)
        atomic_json(output / "history.json", history)
        write_history_csv(output / "history.csv", history)
        print(json.dumps({"stage": "epoch", **row}, ensure_ascii=False), flush=True)

        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            stopped_early = True
            print(
                json.dumps(
                    {
                        "stage": "early_stop",
                        "epoch": epoch,
                        "reason": f"neither accuracy nor macro-F1 improved by {args.min_delta} for {args.patience} epochs",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    best_accuracy_metrics = json.loads(
        (output / "best_accuracy_metrics.json").read_text(encoding="utf-8")
    )
    best_macro_metrics = json.loads(
        (output / "best_macro_f1_metrics.json").read_text(encoding="utf-8")
    )
    summary = {
        "stage": "P32_Step15_fold0_complete",
        "folds_run": [0],
        "folds_1_and_2_run": False,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "elapsed_seconds": time.perf_counter() - run_started,
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_accuracy_metrics": best_accuracy_metrics,
        "best_macro_f1_epoch": best_macro_f1_epoch,
        "best_macro_f1_metrics": best_macro_metrics,
        "reference_p12_fold0": reference,
        "best_accuracy_delta_vs_p12_fold0_pp": 100.0
        * (best_accuracy_metrics["overall"]["accuracy"] - reference["accuracy"]),
        "best_accuracy_delta_vs_p12_oof_pp": 100.0
        * (best_accuracy_metrics["overall"]["accuracy"] - reference["oof_accuracy"]),
        "breaks_previous_fold0_accuracy": bool(
            best_accuracy_metrics["overall"]["accuracy"] > reference["accuracy"]
        ),
        "best_checkpoints": {
            "accuracy": str((output / "best_accuracy.pt").resolve()),
            "macro_f1": str((output / "best_macro_f1.pt").resolve()),
        },
        "checkpoint_contains_optimizer": False,
        "checkpoint_scope": "P32 model plus 40-class head only; P30/YOLO extraction weights remain external",
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
