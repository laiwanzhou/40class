from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from p32_fused_data import (
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
DEFAULT_FOLDS = PROJECT_DIR / "data" / "subject_folds"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p32_steps13_14_short_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short mixed-precision timing run for P32 Steps 13/14 with a disposable CE head"
    )
    parser.add_argument("--visual-run", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--motion-run", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--trials-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--formal-epochs", type=int, nargs="+", default=(20, 30, 40))
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-repeats", type=int, default=2)
    return parser.parse_args()


def stratified_length_sample(
    rows: list[dict[str, str]], trials_per_class: int
) -> set[str]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(row)
    selected: set[str] = set()
    for class_id in sorted(grouped):
        candidates = sorted(grouped[class_id], key=lambda row: int(row["frames"]))
        count = min(trials_per_class, len(candidates))
        if count == len(candidates):
            chosen = candidates
        else:
            positions = np.linspace(0, len(candidates) - 1, count)
            chosen = [candidates[int(round(position))] for position in positions]
        selected.update(row["sample_id"] for row in chosen)
    return selected


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_loader(
    dataset: P32FusedTrialDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> tuple[DataLoader, LengthBucketBatchSampler]:
    sampler = LengthBucketBatchSampler(
        dataset.frame_lengths,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_p32_trials,
    )
    return loader, sampler


def run_train_epoch(
    model: P32PartFusionTemporalModel,
    head: nn.Module,
    loader: DataLoader,
    sampler: LengthBucketBatchSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    log_every: int,
) -> dict[str, float]:
    sampler.set_epoch(epoch)
    model.train()
    head.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    previous_end = started
    loader_wait = 0.0
    total_loss = 0.0
    total_correct = 0
    total_trials = 0
    real_frames = 0
    padded_frames = 0
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
            embedding = model(batch)["trial_embedding"]
            logits = head(embedding)
            loss = nn.functional.cross_entropy(
                logits, batch["label"], label_smoothing=0.05
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_size = len(batch["label"])
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == batch["label"]).sum().item())
        total_trials += batch_size
        real_frames += int(batch["frame_mask"].sum().item())
        padded_frames += int(batch["frame_mask"].numel())
        previous_end = time.perf_counter()
        if log_every > 0 and batch_index % log_every == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "batch": batch_index,
                        "batches": len(loader),
                        "elapsed_seconds": previous_end - started,
                        "last_batch_T": int(batch["frame_mask"].shape[1]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return {
        "epoch": epoch + 1,
        "seconds": elapsed,
        "loader_wait_seconds": loader_wait,
        "compute_and_transfer_seconds": elapsed - loader_wait,
        "trials": total_trials,
        "real_frames": real_frames,
        "padded_frames": padded_frames,
        "padding_efficiency": real_frames / max(padded_frames, 1),
        "loss": total_loss / max(total_trials, 1),
        "accuracy_timing_only": total_correct / max(total_trials, 1),
        "trials_per_second": total_trials / max(elapsed, 1e-6),
        "real_frames_per_second": real_frames / max(elapsed, 1e-6),
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }


@torch.inference_mode()
def run_eval_epoch(
    model: P32PartFusionTemporalModel,
    head: nn.Module,
    loader: DataLoader,
    sampler: LengthBucketBatchSampler,
    device: torch.device,
) -> dict[str, float]:
    sampler.set_epoch(0)
    model.eval()
    head.eval()
    started = time.perf_counter()
    previous_end = started
    loader_wait = 0.0
    total_trials = 0
    real_frames = 0
    padded_frames = 0
    for batch in loader:
        arrived = time.perf_counter()
        loader_wait += arrived - previous_end
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type in {"cuda", "cpu"},
        ):
            head(model(batch)["trial_embedding"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_trials += len(batch["label"])
        real_frames += int(batch["frame_mask"].sum().item())
        padded_frames += int(batch["frame_mask"].numel())
        previous_end = time.perf_counter()
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "loader_wait_seconds": loader_wait,
        "trials": total_trials,
        "real_frames": real_frames,
        "padded_frames": padded_frames,
        "padding_efficiency": real_frames / max(padded_frames, 1),
        "trials_per_second": total_trials / max(elapsed, 1e-6),
        "real_frames_per_second": real_frames / max(elapsed, 1e-6),
    }


def fold_workloads(
    fold_dir: Path,
    available: dict[str, int],
) -> list[dict[str, int]]:
    workloads: list[dict[str, int]] = []
    for fold in range(3):
        path = fold_dir / f"fold_{fold}.csv"
        counts = {"train_trials": 0, "train_frames": 0, "val_trials": 0, "val_frames": 0}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                source_id = f"{row['class_name']}/{row['user_id']}/{row['trial_id']}"
                if source_id not in available:
                    continue
                split = row["split"]
                counts[f"{split}_trials"] += 1
                counts[f"{split}_frames"] += available[source_id]
        workloads.append({"fold": fold, **counts})
    return workloads


def estimate_formal_time(
    train_epochs: list[dict[str, float]],
    evaluation: dict[str, float],
    workloads: list[dict[str, int]],
    formal_epochs: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    steady_epochs = train_epochs[1:] if len(train_epochs) > 1 else train_epochs
    train_seconds_per_frame = float(
        np.median(
            [row["seconds"] / max(row["real_frames"], 1) for row in steady_epochs]
        )
    )
    eval_seconds_per_frame = evaluation["seconds"] / max(evaluation["real_frames"], 1)
    one_epoch_by_fold = []
    for workload in workloads:
        seconds = (
            workload["train_frames"] * train_seconds_per_frame
            + workload["val_frames"] * eval_seconds_per_frame
        )
        one_epoch_by_fold.append({"fold": workload["fold"], "seconds": seconds})
    estimates = {}
    total_one_epoch = sum(row["seconds"] for row in one_epoch_by_fold)
    for epochs in formal_epochs:
        seconds = total_one_epoch * int(epochs)
        estimates[str(epochs)] = {
            "seconds": seconds,
            "hours": seconds / 3600.0,
            "with_20_percent_margin_hours": seconds * 1.2 / 3600.0,
        }
    return {
        "method": (
            "median warm training seconds per real frame plus measured eval seconds per real "
            "frame, multiplied by exact available frame counts in the three fixed folds"
        ),
        "train_seconds_per_real_frame": train_seconds_per_frame,
        "eval_seconds_per_real_frame": eval_seconds_per_frame,
        "one_epoch_all_three_folds": one_epoch_by_fold,
        "estimates": estimates,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    full_dataset = P32FusedTrialDataset(
        args.visual_run.resolve(), args.motion_run.resolve()
    )
    selected_ids = stratified_length_sample(full_dataset.rows, args.trials_per_class)
    dataset = P32FusedTrialDataset(
        args.visual_run.resolve(), args.motion_run.resolve(), sample_ids=selected_ids
    )
    loader, sampler = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    eval_loader, eval_sampler = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    device = torch.device(args.device)
    model = P32PartFusionTemporalModel().to(device)
    head = nn.Linear(384, 40).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.learning_rate,
        weight_decay=0.02,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    train_epochs: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        metrics = run_train_epoch(
            model,
            head,
            loader,
            sampler,
            optimizer,
            scaler,
            device,
            epoch,
            args.log_every,
        )
        train_epochs.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    evaluation_passes = []
    for evaluation_index in range(args.eval_repeats):
        evaluation_passes.append(
            run_eval_epoch(model, head, eval_loader, eval_sampler, device)
        )
        print(
            json.dumps(
                {
                    "evaluation_pass": evaluation_index + 1,
                    "metrics": evaluation_passes[-1],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    evaluation = evaluation_passes[-1]

    available = {
        row["sample_id"]: int(row["frames"]) for row in full_dataset.rows
    }
    workloads = fold_workloads(args.fold_dir.resolve(), available)
    timing = estimate_formal_time(
        train_epochs, evaluation, workloads, args.formal_epochs
    )
    result = {
        "stage": "steps_13_14_short_timing_only",
        "not_a_formal_accuracy_experiment": True,
        "no_checkpoint_saved": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "selected_trials": len(dataset),
        "selected_classes": len({int(row["class_id"]) for row in dataset.rows}),
        "selected_real_frames": sum(dataset.frame_lengths),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "mixed_precision": device.type == "cuda",
        "model_parameters_without_disposable_head": parameter_count(model),
        "model_fp32_mib_without_disposable_head": model_size_mib(model, 4),
        "model_fp16_mib_without_disposable_head": model_size_mib(model, 2),
        "train_epochs": train_epochs,
        "evaluation_passes": evaluation_passes,
        "evaluation": evaluation,
        "fold_workloads": workloads,
        "formal_time_estimate": timing,
    }
    (output / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
