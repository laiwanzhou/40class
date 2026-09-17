from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from p46_event_data import P46EventDataset, collate_p46_events
from p46_protocol import HARD_CLASS_IDS, HARD_CLASS_TO_INDEX
from p46_step10_model import cross_subject_supervised_contrastive
from p46r_event_bottleneck_model import (
    P46R_OFFSET_FRACTIONS,
    P46REventModel,
    circular_shift_visual_batch,
    event_localization_loss,
    parameter_count,
)
from train_p46_step10 import (
    FrameBudgetBatchSampler,
    FullCoverageCrossSubjectBatchSampler,
    atomic_checkpoint,
    atomic_json,
    batch_coverage_report,
    move_batch,
    seed_everything,
    source_coverage_report,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_RUN = PROJECT_DIR / "runs" / "p46_event_inputs_full"
DEFAULT_CONTEXT_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46r_mechanism_short"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P46-R mechanism training with an ordered local-event bottleneck."
    )
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--context-run", type=Path, default=DEFAULT_CONTEXT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--offset-eval-every",
        type=int,
        default=1,
        help="Evaluate all five offsets every N epochs; use 0 to disable during training.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--offset-gate-summary",
        type=Path,
        default=PROJECT_DIR / "runs" / "p46r_offset_overfit_gate_v1" / "summary.json",
        help="Passed fixed-batch learnability gate required before formal training.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--frame-budget", type=int, default=1024)
    parser.add_argument("--eval-frame-budget", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument(
        "--preload-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the local-only P46-R tensors once into RAM before repeated epochs.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--offset-weight", type=float, default=0.50)
    parser.add_argument("--localization-weight", type=float, default=0.15)
    parser.add_argument("--contrast-weight", type=float, default=0.05)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="BF16 is the stable default; FP16 previously produced an infinite gradient norm.",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--skip-mechanism-audit",
        action="store_true",
        help="Development-only numeric/throughput scan; never used for a route decision.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def select_smoke_ids(rows: list[dict[str, str]], per_class: int) -> set[str]:
    selected: set[str] = set()
    count: Counter[int] = Counter()
    for row in rows:
        class_id = int(row["class_id"])
        if count[class_id] >= per_class:
            continue
        selected.add(row["source_id"])
        count[class_id] += 1
    return selected


def make_datasets(args: argparse.Namespace) -> tuple[P46EventDataset, P46EventDataset]:
    train_full = P46EventDataset(
        args.event_run, args.context_run, split="train", load_context=False
    )
    val_full = P46EventDataset(
        args.event_run, args.context_run, split="val", load_context=False
    )
    if not args.smoke:
        return train_full, val_full
    train_ids = select_smoke_ids(train_full.rows, per_class=3)
    val_ids = select_smoke_ids(val_full.rows, per_class=2)
    return (
        P46EventDataset(
            args.event_run,
            args.context_run,
            split="train",
            sample_ids=train_ids,
            load_context=False,
        ),
        P46EventDataset(
            args.event_run,
            args.context_run,
            split="val",
            sample_ids=val_ids,
            load_context=False,
        ),
    )


class PreloadedP46RDataset(Dataset[dict[str, Any]]):
    """RAM-backed view that preserves the P46 dataset/sampler contract."""

    def __init__(self, source: P46EventDataset) -> None:
        started = time.perf_counter()
        self.rows = source.rows
        self.items = [source[index] for index in range(len(source))]
        self.preload_seconds = time.perf_counter() - started

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    @property
    def frame_lengths(self) -> list[int]:
        return [len(item["frame_ids"]) for item in self.items]


def make_loaders(
    train: P46EventDataset | PreloadedP46RDataset,
    val: P46EventDataset | PreloadedP46RDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, FullCoverageCrossSubjectBatchSampler]:
    train_labels = [HARD_CLASS_TO_INDEX[int(row["class_id"])] for row in train.rows]
    train_users = [row["user_id"] for row in train.rows]
    train_ids = [row["source_id"] for row in train.rows]
    train_sampler = FullCoverageCrossSubjectBatchSampler(
        train.frame_lengths,
        train_labels,
        train_users,
        train_ids,
        maximum_batch_size=args.batch_size,
        seed=args.seed,
        frame_budget=args.frame_budget,
    )
    val_sampler = FrameBudgetBatchSampler(
        val.frame_lengths,
        maximum_batch_size=args.eval_batch_size,
        frame_budget=args.eval_frame_budget,
    )
    common = {
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda") and args.workers > 0,
        "persistent_workers": args.workers > 0,
        "collate_fn": partial(collate_p46_events, include_context=False),
    }
    return (
        DataLoader(train, batch_sampler=train_sampler, **common),
        DataLoader(val, batch_sampler=val_sampler, **common),
        train_sampler,
    )


def class_weights(
    dataset: P46EventDataset | PreloadedP46RDataset, power: float
) -> torch.Tensor:
    labels = torch.tensor(
        [HARD_CLASS_TO_INDEX[int(row["class_id"])] for row in dataset.rows],
        dtype=torch.long,
    )
    count = torch.bincount(labels, minlength=len(HARD_CLASS_IDS)).float().clamp_min(1.0)
    weight = count.pow(-power)
    return weight / weight.mean()


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def cosine_lr(epoch: int, epochs: int, maximum: float, minimum: float) -> float:
    if epochs <= 1:
        return maximum
    progress = (epoch - 1) / (epochs - 1)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    return torch.optim.AdamW(
        (
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=args.learning_rate,
    )


def train_epoch(
    model: P46REventModel,
    loader: DataLoader,
    sampler: FullCoverageCrossSubjectBatchSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    weights: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    progress_path: Path | None = None,
    failure_path: Path | None = None,
) -> dict[str, Any]:
    model.train()
    sampler.set_epoch(epoch)
    totals: Counter[str] = Counter()
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    observed_ids: list[str] = []
    samples = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index, raw_batch in enumerate(loader):
        observed_ids.extend(raw_batch["source_id"])
        batch = move_batch(raw_batch, device)
        labels = batch["detail_index"]
        batch_size = len(labels)
        offset_labels = (
            torch.arange(batch_size, device=device) + batch_index + epoch
        ).remainder(len(P46R_OFFSET_FRACTIONS))
        shifted, _ = circular_shift_visual_batch(batch, offset_labels)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda" and args.amp_dtype != "float32"
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(batch)
            shifted_output = model(shifted)
            classification = F.cross_entropy(
                output["detail_logits"],
                labels,
                weight=weights,
                label_smoothing=args.label_smoothing,
            )
            offset = F.cross_entropy(shifted_output["offset_logits"], offset_labels)
            localization = event_localization_loss(output)
            subjects = torch.tensor(
                [hash(value) for value in batch["user_id"]],
                dtype=torch.long,
                device=device,
            )
            contrast = cross_subject_supervised_contrastive(
                output["contrast_embedding"], labels, subjects
            )
            total = (
                classification
                + args.offset_weight * offset
                + args.localization_weight * localization
                + args.contrast_weight * contrast
            )
        finite_losses = {
            "total": bool(torch.isfinite(total).item()),
            "classification": bool(torch.isfinite(classification).item()),
            "offset": bool(torch.isfinite(offset).item()),
            "localization": bool(torch.isfinite(localization).item()),
            "contrast": bool(torch.isfinite(contrast).item()),
        }
        if not all(finite_losses.values()):
            failure = {
                "stage": "non_finite_loss",
                "epoch": epoch,
                "batch": batch_index + 1,
                "source_ids": list(batch["source_id"]),
                "finite_losses": finite_losses,
                "amp_dtype": args.amp_dtype,
            }
            if failure_path is not None:
                atomic_json(failure_path, failure)
            raise FloatingPointError(json.dumps(failure, ensure_ascii=False))
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not torch.isfinite(gradient):
            bad_parameters = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            failure = {
                "stage": "non_finite_gradient",
                "epoch": epoch,
                "batch": batch_index + 1,
                "source_ids": list(batch["source_id"]),
                "gradient_norm": float(gradient.detach()),
                "bad_parameters": bad_parameters,
                "amp_dtype": args.amp_dtype,
            }
            if failure_path is not None:
                atomic_json(failure_path, failure)
            raise FloatingPointError(json.dumps(failure, ensure_ascii=False))
        scaler.step(optimizer)
        scaler.update()

        logged = {
            "total": total,
            "classification": classification,
            "offset": offset,
            "localization": localization,
            "contrast": contrast,
            "gradient_norm": gradient,
            "offset_accuracy": (
                shifted_output["offset_logits"].argmax(1) == offset_labels
            ).float().mean(),
        }
        for key, value in logged.items():
            totals[key] += float(value.detach()) * batch_size
        labels_all.append(labels.detach().cpu())
        logits_all.append(output["detail_logits"].detach().float().cpu())
        samples += batch_size
        if args.log_every and (batch_index + 1) % args.log_every == 0:
            if progress_path is not None:
                atomic_json(
                    progress_path,
                    {
                        "stage": "training",
                        "epoch": epoch,
                        "batch": batch_index + 1,
                        "batches": len(loader),
                        "samples": samples,
                        "expected_samples": len(loader.dataset),
                        "elapsed_seconds": time.perf_counter() - started,
                        "mean_loss": totals["total"] / samples,
                        "mean_offset_accuracy": totals["offset_accuracy"] / samples,
                    },
                )
            print(
                f"train epoch={epoch} batch={batch_index + 1}/{len(loader)} "
                f"samples={samples} loss={totals['total']/samples:.4f} "
                f"offset_acc={totals['offset_accuracy']/samples:.3f}",
                flush=True,
            )

    coverage = source_coverage_report(
        observed_ids, [row["source_id"] for row in loader.dataset.rows]
    )
    if not coverage["exact_once"]:
        raise RuntimeError(f"P46-R epoch coverage failed: {coverage}")
    label_array = torch.cat(labels_all).numpy()
    logit_array = torch.cat(logits_all).numpy()
    return {
        "losses": {key: value / samples for key, value in totals.items()},
        "metrics": metrics(label_array, logit_array),
        "coverage": coverage,
        "seconds": time.perf_counter() - started,
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
    }


@torch.inference_mode()
def evaluate(
    model: P46REventModel,
    loader: DataLoader,
    device: torch.device,
    *,
    fixed_offset_index: int | None = None,
    visual_scale: float = 1.0,
    visual_difference_scale: float = 1.0,
    skeleton_scale: float = 1.0,
    imu_scale: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    source_ids: list[str] = []
    users: list[str] = []
    losses = 0.0
    samples = 0
    started = time.perf_counter()
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if fixed_offset_index is not None:
            index = torch.full(
                (len(batch["detail_index"]),), fixed_offset_index, device=device, dtype=torch.long
            )
            batch, _ = circular_shift_visual_batch(batch, index)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(
                batch,
                visual_scale=visual_scale,
                visual_difference_scale=visual_difference_scale,
                skeleton_scale=skeleton_scale,
                imu_scale=imu_scale,
            )
            loss = F.cross_entropy(output["detail_logits"], batch["detail_index"])
        batch_size = len(batch["detail_index"])
        losses += float(loss) * batch_size
        samples += batch_size
        labels.append(batch["detail_index"].cpu())
        logits.append(output["detail_logits"].float().cpu())
        source_ids.extend(batch["source_id"])
        users.extend(batch["user_id"])
    label_array = torch.cat(labels).numpy()
    logit_array = torch.cat(logits).numpy()
    return {
        "loss": losses / samples,
        "metrics": metrics(label_array, logit_array),
        "labels": label_array,
        "logits": logit_array,
        "source_ids": source_ids,
        "users": users,
        "samples": samples,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_offsets(
    model: P46REventModel, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    users: list[str] = []
    source_ids: list[str] = []
    started = time.perf_counter()
    for raw_batch in loader:
        base = move_batch(raw_batch, device)
        for offset_index in range(len(P46R_OFFSET_FRACTIONS)):
            target = torch.full(
                (len(base["detail_index"]),), offset_index, device=device, dtype=torch.long
            )
            shifted, _ = circular_shift_visual_batch(base, target)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(shifted)
            labels.append(target.cpu())
            logits.append(output["offset_logits"].float().cpu())
            users.extend(base["user_id"])
            source_ids.extend(base["source_id"])
    label_array = torch.cat(labels).numpy()
    logit_array = torch.cat(logits).numpy()
    prediction = logit_array.argmax(1)
    per_offset = {
        str(index): float((prediction[label_array == index] == index).mean())
        for index in range(len(P46R_OFFSET_FRACTIONS))
    }
    return {
        "accuracy": float((prediction == label_array).mean()),
        "macro_f1": float(f1_score(label_array, prediction, average="macro", zero_division=0)),
        "per_offset_accuracy": per_offset,
        "labels": label_array,
        "logits": logit_array,
        "users": users,
        "source_ids": source_ids,
        "seconds": time.perf_counter() - started,
    }


def comparison(
    baseline: dict[str, Any], condition: dict[str, Any]
) -> dict[str, Any]:
    if baseline["source_ids"] != condition["source_ids"]:
        raise RuntimeError("condition prediction order differs from baseline")
    labels = baseline["labels"]
    base_prediction = baseline["logits"].argmax(1)
    condition_prediction = condition["logits"].argmax(1)
    base_correct = base_prediction == labels
    condition_correct = condition_prediction == labels
    result = {
        **condition["metrics"],
        "accuracy_delta_pp": 100.0
        * (condition["metrics"]["accuracy"] - baseline["metrics"]["accuracy"]),
        "balanced_accuracy_delta_pp": 100.0
        * (
            condition["metrics"]["balanced_accuracy"]
            - baseline["metrics"]["balanced_accuracy"]
        ),
        "macro_f1_delta_pp": 100.0
        * (condition["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"]),
        "prediction_changed": int((base_prediction != condition_prediction).sum()),
        "baseline_correct_to_wrong": int((base_correct & ~condition_correct).sum()),
        "baseline_wrong_to_correct": int((~base_correct & condition_correct).sum()),
    }
    per_subject: dict[str, Any] = {}
    user_array = np.asarray(baseline["users"])
    for user in sorted(set(baseline["users"])):
        mask = user_array == user
        base_accuracy = float(base_correct[mask].mean())
        condition_accuracy = float(condition_correct[mask].mean())
        per_subject[user] = {
            "samples": int(mask.sum()),
            "baseline_accuracy": base_accuracy,
            "condition_accuracy": condition_accuracy,
            "accuracy_delta_pp": 100.0 * (condition_accuracy - base_accuracy),
        }
    result["per_subject"] = per_subject
    return result


def save_predictions(output: Path, evaluation: dict[str, Any], name: str) -> None:
    prediction = evaluation["logits"].argmax(1)
    rows = []
    for index, source_id in enumerate(evaluation["source_ids"]):
        rows.append(
            {
                "source_id": source_id,
                "user_id": evaluation["users"][index],
                "true_class_id": HARD_CLASS_IDS[int(evaluation["labels"][index])],
                "predicted_class_id": HARD_CLASS_IDS[int(prediction[index])],
                "correct": int(prediction[index] == evaluation["labels"][index]),
            }
        )
    write_csv(output / f"{name}_predictions.csv", rows)


def run_mechanism_audit(
    model: P46REventModel,
    loader: DataLoader,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    progress_path = output / "progress.json"
    atomic_json(progress_path, {"stage": "mechanism_audit", "condition": "baseline"})
    baseline = evaluate(model, loader, device)
    condition_specs: dict[str, dict[str, Any]] = {
        "visual_shift_minus_quarter": {"fixed_offset_index": 0},
        "visual_shift_minus_eighth": {"fixed_offset_index": 1},
        "visual_shift_plus_eighth": {"fixed_offset_index": 3},
        "visual_shift_plus_quarter": {"fixed_offset_index": 4},
        "visual_difference_removed": {"visual_difference_scale": 0.0},
        "all_local_visual_removed": {"visual_scale": 0.0},
        "skeleton_removed": {"skeleton_scale": 0.0},
        "imu_removed": {"imu_scale": 0.0},
    }
    conditions: dict[str, Any] = {}
    for position, (name, keyword) in enumerate(condition_specs.items(), start=1):
        atomic_json(
            progress_path,
            {
                "stage": "mechanism_audit",
                "condition": name,
                "condition_index": position,
                "condition_count": len(condition_specs) + 1,
            },
        )
        conditions[name] = evaluate(model, loader, device, **keyword)
    atomic_json(
        progress_path,
        {
            "stage": "mechanism_audit",
            "condition": "five_offset_full_evaluation",
            "condition_index": len(condition_specs) + 1,
            "condition_count": len(condition_specs) + 1,
        },
    )
    offset = evaluate_offsets(model, loader, device)
    comparisons = {name: comparison(baseline, value) for name, value in conditions.items()}
    shift_names = [name for name in comparisons if name.startswith("visual_shift_")]
    mean_shift_delta = float(
        np.mean([comparisons[name]["accuracy_delta_pp"] for name in shift_names])
    )
    subject_shift_means = {
        user: float(
            np.mean(
                [comparisons[name]["per_subject"][user]["accuracy_delta_pp"] for name in shift_names]
            )
        )
        for user in sorted(set(baseline["users"]))
    }
    mechanism_gate = {
        "offset_accuracy_at_least_45pct": offset["accuracy"] >= 0.45,
        "mean_visual_shift_drop_at_least_3pp": mean_shift_delta <= -3.0,
        "visual_difference_removal_hurts": comparisons["visual_difference_removed"][
            "accuracy_delta_pp"
        ]
        < 0.0,
        "all_local_visual_removal_hurts": comparisons["all_local_visual_removed"][
            "accuracy_delta_pp"
        ]
        < 0.0,
        "shift_not_single_subject_only": sum(value < 0.0 for value in subject_shift_means.values())
        >= 2,
    }
    summary = {
        "baseline": baseline["metrics"],
        "offset": {
            key: value
            for key, value in offset.items()
            if key not in {"labels", "logits", "users", "source_ids"}
        },
        "conditions": comparisons,
        "mean_visual_shift_accuracy_delta_pp": mean_shift_delta,
        "mean_visual_shift_delta_by_subject_pp": subject_shift_means,
        "mechanism_gate": mechanism_gate,
        "mechanism_gate_pass": all(mechanism_gate.values()),
        "classification_comparison": {
            "p46_accuracy": 0.4068965517241379,
            "p12_restricted_accuracy": 0.45517241379310347,
            "above_p46": baseline["metrics"]["accuracy"] > 0.4068965517241379,
            "above_p12": baseline["metrics"]["accuracy"] > 0.45517241379310347,
        },
    }
    if not summary["mechanism_gate_pass"]:
        summary["route_status"] = "mechanism_not_stable"
    elif summary["classification_comparison"]["above_p12"]:
        summary["route_status"] = "repairs_p46_and_exceeds_p12"
    elif summary["classification_comparison"]["above_p46"]:
        summary["route_status"] = "repairs_p46_but_remains_below_p12"
    else:
        summary["route_status"] = "mechanism_learned_without_p46_accuracy_gain"
    summary["requires_manual_route_decision"] = True
    save_predictions(output, baseline, "mechanism_baseline")
    atomic_json(output / "mechanism_audit.json", summary)
    rows = [
        {"condition": "baseline", **baseline["metrics"], "accuracy_delta_pp": 0.0},
        *[
            {"condition": name, **value}
            for name, value in comparisons.items()
        ],
    ]
    write_csv(output / "mechanism_metrics.csv", rows)
    return summary


def preflight(
    model: P46REventModel,
    train: P46EventDataset | PreloadedP46RDataset,
    val: P46EventDataset | PreloadedP46RDataset,
    train_loader: DataLoader,
    sampler: FullCoverageCrossSubjectBatchSampler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    epoch_reports = []
    for epoch in range(1, max(2, args.epochs) + 1):
        sampler.set_epoch(epoch)
        report = batch_coverage_report(list(iter(sampler)), len(train))
        epoch_reports.append({"epoch": epoch, **report})
        if not report["exact_once"]:
            raise RuntimeError(f"sampler preflight failed at epoch {epoch}: {report}")
    raw = next(iter(train_loader))
    batch = move_batch(raw, device)
    labels = torch.arange(len(batch["detail_index"]), device=device).remainder(
        len(P46R_OFFSET_FRACTIONS)
    )
    shifted, shifts = circular_shift_visual_batch(batch, labels)
    model.train()
    amp_enabled = device.type == "cuda" and args.amp_dtype != "float32"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        output = model(batch)
        shifted_output = model(shifted)
        loss = (
            F.cross_entropy(output["detail_logits"], batch["detail_index"])
            + 0.5 * F.cross_entropy(shifted_output["offset_logits"], labels)
            + 0.15 * event_localization_loss(output)
        )
    loss.backward()
    finite_output = all(
        torch.isfinite(value).all().item()
        for value in output.values()
        if isinstance(value, torch.Tensor)
    )
    finite_gradient = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not finite_output or not finite_gradient:
        raise RuntimeError("P46-R preflight produced a non-finite tensor or gradient")
    model.zero_grad(set_to_none=True)
    return {
        "stage": "P46-R_preflight",
        "train_trials": len(train),
        "val_trials": len(val),
        "train_subjects": sorted({row["user_id"] for row in train.rows}),
        "val_subjects": sorted({row["user_id"] for row in val.rows}),
        "subject_overlap": sorted(
            {row["user_id"] for row in train.rows}
            & {row["user_id"] for row in val.rows}
        ),
        "sampler_epochs": epoch_reports,
        "forward_finite": finite_output,
        "gradient_finite": finite_gradient,
        "first_batch_size": len(batch["detail_index"]),
        "first_batch_frames": int(batch["frame_mask"].sum().item()),
        "example_shift_frames": shifts.cpu().tolist(),
        "model_parameters": parameter_count(model),
        "model_fp32_mib": parameter_count(model) * 4 / 1024**2,
        "event_bottleneck_has_global_pool_bypass": False,
        "old_object_contact_phase_heads_used": False,
    }


def main() -> None:
    args = parse_args()
    if args.torch_threads < 1 or args.interop_threads < 1:
        raise ValueError("torch and interop thread counts must be positive")
    if args.epochs < 1 or args.minimum_epochs < 1:
        raise ValueError("epochs and minimum-epochs must be positive")
    if args.minimum_epochs > args.epochs and not args.smoke:
        raise ValueError("minimum-epochs exceeds epochs")
    if args.patience < 1 or args.offset_eval_every < 0:
        raise ValueError("patience must be positive and offset-eval-every cannot be negative")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.interop_threads)
    requested_workers = args.workers
    workers_forced_to_zero = False
    if os.name == "nt" and args.workers > 0:
        # P46/P30 items are large per-trial NPZ arrays. On this Windows runtime,
        # multiprocessing DataLoader workers repeatedly stall while transferring
        # those arrays. A measured single-process batch takes ~0.15 s, whereas a
        # 20-batch workers=2 read did not finish within 40 s.
        print(
            f"forcing --workers {args.workers} to 0 on Windows for P46-R NPZ loading",
            flush=True,
        )
        args.workers = 0
        workers_forced_to_zero = True
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.minimum_epochs = 1
        args.patience = 2
        args.workers = 0
    seed_everything(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    offset_gate: dict[str, Any] | None = None
    if not args.preflight_only:
        gate_path = args.offset_gate_summary.resolve()
        if not gate_path.is_file():
            raise FileNotFoundError(f"P46-R offset learnability gate is missing: {gate_path}")
        offset_gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not bool(offset_gate.get("passed")):
            raise RuntimeError(f"P46-R offset learnability gate did not pass: {gate_path}")
    train, val = make_datasets(args)
    preload_info: dict[str, Any] = {
        "enabled": False,
        "train_seconds": 0.0,
        "val_seconds": 0.0,
    }
    if args.preload_dataset and not args.preflight_only:
        print(
            f"preloading P46-R local tensors: train={len(train)}, val={len(val)}",
            flush=True,
        )
        train = PreloadedP46RDataset(train)
        val = PreloadedP46RDataset(val)
        preload_info = {
            "enabled": True,
            "train_seconds": train.preload_seconds,
            "val_seconds": val.preload_seconds,
        }
        print(
            "preload completed: "
            f"train={train.preload_seconds:.2f}s, val={val.preload_seconds:.2f}s",
            flush=True,
        )
    train_loader, val_loader, sampler = make_loaders(train, val, args)
    model = P46REventModel(width=args.width, dropout=args.dropout).to(device)
    preflight_result = preflight(
        model, train, val, train_loader, sampler, device, args
    )
    atomic_json(output / "preflight.json", preflight_result)
    if args.preflight_only:
        print(json.dumps(preflight_result, ensure_ascii=False, indent=2), flush=True)
        return

    # Preflight intentionally exercises backward; start the experiment from a
    # fresh random initialisation so the audit does not alter the training state.
    seed_everything(args.seed)
    model = P46REventModel(width=args.width, dropout=args.dropout).to(device)
    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.amp_dtype == "float16",
    )
    weights = class_weights(train, args.class_weight_power).to(device)
    config = {
        "stage": "P46-R_full_mechanism_training",
        "event_run": str(args.event_run.resolve()),
        "context_run": str(args.context_run.resolve()),
        "train_trials": len(train),
        "val_trials": len(val),
        "train_subjects": sorted({row["user_id"] for row in train.rows}),
        "val_subjects": sorted({row["user_id"] for row in val.rows}),
        "epochs": args.epochs,
        "minimum_epochs": args.minimum_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "offset_eval_every": args.offset_eval_every,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "frame_budget": args.frame_budget,
        "eval_frame_budget": args.eval_frame_budget,
        "workers": args.workers,
        "requested_workers": requested_workers,
        "workers_forced_to_zero_on_windows": workers_forced_to_zero,
        "torch_threads": args.torch_threads,
        "interop_threads": args.interop_threads,
        "pin_memory": args.device.startswith("cuda") and args.workers > 0,
        "width": args.width,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weight_power": args.class_weight_power,
        "loss_weights": {
            "detail21": 1.0,
            "offset": args.offset_weight,
            "localization": args.localization_weight,
            "cross_subject_contrast": args.contrast_weight,
        },
        "amp_dtype": args.amp_dtype,
        "offset_fractions": list(P46R_OFFSET_FRACTIONS),
        "offset_learnability_gate": {
            "path": str(args.offset_gate_summary.resolve()),
            "passed": bool(offset_gate and offset_gate.get("passed")),
            "final_accuracy": (
                offset_gate.get("final", {}).get("accuracy") if offset_gate else None
            ),
            "final_loss": offset_gate.get("final", {}).get("loss") if offset_gate else None,
        },
        "sampler": "all unique trials exactly once per epoch; no replacement",
        "all_frames": True,
        "context_loaded": False,
        "context_reason": "P46-R has no global/context bypass; avoid unused P30 cache I/O",
        "dataset_preloaded_to_ram": preload_info["enabled"],
        "train_preload_seconds": preload_info["train_seconds"],
        "val_preload_seconds": preload_info["val_seconds"],
        "old_object_contact_phase_heads_used": False,
        "global_pool_bypass_used": False,
        "model_parameters": parameter_count(model),
        "model_fp32_mib": parameter_count(model) * 4 / 1024**2,
        "seed": args.seed,
        "smoke": args.smoke,
        "device": str(device),
    }
    atomic_json(output / "frozen_config.json", config)

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_accuracy = -math.inf
    best_accuracy_epoch = 0
    best_macro_f1 = -math.inf
    best_macro_f1_epoch = 0
    best_offset_accuracy = -math.inf
    best_offset_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    stopped_early = False
    if args.resume is not None:
        resume_path = args.resume.resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "scaler_state_dict",
            "history",
            "epoch",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"resume checkpoint lacks training state {missing}: {resume_path}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = list(checkpoint["history"])
        best_score = float(checkpoint.get("best_score", -math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_accuracy = float(checkpoint.get("best_accuracy", -math.inf))
        best_accuracy_epoch = int(checkpoint.get("best_accuracy_epoch", 0))
        best_macro_f1 = float(checkpoint.get("best_macro_f1", -math.inf))
        best_macro_f1_epoch = int(checkpoint.get("best_macro_f1_epoch", 0))
        best_offset_accuracy = float(checkpoint.get("best_offset_accuracy", -math.inf))
        best_offset_epoch = int(checkpoint.get("best_offset_epoch", 0))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"resuming P46-R from epoch {start_epoch}: {resume_path}", flush=True)
    experiment_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        learning_rate = cosine_lr(
            epoch, args.epochs, args.learning_rate, args.minimum_learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        training = train_epoch(
            model,
            train_loader,
            sampler,
            optimizer,
            scaler,
            device,
            weights,
            args,
            epoch,
            output / "progress.json",
            output / "numeric_failure.json",
        )
        validation = evaluate(model, val_loader, device)
        offset_validation: dict[str, Any] | None = None
        if args.offset_eval_every and (
            epoch == 1 or epoch % args.offset_eval_every == 0 or epoch == args.epochs
        ):
            offset_validation = evaluate_offsets(model, val_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            **{f"train_{key}": value for key, value in training["losses"].items()},
            **{f"train_{key}": value for key, value in training["metrics"].items()},
            **{f"val_{key}": value for key, value in validation["metrics"].items()},
            "val_loss": validation["loss"],
            "train_seconds": training["seconds"],
            "val_seconds": validation["seconds"],
            "peak_cuda_mib": training["peak_cuda_mib"],
            "coverage_exact_once": training["coverage"]["exact_once"],
        }
        if offset_validation is not None:
            row.update(
                {
                    "val_offset_accuracy": offset_validation["accuracy"],
                    "val_offset_macro_f1": offset_validation["macro_f1"],
                    "val_offset_seconds": offset_validation["seconds"],
                    **{
                        f"val_offset_{key}_accuracy": value
                        for key, value in offset_validation["per_offset_accuracy"].items()
                    },
                }
            )
        history.append(row)
        write_csv(output / "history.csv", history)
        score = validation["metrics"]["macro_f1"] + 0.25 * validation["metrics"]["accuracy"]
        accuracy_improved = (
            validation["metrics"]["accuracy"] > best_accuracy + args.min_delta
        )
        macro_improved = (
            validation["metrics"]["macro_f1"] > best_macro_f1 + args.min_delta
        )
        offset_improved = bool(
            offset_validation is not None
            and offset_validation["accuracy"] > best_offset_accuracy + args.min_delta
        )
        offset_metrics = (
            {
                "accuracy": offset_validation["accuracy"],
                "macro_f1": offset_validation["macro_f1"],
                "per_offset_accuracy": offset_validation["per_offset_accuracy"],
            }
            if offset_validation is not None
            else None
        )
        best_payload = {
            "stage": "P46-R_full_mechanism_training",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config,
            "validation_metrics": validation["metrics"],
            "validation_offset_metrics": offset_metrics,
        }
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            atomic_checkpoint(output / "best.pt", best_payload)
            save_predictions(output, validation, "best")
        if accuracy_improved:
            best_accuracy = validation["metrics"]["accuracy"]
            best_accuracy_epoch = epoch
            atomic_checkpoint(output / "best_accuracy.pt", best_payload)
            save_predictions(output, validation, "best_accuracy")
        if macro_improved:
            best_macro_f1 = validation["metrics"]["macro_f1"]
            best_macro_f1_epoch = epoch
            atomic_checkpoint(output / "best_macro_f1.pt", best_payload)
            save_predictions(output, validation, "best_macro_f1")
        if offset_improved:
            best_offset_accuracy = float(offset_validation["accuracy"])
            best_offset_epoch = epoch
            atomic_checkpoint(output / "best_offset.pt", best_payload)
        stale_epochs = 0 if (accuracy_improved or macro_improved or offset_improved) else stale_epochs + 1
        atomic_json(
            output / "progress.json",
            {
                "stage": "epoch_completed",
                "epoch": epoch,
                "epochs": args.epochs,
                "train_seconds": training["seconds"],
                "val_seconds": validation["seconds"],
                "val_accuracy": validation["metrics"]["accuracy"],
                "val_macro_f1": validation["metrics"]["macro_f1"],
                "val_offset_accuracy": (
                    offset_validation["accuracy"] if offset_validation is not None else None
                ),
                "stale_epochs": stale_epochs,
            },
        )
        atomic_checkpoint(
            output / "last.pt",
            {
                **best_payload,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "best_accuracy_epoch": best_accuracy_epoch,
                "best_macro_f1": best_macro_f1,
                "best_macro_f1_epoch": best_macro_f1_epoch,
                "best_offset_accuracy": best_offset_accuracy,
                "best_offset_epoch": best_offset_epoch,
                "stale_epochs": stale_epochs,
            },
        )
        offset_text = (
            f"{offset_validation['accuracy']:.4f}"
            if offset_validation is not None
            else "not_evaluated"
        )
        epoch_seconds = (
            training["seconds"]
            + validation["seconds"]
            + (offset_validation["seconds"] if offset_validation is not None else 0.0)
        )
        print(
            f"epoch={epoch}/{args.epochs} train_acc={training['metrics']['accuracy']:.4f} "
            f"val_acc={validation['metrics']['accuracy']:.4f} "
            f"val_macro={validation['metrics']['macro_f1']:.4f} "
            f"val_offset={offset_text} "
            f"stale={stale_epochs}/{args.patience} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale_epochs >= args.patience:
            stopped_early = True
            atomic_json(
                output / "progress.json",
                {
                    "stage": "early_stop",
                    "epoch": epoch,
                    "stale_epochs": stale_epochs,
                    "reason": (
                        "validation accuracy, macro-F1 and offset accuracy did not improve "
                        f"by {args.min_delta} for {args.patience} epochs"
                    ),
                },
            )
            break

    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    if args.skip_mechanism_audit:
        summary = {
            "stage": "P46-R_full_mechanism_training",
            "best_epoch": best_epoch,
            "best_validation_metrics": checkpoint["validation_metrics"],
            "mechanism_audit_performed": False,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "best_accuracy_epoch": best_accuracy_epoch,
            "best_accuracy": best_accuracy,
            "best_macro_f1_epoch": best_macro_f1_epoch,
            "best_macro_f1": best_macro_f1,
            "best_offset_epoch": best_offset_epoch,
            "best_offset_accuracy": best_offset_accuracy,
            "all_completed_epochs_exact_once": all(
                bool(row["coverage_exact_once"]) for row in history
            ),
            "elapsed_seconds": time.perf_counter() - experiment_started,
        }
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    audit = run_mechanism_audit(model, val_loader, device, output)
    summary = {
        "stage": "P46-R_full_mechanism_training",
        "best_epoch": best_epoch,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "mechanism_audit": audit,
        "route_status": audit["route_status"],
        "requires_manual_route_decision": True,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_accuracy": best_accuracy,
        "best_macro_f1_epoch": best_macro_f1_epoch,
        "best_macro_f1": best_macro_f1,
        "best_offset_epoch": best_offset_epoch,
        "best_offset_accuracy": best_offset_accuracy,
        "all_completed_epochs_exact_once": all(
            bool(row["coverage_exact_once"]) for row in history
        ),
        "elapsed_seconds": time.perf_counter() - experiment_started,
        "artifacts": {
            "checkpoint": str((output / "best.pt").resolve()),
            "last_checkpoint": str((output / "last.pt").resolve()),
            "history": str((output / "history.csv").resolve()),
            "mechanism_audit": str((output / "mechanism_audit.json").resolve()),
        },
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
