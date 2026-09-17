from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Sampler

from p46_event_data import P46EventDataset, collate_p46_events
from p46_protocol import EXPECTED, HARD_CLASS_IDS, HARD_CLASS_TO_INDEX
from p46_step10_model import (
    P46Step10Model,
    contact_and_phase_losses,
    cross_subject_supervised_contrastive,
    hardest_rival_loss,
    left_right_swap_batch,
    mask_modalities,
    modality_summary_targets,
    parameter_count,
    same_part_alignment_loss,
    selected_modality_reconstruction_loss,
    temporal_order_loss,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_RUN = PROJECT_DIR / "runs" / "p46_event_inputs_full"
DEFAULT_CONTEXT_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_step10_detail21_fullcoverage"
DEFAULT_P12_OOF = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P46 Step 10: weak event pretraining then supervised Detail21 fine-tuning."
    )
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--context-run", type=Path, default=DEFAULT_CONTEXT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12_OOF)
    parser.add_argument(
        "--stage-a-checkpoint",
        type=Path,
        default=None,
        help="Resume directly at Stage B from a completed Stage A checkpoint.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--frame-budget", type=int, default=1024)
    parser.add_argument("--eval-frame-budget", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage-a-epochs", type=int, default=4)
    parser.add_argument("--stage-b-max-epochs", type=int, default=30)
    parser.add_argument("--stage-b-min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--stage-a-learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage-b-learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Power applied to inverse class frequency; 0 disables class weighting.",
    )
    parser.add_argument("--flip-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Audit the exact train/validation coverage contract and exit before model training.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    torch.save(value, temporary)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    observed: set[str] = set()
    for row in rows:
        for key in row:
            if key not in observed:
                observed.add(key)
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def batch_coverage_report(batches: list[list[int]], expected_samples: int) -> dict[str, Any]:
    """Return an auditable exact-once coverage report for a list of index batches."""

    flat = [index for batch in batches for index in batch]
    counts = Counter(flat)
    missing = [index for index in range(expected_samples) if counts[index] == 0]
    duplicated = {index: count for index, count in counts.items() if count > 1}
    out_of_range = sorted(
        index for index in counts if index < 0 or index >= expected_samples
    )
    return {
        "expected_samples": expected_samples,
        "total_positions": len(flat),
        "unique_samples": len(
            {index for index in flat if 0 <= index < expected_samples}
        ),
        "missing_count": len(missing),
        "duplicate_positions": sum(count - 1 for count in duplicated.values()),
        "out_of_range_count": len(out_of_range),
        "max_repeat_count": max(counts.values(), default=0),
        "missing_indices_preview": missing[:20],
        "duplicated_indices_preview": list(sorted(duplicated.items()))[:20],
        "out_of_range_indices_preview": out_of_range[:20],
        "exact_once": (
            len(flat) == expected_samples
            and len(counts) == expected_samples
            and not missing
            and not duplicated
            and not out_of_range
        ),
    }


def source_coverage_report(
    observed_source_ids: list[str], expected_source_ids: list[str]
) -> dict[str, Any]:
    """Audit what the DataLoader actually delivered, independently of the sampler."""

    if len(set(expected_source_ids)) != len(expected_source_ids):
        raise ValueError("expected source IDs are not unique")
    counts = Counter(observed_source_ids)
    expected = set(expected_source_ids)
    missing = sorted(expected - counts.keys())
    unknown = sorted(counts.keys() - expected)
    duplicated = {key: count for key, count in counts.items() if count > 1}
    return {
        "expected_samples": len(expected_source_ids),
        "total_positions": len(observed_source_ids),
        "unique_samples": len(counts.keys() & expected),
        "missing_count": len(missing),
        "duplicate_positions": sum(count - 1 for count in duplicated.values()),
        "unknown_count": len(unknown),
        "max_repeat_count": max(counts.values(), default=0),
        "missing_source_ids_preview": missing[:20],
        "duplicated_source_ids_preview": list(sorted(duplicated.items()))[:20],
        "unknown_source_ids_preview": unknown[:20],
        "exact_once": (
            len(observed_source_ids) == len(expected_source_ids)
            and counts.keys() == expected
            and not duplicated
            and not unknown
        ),
    }


class FullCoverageCrossSubjectBatchSampler(Sampler[list[int]]):
    """Use every trial exactly once while retaining feasible cross-subject pairs.

    Pairing changes only batch order. It never creates replacement draws and never
    drops a trial. Length bucketing controls padded BxT cost after the exact-once
    epoch permutation has been constructed.
    """

    def __init__(
        self,
        lengths: list[int],
        labels: list[int],
        users: list[str],
        source_ids: list[str],
        maximum_batch_size: int,
        seed: int,
        frame_budget: int = 1024,
        bucket_multiplier: int = 12,
    ) -> None:
        if maximum_batch_size < 1:
            raise ValueError("maximum batch size must be positive")
        if not (len(lengths) == len(labels) == len(users) == len(source_ids)):
            raise ValueError("sampler inputs differ in length")
        self.lengths = [int(value) for value in lengths]
        self.labels = [int(value) for value in labels]
        self.users = list(users)
        self.source_ids = list(source_ids)
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("sampler source IDs are not unique")
        if any(length < 1 for length in self.lengths):
            raise ValueError("all frame lengths must be positive")
        self.maximum_batch_size = int(maximum_batch_size)
        self.seed = int(seed)
        self.frame_budget = int(frame_budget)
        self.bucket_units = max(1, bucket_multiplier * maximum_batch_size)
        self.epoch = 0
        self.by_class_user: dict[int, dict[str, list[int]]] = {}
        for index, (label, user) in enumerate(zip(self.labels, self.users)):
            self.by_class_user.setdefault(label, {}).setdefault(user, []).append(index)
        self.classes = sorted(self.by_class_user)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._build_batches())

    def _build_units(self, generator: random.Random) -> list[tuple[int, ...]]:
        units: list[tuple[int, ...]] = []
        for label in self.classes:
            pools = {
                user: values.copy()
                for user, values in self.by_class_user[label].items()
            }
            for values in pools.values():
                generator.shuffle(values)
            while True:
                active = [user for user, values in pools.items() if values]
                if len(active) < 2:
                    break
                generator.shuffle(active)
                active.sort(key=lambda user: len(pools[user]), reverse=True)
                first_user, second_user = active[:2]
                first = pools[first_user].pop()
                second = pools[second_user].pop()
                pair = (first, second)
                pair_cost = 2 * max(self.lengths[index] for index in pair)
                if self.maximum_batch_size >= 2 and pair_cost <= self.frame_budget:
                    units.append(pair)
                else:
                    units.extend(((first,), (second,)))
            for values in pools.values():
                units.extend((index,) for index in values)
        generator.shuffle(units)
        return units

    def _build_batches(self) -> list[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        if max(self.lengths, default=0) > self.frame_budget:
            raise RuntimeError(
                "a single trial exceeds the frame budget; increase --frame-budget "
                "instead of silently dropping or truncating it"
            )
        units = self._build_units(generator)
        batches: list[list[int]] = []
        for start in range(0, len(units), self.bucket_units):
            bucket = units[start : start + self.bucket_units]
            bucket.sort(key=lambda unit: max(self.lengths[index] for index in unit))
            chosen: list[tuple[int, ...]] = []
            for unit in bucket:
                proposed = chosen + [unit]
                flat = [index for value in proposed for index in value]
                cost = len(flat) * max(self.lengths[index] for index in flat)
                if chosen and (
                    len(flat) > self.maximum_batch_size or cost > self.frame_budget
                ):
                    batches.append([index for value in chosen for index in value])
                    chosen = [unit]
                else:
                    chosen = proposed
            if chosen:
                batches.append([index for value in chosen for index in value])
        generator.shuffle(batches)
        coverage = batch_coverage_report(batches, len(self.lengths))
        if not coverage["exact_once"]:
            raise RuntimeError(f"full-coverage sampler contract failed: {coverage}")
        return batches

    def coverage_report(self) -> dict[str, Any]:
        batches = self._build_batches()
        report = batch_coverage_report(batches, len(self.lengths))
        positive_samples = 0
        for batch in batches:
            for index in batch:
                if any(
                    self.labels[other] == self.labels[index]
                    and self.users[other] != self.users[index]
                    for other in batch
                    if other != index
                ):
                    positive_samples += 1
        report.update(
            {
                "epoch": self.epoch,
                "batches": len(batches),
                "cross_subject_positive_samples": positive_samples,
                "cross_subject_positive_fraction": positive_samples
                / max(len(self.lengths), 1),
            }
        )
        return report

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches()


class FrameBudgetBatchSampler(Sampler[list[int]]):
    """Deterministic length buckets with a maximum padded BxT cost."""

    def __init__(
        self,
        lengths: list[int],
        maximum_batch_size: int,
        frame_budget: int,
        bucket_multiplier: int = 12,
    ) -> None:
        self.lengths = [int(value) for value in lengths]
        self.maximum_batch_size = int(maximum_batch_size)
        self.frame_budget = int(frame_budget)
        self.bucket_size = max(1, self.maximum_batch_size * bucket_multiplier)

    def __len__(self) -> int:
        return len(self._build_batches())

    def _build_batches(self) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            chosen: list[int] = []
            for index in bucket:
                proposed = chosen + [index]
                cost = len(proposed) * max(self.lengths[value] for value in proposed)
                if chosen and (
                    len(proposed) > self.maximum_batch_size or cost > self.frame_budget
                ):
                    batches.append(chosen)
                    chosen = [index]
                else:
                    chosen = proposed
            if chosen:
                batches.append(chosen)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches()


def make_loaders(
    train: P46EventDataset,
    val: P46EventDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, FullCoverageCrossSubjectBatchSampler, DataLoader, FrameBudgetBatchSampler]:
    train_sampler = FullCoverageCrossSubjectBatchSampler(
        train.frame_lengths,
        [HARD_CLASS_TO_INDEX[int(row["class_id"])] for row in train.rows],
        [row["user_id"] for row in train.rows],
        [row["source_id"] for row in train.rows],
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
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
        "collate_fn": collate_p46_events,
    }
    return (
        DataLoader(train, batch_sampler=train_sampler, **common),
        train_sampler,
        DataLoader(val, batch_sampler=val_sampler, **common),
        val_sampler,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_optimizer(
    model: torch.nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    decay = [value for value in model.parameters() if value.requires_grad and value.ndim >= 2]
    no_decay = [value for value in model.parameters() if value.requires_grad and value.ndim < 2]
    return torch.optim.AdamW(
        (
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )


def make_class_weights(
    labels: list[int], classes: int, power: float
) -> tuple[torch.Tensor, list[int]]:
    """Balance CE without replacing, duplicating, or dropping training trials."""

    if power < 0.0 or power > 1.0:
        raise ValueError("--class-weight-power must be in [0, 1]")
    counts = torch.bincount(torch.tensor(labels), minlength=classes).float()
    if torch.any(counts <= 0):
        raise RuntimeError(f"training split lacks a Detail21 class: {counts.tolist()}")
    weights = counts.pow(-power)
    weights = weights / weights.mean()
    return weights, [int(value) for value in counts.tolist()]


def cosine_lr(
    epoch: int, epochs: int, maximum: float, minimum: float, warmup: int = 1
) -> float:
    if epoch <= warmup:
        return maximum * epoch / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def event_structure_losses(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    contact, phase, coverage = contact_and_phase_losses(output, batch)
    return {
        "alignment": same_part_alignment_loss(output),
        "contact": contact,
        "phase": phase,
        "order": temporal_order_loss(output),
        "phase_coverage": coverage,
    }


def run_stage_a_epoch(
    model: P46Step10Model,
    loader: DataLoader,
    sampler: FullCoverageCrossSubjectBatchSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    flip_every: int,
    log_every: int,
    maximum_batches: int = 0,
) -> dict[str, float]:
    model.train()
    sampler.set_epoch(epoch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    totals = Counter()
    samples = 0
    observed_source_ids: list[str] = []
    started = time.perf_counter()
    for batch_index, raw_batch in enumerate(loader):
        if maximum_batches and batch_index >= maximum_batches:
            break
        observed_source_ids.extend(raw_batch["source_id"])
        batch = move_batch(raw_batch, device)
        batch_size = len(batch["detail_index"])
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            reconstruction_target = modality_summary_targets(batch).float()
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            full = model(batch)
            structural = event_structure_losses(full, batch)
            structural_total = (
                structural["alignment"]
                + 0.30 * structural["contact"]
                + 0.30 * structural["phase"]
                + 0.20 * structural["order"]
            )
        scaler.scale(structural_total).backward()

        assignment = (
            torch.arange(batch_size, device=device) + batch_index + epoch
        ).remainder(3)
        assignment = assignment[torch.randperm(batch_size, device=device)]
        masked = mask_modalities(batch, assignment)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            masked_output = model(masked)  # type: ignore[arg-type]
            reconstruction = selected_modality_reconstruction_loss(
                masked_output["modality_reconstruction"],
                reconstruction_target,
                assignment,
            )
        scaler.scale(0.50 * reconstruction).backward()

        flip = structural_total.detach() * 0.0
        if flip_every > 0 and (batch_index + 1) % flip_every == 0:
            swapped = left_right_swap_batch(batch)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                swapped_output = model(swapped)  # type: ignore[arg-type]
                flip = (
                    1.0
                    - F.cosine_similarity(
                        swapped_output["trial_embedding"].float(),
                        full["trial_embedding"].detach().float(),
                        dim=1,
                    )
                ).mean()
            scaler.scale(0.10 * flip).backward()

        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        logged = {
            "total": structural_total.detach() + 0.50 * reconstruction.detach() + 0.10 * flip.detach(),
            **{key: value.detach() for key, value in structural.items()},
            "reconstruction": reconstruction.detach(),
            "flip": flip.detach(),
            "gradient_norm": gradient.detach(),
        }
        for key, value in logged.items():
            totals[key] += float(value) * batch_size
        samples += batch_size
        if log_every and (batch_index + 1) % log_every == 0:
            print(
                json.dumps(
                    {
                        "stage": "A_batch",
                        "epoch": epoch,
                        "batch": batch_index + 1,
                        "batches": len(loader),
                        "mean_total": totals["total"] / max(samples, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    coverage = source_coverage_report(observed_source_ids, sampler.source_ids)
    if not maximum_batches and not coverage["exact_once"]:
        raise RuntimeError(f"Stage A epoch {epoch} DataLoader coverage failed: {coverage}")
    return {
        **{key: value / max(samples, 1) for key, value in totals.items()},
        "samples": float(samples),
        "coverage_total_positions": coverage["total_positions"],
        "coverage_unique_samples": coverage["unique_samples"],
        "coverage_missing_count": coverage["missing_count"],
        "coverage_duplicate_positions": coverage["duplicate_positions"],
        "coverage_exact_once": coverage["exact_once"],
        "seconds": time.perf_counter() - started,
        "gpu_peak_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
    }


def detail_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(len(HARD_CLASS_IDS)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def compute_p12_restricted_baseline(
    val: P46EventDataset, path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Align the fold-pure P12 OOF logits to the current frozen P46 validation set."""

    sample_ids = [row["sample_id"] for row in val.rows]
    class_labels = np.asarray([int(row["class_id"]) for row in val.rows], dtype=np.int64)
    detail_labels = np.asarray(
        [HARD_CLASS_TO_INDEX[int(value)] for value in class_labels], dtype=np.int64
    )
    with np.load(path, allow_pickle=False) as baseline:
        baseline_index = {
            str(value): index for index, value in enumerate(baseline["sample_ids"])
        }
        missing = [sample_id for sample_id in sample_ids if sample_id not in baseline_index]
        if missing:
            raise RuntimeError(f"P12 OOF lacks P46 validation samples: {missing[:10]}")
        indices = np.asarray([baseline_index[sample_id] for sample_id in sample_ids])
        oof_labels = baseline["labels"][indices].astype(np.int64)
        logits = baseline["sd_imu_logits"][indices][:, HARD_CLASS_IDS].astype(np.float32)
    if not np.array_equal(class_labels, oof_labels):
        raise RuntimeError("P12 OOF labels do not align with P46 validation labels")
    predictions = logits.argmax(1)
    metrics = detail_metrics(detail_labels, logits)
    summary: dict[str, Any] = {
        "protocol": (
            "P12 Skeleton+Depth+RF-IMU fold-pure OOF logits restricted to Detail21 "
            "on the frozen P46-v2 validation subjects"
        ),
        "oof_path": str(path.resolve()),
        "trials": len(val),
        "validation_subjects": sorted({row["user_id"] for row in val.rows}),
        **metrics,
        "correct": int((predictions == detail_labels).sum()),
    }
    rows = [
        {
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "user_id": row["user_id"],
            "true_class_id": int(row["class_id"]),
            "predicted_class_id": HARD_CLASS_IDS[int(predictions[index])],
            "correct": int(predictions[index] == detail_labels[index]),
        }
        for index, row in enumerate(val.rows)
    ]
    return summary, rows


def run_stage_b_train_epoch(
    model: P46Step10Model,
    loader: DataLoader,
    sampler: FullCoverageCrossSubjectBatchSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    user_to_index: dict[str, int],
    class_weights: torch.Tensor,
    label_smoothing: float,
    log_every: int,
    maximum_batches: int = 0,
) -> dict[str, Any]:
    model.train()
    sampler.set_epoch(1000 + epoch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    totals = Counter()
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    samples = 0
    observed_source_ids: list[str] = []
    started = time.perf_counter()
    for batch_index, raw_batch in enumerate(loader):
        if maximum_batches and batch_index >= maximum_batches:
            break
        observed_source_ids.extend(raw_batch["source_id"])
        batch = move_batch(raw_batch, device)
        labels = batch["detail_index"]
        subjects = torch.tensor(
            [user_to_index[value] for value in batch["user_id"]],
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(batch, subject_adversarial_scale=1.0)
            structure = event_structure_losses(output, batch)
            classification = F.cross_entropy(
                output["detail_logits"],
                labels,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
            rival = hardest_rival_loss(output["detail_logits"], labels)
            contrast = cross_subject_supervised_contrastive(
                output["contrast_embedding"], labels, subjects
            )
            subject = F.cross_entropy(output["subject_logits"], subjects)
            total = (
                classification
                + 0.15 * rival
                + 0.08 * contrast
                + 0.03 * subject
                + 0.05 * structure["alignment"]
                + 0.05 * structure["contact"]
                + 0.05 * structure["phase"]
                + 0.02 * structure["order"]
            )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(labels)
        logged = {
            "total": total,
            "classification": classification,
            "rival": rival,
            "contrast": contrast,
            "subject": subject,
            "alignment": structure["alignment"],
            "contact": structure["contact"],
            "phase": structure["phase"],
            "order": structure["order"],
            "phase_coverage": structure["phase_coverage"],
            "gradient_norm": gradient,
        }
        for key, value in logged.items():
            totals[key] += float(value.detach()) * batch_size
        samples += batch_size
        labels_all.append(labels.detach().cpu())
        logits_all.append(output["detail_logits"].detach().float().cpu())
        if log_every and (batch_index + 1) % log_every == 0:
            print(
                json.dumps(
                    {
                        "stage": "B_batch",
                        "epoch": epoch,
                        "batch": batch_index + 1,
                        "batches": len(loader),
                        "mean_total": totals["total"] / max(samples, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    coverage = source_coverage_report(observed_source_ids, sampler.source_ids)
    if not maximum_batches and not coverage["exact_once"]:
        raise RuntimeError(f"Stage B epoch {epoch} DataLoader coverage failed: {coverage}")
    labels_array = torch.cat(labels_all).numpy()
    logits_array = torch.cat(logits_all).numpy()
    return {
        "losses": {key: value / max(samples, 1) for key, value in totals.items()},
        "metrics": detail_metrics(labels_array, logits_array),
        "samples": samples,
        "coverage": coverage,
        "seconds": time.perf_counter() - started,
        "gpu_peak_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
    }


@torch.inference_mode()
def evaluate(
    model: P46Step10Model,
    loader: DataLoader,
    device: torch.device,
    maximum_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    source_ids: list[str] = []
    users: list[str] = []
    losses = 0.0
    samples = 0
    started = time.perf_counter()
    for batch_index, raw_batch in enumerate(loader):
        if maximum_batches and batch_index >= maximum_batches:
            break
        batch = move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(batch)
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
        "loss": losses / max(samples, 1),
        "metrics": detail_metrics(label_array, logit_array),
        "labels": label_array,
        "logits": logit_array,
        "source_ids": source_ids,
        "users": users,
        "samples": samples,
        "seconds": time.perf_counter() - started,
    }


def save_evaluation(
    output: Path,
    prefix: str,
    evaluation: dict[str, Any],
    epoch: int,
) -> None:
    prediction = evaluation["logits"].argmax(1)
    rows = []
    for index, source_id in enumerate(evaluation["source_ids"]):
        rows.append(
            {
                "source_id": source_id,
                "user_id": evaluation["users"][index],
                "true_detail_index": int(evaluation["labels"][index]),
                "true_class_id": HARD_CLASS_IDS[int(evaluation["labels"][index])],
                "predicted_detail_index": int(prediction[index]),
                "predicted_class_id": HARD_CLASS_IDS[int(prediction[index])],
                "correct": int(prediction[index] == evaluation["labels"][index]),
            }
        )
    write_csv(output / f"{prefix}_predictions.csv", rows)
    matrix = confusion_matrix(
        evaluation["labels"], prediction, labels=np.arange(len(HARD_CLASS_IDS))
    )
    matrix_rows = []
    for row_index, class_id in enumerate(HARD_CLASS_IDS):
        matrix_rows.append(
            {"true_class_id": class_id, **{str(value): int(matrix[row_index, column]) for column, value in enumerate(HARD_CLASS_IDS)}}
        )
    write_csv(output / f"{prefix}_confusion.csv", matrix_rows)
    atomic_json(
        output / f"{prefix}_metrics.json",
        {"epoch": epoch, **evaluation["metrics"], "loss": evaluation["loss"]},
    )


def checkpoint_payload(
    model: P46Step10Model,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "P46_step10_stageB_detail21",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "hard_class_ids": HARD_CLASS_IDS,
        "config": config,
    }


def main() -> None:
    args = parse_args()
    if args.stage_b_min_epochs > args.stage_b_max_epochs:
        raise ValueError("stage-b-min-epochs exceeds stage-b-max-epochs")
    if args.smoke:
        args.stage_a_epochs = 1
        args.stage_b_max_epochs = 2
        args.stage_b_min_epochs = 1
        args.patience = 2
    maximum_train_batches = 2 if args.smoke else 0
    maximum_val_batches = 2 if args.smoke else 0
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = P46EventDataset(args.event_run.resolve(), args.context_run.resolve(), split="train")
    val = P46EventDataset(args.event_run.resolve(), args.context_run.resolve(), split="val")
    if len(train) != EXPECTED["train_detail_trials"] or len(val) != EXPECTED["val_detail_trials"]:
        raise RuntimeError(f"P46 split changed: train={len(train)} val={len(val)}")
    train_users = sorted({row["user_id"] for row in train.rows})
    val_users = sorted({row["user_id"] for row in val.rows})
    if set(train_users) & set(val_users):
        raise RuntimeError("P46 Step10 subject leakage")
    train_source_ids = [row["source_id"] for row in train.rows]
    val_source_ids = [row["source_id"] for row in val.rows]
    if len(set(train_source_ids)) != len(train_source_ids):
        raise RuntimeError("P46 training source IDs are not unique")
    if len(set(val_source_ids)) != len(val_source_ids):
        raise RuntimeError("P46 validation source IDs are not unique")
    user_to_index = {value: index for index, value in enumerate(train_users)}
    train_loader, train_sampler, val_loader, val_sampler = make_loaders(train, val, args)
    train_labels = [
        HARD_CLASS_TO_INDEX[int(row["class_id"])] for row in train.rows
    ]
    class_weights_cpu, class_counts = make_class_weights(
        train_labels, len(HARD_CLASS_IDS), args.class_weight_power
    )
    p12_restricted_baseline, p12_prediction_rows = compute_p12_restricted_baseline(
        val, args.p12_oof.resolve()
    )
    atomic_json(output / "p12_restricted_baseline.json", p12_restricted_baseline)
    write_csv(output / "p12_restricted_predictions.csv", p12_prediction_rows)
    planned_sampler_epochs = list(range(1, args.stage_a_epochs + 1)) + list(
        range(1001, 1000 + args.stage_b_max_epochs + 1)
    )
    train_preflight = []
    for sampler_epoch in planned_sampler_epochs:
        train_sampler.set_epoch(sampler_epoch)
        report = train_sampler.coverage_report()
        if not report["exact_once"]:
            raise RuntimeError(f"preflight coverage failed: {report}")
        train_preflight.append(report)
    validation_batches = list(val_sampler)
    validation_preflight = batch_coverage_report(validation_batches, len(val))
    if not validation_preflight["exact_once"]:
        raise RuntimeError(f"validation coverage failed: {validation_preflight}")
    preflight = {
        "protocol": "full_coverage_exact_once_v2",
        "formal_training_started": False,
        "train_trials": len(train),
        "validation_trials": len(val),
        "audited_train_epochs": len(train_preflight),
        "all_train_epochs_exact_once": all(
            report["exact_once"] for report in train_preflight
        ),
        "minimum_train_unique_samples": min(
            (report["unique_samples"] for report in train_preflight), default=0
        ),
        "maximum_train_duplicate_positions": max(
            (report["duplicate_positions"] for report in train_preflight), default=0
        ),
        "maximum_train_missing_count": max(
            (report["missing_count"] for report in train_preflight), default=0
        ),
        "train_epoch_reports": train_preflight,
        "validation_report": validation_preflight,
        "class_counts": {
            str(class_id): class_counts[index]
            for index, class_id in enumerate(HARD_CLASS_IDS)
        },
        "class_weights": {
            str(class_id): float(class_weights_cpu[index])
            for index, class_id in enumerate(HARD_CLASS_IDS)
        },
        "class_weight_power": args.class_weight_power,
        "p12_restricted_baseline": p12_restricted_baseline,
    }
    atomic_json(output / "sampler_preflight.json", preflight)
    print(json.dumps({"stage": "sampler_preflight", **preflight}, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return
    train_sampler.set_epoch(0)
    device = torch.device(args.device)
    model = P46Step10Model(subjects=len(train_users)).to(device)
    class_weights = class_weights_cpu.to(device)
    stage_a_epochs_completed = 0
    if args.stage_a_checkpoint is not None:
        stage_a_checkpoint = torch.load(
            args.stage_a_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        if stage_a_checkpoint.get("stage") != "P46_step10_stageA":
            raise RuntimeError("--stage-a-checkpoint is not a P46 Stage A checkpoint")
        if (
            stage_a_checkpoint.get("config", {}).get("sampler_protocol")
            != "full_coverage_exact_once_v2"
        ):
            raise RuntimeError(
                "refusing a Stage A checkpoint created by the old replacement sampler"
            )
        model.load_state_dict(stage_a_checkpoint["model_state_dict"], strict=True)
        stage_a_epochs_completed = int(stage_a_checkpoint["epoch"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = {
        "stage": "P46_step10",
        "stage10_scope": "A weak event pretraining + B supervised Detail21 probe",
        "step11_expert_or_base_routing_started": False,
        "event_run": str(args.event_run.resolve()),
        "context_run": str(args.context_run.resolve()),
        "stage_a_checkpoint": (
            str(args.stage_a_checkpoint.resolve())
            if args.stage_a_checkpoint is not None
            else None
        ),
        "train_trials": len(train),
        "val_trials": len(val),
        "train_frames": sum(train.frame_lengths),
        "val_frames": sum(val.frame_lengths),
        "train_subjects": train_users,
        "val_subjects": val_users,
        "subject_overlap": [],
        "hard_class_ids": list(HARD_CLASS_IDS),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "train_frame_budget": args.frame_budget,
        "eval_frame_budget": args.eval_frame_budget,
        "workers": args.workers,
        "stage_a_epochs": args.stage_a_epochs,
        "stage_b_max_epochs": args.stage_b_max_epochs,
        "stage_b_min_epochs": args.stage_b_min_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "stage_a_lr": args.stage_a_learning_rate,
        "stage_b_lr": args.stage_b_learning_rate,
        "minimum_lr": args.minimum_learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weight_power": args.class_weight_power,
        "class_counts": class_counts,
        "class_weights": [float(value) for value in class_weights_cpu],
        "stage_a_losses": {
            "alignment": 1.0,
            "masked_modality_reconstruction": 0.50,
            "weak_contact": 0.30,
            "weak_phase": 0.30,
            "time_order": 0.20,
            "left_right_consistency": 0.10,
        },
        "stage_b_losses": {
            "detail21_weighted_ce": 1.0,
            "hardest_rival": 0.15,
            "cross_subject_supcon": 0.08,
            "subject_adversarial": 0.03,
            "alignment": 0.05,
            "weak_contact": 0.05,
            "weak_phase": 0.05,
            "time_order": 0.02,
        },
        "sampler_protocol": "full_coverage_exact_once_v2",
        "sampler": (
            f"all {len(train)} unique trials exactly once per epoch; length-bucketed; "
            "feasible same-class cross-subject pairs only reorder unique trials; no replacement"
        ),
        "sampler_preflight": str((output / "sampler_preflight.json").resolve()),
        "all_frames": True,
        "mixed_precision": device.type == "cuda",
        "seed": args.seed,
        "smoke": args.smoke,
        "model_parameters": parameter_count(model),
        "model_fp32_mib": parameter_count(model) * 4 / 1024**2,
        "p12_restricted_baseline": p12_restricted_baseline,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    atomic_json(output / "frozen_config.json", config)
    print(json.dumps({"stage": "start", **config}, ensure_ascii=False), flush=True)

    stage_a_optimizer = make_optimizer(
        model, args.stage_a_learning_rate, args.weight_decay
    )
    stage_a_history: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    if args.stage_a_checkpoint is not None:
        print(
            json.dumps(
                {
                    "stage": "A_resume",
                    "checkpoint": str(args.stage_a_checkpoint.resolve()),
                    "epochs_completed": stage_a_epochs_completed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for epoch in range(
        stage_a_epochs_completed + 1,
        args.stage_a_epochs + 1,
    ):
        rate = cosine_lr(
            epoch,
            args.stage_a_epochs,
            args.stage_a_learning_rate,
            args.minimum_learning_rate,
        )
        for group in stage_a_optimizer.param_groups:
            group["lr"] = rate
        result = run_stage_a_epoch(
            model,
            train_loader,
            train_sampler,
            stage_a_optimizer,
            scaler,
            device,
            epoch,
            args.flip_every,
            args.log_every,
            maximum_batches=maximum_train_batches,
        )
        row = {"epoch": epoch, "learning_rate": rate, **result}
        stage_a_history.append(row)
        stage_a_epochs_completed = epoch
        write_csv(output / "stage_a_history.csv", stage_a_history)
        atomic_checkpoint(
            output / "stage_a_last.pt",
            {
                "stage": "P46_step10_stageA",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
            },
        )
        print(json.dumps({"stage": "A_epoch", **row}, ensure_ascii=False), flush=True)

    stage_b_optimizer = make_optimizer(
        model, args.stage_b_learning_rate, args.weight_decay
    )
    # Stage B introduces several supervised heads at once.  A conservative
    # fresh AMP scale avoids a known first-step overflow at the default 65536.
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
        init_scale=4096.0,
        growth_interval=1000,
    )
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_accuracy_epoch = 0
    best_macro_f1_epoch = 0
    stale = 0
    stopped_early = False
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.stage_b_max_epochs + 1):
        rate = cosine_lr(
            epoch,
            args.stage_b_max_epochs,
            args.stage_b_learning_rate,
            args.minimum_learning_rate,
            warmup=2,
        )
        for group in stage_b_optimizer.param_groups:
            group["lr"] = rate
        training = run_stage_b_train_epoch(
            model,
            train_loader,
            train_sampler,
            stage_b_optimizer,
            scaler,
            device,
            epoch,
            user_to_index,
            class_weights,
            args.label_smoothing,
            args.log_every,
            maximum_batches=maximum_train_batches,
        )
        evaluation = evaluate(
            model, val_loader, device, maximum_batches=maximum_val_batches
        )
        accuracy = float(evaluation["metrics"]["accuracy"])
        macro_f1 = float(evaluation["metrics"]["macro_f1"])
        accuracy_improved = accuracy > best_accuracy + args.min_delta
        macro_improved = macro_f1 > best_macro_f1 + args.min_delta
        if accuracy_improved:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch
            atomic_checkpoint(
                output / "best_accuracy.pt",
                checkpoint_payload(model, epoch, evaluation["metrics"], config),
            )
            save_evaluation(output, "best_accuracy", evaluation, epoch)
        if macro_improved:
            best_macro_f1 = macro_f1
            best_macro_f1_epoch = epoch
            atomic_checkpoint(
                output / "best_macro_f1.pt",
                checkpoint_payload(model, epoch, evaluation["metrics"], config),
            )
            save_evaluation(output, "best_macro_f1", evaluation, epoch)
        stale = 0 if (accuracy_improved or macro_improved) else stale + 1
        row = {
            "epoch": epoch,
            "learning_rate": rate,
            **{f"train_{key}": value for key, value in training["losses"].items()},
            "train_sampled_accuracy": training["metrics"]["accuracy"],
            "train_sampled_macro_f1": training["metrics"]["macro_f1"],
            "train_samples": training["samples"],
            "train_unique_samples": training["coverage"]["unique_samples"],
            "train_missing_count": training["coverage"]["missing_count"],
            "train_duplicate_positions": training["coverage"]["duplicate_positions"],
            "train_coverage_exact_once": training["coverage"]["exact_once"],
            "train_seconds": training["seconds"],
            "val_loss": evaluation["loss"],
            "val_accuracy": accuracy,
            "val_balanced_accuracy": evaluation["metrics"]["balanced_accuracy"],
            "val_macro_f1": macro_f1,
            "val_seconds": evaluation["seconds"],
            "best_accuracy_epoch": best_accuracy_epoch,
            "best_macro_f1_epoch": best_macro_f1_epoch,
            "stale_epochs": stale,
        }
        history.append(row)
        write_csv(output / "stage_b_history.csv", history)
        print(json.dumps({"stage": "B_epoch", **row}, ensure_ascii=False), flush=True)
        if epoch >= args.stage_b_min_epochs and stale >= args.patience:
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
        "stage": "P46_step10_complete",
        "step11_started": False,
        "stage_a_epochs_completed": stage_a_epochs_completed,
        "stage_b_epochs_completed": len(history),
        "stopped_early": stopped_early,
        "elapsed_seconds": time.perf_counter() - run_started,
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_accuracy_metrics": best_accuracy_metrics,
        "best_macro_f1_epoch": best_macro_f1_epoch,
        "best_macro_f1_metrics": best_macro_metrics,
        "p12_restricted_baseline": p12_restricted_baseline,
        "best_accuracy_delta_vs_p12_pp": 100.0
        * (best_accuracy_metrics["accuracy"] - p12_restricted_baseline["accuracy"]),
        "best_balanced_accuracy_delta_vs_p12_pp": 100.0
        * (
            best_macro_metrics["balanced_accuracy"]
            - p12_restricted_baseline["balanced_accuracy"]
        ),
        "best_macro_f1_delta_vs_p12_pp": 100.0
        * (best_macro_metrics["macro_f1"] - p12_restricted_baseline["macro_f1"]),
        "step10_shows_accuracy_improvement": bool(
            best_accuracy_metrics["accuracy"] > p12_restricted_baseline["accuracy"]
        ),
        "step10_shows_macro_f1_improvement": bool(
            best_macro_metrics["macro_f1"] > p12_restricted_baseline["macro_f1"]
        ),
        "checkpoints": {
            "stage_a": str(
                args.stage_a_checkpoint.resolve()
                if args.stage_a_checkpoint is not None
                else (output / "stage_a_last.pt").resolve()
            ),
            "best_accuracy": str((output / "best_accuracy.pt").resolve()),
            "best_macro_f1": str((output / "best_macro_f1.pt").resolve()),
        },
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
