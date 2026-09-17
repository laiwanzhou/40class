from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from p31_skeleton_imu_model import IMUIntervalPartEncoder, SkeletonPartEncoder
from p46_event_data import P46EventDataset, collate_p46_events
from p46_event_model import LocalVisualObjectEncoder
from p46_protocol import EXPECTED, HARD_CLASS_IDS, HARD_CLASS_TO_INDEX
from p46_unified_repair_model import (
    P46UnifiedRepairModel,
    P46UnifiedRepairV2Model,
    P46UnifiedRepairV3Model,
    SharedSameTimeRelationshipRefiner,
    parameter_count,
    relationship_localization_loss,
)
from p46r_event_bottleneck_model import P46R_OFFSET_FRACTIONS
from train_p46_full_repair import (
    amp_context,
    make_datasets,
    release_epoch_memory,
)
from train_p46_step10 import (
    atomic_checkpoint,
    atomic_json,
    batch_coverage_report,
    compute_p12_restricted_baseline,
    cosine_lr,
    cross_subject_supervised_contrastive,
    detail_metrics,
    event_structure_losses,
    hardest_rival_loss,
    make_class_weights,
    make_optimizer,
    move_batch,
    FrameBudgetBatchSampler,
    FullCoverageCrossSubjectBatchSampler,
    run_stage_a_epoch,
    save_evaluation,
    seed_everything,
    source_coverage_report,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_RUN = PROJECT_DIR / "runs" / "p46_event_inputs_full"
DEFAULT_CONTEXT_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_P12_OOF = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_unified_repair_from_scratch_v1"
DEFAULT_OUTPUT_V2 = PROJECT_DIR / "runs" / "p46_unified_repair_v2"
DEFAULT_OUTPUT_V3 = PROJECT_DIR / "runs" / "p46_unified_repair_v3_clean"


def checkpoint_stage(protocol: str, phase: str) -> str:
    if phase not in {"stageA", "stageB"}:
        raise ValueError(f"unsupported checkpoint phase: {phase}")
    if protocol == "v1":
        return f"P46_unified_repair_{phase}"
    return f"P46_unified_repair_{protocol}_{phase}"


def parse_args(default_protocol: str = "v1") -> argparse.Namespace:
    if default_protocol not in {"v1", "v2", "v3"}:
        raise ValueError("default protocol must be v1, v2 or v3")
    is_v2 = default_protocol == "v2"
    is_v3 = default_protocol == "v3"
    parser = argparse.ArgumentParser(
        description=(
            "Train P46 with one shared time-aware, raw-IMU-aware, ROI-geometry-aware "
            "relationship trunk from random initialisation."
        )
    )
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--context-run", type=Path, default=DEFAULT_CONTEXT_RUN)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12_OOF)
    parser.add_argument(
        "--protocol", choices=("v1", "v2", "v3"), default=default_protocol
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(DEFAULT_OUTPUT_V3 if is_v3 else DEFAULT_OUTPUT_V2 if is_v2 else DEFAULT_OUTPUT),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stage-a-resume", type=Path, default=None)
    parser.add_argument("--stop-after-stage-a-epoch", type=int, default=0)
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--frame-budget", type=int, default=1024)
    parser.add_argument("--eval-frame-budget", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage-a-epochs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stage-a-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.06 if is_v2 else 0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--offset-eval-every", type=int, default=1)
    parser.add_argument("--flip-every", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--main-weight", type=float, default=1.0)
    parser.add_argument("--rival-weight", type=float, default=0.05 if is_v3 else 0.15)
    parser.add_argument("--contrast-weight", type=float, default=0.02 if is_v3 else 0.08)
    parser.add_argument(
        "--subject-weight", type=float, default=0.0 if is_v3 else 0.08 if is_v2 else 0.03
    )
    parser.add_argument("--offset-weight", type=float, default=0.50)
    parser.add_argument(
        "--localization-weight", type=float, default=0.0 if is_v3 else 0.03 if is_v2 else 0.15
    )
    parser.add_argument("--model-width", type=int, default=180 if is_v3 else 168 if is_v2 else 192)
    parser.add_argument("--model-dropout", type=float, default=0.14 if is_v3 else 0.18 if is_v2 else 0.12)
    parser.add_argument("--raw-axis-rotation-degrees", type=float, default=10.0 if is_v3 else 20.0)
    parser.add_argument("--raw-coordinate-dropout", type=float, default=0.05 if is_v3 else 0.10)
    parser.add_argument("--relationship-maximum-scale", type=float, default=0.30 if is_v3 else 0.25)
    parser.add_argument("--relationship-initial-scale", type=float, default=0.10 if is_v3 else 0.05)
    parser.add_argument("--group-dro-eta", type=float, default=0.02 if is_v2 else 0.0)
    parser.add_argument("--subject-max-adversarial-scale", type=float, default=0.0 if is_v3 else 2.0)
    parser.add_argument("--offset-saturation-accuracy", type=float, default=0.95)
    parser.add_argument("--offset-minimum-epochs", type=int, default=2)
    parser.add_argument("--localization-decay-epochs", type=int, default=8)
    parser.add_argument("--protected-boundary-weight", type=float, default=0.10 if is_v2 else 0.0)
    parser.add_argument("--oof-teacher", type=Path, default=DEFAULT_P12_OOF if is_v2 else None)
    parser.add_argument("--teacher-logits-key", default="sd_imu_logits")
    parser.add_argument("--teacher-weight", type=float, default=0.15 if is_v2 else 0.0)
    parser.add_argument("--alignment-weight", type=float, default=0.0 if is_v3 else 0.05)
    parser.add_argument("--contact-weight", type=float, default=0.0 if is_v3 else 0.05)
    parser.add_argument("--phase-weight", type=float, default=0.0 if is_v3 else 0.05)
    parser.add_argument("--order-weight", type=float, default=0.0 if is_v3 else 0.02)
    parser.add_argument("--teacher-confidence", type=float, default=0.60)
    parser.add_argument("--teacher-temperature", type=float, default=2.0)
    parser.add_argument("--minimum-epochs", type=int, default=8 if (is_v2 or is_v3) else 0)
    parser.add_argument("--patience", type=int, default=5 if (is_v2 or is_v3) else 0)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--train-users",
        nargs="+",
        default=None,
        help="Optional explicit user IDs for an inner/OOF training split.",
    )
    parser.add_argument(
        "--val-users",
        nargs="+",
        default=None,
        help="Optional explicit held-user IDs for an inner/OOF validation split.",
    )
    return parser.parse_args()


def make_unified_datasets(
    args: argparse.Namespace,
) -> tuple[P46EventDataset, P46EventDataset, dict[str, Any] | None]:
    if args.train_users is None and args.val_users is None:
        train, val = make_datasets(args)
        return train, val, None
    if not args.train_users or not args.val_users:
        raise ValueError("--train-users and --val-users must be supplied together")
    train_users = set(args.train_users)
    val_users = set(args.val_users)
    overlap = sorted(train_users & val_users)
    if overlap:
        raise ValueError(f"custom P46 user split overlaps: {overlap}")
    universe = P46EventDataset(
        args.event_run.resolve(),
        args.context_run.resolve(),
        split=None,
        load_context=True,
    )
    observed_users = {str(row["user_id"]) for row in universe.rows}
    unknown = sorted((train_users | val_users) - observed_users)
    if unknown:
        raise ValueError(f"unknown custom P46 users: {unknown}")
    train_ids = {
        str(row["sample_id"])
        for row in universe.rows
        if str(row["user_id"]) in train_users
    }
    val_ids = {
        str(row["sample_id"])
        for row in universe.rows
        if str(row["user_id"]) in val_users
    }
    train = P46EventDataset(
        args.event_run.resolve(),
        args.context_run.resolve(),
        split=None,
        sample_ids=train_ids,
        load_context=True,
    )
    val = P46EventDataset(
        args.event_run.resolve(),
        args.context_run.resolve(),
        split=None,
        sample_ids=val_ids,
        load_context=True,
    )
    protocol = {
        "purpose": "inner held-user OOF generation",
        "train_users": sorted(train_users),
        "val_users": sorted(val_users),
        "train_trials": len(train),
        "val_trials": len(val),
        "unused_users": sorted(observed_users - train_users - val_users),
    }
    return train, val, protocol


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class OOFCorrectnessTeacher:
    """Strict fold-pure teacher targets aligned by immutable sample ID."""

    def __init__(
        self,
        path: Path,
        dataset: P46EventDataset,
        logits_key: str,
        confidence_threshold: float,
        temperature: float,
    ) -> None:
        self.path = path.resolve()
        self.confidence_threshold = float(confidence_threshold)
        self.temperature = float(temperature)
        with np.load(self.path, allow_pickle=False) as data:
            required = {"sample_ids", "labels", "folds", logits_key}
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"OOF teacher is missing fields: {sorted(missing)}")
            sample_ids = [str(value) for value in data["sample_ids"]]
            labels = np.asarray(data["labels"], dtype=np.int64)
            folds = np.asarray(data["folds"], dtype=np.int64)
            logits = np.asarray(data[logits_key], dtype=np.float32)
        if len(set(sample_ids)) != len(sample_ids):
            raise RuntimeError("OOF teacher contains duplicate sample IDs")
        if logits.shape != (len(sample_ids), 40):
            raise ValueError(f"OOF teacher logits must be [N,40], got {logits.shape}")
        hard_logits = logits[:, list(HARD_CLASS_IDS)]
        self.rows = {
            sample_id: (int(labels[index]), int(folds[index]), hard_logits[index])
            for index, sample_id in enumerate(sample_ids)
        }
        missing_ids: list[str] = []
        wrong_labels: list[str] = []
        fold_ids: set[int] = set()
        user_folds: dict[str, set[int]] = {}
        for row in dataset.rows:
            sample_id = str(row["sample_id"])
            teacher = self.rows.get(sample_id)
            if teacher is None:
                missing_ids.append(sample_id)
                continue
            if teacher[0] != int(row["class_id"]):
                wrong_labels.append(sample_id)
            fold_ids.add(teacher[1])
            user_folds.setdefault(str(row["user_id"]), set()).add(teacher[1])
        if missing_ids or wrong_labels:
            raise RuntimeError(
                "OOF teacher alignment failed: "
                f"missing={missing_ids[:3]} wrong_labels={wrong_labels[:3]}"
            )
        if len(fold_ids) < 2:
            raise RuntimeError("OOF teacher does not contain multiple held-out folds")
        split_users = {
            user: sorted(folds) for user, folds in user_folds.items() if len(folds) != 1
        }
        if split_users:
            raise RuntimeError(
                f"OOF teacher is not subject-disjoint for users: {split_users}"
            )
        self.coverage = {
            "path": str(self.path),
            "logits_key": logits_key,
            "dataset_trials": len(dataset),
            "matched_trials": len(dataset),
            "folds": sorted(fold_ids),
            "subject_disjoint_verified": True,
            "confidence_threshold": self.confidence_threshold,
            "temperature": self.temperature,
        }

    def loss(
        self,
        student_logits: torch.Tensor,
        sample_ids: list[str],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher = torch.from_numpy(
            np.stack([self.rows[sample_id][2] for sample_id in sample_ids])
        ).to(device=student_logits.device, dtype=torch.float32)
        probability = torch.softmax(teacher, dim=1)
        confidence, prediction = probability.max(dim=1)
        active = prediction.eq(labels) & confidence.ge(self.confidence_threshold)
        temperature = self.temperature
        target = torch.softmax(teacher / temperature, dim=1)
        divergence = F.kl_div(
            F.log_softmax(student_logits.float() / temperature, dim=1),
            target,
            reduction="none",
        ).sum(dim=1) * (temperature**2)
        weight = active.to(divergence.dtype) * (
            (confidence - self.confidence_threshold)
            / max(1.0 - self.confidence_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        loss = (divergence * weight).sum() / weight.sum().clamp_min(1.0)
        active_confidence = (
            confidence[active].mean() if active.any() else confidence.new_zeros(())
        )
        return loss, active.float().mean(), active_confidence


def group_dro_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    class_weights: torch.Tensor,
    label_smoothing: float,
    group_weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    per_sample = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    target_weight = class_weights.index_select(0, labels)
    if eta <= 0.0:
        return per_sample.sum() / target_weight.sum().clamp_min(1e-8)
    with torch.no_grad():
        unique = subjects.unique()
        for subject_index in unique.tolist():
            selected = subjects == subject_index
            group_loss = per_sample[selected].float().mean().clamp(max=10.0)
            group_weights[subject_index] *= torch.exp(float(eta) * group_loss)
        group_weights /= group_weights.sum().clamp_min(1e-8)
    sample_weight = group_weights.index_select(0, subjects)
    sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-8)
    detached_weight = sample_weight.detach()
    return (per_sample * detached_weight).sum() / (
        target_weight * detached_weight
    ).sum().clamp_min(1e-8)


def protected_boundary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_ids: tuple[int, ...] = (7, 9, 10),
    margin: float = 0.50,
) -> torch.Tensor:
    indices = torch.tensor(
        [HARD_CLASS_TO_INDEX[class_id] for class_id in class_ids],
        device=logits.device,
    )
    selected = (labels[:, None] == indices[None]).any(dim=1)
    if not selected.any():
        return logits.sum() * 0.0
    protected_logits = logits[selected].index_select(1, indices)
    protected_labels = labels[selected]
    local_target = (protected_labels[:, None] == indices[None]).long().argmax(dim=1)
    true = protected_logits.gather(1, local_target[:, None]).squeeze(1)
    rival = protected_logits.masked_fill(
        F.one_hot(local_target, len(indices)).bool(),
        torch.finfo(protected_logits.dtype).min,
    ).amax(dim=1)
    return F.softplus(rival - true + margin).mean()


def subject_adversarial_scale(epoch: int, epochs: int, maximum: float) -> float:
    progress = float(epoch - 1) / max(epochs - 1, 1)
    return float(maximum) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def dynamic_auxiliary_weights(
    args: argparse.Namespace, epoch: int, state: dict[str, Any]
) -> dict[str, float]:
    if args.protocol == "v1":
        return {
            "offset": float(args.offset_weight),
            "localization": float(args.localization_weight),
        }
    previous_accuracy = float(state.get("previous_offset_accuracy", 0.0))
    offset = float(args.offset_weight)
    if (
        epoch > int(args.offset_minimum_epochs)
        and previous_accuracy >= float(args.offset_saturation_accuracy)
    ):
        offset = 0.0
    decay_epochs = max(int(args.localization_decay_epochs), 1)
    localization = float(args.localization_weight) * max(
        0.0, 1.0 - float(epoch - 1) / decay_epochs
    )
    return {"offset": offset, "localization": localization}


def _gradient_report(module: torch.nn.Module) -> dict[str, Any]:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return {
        "present": bool(gradients),
        "finite": bool(gradients)
        and all(torch.isfinite(value).all().item() for value in gradients),
        "nonzero": bool(gradients)
        and any((value.abs().sum() > 0).item() for value in gradients),
    }


@torch.inference_mode()
def feature_sensitivity(
    model: P46UnifiedRepairModel,
    raw_batch: dict[str, Any],
    device: torch.device,
    amp_dtype: str,
) -> dict[str, float]:
    model.eval()
    batch = move_batch(raw_batch, device)
    with amp_context(device, amp_dtype):
        baseline = model(batch)["detail_logits"].float()
    replacements = {
        "time_position": torch.zeros_like(batch["time_position"]),
        "frame_time_seconds": batch["frame_time_seconds"] * 2.0,
        "imu_raw_vectors": torch.zeros_like(batch["imu_raw_vectors"]),
        "oriented_roi_geometry": torch.zeros_like(batch["oriented_roi_geometry"]),
        "oriented_angle_valid": torch.zeros_like(batch["oriented_angle_valid"]),
        "local_roi_source": (batch["local_roi_source"] + 1).remainder(7),
    }
    result: dict[str, float] = {}
    for key, replacement in replacements.items():
        changed = dict(batch)
        changed[key] = replacement
        with amp_context(device, amp_dtype):
            logits = model(changed)["detail_logits"].float()
        result[key] = float((baseline - logits).abs().amax().cpu())
    if not all(value > 0.0 for value in result.values()):
        raise RuntimeError(f"unconsumed unified input detected: {result}")
    return result


def architecture_report(model: P46UnifiedRepairModel) -> dict[str, Any]:
    report = {
        "skeleton_encoder_instances": sum(
            isinstance(module, SkeletonPartEncoder) for module in model.modules()
        ),
        "compensated_imu_encoder_instances": sum(
            isinstance(module, IMUIntervalPartEncoder) for module in model.modules()
        ),
        "visual_encoder_instances": sum(
            isinstance(module, LocalVisualObjectEncoder) for module in model.modules()
        ),
        "relationship_refiner_instances": sum(
            isinstance(module, SharedSameTimeRelationshipRefiner)
            for module in model.modules()
        ),
        "detail_head_instances": 1,
        "parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
    }
    required = (
        report["skeleton_encoder_instances"] == 1
        and report["compensated_imu_encoder_instances"] == 1
        and report["visual_encoder_instances"] == 1
        and report["relationship_refiner_instances"] == 1
        and report["parameters"] == report["trainable_parameters"]
    )
    report["single_shared_trunk_verified"] = required
    if not required:
        raise RuntimeError(f"unified architecture contract failed: {report}")
    return report


def make_unified_loaders(
    train: P46EventDataset,
    val: P46EventDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, Any, DataLoader, Any]:
    """Keep variable-size NPZ allocations out of the long-lived main process.

    On Windows, workers=0 makes the main PyTorch CPU allocator retain many
    differently sized all-frame buffers across epochs.  Non-persistent workers
    are intentionally restarted for every iterator so the OS reclaims their
    heaps at every train/validation boundary.
    """

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
    common: dict[str, Any] = {
        "num_workers": args.workers,
        "pin_memory": False,
        "persistent_workers": False,
        "collate_fn": collate_p46_events,
    }
    if args.workers > 0:
        common["prefetch_factor"] = 1
    return (
        DataLoader(train, batch_sampler=train_sampler, **common),
        train_sampler,
        DataLoader(val, batch_sampler=val_sampler, **common),
        val_sampler,
    )


def train_epoch(
    model: P46UnifiedRepairModel,
    loader: DataLoader,
    sampler: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    class_weights: torch.Tensor,
    user_to_index: dict[str, int],
    group_weights: torch.Tensor,
    teacher: OOFCorrectnessTeacher | None,
    auxiliary_weights: dict[str, float],
    *,
    maximum_batches: int = 0,
) -> dict[str, Any]:
    model.train()
    sampler.set_epoch(1000 + epoch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    totals: Counter[str] = Counter()
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    observed: list[str] = []
    samples = 0
    started = time.perf_counter()
    adversarial_scale = (
        subject_adversarial_scale(
            epoch, args.epochs, args.subject_max_adversarial_scale
        )
        if args.protocol in {"v2", "v3"}
        else 1.0
    )
    for batch_index, raw_batch in enumerate(loader):
        if maximum_batches and batch_index >= maximum_batches:
            break
        observed.extend(raw_batch["source_id"])
        batch = move_batch(raw_batch, device)
        labels = batch["detail_index"]
        batch_size = len(labels)
        subjects = torch.tensor(
            [user_to_index[value] for value in batch["user_id"]],
            dtype=torch.long,
            device=device,
        )
        offset_labels = (
            torch.arange(batch_size, device=device) + batch_index + epoch
        ).remainder(len(P46R_OFFSET_FRACTIONS))
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.amp_dtype):
            output = model(
                batch, subject_adversarial_scale=adversarial_scale
            )
            structure = event_structure_losses(output, batch)
            classification = group_dro_classification_loss(
                output["detail_logits"],
                labels,
                subjects,
                class_weights,
                args.label_smoothing,
                group_weights,
                args.group_dro_eta,
            )
            rival = hardest_rival_loss(output["detail_logits"], labels)
            contrast = cross_subject_supervised_contrastive(
                output["contrast_embedding"], labels, subjects
            )
            subject = F.cross_entropy(output["subject_logits"], subjects)
            offset_logits = model.synthetic_offset_logits(
                output, offset_labels, batch["frame_mask"]
            )
            offset = F.cross_entropy(offset_logits, offset_labels)
            localization = relationship_localization_loss(
                output, confidence_weighted=args.protocol in {"v2", "v3"}
            )
            boundary = protected_boundary_loss(output["detail_logits"], labels)
            if teacher is not None and args.teacher_weight > 0.0:
                teacher_anchor, teacher_coverage, teacher_confidence = teacher.loss(
                    output["detail_logits"], batch["sample_id"], labels
                )
            else:
                teacher_anchor = output["detail_logits"].sum() * 0.0
                teacher_coverage = teacher_anchor.detach()
                teacher_confidence = teacher_anchor.detach()
            total = (
                args.main_weight * classification
                + args.rival_weight * rival
                + args.contrast_weight * contrast
                + args.subject_weight * subject
                + args.alignment_weight * structure["alignment"]
                + args.contact_weight * structure["contact"]
                + args.phase_weight * structure["phase"]
                + args.order_weight * structure["order"]
                + auxiliary_weights["offset"] * offset
                + auxiliary_weights["localization"] * localization
                + args.protected_boundary_weight * boundary
                + args.teacher_weight * teacher_anchor
            )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite unified P46 loss at epoch {epoch}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite unified P46 gradients at epoch {epoch}")
        scaler.step(optimizer)
        scaler.update()
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
            "offset": offset,
            "localization": localization,
            "protected_boundary": boundary,
            "teacher_anchor": teacher_anchor,
            "teacher_coverage": teacher_coverage,
            "teacher_confidence": teacher_confidence,
            "offset_accuracy": (offset_logits.argmax(dim=1) == offset_labels)
            .float()
            .mean(),
            "gradient_norm": gradient,
            "subject_adversarial_scale": torch.as_tensor(
                adversarial_scale, device=device
            ),
            "effective_offset_weight": torch.as_tensor(
                auxiliary_weights["offset"], device=device
            ),
            "effective_localization_weight": torch.as_tensor(
                auxiliary_weights["localization"], device=device
            ),
            "maximum_group_weight": group_weights.max(),
            "mean_relationship_residual_scale": output[
                "relationship_residual_scale"
            ].mean(),
        }
        for key, value in logged.items():
            totals[key] += float(value.detach()) * batch_size
        samples += batch_size
        labels_all.append(labels.detach().cpu())
        logits_all.append(output["detail_logits"].detach().float().cpu())
        if args.log_every and (batch_index + 1) % args.log_every == 0:
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
        del raw_batch, batch, output, offset_logits, total
    coverage = source_coverage_report(observed, sampler.source_ids)
    if not maximum_batches and not coverage["exact_once"]:
        raise RuntimeError(f"unified P46 epoch coverage failed: {coverage}")
    label_array = torch.cat(labels_all).numpy()
    logit_array = torch.cat(logits_all).numpy()
    return {
        "losses": {
            key: value / max(samples, 1) for key, value in totals.items()
        },
        "metrics": detail_metrics(label_array, logit_array),
        "coverage": coverage,
        "samples": samples,
        "seconds": time.perf_counter() - started,
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }


@torch.inference_mode()
def evaluate(
    model: P46UnifiedRepairModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    *,
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
        with amp_context(device, amp_dtype):
            output = model(batch)
            loss = F.cross_entropy(output["detail_logits"], batch["detail_index"])
        batch_size = len(batch["detail_index"])
        losses += float(loss) * batch_size
        samples += batch_size
        labels.append(batch["detail_index"].cpu())
        logits.append(output["detail_logits"].float().cpu())
        source_ids.extend(batch["source_id"])
        users.extend(batch["user_id"])
        del raw_batch, batch, output, loss
    label_array = torch.cat(labels).numpy()
    logit_array = torch.cat(logits).numpy()
    metrics = detail_metrics(label_array, logit_array)
    prediction = logit_array.argmax(axis=1)
    user_array = np.asarray(users)
    per_user_accuracy = {
        user: float((prediction[user_array == user] == label_array[user_array == user]).mean())
        for user in sorted(set(users))
    }
    metrics["worst_user_accuracy"] = min(per_user_accuracy.values())
    metrics["mean_user_accuracy"] = float(np.mean(list(per_user_accuracy.values())))
    return {
        "loss": losses / max(samples, 1),
        "metrics": metrics,
        "per_user_accuracy": per_user_accuracy,
        "labels": label_array,
        "logits": logit_array,
        "source_ids": source_ids,
        "users": users,
        "samples": samples,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_offsets(
    model: P46UnifiedRepairModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    *,
    maximum_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    started = time.perf_counter()
    for batch_index, raw_batch in enumerate(loader):
        if maximum_batches and batch_index >= maximum_batches:
            break
        batch = move_batch(raw_batch, device)
        with amp_context(device, amp_dtype):
            output = model(batch)
            for offset_index in range(len(P46R_OFFSET_FRACTIONS)):
                target = torch.full(
                    (len(batch["detail_index"]),),
                    offset_index,
                    dtype=torch.long,
                    device=device,
                )
                offset_logits = model.synthetic_offset_logits(
                    output, target, batch["frame_mask"]
                )
                labels.append(target.cpu())
                logits.append(offset_logits.float().cpu())
        del raw_batch, batch, output, offset_logits
    label_array = torch.cat(labels).numpy()
    logit_array = torch.cat(logits).numpy()
    prediction = logit_array.argmax(1)
    return {
        "accuracy": float((prediction == label_array).mean()),
        "macro_f1": float(
            f1_score(label_array, prediction, average="macro", zero_division=0)
        ),
        "per_offset_accuracy": {
            str(index): float((prediction[label_array == index] == index).mean())
            for index in range(len(P46R_OFFSET_FRACTIONS))
        },
        "seconds": time.perf_counter() - started,
    }


def checkpoint_payload(
    model: P46UnifiedRepairModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    evaluation: dict[str, Any],
    history: list[dict[str, Any]],
    config: dict[str, Any],
    best: dict[str, Any],
    training_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": config["checkpoint_stage"],
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": evaluation["metrics"],
        "history": history,
        "config": config,
        "best": best,
        "training_state": training_state,
        "rng_state": capture_rng_state(),
    }


def save_oof_logits(
    output: Path,
    evaluation: dict[str, Any],
    epoch: int,
    train_users: list[str],
    val_users: list[str],
) -> None:
    """Persist fixed-final held-user logits, not label-selected best-epoch logits."""
    path = output / "fixed_final_oof_logits.npz"
    temporary = path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_ids=np.asarray(evaluation["source_ids"]),
            users=np.asarray(evaluation["users"]),
            labels=np.asarray(evaluation["labels"], dtype=np.int64),
            logits=np.asarray(evaluation["logits"], dtype=np.float32),
            epoch=np.asarray(epoch, dtype=np.int64),
            train_users=np.asarray(train_users),
            val_users=np.asarray(val_users),
        )
    temporary.replace(path)


def main(default_protocol: str = "v1") -> None:
    args = parse_args(default_protocol)
    if args.stage_a_epochs < 1 or args.epochs < 1:
        raise ValueError("stage-a-epochs and epochs must be positive")
    if args.resume is not None and args.stage_a_resume is not None:
        raise ValueError("--resume and --stage-a-resume are mutually exclusive")
    if args.model_width % 6 or args.model_width % 2:
        raise ValueError("model-width must be divisible by 6 and even")
    if args.resume is not None and args.smoke:
        raise ValueError("Stage-B resume cannot be combined with --smoke")
    if args.smoke:
        args.stage_a_epochs = 1
        args.epochs = 2
    maximum_train_batches = 2 if args.smoke else 0
    maximum_val_batches = 2 if args.smoke else 0
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, custom_split = make_unified_datasets(args)
    if not args.smoke and custom_split is None and (
        len(train) != EXPECTED["train_detail_trials"]
        or len(val) != EXPECTED["val_detail_trials"]
    ):
        raise RuntimeError(f"P46 split changed: train={len(train)} val={len(val)}")
    train_users = sorted({row["user_id"] for row in train.rows})
    val_users = sorted({row["user_id"] for row in val.rows})
    if set(train_users) & set(val_users):
        raise RuntimeError("unified P46 subject leakage")
    user_to_index = {value: index for index, value in enumerate(train_users)}
    train_loader, train_sampler, val_loader, val_sampler = make_unified_loaders(
        train, val, args
    )
    train_labels = [HARD_CLASS_TO_INDEX[int(row["class_id"])] for row in train.rows]
    class_weights_cpu, class_counts = make_class_weights(
        train_labels, len(HARD_CLASS_IDS), args.class_weight_power
    )
    class_weights = class_weights_cpu.to(device)
    teacher = (
        OOFCorrectnessTeacher(
            args.oof_teacher,
            train,
            args.teacher_logits_key,
            args.teacher_confidence,
            args.teacher_temperature,
        )
        if args.oof_teacher is not None and args.teacher_weight > 0.0
        else None
    )
    p12_baseline = (
        compute_p12_restricted_baseline(val, args.p12_oof.resolve())[0]
        if not args.smoke
        else None
    )

    planned_epochs = list(range(1, args.stage_a_epochs + 1)) + list(
        range(1001, 1000 + args.epochs + 1)
    )
    train_reports: list[dict[str, Any]] = []
    for sampler_epoch in planned_epochs:
        train_sampler.set_epoch(sampler_epoch)
        report = train_sampler.coverage_report()
        if not report["exact_once"]:
            raise RuntimeError(f"train sampler coverage failed: {report}")
        train_reports.append(report)
    validation_report = batch_coverage_report(list(val_sampler), len(val))
    if not validation_report["exact_once"]:
        raise RuntimeError(f"validation sampler coverage failed: {validation_report}")

    if args.protocol == "v2":
        model = P46UnifiedRepairV2Model(
            width=args.model_width,
            dropout=args.model_dropout,
            subjects=len(train_users),
            raw_axis_rotation_degrees=args.raw_axis_rotation_degrees,
            raw_coordinate_dropout=args.raw_coordinate_dropout,
            relationship_maximum_scale=args.relationship_maximum_scale,
            relationship_initial_scale=args.relationship_initial_scale,
        ).to(device)
    elif args.protocol == "v3":
        model = P46UnifiedRepairV3Model(
            width=args.model_width,
            dropout=args.model_dropout,
            subjects=len(train_users),
            raw_axis_rotation_degrees=args.raw_axis_rotation_degrees,
            raw_coordinate_dropout=args.raw_coordinate_dropout,
            relationship_maximum_scale=args.relationship_maximum_scale,
            relationship_initial_scale=args.relationship_initial_scale,
        ).to(device)
    else:
        model = P46UnifiedRepairModel(
            width=args.model_width,
            dropout=args.model_dropout,
            subjects=len(train_users),
        ).to(device)
    architecture = architecture_report(model)
    first_validation_batch = next(iter(val_loader))
    sensitivity = feature_sensitivity(
        model, first_validation_batch, device, args.amp_dtype
    )
    preflight = {
        "protocol": f"P46_unified_single_trunk_from_scratch_{args.protocol}",
        "train_trials": len(train),
        "val_trials": len(val),
        "subject_overlap": [],
        "all_train_epochs_exact_once": all(
            report["exact_once"] for report in train_reports
        ),
        "validation_exact_once": validation_report["exact_once"],
        "architecture": architecture,
        "feature_logit_sensitivity_max_abs": sensitivity,
        "all_new_fields_affect_logits": all(value > 0.0 for value in sensitivity.values()),
        "oof_teacher": teacher.coverage if teacher is not None else None,
    }
    atomic_json(output_dir / "preflight.json", preflight)
    print(json.dumps({"stage": "preflight", **preflight}, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return

    config = {
        "stage": f"P46_unified_repair_from_scratch_{args.protocol}",
        "checkpoint_stage": checkpoint_stage(args.protocol, "stageB"),
        "architecture": (
            "raw+compensated IMU and Skeleton + geometry/provenance-aware local visual "
            "-> shared same-part/same-time relationship refiner -> explicit continuous-time "
            "temporal trunk -> one Detail21 head"
        ),
        "random_initialisation": True,
        "pretrained_checkpoint_loaded": False,
        "frozen_modules": False,
        "duplicated_modality_encoders": False,
        "output_gate": False,
        "high_level_embedding_concat": False,
        "torso_relative_roi_geometry": args.protocol in {"v2", "v3"},
        "raw_imu_axis_augmentation": args.protocol in {"v2", "v3"},
        "relationship_residual_bounded": args.protocol in {"v2", "v3"},
        "clean_main_classification_protocol": args.protocol == "v3",
        "synthetic_offset_uses_shared_encoded_tokens": True,
        "train_trials": len(train),
        "val_trials": len(val),
        "train_subjects": train_users,
        "val_subjects": val_users,
        "stage_a_epochs": args.stage_a_epochs,
        "stage_b_epochs": args.epochs,
        "stage_a_learning_rate": args.stage_a_learning_rate,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "weight_decay": args.weight_decay,
        "model_width": args.model_width,
        "model_dropout": args.model_dropout,
        "raw_axis_rotation_degrees": args.raw_axis_rotation_degrees,
        "raw_coordinate_dropout": args.raw_coordinate_dropout,
        "relationship_initial_scale": args.relationship_initial_scale,
        "relationship_maximum_scale": args.relationship_maximum_scale,
        "label_smoothing": args.label_smoothing,
        "class_counts": class_counts,
        "class_weights": [float(value) for value in class_weights_cpu],
        "loss_weights": {
            "main": args.main_weight,
            "rival": args.rival_weight,
            "contrast": args.contrast_weight,
            "subject": args.subject_weight,
            "alignment": args.alignment_weight,
            "contact": args.contact_weight,
            "phase": args.phase_weight,
            "order": args.order_weight,
            "relationship_offset": args.offset_weight,
            "relationship_localization": args.localization_weight,
            "protected_7_9_10_boundary": args.protected_boundary_weight,
            "oof_correctness_teacher": args.teacher_weight,
        },
        "dynamic_loss_protocol": {
            "offset_saturation_accuracy": args.offset_saturation_accuracy,
            "offset_minimum_epochs": args.offset_minimum_epochs,
            "localization_decay_epochs": args.localization_decay_epochs,
            "localization_confidence_weighted": args.protocol in {"v2", "v3"},
        },
        "cross_subject_protocol": {
            "group_dro_eta": args.group_dro_eta,
            "subject_max_adversarial_scale": args.subject_max_adversarial_scale,
        },
        "oof_teacher": teacher.coverage if teacher is not None else None,
        "early_stopping": {
            "minimum_epochs": args.minimum_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "selection_score": (
                "accuracy + 0.50*macro_f1 + 0.25*worst_user_accuracy"
                if args.protocol in {"v2", "v3"}
                else "macro_f1 + 0.25*accuracy"
            ),
        },
        "offset_fractions": list(P46R_OFFSET_FRACTIONS),
        "architecture_report": architecture,
        "feature_sensitivity": sensitivity,
        "model_parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "amp_dtype": args.amp_dtype,
        "workers": args.workers,
        "persistent_workers": False,
        "prefetch_factor": 1 if args.workers > 0 else None,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "p12_restricted_baseline": p12_baseline,
        "smoke": args.smoke,
        "custom_oof_split": custom_split,
    }
    atomic_json(output_dir / "frozen_config.json", config)
    print(json.dumps({"stage": "start", **config}, ensure_ascii=False), flush=True)
    run_started = time.perf_counter()

    stage_a_history: list[dict[str, Any]] = []
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best = {
        "score": -1.0,
        "score_epoch": 0,
        "accuracy": -1.0,
        "accuracy_epoch": 0,
        "macro_f1": -1.0,
        "macro_f1_epoch": 0,
    }
    group_weights = torch.full(
        (len(train_users),), 1.0 / len(train_users), device=device
    )
    auxiliary_state: dict[str, Any] = {"previous_offset_accuracy": 0.0}
    epochs_without_improvement = 0
    if args.resume is None:
        stage_a_start_epoch = 1
        stage_a_optimizer = make_optimizer(
            model, args.stage_a_learning_rate, args.weight_decay
        )
        stage_a_scaler = torch.amp.GradScaler(
            device.type, enabled=device.type == "cuda", init_scale=4096.0
        )
        if args.stage_a_resume is not None:
            stage_a_checkpoint = torch.load(
                args.stage_a_resume.resolve(), map_location="cpu", weights_only=False
            )
            expected_stage = checkpoint_stage(args.protocol, "stageA")
            if stage_a_checkpoint.get("stage") != expected_stage:
                raise RuntimeError("--stage-a-resume is not a compatible Stage-A checkpoint")
            checkpoint_config = stage_a_checkpoint.get("config", {})
            for key, value in {
                "train_trials": len(train),
                "val_trials": len(val),
                "stage_a_epochs": args.stage_a_epochs,
                "stage_b_epochs": args.epochs,
                "seed": args.seed,
            }.items():
                if checkpoint_config.get(key) != value:
                    raise RuntimeError(
                        f"Stage-A resume protocol changed for {key}: "
                        f"{checkpoint_config.get(key)} != {value}"
                    )
            required_state = {
                "optimizer_state_dict",
                "scaler_state_dict",
                "rng_state",
                "history",
            }
            missing_state = required_state - set(stage_a_checkpoint)
            if missing_state:
                raise RuntimeError(
                    f"Stage-A checkpoint predates exact resume support: {sorted(missing_state)}"
                )
            model.load_state_dict(stage_a_checkpoint["model_state_dict"], strict=True)
            stage_a_optimizer.load_state_dict(
                stage_a_checkpoint["optimizer_state_dict"]
            )
            stage_a_scaler.load_state_dict(stage_a_checkpoint["scaler_state_dict"])
            stage_a_history = list(stage_a_checkpoint["history"])
            stage_a_start_epoch = int(stage_a_checkpoint["epoch"]) + 1
            restore_rng_state(stage_a_checkpoint["rng_state"])
            del stage_a_checkpoint
            release_epoch_memory(device)
        for epoch in range(stage_a_start_epoch, args.stage_a_epochs + 1):
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
                stage_a_scaler,
                device,
                epoch,
                args.flip_every,
                args.log_every,
                maximum_batches=maximum_train_batches,
            )
            row = {"epoch": epoch, "learning_rate": rate, **result}
            stage_a_history.append(row)
            write_csv(output_dir / "stage_a_history.csv", stage_a_history)
            atomic_checkpoint(
                output_dir / "stage_a_last.pt",
                {
                    "stage": checkpoint_stage(args.protocol, "stageA"),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": stage_a_optimizer.state_dict(),
                    "scaler_state_dict": stage_a_scaler.state_dict(),
                    "history": stage_a_history,
                    "rng_state": capture_rng_state(),
                    "config": config,
                },
            )
            print(json.dumps({"stage": "A_epoch", **row}, ensure_ascii=False), flush=True)
            release_epoch_memory(device)
            if (
                args.stop_after_stage_a_epoch
                and epoch >= args.stop_after_stage_a_epoch
            ):
                stage_a_summary = {
                    "stage": "P46_unified_repair_stageA_controlled_stop",
                    "stage_a_epochs_completed": len(stage_a_history),
                    "stop_reason": "requested_stage_a_epoch_stop",
                    "checkpoint": str((output_dir / "stage_a_last.pt").resolve()),
                }
                atomic_json(output_dir / "stage_a_progress.json", stage_a_summary)
                print(json.dumps(stage_a_summary, ensure_ascii=False), flush=True)
                return

    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
        init_scale=4096.0,
        growth_interval=1000,
    )
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume.resolve(), map_location="cpu", weights_only=False
        )
        if checkpoint.get("stage") != config["checkpoint_stage"]:
            raise RuntimeError("--resume is not a unified P46 Stage-B checkpoint")
        checkpoint_config = checkpoint.get("config", {})
        resume_protocol = {
            "train_trials": len(train),
            "val_trials": len(val),
            "stage_b_epochs": args.epochs,
            "seed": args.seed,
        }
        if args.protocol in {"v2", "v3"}:
            resume_protocol["model_width"] = args.model_width
        for key, value in resume_protocol.items():
            if checkpoint_config.get(key) != value:
                raise RuntimeError(
                    f"resume protocol changed for {key}: "
                    f"{checkpoint_config.get(key)} != {value}"
                )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = list(checkpoint["history"])
        best = dict(checkpoint["best"])
        restored_training_state = checkpoint.get("training_state", {})
        if "group_dro_weights" in restored_training_state:
            group_weights.copy_(
                torch.as_tensor(
                    restored_training_state["group_dro_weights"], device=device
                )
            )
        auxiliary_state = dict(
            restored_training_state.get("auxiliary_state", auxiliary_state)
        )
        epochs_without_improvement = int(
            restored_training_state.get("epochs_without_improvement", 0)
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"])
        print(
            json.dumps(
                {"stage": "resume", "next_epoch": start_epoch}, ensure_ascii=False
            ),
            flush=True,
        )
        del checkpoint
        release_epoch_memory(device)

    controlled_stop = False
    stop_reason: str | None = None
    for epoch in range(start_epoch, args.epochs + 1):
        rate = cosine_lr(
            epoch,
            args.epochs,
            args.learning_rate,
            args.minimum_learning_rate,
            warmup=2,
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        auxiliary_weights = dynamic_auxiliary_weights(
            args, epoch, auxiliary_state
        )
        training = train_epoch(
            model,
            train_loader,
            train_sampler,
            optimizer,
            scaler,
            device,
            args,
            epoch,
            class_weights,
            user_to_index,
            group_weights,
            teacher,
            auxiliary_weights,
            maximum_batches=maximum_train_batches,
        )
        auxiliary_state["previous_offset_accuracy"] = float(
            training["losses"]["offset_accuracy"]
        )
        validation = evaluate(
            model,
            val_loader,
            device,
            args.amp_dtype,
            maximum_batches=maximum_val_batches,
        )
        offset_validation = None
        if args.offset_eval_every and (
            epoch == 1
            or epoch % args.offset_eval_every == 0
            or epoch == args.epochs
        ):
            offset_validation = evaluate_offsets(
                model,
                val_loader,
                device,
                args.amp_dtype,
                maximum_batches=maximum_val_batches,
            )
        accuracy = float(validation["metrics"]["accuracy"])
        macro_f1 = float(validation["metrics"]["macro_f1"])
        score = (
            accuracy
            + 0.50 * macro_f1
            + 0.25 * float(validation["metrics"]["worst_user_accuracy"])
            if args.protocol in {"v2", "v3"}
            else macro_f1 + 0.25 * accuracy
        )
        row = {
            "epoch": epoch,
            "learning_rate": rate,
            **{f"train_{key}": value for key, value in training["losses"].items()},
            **{f"train_{key}": value for key, value in training["metrics"].items()},
            **{f"val_{key}": value for key, value in validation["metrics"].items()},
            "val_loss": validation["loss"],
            "val_offset_accuracy": (
                offset_validation["accuracy"] if offset_validation else None
            ),
            "val_offset_macro_f1": (
                offset_validation["macro_f1"] if offset_validation else None
            ),
            "train_seconds": training["seconds"],
            "val_seconds": validation["seconds"],
            "offset_val_seconds": (
                offset_validation["seconds"] if offset_validation else 0.0
            ),
            "peak_cuda_mib": training["peak_cuda_mib"],
            "coverage_exact_once": training["coverage"]["exact_once"],
        }
        history.append(row)
        improved_score = score > float(best["score"]) + float(args.min_delta)
        improved_accuracy = accuracy > float(best["accuracy"])
        improved_macro = macro_f1 > float(best["macro_f1"])
        if improved_score:
            best["score"] = score
            best["score_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if improved_accuracy:
            best["accuracy"] = accuracy
            best["accuracy_epoch"] = epoch
        if improved_macro:
            best["macro_f1"] = macro_f1
            best["macro_f1_epoch"] = epoch
        write_csv(output_dir / "history.csv", history)
        training_state = {
            "group_dro_weights": group_weights.detach().cpu(),
            "auxiliary_state": dict(auxiliary_state),
            "epochs_without_improvement": epochs_without_improvement,
        }
        payload = checkpoint_payload(
            model,
            optimizer,
            scaler,
            epoch,
            validation,
            history,
            config,
            best,
            training_state,
        )
        atomic_checkpoint(output_dir / "last.pt", payload)
        if custom_split is not None:
            save_evaluation(output_dir, "fixed_final", validation, epoch)
            save_oof_logits(
                output_dir,
                validation,
                epoch,
                train_users,
                val_users,
            )
        if improved_score:
            atomic_checkpoint(output_dir / "best.pt", payload)
            save_evaluation(output_dir, "best", validation, epoch)
        if improved_accuracy:
            atomic_checkpoint(output_dir / "best_accuracy.pt", payload)
            save_evaluation(output_dir, "best_accuracy", validation, epoch)
        if improved_macro:
            atomic_checkpoint(output_dir / "best_macro_f1.pt", payload)
            save_evaluation(output_dir, "best_macro_f1", validation, epoch)
        print(json.dumps({"stage": "B_epoch", **row, "best": best}, ensure_ascii=False), flush=True)
        del training, validation, offset_validation, payload
        release_epoch_memory(device)
        if (
            args.patience > 0
            and epoch >= args.minimum_epochs
            and epochs_without_improvement >= args.patience
        ):
            controlled_stop = True
            stop_reason = "validation_plateau"
            break
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            controlled_stop = True
            stop_reason = "requested_epoch_stop"
            break

    summary = {
        "stage": "P46_unified_repair_complete" if not controlled_stop else "P46_unified_repair_controlled_stop",
        "stage_a_epochs_completed": args.stage_a_epochs,
        "stage_b_epochs_completed": len(history),
        "controlled_stop": controlled_stop,
        "stop_reason": stop_reason,
        "best": best,
        "elapsed_seconds": time.perf_counter() - run_started,
        "p12_restricted_baseline": p12_baseline,
        "architecture_report": architecture,
        "feature_sensitivity": sensitivity,
        "checkpoints": {
            "last": str((output_dir / "last.pt").resolve()),
            "best": str((output_dir / "best.pt").resolve()),
            "best_accuracy": str((output_dir / "best_accuracy.pt").resolve()),
            "best_macro_f1": str((output_dir / "best_macro_f1.pt").resolve()),
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
