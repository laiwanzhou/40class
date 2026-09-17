from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
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

from p46_event_data import P46EventDataset, collate_p46_events
from p46_full_repair_model import (
    P46FullRepairModel,
    parameter_count,
    relationship_localization_loss,
)
from p46_protocol import EXPECTED, HARD_CLASS_TO_INDEX
from p46_step10_model import (
    cross_subject_supervised_contrastive,
    hardest_rival_loss,
)
from p46r_event_bottleneck_model import (
    P46R_OFFSET_FRACTIONS,
    circular_shift_visual_batch,
)
from train_p46_step10 import (
    atomic_checkpoint,
    atomic_json,
    batch_coverage_report,
    compute_p12_restricted_baseline,
    cosine_lr,
    detail_metrics,
    event_structure_losses,
    FrameBudgetBatchSampler,
    FullCoverageCrossSubjectBatchSampler,
    make_class_weights,
    make_optimizer,
    move_batch,
    run_stage_a_epoch,
    save_evaluation,
    seed_everything,
    source_coverage_report,
    write_csv,
)
from train_p46r_mechanism import select_smoke_ids


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_RUN = PROJECT_DIR / "runs" / "p46_event_inputs_full"
DEFAULT_CONTEXT_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_P12_OOF = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_full_repair_from_scratch_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the complete P46+P46-R feature-input model from random "
            "initialisation with no checkpoint loading or frozen branch."
        )
    )
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--context-run", type=Path, default=DEFAULT_CONTEXT_RUN)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume the same from-scratch run from a full last.pt training checkpoint.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=0,
        help="Controlled diagnostic boundary; 0 runs through --epochs.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--frame-budget", type=int, default=1024)
    parser.add_argument("--eval-frame-budget", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stage-a-epochs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stage-a-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--offset-eval-every", type=int, default=1)
    parser.add_argument("--flip-every", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--main-weight", type=float, default=1.0)
    parser.add_argument("--p46-aux-weight", type=float, default=0.10)
    parser.add_argument("--relationship-aux-weight", type=float, default=0.10)
    parser.add_argument("--rival-weight", type=float, default=0.15)
    parser.add_argument("--contrast-weight", type=float, default=0.08)
    parser.add_argument("--relationship-contrast-weight", type=float, default=0.05)
    parser.add_argument("--subject-weight", type=float, default=0.03)
    parser.add_argument("--offset-weight", type=float, default=0.50)
    parser.add_argument("--localization-weight", type=float, default=0.15)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def amp_context(device: torch.device, name: str) -> torch.autocast:
    dtype = torch.bfloat16 if name == "bfloat16" else torch.float16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and name != "float32",
    )


def make_datasets(args: argparse.Namespace) -> tuple[P46EventDataset, P46EventDataset]:
    train_full = P46EventDataset(
        args.event_run.resolve(),
        args.context_run.resolve(),
        split="train",
        load_context=True,
    )
    val_full = P46EventDataset(
        args.event_run.resolve(),
        args.context_run.resolve(),
        split="val",
        load_context=True,
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
            load_context=True,
        ),
        P46EventDataset(
            args.event_run,
            args.context_run,
            split="val",
            sample_ids=val_ids,
            load_context=True,
        ),
    )


def make_full_repair_loaders(
    train: P46EventDataset,
    val: P46EventDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, Any, DataLoader, Any]:
    """Windows-safe loaders with no growing pinned-memory pool.

    The original shared loader enables pin_memory whenever CUDA exists, including
    workers=0.  With highly variable all-frame batches that can retain a large set
    of differently sized page-locked buffers across epochs.  This full-repair run
    values bounded host memory over the small transfer-speed benefit.
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
    common = {
        "num_workers": args.workers,
        "pin_memory": False,
        "persistent_workers": args.workers > 0,
        "collate_fn": collate_p46_events,
    }
    return (
        DataLoader(train, batch_sampler=train_sampler, **common),
        train_sampler,
        DataLoader(val, batch_sampler=val_sampler, **common),
        val_sampler,
    )


def memory_snapshot(device: torch.device) -> dict[str, float]:
    result: dict[str, float] = {}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        result["process_rss_mib"] = float(memory.rss / 1024**2)
        result["process_vms_mib"] = float(memory.vms / 1024**2)
        private = getattr(memory, "private", None)
        if private is not None:
            result["process_private_mib"] = float(private / 1024**2)
        result["host_available_mib"] = float(psutil.virtual_memory().available / 1024**2)
    except (ImportError, OSError):
        pass
    if device.type == "cuda":
        result.update(
            {
                "cuda_allocated_mib": float(torch.cuda.memory_allocated(device) / 1024**2),
                "cuda_reserved_mib": float(torch.cuda.memory_reserved(device) / 1024**2),
            }
        )
    return result


def release_epoch_memory(device: torch.device) -> dict[str, float]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return memory_snapshot(device)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def load_resume_checkpoint(
    path: Path,
    args: argparse.Namespace,
    *,
    train_trials: int,
    val_trials: int,
) -> dict[str, Any]:
    resolved = path.resolve()
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    required = {
        "stage",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "history",
        "best_epoch",
        "best_score",
        "best_accuracy",
        "best_macro_f1",
        "config",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(f"resume checkpoint lacks training state {missing}: {resolved}")
    if checkpoint["stage"] != "P46_full_repair_from_scratch":
        raise RuntimeError(f"not a P46 full-repair training checkpoint: {resolved}")
    epoch = int(checkpoint["epoch"])
    history = list(checkpoint["history"])
    if epoch < 1 or epoch >= args.epochs:
        raise RuntimeError(f"resume epoch {epoch} is outside 1..{args.epochs - 1}")
    if len(history) != epoch or int(history[-1]["epoch"]) != epoch:
        raise RuntimeError(
            f"resume history/epoch mismatch: epoch={epoch} rows={len(history)}"
        )
    config = checkpoint["config"]
    expected = {
        "train_trials": train_trials,
        "val_trials": val_trials,
        "stage_a_epochs": args.stage_a_epochs,
        "stage_b_epochs": args.epochs,
        "seed": args.seed,
    }
    changed = {
        key: {"checkpoint": config.get(key), "runtime": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if changed:
        raise RuntimeError(f"resume protocol changed: {changed}")
    return checkpoint


def train_epoch(
    model: P46FullRepairModel,
    loader: DataLoader,
    sampler: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    class_weights: torch.Tensor,
    user_to_index: dict[str, int],
) -> dict[str, Any]:
    model.train()
    model.set_p46_pretraining(False)
    sampler.set_epoch(1000 + epoch)
    totals: Counter[str] = Counter()
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    observed: list[str] = []
    samples = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index, raw_batch in enumerate(loader):
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
        shifted, _ = circular_shift_visual_batch(batch, offset_labels)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.amp_dtype):
            output = model(batch, subject_adversarial_scale=1.0)
            shifted_relationship = model.forward_relationship(shifted)
            structure = event_structure_losses(output, batch)
            classification = F.cross_entropy(
                output["detail_logits"],
                labels,
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
            p46_aux = F.cross_entropy(
                output["p46_detail_logits"],
                labels,
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
            relationship_aux = F.cross_entropy(
                output["relationship_detail_logits"],
                labels,
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
            rival = hardest_rival_loss(output["detail_logits"], labels)
            contrast = cross_subject_supervised_contrastive(
                output["contrast_embedding"], labels, subjects
            )
            relationship_contrast = cross_subject_supervised_contrastive(
                output["relationship_contrast_embedding"], labels, subjects
            )
            subject = F.cross_entropy(output["subject_logits"], subjects)
            offset = F.cross_entropy(
                shifted_relationship["offset_logits"], offset_labels
            )
            localization = relationship_localization_loss(output)
            total = (
                args.main_weight * classification
                + args.p46_aux_weight * p46_aux
                + args.relationship_aux_weight * relationship_aux
                + args.rival_weight * rival
                + args.contrast_weight * contrast
                + args.relationship_contrast_weight * relationship_contrast
                + args.subject_weight * subject
                + 0.05 * structure["alignment"]
                + 0.05 * structure["contact"]
                + 0.05 * structure["phase"]
                + 0.02 * structure["order"]
                + args.offset_weight * offset
                + args.localization_weight * localization
            )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite P46 full-repair loss at epoch {epoch}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not torch.isfinite(gradient):
            bad = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            raise FloatingPointError(f"non-finite gradients at epoch {epoch}: {bad}")
        scaler.step(optimizer)
        scaler.update()

        logged = {
            "total": total,
            "classification": classification,
            "p46_aux": p46_aux,
            "relationship_aux": relationship_aux,
            "rival": rival,
            "contrast": contrast,
            "relationship_contrast": relationship_contrast,
            "subject": subject,
            "alignment": structure["alignment"],
            "contact": structure["contact"],
            "phase": structure["phase"],
            "order": structure["order"],
            "offset": offset,
            "localization": localization,
            "offset_accuracy": (
                shifted_relationship["offset_logits"].argmax(1) == offset_labels
            ).float().mean(),
            "gradient_norm": gradient,
        }
        for key, value in logged.items():
            totals[key] += float(value.detach()) * batch_size
        labels_all.append(labels.detach().cpu())
        logits_all.append(output["detail_logits"].detach().float().cpu())
        samples += batch_size
        if args.log_every and (batch_index + 1) % args.log_every == 0:
            print(
                f"full-repair epoch={epoch} batch={batch_index+1}/{len(loader)} "
                f"samples={samples} loss={totals['total']/samples:.4f} "
                f"offset_acc={totals['offset_accuracy']/samples:.3f}",
                flush=True,
            )

    coverage = source_coverage_report(observed, sampler.source_ids)
    if not coverage["exact_once"]:
        raise RuntimeError(f"P46 full-repair epoch coverage failed: {coverage}")
    labels_array = torch.cat(labels_all).numpy()
    logits_array = torch.cat(logits_all).numpy()
    return {
        "losses": {key: value / samples for key, value in totals.items()},
        "metrics": detail_metrics(labels_array, logits_array),
        "coverage": coverage,
        "seconds": time.perf_counter() - started,
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }


@torch.inference_mode()
def evaluate(
    model: P46FullRepairModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    *,
    relationship_scale: float = 1.0,
    relationship_offset_index: int | None = None,
    visual_scale: float = 1.0,
    visual_difference_scale: float = 1.0,
    skeleton_scale: float = 1.0,
    imu_scale: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    model.set_p46_pretraining(False)
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    p46_logits: list[torch.Tensor] = []
    relationship_logits: list[torch.Tensor] = []
    source_ids: list[str] = []
    users: list[str] = []
    losses = 0.0
    samples = 0
    started = time.perf_counter()
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        relationship_batch = None
        if relationship_offset_index is not None:
            target = torch.full(
                (len(batch["detail_index"]),),
                relationship_offset_index,
                dtype=torch.long,
                device=device,
            )
            relationship_batch, _ = circular_shift_visual_batch(batch, target)
        with amp_context(device, amp_dtype):
            output = model(
                batch,
                relationship_batch=relationship_batch,
                relationship_scale=relationship_scale,
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
        p46_logits.append(output["p46_detail_logits"].float().cpu())
        relationship_logits.append(output["relationship_detail_logits"].float().cpu())
        source_ids.extend(batch["source_id"])
        users.extend(batch["user_id"])
    label_array = torch.cat(labels).numpy()
    logit_array = torch.cat(logits).numpy()
    return {
        "loss": losses / samples,
        "metrics": detail_metrics(label_array, logit_array),
        "p46_aux_metrics": detail_metrics(label_array, torch.cat(p46_logits).numpy()),
        "relationship_aux_metrics": detail_metrics(
            label_array, torch.cat(relationship_logits).numpy()
        ),
        "labels": label_array,
        "logits": logit_array,
        "source_ids": source_ids,
        "users": users,
        "samples": samples,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_offsets(
    model: P46FullRepairModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, Any]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    started = time.perf_counter()
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        for offset_index in range(len(P46R_OFFSET_FRACTIONS)):
            target = torch.full(
                (len(batch["detail_index"]),),
                offset_index,
                dtype=torch.long,
                device=device,
            )
            shifted, _ = circular_shift_visual_batch(batch, target)
            with amp_context(device, amp_dtype):
                output = model.forward_relationship(shifted)
            labels.append(target.cpu())
            logits.append(output["offset_logits"].float().cpu())
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


def compare_condition(
    baseline: dict[str, Any], condition: dict[str, Any]
) -> dict[str, Any]:
    if baseline["source_ids"] != condition["source_ids"]:
        raise RuntimeError("audit prediction order differs")
    base_prediction = baseline["logits"].argmax(1)
    condition_prediction = condition["logits"].argmax(1)
    labels = baseline["labels"]
    base_correct = base_prediction == labels
    condition_correct = condition_prediction == labels
    return {
        "metrics": condition["metrics"],
        "accuracy_delta_pp": 100.0
        * (condition["metrics"]["accuracy"] - baseline["metrics"]["accuracy"]),
        "macro_f1_delta_pp": 100.0
        * (condition["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"]),
        "prediction_changed": int((base_prediction != condition_prediction).sum()),
        "baseline_correct_to_wrong": int((base_correct & ~condition_correct).sum()),
        "baseline_wrong_to_correct": int((~base_correct & condition_correct).sum()),
    }


def run_audit(
    model: P46FullRepairModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    output_dir: Path,
) -> dict[str, Any]:
    baseline = evaluate(model, loader, device, amp_dtype)
    specifications: dict[str, dict[str, Any]] = {
        "relationship_removed": {"relationship_scale": 0.0},
        "visual_shift_minus_quarter": {"relationship_offset_index": 0},
        "visual_shift_minus_eighth": {"relationship_offset_index": 1},
        "visual_shift_plus_eighth": {"relationship_offset_index": 3},
        "visual_shift_plus_quarter": {"relationship_offset_index": 4},
        "relationship_visual_difference_removed": {"visual_difference_scale": 0.0},
        "relationship_local_visual_removed": {"visual_scale": 0.0},
        "relationship_skeleton_removed": {"skeleton_scale": 0.0},
        "relationship_imu_removed": {"imu_scale": 0.0},
    }
    comparisons: dict[str, Any] = {}
    for index, (name, keyword) in enumerate(specifications.items(), start=1):
        atomic_json(
            output_dir / "progress.json",
            {
                "stage": "relationship_audit",
                "condition": name,
                "index": index,
                "count": len(specifications),
            },
        )
        comparisons[name] = compare_condition(
            baseline,
            evaluate(model, loader, device, amp_dtype, **keyword),
        )
    offset = evaluate_offsets(model, loader, device, amp_dtype)
    shift_deltas = [
        value["accuracy_delta_pp"]
        for name, value in comparisons.items()
        if name.startswith("visual_shift_")
    ]
    result = {
        "baseline": baseline["metrics"],
        "p46_auxiliary": baseline["p46_aux_metrics"],
        "relationship_auxiliary": baseline["relationship_aux_metrics"],
        "conditions": comparisons,
        "mean_visual_shift_accuracy_delta_pp": float(np.mean(shift_deltas)),
        "offset": offset,
    }
    atomic_json(output_dir / "relationship_audit.json", result)
    save_evaluation(output_dir, "final_best", baseline, epoch=-1)
    return result


def preflight(
    model: P46FullRepairModel,
    train: P46EventDataset,
    val: P46EventDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sampler: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reports = []
    for sampler_epoch in list(range(1, args.stage_a_epochs + 1)) + list(
        range(1001, 1001 + args.epochs)
    ):
        sampler.set_epoch(sampler_epoch)
        report = sampler.coverage_report()
        if not report["exact_once"]:
            raise RuntimeError(f"full-repair preflight coverage failed: {report}")
        reports.append(report)

    batch = move_batch(next(iter(train_loader)), device)
    labels = batch["detail_index"]
    offset_labels = torch.arange(len(labels), device=device).remainder(
        len(P46R_OFFSET_FRACTIONS)
    )
    shifted, _ = circular_shift_visual_batch(batch, offset_labels)
    model.train()
    model.set_p46_pretraining(False)
    model.zero_grad(set_to_none=True)
    with amp_context(device, args.amp_dtype):
        output = model(batch, subject_adversarial_scale=1.0)
        shifted_relation = model.forward_relationship(shifted)
        loss = (
            F.cross_entropy(output["detail_logits"], labels)
            + 0.5 * F.cross_entropy(shifted_relation["offset_logits"], offset_labels)
            + 0.15 * relationship_localization_loss(output)
        )
    loss.backward()

    def gradient_report(module: torch.nn.Module) -> dict[str, Any]:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        return {
            "present": bool(gradients),
            "finite": bool(gradients)
            and all(torch.isfinite(value).all() for value in gradients),
            "nonzero": any(float(value.abs().sum()) > 0.0 for value in gradients),
        }

    validation_batches = list(val_loader.batch_sampler)
    result = {
        "stage": "P46_full_repair_from_scratch_preflight",
        "pretrained_checkpoint_loaded": False,
        "frozen_parameters": [
            name for name, parameter in model.named_parameters() if not parameter.requires_grad
        ],
        "train_trials": len(train),
        "val_trials": len(val),
        "subject_overlap": sorted(
            {row["user_id"] for row in train.rows}
            & {row["user_id"] for row in val.rows}
        ),
        "sampler_epochs": reports,
        "all_planned_epochs_exact_once": all(value["exact_once"] for value in reports),
        "validation_coverage": batch_coverage_report(validation_batches, len(val)),
        "forward_finite": bool(torch.isfinite(output["detail_logits"]).all()),
        "relationship_input_shape": list(output["relationship_input"].shape),
        "p46_gradient": gradient_report(model.p46),
        "relationship_gradient": gradient_report(model.relationship),
        "fusion_gradient": gradient_report(model.fusion),
        "detail_head_gradient": gradient_report(model.detail_head),
        "all_parameters_trainable": all(
            parameter.requires_grad for parameter in model.parameters()
        ),
    }
    model.zero_grad(set_to_none=True)
    return result


def checkpoint_payload(
    model: P46FullRepairModel,
    epoch: int,
    evaluation: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "P46_full_repair_from_scratch",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": evaluation["metrics"],
        "p46_aux_metrics": evaluation["p46_aux_metrics"],
        "relationship_aux_metrics": evaluation["relationship_aux_metrics"],
        "config": config,
    }


def main() -> None:
    args = parse_args()
    if args.stage_a_epochs < 1 or args.epochs < 1:
        raise ValueError("stage-a-epochs and epochs must be positive")
    if args.offset_eval_every < 0:
        raise ValueError("offset-eval-every cannot be negative")
    if args.resume is not None and args.smoke:
        raise ValueError("--resume cannot be combined with --smoke")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        raise ValueError("--stop-after-epoch must be 0 or within the Stage-B epoch range")
    if os.name == "nt" and args.workers > 0:
        print(f"forcing --workers {args.workers} to 0 on Windows", flush=True)
        args.workers = 0
    if args.smoke:
        args.stage_a_epochs = 1
        args.epochs = 2
        args.offset_eval_every = 1

    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    train, val = make_datasets(args)
    if not args.smoke and (
        len(train) != EXPECTED["train_detail_trials"]
        or len(val) != EXPECTED["val_detail_trials"]
    ):
        raise RuntimeError(f"full-repair split changed: {len(train)}/{len(val)}")
    train_users = sorted({row["user_id"] for row in train.rows})
    val_users = sorted({row["user_id"] for row in val.rows})
    if set(train_users) & set(val_users):
        raise RuntimeError("P46 full-repair subject leakage")
    user_to_index = {value: index for index, value in enumerate(train_users)}
    train_loader, train_sampler, val_loader, _ = make_full_repair_loaders(
        train, val, args
    )

    # A new run starts from this random constructor. A resume checkpoint is only
    # accepted when it contains the full state of this same from-scratch run.
    model = P46FullRepairModel(subjects=len(train_users), dropout=0.12).to(device)
    resume_checkpoint: dict[str, Any] | None = None
    resume_epoch = 0
    if args.resume is not None:
        resume_checkpoint = load_resume_checkpoint(
            args.resume,
            args,
            train_trials=len(train),
            val_trials=len(val),
        )
        resume_epoch = int(resume_checkpoint["epoch"])
        if args.stop_after_epoch and args.stop_after_epoch <= resume_epoch:
            raise ValueError(
                f"--stop-after-epoch {args.stop_after_epoch} is not after resume epoch "
                f"{resume_epoch}"
            )
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
    preflight_result = preflight(
        model,
        train,
        val,
        train_loader,
        val_loader,
        train_sampler,
        device,
        args,
    )
    preflight_result.update(
        {
            "resume_checkpoint_loaded": resume_checkpoint is not None,
            "resume_epoch": resume_epoch,
            "resume_path": str(args.resume.resolve()) if args.resume else None,
            "pin_memory": False,
            "memory_after_preflight": release_epoch_memory(device),
        }
    )
    atomic_json(output / "preflight.json", preflight_result)
    required = (
        not preflight_result["pretrained_checkpoint_loaded"],
        not preflight_result["frozen_parameters"],
        not preflight_result["subject_overlap"],
        preflight_result["all_planned_epochs_exact_once"],
        preflight_result["validation_coverage"]["exact_once"],
        preflight_result["forward_finite"],
        preflight_result["all_parameters_trainable"],
        *(
            preflight_result[key][criterion]
            for key in (
                "p46_gradient",
                "relationship_gradient",
                "fusion_gradient",
                "detail_head_gradient",
            )
            for criterion in ("present", "finite", "nonzero")
        ),
    )
    if not all(required):
        raise RuntimeError(f"P46 full-repair preflight failed: {preflight_result}")
    if args.preflight_only:
        print(json.dumps(preflight_result, ensure_ascii=False, indent=2), flush=True)
        return

    train_labels = [HARD_CLASS_TO_INDEX[int(row["class_id"])] for row in train.rows]
    class_weights_cpu, class_counts = make_class_weights(
        train_labels, 21, args.class_weight_power
    )
    class_weights = class_weights_cpu.to(device)
    p12_baseline = (
        compute_p12_restricted_baseline(val, args.p12_oof.resolve())[0]
        if not args.smoke
        else None
    )
    config = {
        "stage": "P46_full_repair_from_scratch",
        "architecture": (
            "concat(P46 trial embedding, P46-R ordered relationship embedding, "
            "five-part lag correlations, event centres) -> shared fusion -> Detail21"
        ),
        "pretrained_checkpoint_loaded": False,
        "frozen_branches": False,
        "output_gating_used": False,
        "random_initialisation": True,
        "resume_checkpoint_loaded": resume_checkpoint is not None,
        "resume_epoch": resume_epoch,
        "resume_path": str(args.resume.resolve()) if args.resume else None,
        "resume_is_same_training_run_not_pretraining": True,
        "train_trials": len(train),
        "val_trials": len(val),
        "train_subjects": train_users,
        "val_subjects": val_users,
        "stage_a_epochs": args.stage_a_epochs,
        "stage_b_epochs": args.epochs,
        "early_stopping": False,
        "stage_a_learning_rate": args.stage_a_learning_rate,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_counts": class_counts,
        "class_weights": [float(value) for value in class_weights_cpu],
        "offset_eval_every": args.offset_eval_every,
        "offset_fractions": list(P46R_OFFSET_FRACTIONS),
        "loss_weights": {
            "main": args.main_weight,
            "p46_aux": args.p46_aux_weight,
            "relationship_aux": args.relationship_aux_weight,
            "rival": args.rival_weight,
            "contrast": args.contrast_weight,
            "relationship_contrast": args.relationship_contrast_weight,
            "subject": args.subject_weight,
            "p46_alignment": 0.05,
            "p46_contact": 0.05,
            "p46_phase": 0.05,
            "p46_order": 0.02,
            "relationship_offset": args.offset_weight,
            "relationship_localization": args.localization_weight,
        },
        "all_train_epochs_exact_once_planned": True,
        "pin_memory": False,
        "host_memory_cleanup_each_epoch": True,
        "model_parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "amp_dtype": args.amp_dtype,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "p12_restricted_baseline": p12_baseline,
        "smoke": args.smoke,
    }
    atomic_json(
        output / ("resume_config.json" if resume_checkpoint else "frozen_config.json"),
        config,
    )
    print(json.dumps({"stage": "start", **config}, ensure_ascii=False), flush=True)

    run_started = time.perf_counter()
    stage_a_history: list[dict[str, Any]] = []
    stage_a_epochs_completed = args.stage_a_epochs if resume_checkpoint else 0
    if resume_checkpoint is None:
        model.set_p46_pretraining(True)
        stage_a_optimizer = make_optimizer(
            model, args.stage_a_learning_rate, args.weight_decay
        )
        stage_a_scaler = torch.amp.GradScaler(
            device.type,
            enabled=device.type == "cuda",
            init_scale=4096.0,
            growth_interval=1000,
        )
        for epoch in range(1, args.stage_a_epochs + 1):
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
            )
            row = {"epoch": epoch, "learning_rate": rate, **result}
            stage_a_history.append(row)
            stage_a_epochs_completed = epoch
            write_csv(output / "stage_a_history.csv", stage_a_history)
            atomic_checkpoint(
                output / "stage_a_last.pt",
                {
                    "stage": "P46_full_repair_stage_a_from_scratch",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
            )
            print(json.dumps({"stage": "A_epoch", **row}, ensure_ascii=False), flush=True)
        del stage_a_optimizer, stage_a_scaler
        release_epoch_memory(device)
    else:
        print(
            json.dumps(
                {
                    "stage": "resume",
                    "checkpoint": str(args.resume.resolve()),
                    "completed_stage_a_epochs": stage_a_epochs_completed,
                    "completed_stage_b_epochs": resume_epoch,
                    "next_epoch": resume_epoch + 1,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    model.set_p46_pretraining(False)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.amp_dtype == "float16",
        init_scale=4096.0,
        growth_interval=1000,
    )
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_accuracy = -math.inf
    best_macro_f1 = -math.inf
    best_epoch = 0
    best_accuracy_epoch = 0
    best_macro_epoch = 0
    start_epoch = 1
    rng_restored = False
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        history = list(resume_checkpoint["history"])
        best_score = float(resume_checkpoint["best_score"])
        best_accuracy = float(resume_checkpoint["best_accuracy"])
        best_macro_f1 = float(resume_checkpoint["best_macro_f1"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        best_accuracy_epoch = int(
            max(history, key=lambda row: float(row["val_accuracy"]))["epoch"]
        )
        best_macro_epoch = int(
            max(history, key=lambda row: float(row["val_macro_f1"]))["epoch"]
        )
        start_epoch = resume_epoch + 1
        if "rng_state" in resume_checkpoint:
            restore_rng_state(resume_checkpoint["rng_state"])
            rng_restored = True
        else:
            # The interrupted epoch-6 checkpoint predates RNG capture. Sampler and
            # LR are still exact by epoch; this deterministically restarts dropout.
            seed_everything(args.seed + resume_epoch)
        del resume_checkpoint
        release_epoch_memory(device)
        atomic_json(
            output / "resume_state.json",
            {
                "resume_epoch": resume_epoch,
                "next_epoch": start_epoch,
                "optimizer_restored": True,
                "scaler_restored": True,
                "history_rows": len(history),
                "rng_state_restored": rng_restored,
                "rng_fallback_seed": None if rng_restored else args.seed + resume_epoch,
                "memory": memory_snapshot(device),
            },
        )

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
        )
        validation = evaluate(model, val_loader, device, args.amp_dtype)
        offset_validation = None
        if args.offset_eval_every and (
            epoch == 1
            or epoch % args.offset_eval_every == 0
            or epoch == args.epochs
        ):
            offset_validation = evaluate_offsets(
                model, val_loader, device, args.amp_dtype
            )
        accuracy = validation["metrics"]["accuracy"]
        macro_f1 = validation["metrics"]["macro_f1"]
        train_accuracy = training["metrics"]["accuracy"]
        offset_accuracy = offset_validation["accuracy"] if offset_validation else None
        score = macro_f1 + 0.25 * accuracy
        payload = checkpoint_payload(model, epoch, validation, config)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            atomic_checkpoint(output / "best.pt", payload)
            save_evaluation(output, "best", validation, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch
            atomic_checkpoint(output / "best_accuracy.pt", payload)
            save_evaluation(output, "best_accuracy", validation, epoch)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_macro_epoch = epoch
            atomic_checkpoint(output / "best_macro_f1.pt", payload)
            save_evaluation(output, "best_macro_f1", validation, epoch)

        row = {
            "epoch": epoch,
            "learning_rate": rate,
            **{f"train_{key}": value for key, value in training["losses"].items()},
            **{f"train_{key}": value for key, value in training["metrics"].items()},
            **{f"val_{key}": value for key, value in validation["metrics"].items()},
            "val_p46_aux_accuracy": validation["p46_aux_metrics"]["accuracy"],
            "val_relationship_aux_accuracy": validation["relationship_aux_metrics"][
                "accuracy"
            ],
            "val_offset_accuracy": (
                offset_validation["accuracy"] if offset_validation else None
            ),
            "train_seconds": training["seconds"],
            "val_seconds": validation["seconds"],
            "offset_val_seconds": (
                offset_validation["seconds"] if offset_validation else 0.0
            ),
            "peak_cuda_mib": training["peak_cuda_mib"],
            "coverage_exact_once": training["coverage"]["exact_once"],
            "best_epoch": best_epoch,
            "best_accuracy_epoch": best_accuracy_epoch,
            "best_macro_epoch": best_macro_epoch,
        }
        del training, validation, offset_validation
        row.update(
            {
                f"post_epoch_{key}": value
                for key, value in release_epoch_memory(device).items()
            }
        )
        history.append(row)
        write_csv(output / "history.csv", history)
        atomic_checkpoint(
            output / "last.pt",
            {
                **payload,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_accuracy": best_accuracy,
                "best_macro_f1": best_macro_f1,
                "rng_state": capture_rng_state(),
            },
        )
        atomic_json(
            output / "progress.json",
            {
                "stage": "training",
                "epoch": epoch,
                "epochs": args.epochs,
                "val_accuracy": accuracy,
                "val_macro_f1": macro_f1,
                "val_offset_accuracy": offset_accuracy,
                "best_epoch": best_epoch,
                "memory": {
                    key.removeprefix("post_epoch_"): value
                    for key, value in row.items()
                    if key.startswith("post_epoch_")
                },
            },
        )
        print(
            f"full-repair epoch={epoch}/{args.epochs} "
            f"train_acc={train_accuracy:.4f} "
            f"val_acc={accuracy:.4f} val_macro={macro_f1:.4f} "
            f"val_offset={offset_accuracy} best_epoch={best_epoch} "
            f"rss_mib={row.get('post_epoch_process_rss_mib')} "
            f"host_available_mib={row.get('post_epoch_host_available_mib')}",
            flush=True,
        )
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            paused = {
                "stage": "controlled_pause",
                "completed_epoch": epoch,
                "planned_epochs": args.epochs,
                "next_epoch": epoch + 1,
                "last_checkpoint": str((output / "last.pt").resolve()),
                "formal_run_complete": False,
                "memory": {
                    key.removeprefix("post_epoch_"): value
                    for key, value in row.items()
                    if key.startswith("post_epoch_")
                },
            }
            atomic_json(output / "progress.json", paused)
            atomic_json(output / "controlled_pause.json", paused)
            print(json.dumps(paused, ensure_ascii=False, indent=2), flush=True)
            return

    if len(history) != args.epochs:
        raise RuntimeError(
            f"formal run ended at {len(history)}/{args.epochs} completed epochs"
        )
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    audit = run_audit(model, val_loader, device, args.amp_dtype, output)
    best_metrics = checkpoint["metrics"]
    summary = {
        "stage": "P46_full_repair_from_scratch_complete",
        "random_initialisation": True,
        "pretrained_checkpoint_loaded": False,
        "resumed_same_training_run": resume_epoch > 0,
        "resumed_from_epoch": resume_epoch,
        "frozen_branches": False,
        "output_gating_used": False,
        "stage_a_epochs_completed": stage_a_epochs_completed,
        "stage_b_epochs_completed": len(history),
        "all_epochs_exact_once": all(
            bool(row["coverage_exact_once"]) for row in history
        ),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_accuracy": best_accuracy,
        "best_macro_epoch": best_macro_epoch,
        "best_macro_f1": best_macro_f1,
        "p46_original_accuracy": 0.4068965517241379,
        "p12_restricted_accuracy": (
            p12_baseline["accuracy"] if p12_baseline else None
        ),
        "accuracy_delta_vs_original_p46_pp": 100.0
        * (best_metrics["accuracy"] - 0.4068965517241379),
        "accuracy_delta_vs_p12_pp": (
            100.0 * (best_metrics["accuracy"] - p12_baseline["accuracy"])
            if p12_baseline
            else None
        ),
        "relationship_audit": audit,
        "elapsed_seconds": time.perf_counter() - run_started,
        "artifacts": {
            "stage_a": str((output / "stage_a_last.pt").resolve()),
            "best": str((output / "best.pt").resolve()),
            "last": str((output / "last.pt").resolve()),
            "history": str((output / "history.csv").resolve()),
            "audit": str((output / "relationship_audit.json").resolve()),
        },
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "progress.json",
        {
            "stage": "complete",
            "completed_epoch": len(history),
            "planned_epochs": args.epochs,
            "best_epoch": best_epoch,
            "formal_run_complete": True,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
