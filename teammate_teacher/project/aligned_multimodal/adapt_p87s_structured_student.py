from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from p86_cached_motion_data import (
    P86CachedSequenceMotionDataset,
    collate_p86_cached_motion,
)
from train_p86_mobind_fusion_proxy import (
    build_model,
    evaluate,
    model_forward,
    visual_head_parameters,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGETS = (
    PROJECT_DIR / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short label-free P87-S adaptation from one common subject-disjoint P86 "
            "fusion checkpoint. The training dataset deletes true labels before collation."
        )
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--structured-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--evaluation-ids",
        type=Path,
        default=None,
        help=(
            "Optional .npy sample-ID list for honest inductive evaluation. When set, "
            "adaptation uses only structured-target rows and evaluates this disjoint list."
        ),
    )
    parser.add_argument("--target", choices=("emission", "structured"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fusion-learning-rate", type=float, default=5e-5)
    parser.add_argument("--visual-head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--emission-warmup-epochs",
        type=int,
        default=0,
        help=(
            "For a structured run, spend this many of the same total epochs on P85 "
            "emissions before switching to structured targets."
        ),
    )
    parser.add_argument(
        "--confidence-power",
        type=float,
        default=0.0,
        help=(
            "Optional structured-confidence weighting. Zero is the mechanism baseline; "
            "nonzero values are reserved for a result-driven follow-up."
        ),
    )
    parser.add_argument(
        "--adaptation-scope",
        choices=("heads", "heads_motion_encoder"),
        default="heads",
        help=(
            "heads preserves the frozen P87-S recipe; heads_motion_encoder also "
            "updates the compact Skeleton/IMU encoder while the cached visual "
            "backbone remains frozen."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def model_build_args(
    base_checkpoint: Path, summary: dict[str, Any]
) -> SimpleNamespace:
    config = summary["config"]
    repository_root = PROJECT_DIR.parent
    return SimpleNamespace(
        visual_checkpoint=resolve_config_path(config["visual_checkpoint"], repository_root),
        pretrain_checkpoint=resolve_config_path(
            config["pretrain_checkpoint"], repository_root
        ),
        modality=summary["modality"],
        skeleton_time_position=bool(config.get("skeleton_time_position", False)),
        skeleton_multistream=bool(config.get("skeleton_multistream", False)),
        skeleton_multistream_strength=float(
            config.get("skeleton_multistream_strength", 0.1)
        ),
        skeleton_adaptive_graph=bool(config.get("skeleton_adaptive_graph", False)),
        separate_modality_dropout=float(config.get("separate_modality_dropout", 0.0)),
        joint_initial_imu_gate=float(config.get("joint_initial_imu_gate", 0.35)),
        joint_max_imu_residual=float(config.get("joint_max_imu_residual", 1.0)),
        imu_event_features=(
            resolve_config_path(config["imu_event_features"], repository_root)
            if config.get("imu_event_features")
            else None
        ),
        initial_residual_strength=float(config.get("initial_residual_strength", 0.25)),
        global_fusion_mode=str(config.get("global_fusion_mode", "additive")),
        reliability_groups=int(config.get("reliability_groups", 1)),
        # P93 extended the shared model factory after the original P87-S
        # checkpoints and adaptation runner were frozen.  Old checkpoints have
        # no fusion-position fields and must continue to reconstruct the exact
        # P86 clip-level graph rather than failing or silently selecting P93.
        fusion_position=str(config.get("fusion_position", "clip")),
        temporal_radius=int(config.get("temporal_radius", 1)),
        temporal_residual_budget=float(config.get("temporal_residual_budget", 0.10)),
        temporal_attention_logit_limit=float(
            config.get("temporal_attention_logit_limit", 1.0)
        ),
        spatial_grid=int(config.get("spatial_grid", 5)),
        spatial_attention_logit_limit=float(
            config.get("spatial_attention_logit_limit", 1.0)
        ),
        base_checkpoint=base_checkpoint,
    )


class LabelFreePseudoDataset(Dataset[dict[str, Any]]):
    """Remove the ground-truth field before it can enter an adaptation batch."""

    def __init__(
        self,
        base: P86CachedSequenceMotionDataset,
        probability_by_id: dict[str, np.ndarray],
        confidence_by_id: dict[str, float],
        secondary_probability_by_id: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.base = base
        self.probability_by_id = probability_by_id
        self.confidence_by_id = confidence_by_id
        self.secondary_probability_by_id = secondary_probability_by_id

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        sample_id = str(item["sample_id"])
        # P86CachedSequenceMotionDataset exposes labels for supervised training. Delete
        # them here so the pseudo adaptation loss cannot consume them by accident.
        item.pop("label", None)
        item["pseudo_probability"] = torch.from_numpy(
            self.probability_by_id[sample_id].astype(np.float32, copy=True)
        )
        if self.secondary_probability_by_id is not None:
            item["pseudo_emission_probability"] = torch.from_numpy(
                self.secondary_probability_by_id[sample_id].astype(
                    np.float32, copy=True
                )
            )
        item["pseudo_confidence"] = torch.tensor(
            self.confidence_by_id[sample_id], dtype=torch.float32
        )
        return item


def make_loader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        collate_fn=collate_p86_cached_motion,
        drop_last=False,
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def configure_adaptation_parameters(
    model, adaptation_scope: str = "heads"
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    if adaptation_scope not in {"heads", "heads_motion_encoder"}:
        raise ValueError(f"unknown adaptation scope: {adaptation_scope}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual_parameters = visual_head_parameters(model.visual)
    for parameter in visual_parameters:
        parameter.requires_grad_(True)
    fusion_parameters = []
    for name, parameter in model.motion_residual.named_parameters():
        if adaptation_scope == "heads_motion_encoder" or not name.startswith("encoder."):
            parameter.requires_grad_(True)
            fusion_parameters.append(parameter)
    if not visual_parameters or not fusion_parameters:
        raise RuntimeError("P87-S adaptation parameter groups are unexpectedly empty")
    return fusion_parameters, visual_parameters


def tempered_probability(probability: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return probability
    scaled = probability.clamp_min(1e-12).pow(1.0 / temperature)
    return scaled / scaled.sum(dim=1, keepdim=True)


def train_label_free(
    model,
    data: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    fusion_parameters, visual_parameters = configure_adaptation_parameters(
        model, args.adaptation_scope
    )
    groups = [
        {
            "params": fusion_parameters,
            "lr": args.fusion_learning_rate,
            "initial_lr": args.fusion_learning_rate,
        },
        {
            "params": visual_parameters,
            "lr": args.visual_head_learning_rate,
            "initial_lr": args.visual_head_learning_rate,
        },
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = args.minimum_learning_rate + (
                float(group["initial_lr"]) - args.minimum_learning_rate
            ) * cosine
        loss_sum = entropy_sum = confidence_sum = 0.0
        agreement = samples = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(data):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            if "label" in batch:
                raise RuntimeError("Ground-truth label entered label-free P87-S adaptation")
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model_forward(model, batch)
                target_kind = (
                    "emission"
                    if epoch <= args.emission_warmup_epochs
                    else args.target
                )
                target_tensor = (
                    batch["pseudo_emission_probability"]
                    if target_kind == "emission"
                    and "pseudo_emission_probability" in batch
                    else batch["pseudo_probability"]
                )
                target = tempered_probability(target_tensor.float(), args.temperature)
                per_sample = F.kl_div(
                    F.log_softmax(output["logits"].float() / args.temperature, dim=1),
                    target,
                    reduction="none",
                ).sum(dim=1) * args.temperature**2
                if args.confidence_power > 0:
                    sample_weight = batch["pseudo_confidence"].float().clamp_min(1e-3).pow(
                        args.confidence_power
                    )
                    loss = (per_sample * sample_weight).sum() / sample_weight.sum()
                else:
                    loss = per_sample.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["sample_id"])
            samples += count
            loss_sum += float(loss.detach()) * count
            entropy_sum += float(
                (-(target * target.clamp_min(1e-12).log()).sum(dim=1)).sum().detach()
            )
            confidence_sum += float(batch["pseudo_confidence"].sum().detach())
            agreement += int(
                output["logits"].argmax(dim=1).eq(target.argmax(dim=1)).sum().detach()
            )
        record = {
            "epoch": epoch,
            "target": (
                "emission"
                if epoch <= args.emission_warmup_epochs
                else args.target
            ),
            "fusion_learning_rate": optimizer.param_groups[0]["lr"],
            "visual_head_learning_rate": optimizer.param_groups[1]["lr"],
            "train_kl": loss_sum / max(samples, 1),
            "target_entropy": entropy_sum / max(samples, 1),
            "mean_target_confidence": confidence_sum / max(samples, 1),
            "student_target_agreement": agreement / max(samples, 1),
            "train_samples": samples,
            "ground_truth_fields_seen": 0,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return history


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_delta(base: dict[str, Any], adapted: dict[str, Any]) -> dict[str, float | int]:
    return {
        "correct": int(adapted["correct"] - base["correct"]),
        "accuracy_pp": float(100.0 * (adapted["accuracy"] - base["accuracy"])),
        "macro_f1_pp": float(100.0 * (adapted["macro_f1"] - base["macro_f1"])),
        "worst_subject_accuracy_pp": float(
            100.0
            * (
                adapted["worst_subject_accuracy"]
                - base["worst_subject_accuracy"]
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.confidence_power < 0:
        raise ValueError("confidence-power must be nonnegative")
    if args.emission_warmup_epochs < 0 or args.emission_warmup_epochs >= args.epochs:
        if args.emission_warmup_epochs != 0:
            raise ValueError(
                "emission-warmup-epochs must be nonnegative and smaller than epochs"
            )
    if args.emission_warmup_epochs and args.target != "structured":
        raise ValueError("emission warmup is only meaningful for a structured run")
    if args.emission_warmup_epochs and args.confidence_power > 0:
        raise ValueError(
            "curriculum and confidence weighting are separate mechanisms; test them "
            "in separate controlled runs"
        )
    if args.smoke:
        args.epochs = 1
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
    seed_all(args.seed)

    base_checkpoint_path = args.base_checkpoint.resolve()
    base_dir = base_checkpoint_path.parent
    base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    if base_summary.get("stage") != "P87S_mobind_fusion_subject_holdout":
        raise ValueError("base checkpoint is not a P87-S subject-holdout fusion model")
    build_args = model_build_args(base_checkpoint_path, base_summary)
    model, visual_config, pretrain_config = build_model(build_args)
    base_checkpoint = torch.load(
        base_checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(base_checkpoint["model_state"], strict=True)

    config = base_summary["config"]
    repository_root = PROJECT_DIR.parent
    full = P86CachedSequenceMotionDataset(
        resolve_config_path(config["sequence_cache"], repository_root),
        resolve_config_path(config["motion_cache"], repository_root),
        resolve_config_path(config["pixel_cache"], repository_root),
        resolve_config_path(config["teacher_features"], repository_root),
        resolve_config_path(config["teacher_logits"], repository_root),
        imu_teacher_logits=(
            resolve_config_path(config["imu_teacher_logits"], repository_root)
            if config.get("imu_teacher_logits")
            else None
        ),
        imu_event_features=(
            resolve_config_path(config["imu_event_features"], repository_root)
            if config.get("imu_event_features")
            else None
        ),
    )
    targets = np.load(args.structured_targets.resolve(), allow_pickle=False)
    target_ids = targets["sample_ids"].astype(str)
    target_mask = targets["target_mask"].astype(bool)
    selected_ids = target_ids[target_mask]
    missing = sorted(set(selected_ids.tolist()) - set(full.index_lookup))
    if missing:
        raise ValueError(f"Student cache is missing {len(missing)} pseudo-target rows")
    adaptation_indices = np.asarray(
        [full.index_lookup[sample_id] for sample_id in selected_ids], dtype=np.int64
    )
    if args.evaluation_ids is None:
        evaluation_ids = selected_ids
    else:
        evaluation_ids = np.load(args.evaluation_ids.resolve(), allow_pickle=False).astype(str)
        overlap = set(selected_ids.tolist()) & set(evaluation_ids.tolist())
        if overlap:
            raise ValueError(
                f"inductive evaluation leaked {len(overlap)} evaluation IDs into adaptation"
            )
        missing_evaluation = sorted(set(evaluation_ids.tolist()) - set(full.index_lookup))
        if missing_evaluation:
            raise ValueError(
                f"Student cache is missing {len(missing_evaluation)} evaluation rows"
            )
    evaluation_indices = np.asarray(
        [full.index_lookup[sample_id] for sample_id in evaluation_ids], dtype=np.int64
    )
    target_key = (
        "emission_probability"
        if args.target == "emission"
        else "structured_distillation_probability"
    )
    probability = targets[target_key].astype(np.float32)[target_mask]
    confidence = (
        probability.max(axis=1)
        if args.target == "emission"
        else targets["structured_confidence"].astype(np.float32)[target_mask]
    )
    probability_by_id = {
        sample_id: value for sample_id, value in zip(selected_ids, probability)
    }
    emission_probability_by_id = None
    if args.emission_warmup_epochs:
        emission_probability = targets["emission_probability"].astype(np.float32)[
            target_mask
        ]
        emission_probability_by_id = {
            sample_id: value
            for sample_id, value in zip(selected_ids, emission_probability)
        }
    confidence_by_id = {
        sample_id: float(value) for sample_id, value in zip(selected_ids, confidence)
    }
    supervised_evaluation = P86CachedSequenceMotionDataset(
        full.sequence_cache,
        full.motion_cache,
        resolve_config_path(config["pixel_cache"], repository_root),
        resolve_config_path(config["teacher_features"], repository_root),
        resolve_config_path(config["teacher_logits"], repository_root),
        indices=evaluation_indices,
        temporal_augment=False,
        imu_teacher_logits=(
            resolve_config_path(config["imu_teacher_logits"], repository_root)
            if config.get("imu_teacher_logits")
            else None
        ),
        imu_event_features=(
            resolve_config_path(config["imu_event_features"], repository_root)
            if config.get("imu_event_features")
            else None
        ),
    )
    pseudo_base = P86CachedSequenceMotionDataset(
        full.sequence_cache,
        full.motion_cache,
        resolve_config_path(config["pixel_cache"], repository_root),
        resolve_config_path(config["teacher_features"], repository_root),
        resolve_config_path(config["teacher_logits"], repository_root),
        indices=adaptation_indices,
        temporal_augment=True,
        imu_teacher_logits=(
            resolve_config_path(config["imu_teacher_logits"], repository_root)
            if config.get("imu_teacher_logits")
            else None
        ),
        imu_event_features=(
            resolve_config_path(config["imu_event_features"], repository_root)
            if config.get("imu_event_features")
            else None
        ),
    )
    pseudo_dataset = LabelFreePseudoDataset(
        pseudo_base,
        probability_by_id,
        confidence_by_id,
        secondary_probability_by_id=emission_probability_by_id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = train_label_free(
        model,
        make_loader(pseudo_dataset, args.batch_size, args.workers, shuffle=True),
        args,
        device,
    )
    metrics, rows, logits = evaluate(
        model,
        make_loader(
            supervised_evaluation, args.batch_size, args.workers, shuffle=False
        ),
        device,
        args.max_eval_batches,
    )
    base_metrics = base_checkpoint["subject_holdout_metrics"]
    base_logits = np.load(base_dir / "subject_holdout_logits.npy")
    base_rows = list(csv.DictReader((base_dir / "subject_holdout_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    )))
    if not args.max_eval_batches:
        base_ids = [row["sample_id"] for row in base_rows]
        adapted_ids = [row["sample_id"] for row in rows]
        if base_ids != adapted_ids or base_ids != evaluation_ids.tolist():
            raise RuntimeError("Base/adapted/evaluation row order differs")
    compared = min(len(rows), len(base_rows))
    labels = np.asarray([int(row["label"]) for row in rows[:compared]])
    base_prediction = base_logits[:compared].argmax(axis=1)
    adapted_prediction = logits[:compared].argmax(axis=1)
    base_correct = base_prediction == labels
    adapted_correct = adapted_prediction == labels
    audit = {
        "base_to_adapted_rescue": int(np.sum(~base_correct & adapted_correct)),
        "base_to_adapted_harm": int(np.sum(base_correct & ~adapted_correct)),
        "evaluation_rows_seen_during_adaptation": 0 if args.evaluation_ids else compared,
        "inductive_evaluation": bool(args.evaluation_ids),
    }
    if args.evaluation_ids is None:
        target_prediction = probability[:compared].argmax(axis=1)
        target_correct = target_prediction == labels
        audit.update(
            {
                "student_target_agreement": float(
                    np.mean(adapted_prediction == target_prediction)
                ),
                "target_errors_copied": int(
                    np.sum(~target_correct & (adapted_prediction == target_prediction))
                ),
                "student_correct_when_target_wrong": int(
                    np.sum(~target_correct & adapted_correct)
                ),
                "student_exceeds_target_hard_correct": bool(
                    adapted_correct.sum() > target_correct.sum()
                ),
                "target_hard_correct": int(target_correct.sum()),
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "subject_holdout_predictions.csv", rows)
    np.save(output / "subject_holdout_logits.npy", logits)
    with (output / "training_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = {
        "stage": "P87S_label_free_adaptation",
        "target": args.target,
        "adaptation_scope": args.adaptation_scope,
        "emission_warmup_epochs": args.emission_warmup_epochs,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "visual_config": visual_config,
        "pretrain_config": pretrain_config,
        "modality": base_summary["modality"],
        "base_checkpoint": str(base_checkpoint_path),
        "subject_holdout_metrics": metrics,
    }
    torch.save(checkpoint, output / "unified_student.pt")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "stage": "P87S_label_free_adaptation",
        "status": "smoke" if args.smoke else "formal",
        "target": args.target,
        "adaptation_scope": args.adaptation_scope,
        "emission_warmup_epochs": args.emission_warmup_epochs,
        "protocol": (
            "Start from the identical labeled-only P87-S C0 checkpoint. Adapt on the "
            "held pseudo-Test inputs after deleting ground-truth labels from every batch. "
            "The visual backbone stays frozen because adaptation consumes its cached "
            "sequence. adaptation_scope controls whether the compact Skeleton/IMU "
            "encoder is frozen or updated with the visual and fusion heads."
        ),
        "base_checkpoint": str(base_checkpoint_path),
        "structured_targets": str(args.structured_targets.resolve()),
        "pseudo_rows": len(selected_ids),
        "evaluation_rows": len(evaluation_ids),
        "evaluation_ids": str(args.evaluation_ids.resolve()) if args.evaluation_ids else None,
        "ground_truth_fields_seen_during_training": 0,
        "base_metrics": base_metrics,
        "adapted_metrics": metrics,
        "delta_vs_base": metric_delta(base_metrics, metrics),
        "audit": audit,
        "parameters": parameters,
        "fp32_mib": parameters * 4 / 1024**2,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
