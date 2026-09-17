from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from p86_visual_pixel_data import P86VisualPixelDataset, collate_p86_pixels
from p86_mc3_visual_model import P86MC3VisualStudent
from p86_videomae_small_visual_model import (
    DEFAULT_VIDEOMAE_SMALL,
    P86VideoMAESmallVisualStudent,
)
from p86_visual_pixel_model import (
    P86TrainableVisualStudent,
    model_size_mib,
    parameter_count,
)
from train_p86_visual_student_oof import (
    class_weights,
    metric_dict,
    relation_loss,
    split_universe,
    write_rows,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = PROJECT_DIR / "runs/p86_visual_pixel_cache_v2"
DEFAULT_TEACHER_FEATURES = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR
    / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86_visual_pixel_fold0_hybrid_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested subject-disjoint training for the P86 V2 trainable visual student."
    )
    parser.add_argument(
        "--mode", choices=("mechanism", "kd", "hybrid", "feature"), default="hybrid"
    )
    parser.add_argument("--outer-fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--fusion-mode", choices=("gated", "structured", "gated_residual"), default="gated"
    )
    parser.add_argument(
        "--backbone",
        choices=("resnet18_tsm", "mc3_18", "mc3_18_temporal", "videomae_small"),
        default="resnet18_tsm",
    )
    parser.add_argument(
        "--freeze-through",
        choices=("layer1", "layer2", "layer3", "layer6", "layer8", "layer10"),
        default="layer2",
    )
    parser.add_argument("--frames", type=int, choices=(8, 12, 16), default=8)
    parser.add_argument("--input-resolution", type=int, default=112)
    parser.add_argument("--pretrained-model", default=DEFAULT_VIDEOMAE_SMALL)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-epochs", type=int, default=18)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=8e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.6)
    parser.add_argument("--stage-distillation-weight", type=float, default=0.0)
    parser.add_argument(
        "--exact-time-position",
        action="store_true",
        help="Add exact normalized source-frame time on top of the learned local slot position.",
    )
    parser.add_argument(
        "--same-time-cross-view",
        action="store_true",
        help=(
            "Fuse aligned scene/person/workspace layer4 tokens at each source time before "
            "per-view temporal encoding."
        ),
    )
    parser.add_argument(
        "--spatial-region-modeling",
        action="store_true",
        help="Retain a 2x2 layer4 region grid at each frame before temporal modeling.",
    )
    parser.add_argument(
        "--region-local-temporal",
        action="store_true",
        help="Track every 2x2 region across time before spatial and clip fusion.",
    )
    parser.add_argument(
        "--structured-spatial-region",
        action="store_true",
        help="Encode explicit vertical, horizontal and diagonal 2x2 region contrasts.",
    )
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--feature-weight", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--augmentation-mode",
        choices=("basic", "subject_robust", "subject_robust_no_flip"),
        default="basic",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--inner-gate-score",
        type=float,
        default=0.0,
        help=(
            "Stop before refit/outer-held evaluation unless the selected inner subject-disjoint "
            "score strictly exceeds this rollback score. Zero disables the gate."
        ),
    )
    parser.add_argument(
        "--save-inner-checkpoint",
        action="store_true",
        help="Persist the subject-pure selected inner model for downstream modality probes.",
    )
    parser.add_argument(
        "--stop-after-inner",
        action="store_true",
        help="Stop after inner selection without refit or outer-held evaluation.",
    )
    parser.add_argument(
        "--fixed-refit-epochs",
        type=int,
        default=0,
        help=(
            "Skip inner selection and train outer-train for this pre-frozen epoch count before "
            "one outer-held confirmation. Used only after architecture/epoch selection is frozen."
        ),
    )
    parser.add_argument(
        "--single-fold-fixed-epochs",
        type=int,
        default=0,
        help=(
            "Permanent one-fold protocol: train all subjects except fixed user1/user2/user21 "
            "for this pre-frozen epoch count, then evaluate that validation split once. No "
            "per-epoch validation, outer-held evaluation or multi-fold OOF."
        ),
    )
    parser.add_argument(
        "--proxy-fixed-epochs",
        type=int,
        default=0,
        help=(
            "Training-only architecture proxy: train the fixed nine inner-train subjects and "
            "evaluate the six outer-held subjects once. Fixed user1/user2/user21 validation is "
            "never read. This is one proxy split, not cross-validation."
        ),
    )
    parser.add_argument(
        "--subject-holdout-users",
        nargs="+",
        help=(
            "Leakage-safe P87-S protocol: train on every subject except this explicit "
            "list, then evaluate the held subjects once after fixed-epoch training."
        ),
    )
    parser.add_argument(
        "--subject-holdout-fixed-epochs",
        type=int,
        default=0,
        help="Fixed epoch budget for --subject-holdout-users; no per-epoch holdout reads.",
    )
    parser.add_argument(
        "--all-label-fixed-epochs",
        type=int,
        default=0,
        help=(
            "Terminal refit only: train once on all 2914 labeled rows for this already-"
            "frozen epoch count. No validation rows or checkpoint selection are used."
        ),
    )
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def make_dataset(
    args: argparse.Namespace,
    canonical_ids: list[str],
    index_lookup: dict[str, int],
    augment: bool,
) -> P86VisualPixelDataset:
    missing = set(canonical_ids) - set(index_lookup)
    if missing:
        raise RuntimeError(f"pixel cache is missing split rows: {sorted(missing)[:3]}")
    indices = np.asarray([index_lookup[sample_id] for sample_id in canonical_ids], dtype=np.int64)
    return P86VisualPixelDataset(
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        indices=indices,
        augment=augment,
        augmentation_mode=args.augmentation_mode,
    )


def make_loader(
    dataset: P86VisualPixelDataset,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_p86_pixels,
        # A one-sample tail is unsafe for training-time normalization, but larger
        # tails are valid.  In particular the final 2470-sample refit with a
        # batch size of four must consume its final two samples every epoch.
        drop_last=shuffle and len(dataset) % args.batch_size == 1,
    )


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_model(model: nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    if bool(getattr(model, "exact_time_modeling", False)):
        return model(
            batch["images"],
            batch["view_valid"],
            batch["view_quality"],
            batch["global_time_position"],
        )
    return model(batch["images"], batch["view_valid"], batch["view_quality"])


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    maximum_batches: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if maximum_batches and batch_index >= maximum_batches:
                break
            batch = batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = forward_model(model, batch)
            probability = torch.softmax(output["logits"].float(), dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "user_id": batch["user_id"][index],
                        "label": int(labels[index]),
                        "prediction": int(probability[index].argmax()),
                        "confidence": float(probability[index].max()),
                    }
                )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    return metric_dict(labels, predictions, [row["user_id"] for row in rows]), rows


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.stage_distillation_weight > 0 and args.backbone != "mc3_18_temporal":
        raise ValueError("stage distillation requires the temporal MC3 backbone")
    if (
        args.exact_time_position
        or args.same_time_cross_view
        or args.spatial_region_modeling
        or args.region_local_temporal
        or args.structured_spatial_region
    ) and args.backbone != "mc3_18_temporal":
        raise ValueError("temporal position, cross-view and region modeling require temporal MC3")
    if args.region_local_temporal and not args.spatial_region_modeling:
        raise ValueError("region-local temporal modeling requires --spatial-region-modeling")
    if args.structured_spatial_region and not args.spatial_region_modeling:
        raise ValueError("structured spatial modeling requires --spatial-region-modeling")
    common = {
        "fusion_mode": args.fusion_mode,
        "enable_distillation_projection": args.mode == "feature",
        "frames": args.frames,
    }
    if args.backbone == "resnet18_tsm":
        if args.freeze_through not in {"layer1", "layer2"}:
            raise ValueError("ResNet18-TSM freeze boundary must be layer1 or layer2")
        model: nn.Module = P86TrainableVisualStudent(**common)
    elif args.backbone in {"mc3_18", "mc3_18_temporal"}:
        if args.freeze_through not in {"layer1", "layer2", "layer3"}:
            raise ValueError("MC3-18 freeze boundary must be layer1, layer2 or layer3")
        model = P86MC3VisualStudent(
            **common,
            temporal_modeling=args.backbone == "mc3_18_temporal",
            exact_time_modeling=args.exact_time_position,
            cross_view_time_modeling=args.same_time_cross_view,
            spatial_region_modeling=args.spatial_region_modeling,
            region_temporal_modeling=args.region_local_temporal,
            structured_region_modeling=args.structured_spatial_region,
        )
    elif args.backbone == "videomae_small":
        if args.freeze_through not in {"layer6", "layer8", "layer10"}:
            raise ValueError("VideoMAE-Small freeze boundary must be layer6, layer8 or layer10")
        model = P86VideoMAESmallVisualStudent(
            **common,
            width=384,
            resolution=args.input_resolution,
            pretrained_model=args.pretrained_model,
        )
    else:
        raise ValueError(f"unknown visual backbone: {args.backbone}")
    model.freeze_low_level(args.freeze_through)
    return model


def optimizer_for(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone = [parameter for parameter in model.backbone_parameters() if parameter.requires_grad]
    head = [parameter for parameter in model.head_parameters() if parameter.requires_grad]
    if not backbone or not head:
        raise RuntimeError("P86 V2 optimizer groups are empty")
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.backbone_learning_rate, "initial_lr": args.backbone_learning_rate},
            {"params": head, "lr": args.head_learning_rate, "initial_lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
        foreach=False,
    )


def model_config_for(model: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "classes": 40,
        "width": int(getattr(model, "width", 512)),
        "dropout": 0.18,
        "fusion_mode": args.fusion_mode,
        "enable_distillation_projection": args.mode == "feature",
        "freeze_through": args.freeze_through,
        "frames": args.frames,
        "input_resolution": args.input_resolution,
        "backbone": args.backbone,
        "temporal_modeling": bool(getattr(model, "temporal_modeling", False)),
        "exact_time_modeling": bool(getattr(model, "exact_time_modeling", False)),
        "cross_view_time_modeling": bool(
            getattr(model, "cross_view_time_modeling", False)
        ),
        "spatial_region_modeling": bool(
            getattr(model, "spatial_region_modeling", False)
        ),
        "region_temporal_modeling": bool(
            getattr(model, "region_temporal_modeling", False)
        ),
        "structured_region_modeling": bool(
            getattr(model, "structured_region_modeling", False)
        ),
    }
    if args.backbone == "videomae_small":
        config["pretrained_model"] = args.pretrained_model
        config["pretrained_load_audit"] = getattr(model, "pretrained_load_audit", {})
    return config


def learning_rate_scale(epoch: int, maximum_epochs: int, minimum_ratio: float) -> float:
    if epoch <= 2:
        return epoch / 2.0
    progress = (epoch - 2) / max(maximum_epochs - 2, 1)
    return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (
        1.0 + math.cos(math.pi * progress)
    )


def selection_score(metrics: dict[str, Any]) -> float:
    return (
        float(metrics["accuracy"])
        + 0.5 * float(metrics["macro_f1"])
        + 0.25 * float(metrics["worst_subject_accuracy"])
    )


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    labels_for_weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    early_stop: bool,
) -> tuple[nn.Module, list[dict[str, Any]], int]:
    optimizer = optimizer_for(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = class_weights(labels_for_weights, args.class_weight_power, device)
    history: list[dict[str, Any]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 1
    best_score = -math.inf
    stale = 0
    accumulation = max(int(args.gradient_accumulation), 1)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        ratio = learning_rate_scale(
            epoch,
            max(epochs, 2),
            args.minimum_learning_rate / args.head_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * ratio
        sums = {
            "loss": 0.0,
            "ce": 0.0,
            "kd": 0.0,
            "stage_kd": 0.0,
            "relation": 0.0,
            "feature": 0.0,
            "samples": 0,
        }
        optimizer.zero_grad(set_to_none=True)
        completed_batches = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = forward_model(model, batch)
                ce = F.cross_entropy(
                    output["logits"],
                    batch["label"],
                    weight=weights,
                    label_smoothing=args.label_smoothing,
                )
                temperature = args.distillation_temperature
                kd = F.kl_div(
                    F.log_softmax(output["logits"] / temperature, dim=1),
                    F.softmax(batch["teacher_logits"] / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature**2
                if args.stage_distillation_weight > 0:
                    teacher_stage = torch.stack(
                        (
                            batch["teacher_early_logits"],
                            batch["teacher_late_logits"],
                            batch["teacher_temporal_delta_logits"],
                        ),
                        dim=1,
                    )
                    stage_kd = F.kl_div(
                        F.log_softmax(output["stage_logits"] / temperature, dim=-1),
                        F.softmax(teacher_stage / temperature, dim=-1),
                        reduction="batchmean",
                    ) * temperature**2 / teacher_stage.shape[1]
                else:
                    stage_kd = output["logits"].sum() * 0.0
                relation = relation_loss(
                    output["clip_embeddings"], batch["teacher_features"], output["clip_mask"]
                )
                if args.mode == "feature":
                    feature_distance = 1.0 - F.cosine_similarity(
                        output["projected_clip_embeddings"].float(),
                        batch["teacher_features"].float(),
                        dim=-1,
                    )
                    feature = (
                        feature_distance * output["clip_mask"].to(feature_distance.dtype)
                    ).sum() / output["clip_mask"].sum().clamp_min(1)
                else:
                    feature = output["clip_embeddings"].sum() * 0.0
                loss = ce
                if args.mode in {"kd", "hybrid", "feature"}:
                    loss = loss + args.distillation_weight * kd
                    loss = loss + args.stage_distillation_weight * stage_kd
                if args.mode in {"hybrid", "feature"}:
                    loss = loss + args.relation_weight * relation
                if args.mode == "feature":
                    loss = loss + args.feature_weight * feature
            scaler.scale(loss / accumulation).backward()
            completed_batches += 1
            if completed_batches % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            count = len(batch["label"])
            sums["loss"] += float(loss.detach()) * count
            sums["ce"] += float(ce.detach()) * count
            sums["kd"] += float(kd.detach()) * count
            sums["stage_kd"] += float(stage_kd.detach()) * count
            sums["relation"] += float(relation.detach()) * count
            sums["feature"] += float(feature.detach()) * count
            sums["samples"] += count
        if completed_batches and completed_batches % accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        record: dict[str, Any] = {
            "epoch": epoch,
            "backbone_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
            "train_loss": sums["loss"] / max(sums["samples"], 1),
            "train_ce": sums["ce"] / max(sums["samples"], 1),
            "train_kd": sums["kd"] / max(sums["samples"], 1),
            "train_stage_kd": sums["stage_kd"] / max(sums["samples"], 1),
            "train_relation": sums["relation"] / max(sums["samples"], 1),
            "train_feature": sums["feature"] / max(sums["samples"], 1),
            "train_samples": sums["samples"],
            "seconds": time.perf_counter() - started,
        }
        if eval_loader is not None:
            metrics, _ = evaluate(model, eval_loader, device, args.max_eval_batches)
            record.update(
                {f"val_{key}": value for key, value in metrics.items() if key != "per_user_accuracy"}
            )
            score = selection_score(metrics)
            if score > best_score + 1e-4:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if early_stop and epoch >= args.min_epochs and stale >= args.patience:
            break
    model.load_state_dict(best_state, strict=True)
    return model, history, best_epoch


def run_fixed_epoch_confirmation(
    args: argparse.Namespace,
    split: dict[str, Any],
    sample_ids: np.ndarray,
    labels: np.ndarray,
    index_lookup: dict[str, int],
    device: torch.device,
    output: Path,
) -> None:
    if args.fixed_refit_epochs <= 0:
        raise ValueError("fixed refit epochs must be positive")

    def selected_ids(key: str) -> list[str]:
        return sample_ids[split[key]].tolist()

    seed_all(args.seed + 1000 + args.outer_fold)
    outer_train = make_dataset(
        args, selected_ids("outer_train"), index_lookup, augment=True
    )
    model = build_model(args).to(device)
    model, refit_history, _ = train_epochs(
        model,
        make_loader(outer_train, args, True),
        None,
        labels[split["outer_train"]],
        args,
        device,
        args.fixed_refit_epochs,
        early_stop=False,
    )
    outer_held = make_dataset(
        args, selected_ids("outer_held"), index_lookup, augment=False
    )
    outer_metrics, prediction_rows = evaluate(
        model,
        make_loader(outer_held, args, False),
        device,
        args.max_eval_batches,
    )
    write_rows(output / "outer_predictions.csv", prediction_rows)
    write_rows(output / "refit_history.csv", refit_history)
    checkpoint = {
        "stage": "P86_visual_fixed_epoch_confirmation",
        "mode": args.mode,
        "outer_fold": args.outer_fold,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model_config_for(model, args),
        "fixed_refit_epochs": args.fixed_refit_epochs,
        "outer_metrics": outer_metrics,
    }
    torch.save(checkpoint, output / "visual_student.pt")
    summary = {
        "stage": "P86_visual_fixed_epoch_outer_confirmation",
        "status": "smoke" if args.smoke else "formal",
        "mode": args.mode,
        "outer_fold": args.outer_fold,
        "protocol": (
            "Architecture, losses, optimizer and epoch count were frozen from fold0. This fold "
            "skips inner search, trains all outer-train subjects once, and evaluates outer-held "
            "subjects exactly once."
        ),
        "visual_backbone": args.backbone,
        "visual_fusion_mode": args.fusion_mode,
        "counts": {
            "outer_train": int(len(split["outer_train"])),
            "outer_held": int(len(split["outer_held"])),
        },
        "fixed_refit_epochs": args.fixed_refit_epochs,
        "outer_metrics": outer_metrics,
        "student_parameters": parameter_count(model),
        "student_fp32_mib": model_size_mib(model, 4),
        "student_fp16_mib": model_size_mib(model, 2),
        "pretrained_load_audit": getattr(model, "pretrained_load_audit", None),
        "large_videomae_required_at_inference": False,
        "large_videomae_role": (
            "training-only OOF logits and six-clip relation targets; none are inference inputs"
        ),
        "test_used_for_selection": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run_single_fold_fixed_validation(
    args: argparse.Namespace,
    split: dict[str, Any],
    sample_ids: np.ndarray,
    labels: np.ndarray,
    index_lookup: dict[str, int],
    device: torch.device,
    output: Path,
) -> None:
    epochs = int(args.single_fold_fixed_epochs)
    if epochs <= 0:
        raise ValueError("single-fold fixed epochs must be positive")
    validation_ids = sample_ids[split["inner_dev"]]
    validation_set = set(validation_ids.tolist())
    training_indices = np.asarray(
        [index for index, value in enumerate(sample_ids) if value not in validation_set],
        dtype=np.int64,
    )
    training_ids = sample_ids[training_indices]
    if len(training_ids) != 2470 or len(validation_ids) != 444:
        raise RuntimeError(
            f"fixed single-fold counts changed: train={len(training_ids)}, val={len(validation_ids)}"
        )
    seed_all(args.seed)
    training = make_dataset(args, training_ids.tolist(), index_lookup, augment=True)
    validation = make_dataset(args, validation_ids.tolist(), index_lookup, augment=False)
    model = build_model(args).to(device)
    model, history, _ = train_epochs(
        model,
        make_loader(training, args, True),
        None,
        labels[training_indices],
        args,
        device,
        epochs,
        early_stop=False,
    )
    validation_metrics, prediction_rows = evaluate(
        model,
        make_loader(validation, args, False),
        device,
        args.max_eval_batches,
    )
    write_rows(output / "fixed_validation_predictions.csv", prediction_rows)
    write_rows(output / "training_history.csv", history)
    checkpoint = {
        "stage": "P86_visual_permanent_singlefold",
        "mode": args.mode,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model_config_for(model, args),
        "fixed_epochs": epochs,
        "validation_subjects": sorted({row["user_id"] for row in prediction_rows}),
        "validation_metrics": validation_metrics,
    }
    torch.save(checkpoint, output / "visual_student.pt")
    summary = {
        "stage": "P86_visual_permanent_singlefold",
        "status": "smoke" if args.smoke else "formal",
        "protocol": (
            "Permanent one-fold subject-disjoint protocol. Train all 15 non-validation "
            "subjects for a pre-frozen epoch budget without reading validation per epoch, then "
            "evaluate fixed user1/user2/user21 exactly once. No outer-held or three-fold OOF."
        ),
        "visual_backbone": args.backbone,
        "visual_fusion_mode": args.fusion_mode,
        "counts": {"candidate_train": len(training_ids), "fixed_validation": len(validation_ids)},
        "validation_subjects": sorted({row["user_id"] for row in prediction_rows}),
        "fixed_epochs": epochs,
        "validation_metrics": validation_metrics,
        "selection_score": selection_score(validation_metrics),
        "student_parameters": parameter_count(model),
        "student_fp32_mib": model_size_mib(model, 4),
        "student_fp16_mib": model_size_mib(model, 2),
        "pretrained_load_audit": getattr(model, "pretrained_load_audit", None),
        "large_videomae_required_at_inference": False,
        "large_videomae_role": (
            "training-only OOF logits and six-clip relation targets; none are inference inputs"
        ),
        "three_fold_oof": False,
        "validation_used_during_training": False,
        "test_used_for_selection": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run_training_only_proxy(
    args: argparse.Namespace,
    split: dict[str, Any],
    sample_ids: np.ndarray,
    labels: np.ndarray,
    index_lookup: dict[str, int],
    device: torch.device,
    output: Path,
) -> None:
    epochs = int(args.proxy_fixed_epochs)
    if epochs <= 0:
        raise ValueError("proxy fixed epochs must be positive")
    training_indices = np.asarray(split["inner_train"], dtype=np.int64)
    proxy_indices = np.asarray(split["outer_held"], dtype=np.int64)
    forbidden_indices = np.asarray(split["inner_dev"], dtype=np.int64)
    training_users = set(split["users"][training_indices].tolist())
    proxy_users = set(split["users"][proxy_indices].tolist())
    forbidden_users = set(split["users"][forbidden_indices].tolist())
    if training_users & proxy_users or (training_users | proxy_users) & forbidden_users:
        raise RuntimeError("training-only proxy subject isolation failed")
    if len(training_indices) != 1497 or len(proxy_indices) != 973:
        raise RuntimeError(
            f"training-only proxy counts changed: train={len(training_indices)}, "
            f"proxy={len(proxy_indices)}"
        )
    seed_all(args.seed)
    training = make_dataset(
        args, sample_ids[training_indices].tolist(), index_lookup, augment=True
    )
    proxy = make_dataset(
        args, sample_ids[proxy_indices].tolist(), index_lookup, augment=False
    )
    model = build_model(args).to(device)
    model, history, _ = train_epochs(
        model,
        make_loader(training, args, True),
        None,
        labels[training_indices],
        args,
        device,
        epochs,
        early_stop=False,
    )
    proxy_metrics, prediction_rows = evaluate(
        model,
        make_loader(proxy, args, False),
        device,
        args.max_eval_batches,
    )
    write_rows(output / "proxy_validation_predictions.csv", prediction_rows)
    write_rows(output / "training_history.csv", history)
    checkpoint = {
        "stage": "P86_visual_training_only_proxy",
        "mode": args.mode,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model_config_for(model, args),
        "fixed_epochs": epochs,
        "training_users": sorted(training_users),
        "proxy_users": sorted(proxy_users),
        "proxy_metrics": proxy_metrics,
    }
    torch.save(checkpoint, output / "visual_student.pt")
    summary = {
        "stage": "P86_visual_training_only_proxy",
        "status": "smoke" if args.smoke else "formal",
        "protocol": (
            "One fixed training-only subject-disjoint proxy. Train nine candidate-train "
            "subjects without per-epoch evaluation, then evaluate the other six candidate-train "
            "subjects exactly once. Permanent user1/user2/user21 validation is never read."
        ),
        "visual_backbone": args.backbone,
        "visual_fusion_mode": args.fusion_mode,
        "counts": {
            "proxy_train": int(len(training_indices)),
            "proxy_validation": int(len(proxy_indices)),
            "untouched_permanent_validation": int(len(forbidden_indices)),
        },
        "training_users": sorted(training_users),
        "proxy_users": sorted(proxy_users),
        "untouched_permanent_validation_users": sorted(forbidden_users),
        "fixed_epochs": epochs,
        "proxy_metrics": proxy_metrics,
        "selection_score": selection_score(proxy_metrics),
        "student_parameters": parameter_count(model),
        "student_fp32_mib": model_size_mib(model, 4),
        "student_fp16_mib": model_size_mib(model, 2),
        "large_videomae_required_at_inference": False,
        "permanent_validation_used": False,
        "three_fold_oof": False,
        "test_used_for_selection": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run_subject_holdout_fixed_validation(
    args: argparse.Namespace,
    split: dict[str, Any],
    sample_ids: np.ndarray,
    labels: np.ndarray,
    index_lookup: dict[str, int],
    device: torch.device,
    output: Path,
) -> None:
    epochs = int(args.subject_holdout_fixed_epochs)
    if epochs <= 0 or not args.subject_holdout_users:
        raise ValueError("subject holdout users and a positive fixed epoch count are required")
    requested_users = set(map(str, args.subject_holdout_users))
    users = np.asarray(split["users"]).astype(str)
    observed_users = set(users[np.isin(users, sorted(requested_users))].tolist())
    if observed_users != requested_users:
        raise ValueError(
            f"Requested holdout users {sorted(requested_users)}, observed {sorted(observed_users)}"
        )
    holdout_indices = np.flatnonzero(np.isin(users, sorted(requested_users)))
    training_indices = np.flatnonzero(~np.isin(users, sorted(requested_users)))
    training_users = set(users[training_indices].tolist())
    if training_users & requested_users:
        raise RuntimeError("subject holdout isolation failed")
    if len(training_indices) + len(holdout_indices) != len(sample_ids):
        raise RuntimeError("subject holdout split does not cover the full training universe")

    seed_all(args.seed)
    training = make_dataset(
        args, sample_ids[training_indices].tolist(), index_lookup, augment=True
    )
    holdout = make_dataset(
        args, sample_ids[holdout_indices].tolist(), index_lookup, augment=False
    )
    model = build_model(args).to(device)
    model, history, _ = train_epochs(
        model,
        make_loader(training, args, True),
        None,
        labels[training_indices],
        args,
        device,
        epochs,
        early_stop=False,
    )
    holdout_metrics, prediction_rows = evaluate(
        model,
        make_loader(holdout, args, False),
        device,
        args.max_eval_batches,
    )
    write_rows(output / "subject_holdout_predictions.csv", prediction_rows)
    write_rows(output / "training_history.csv", history)
    checkpoint = {
        "stage": "P87S_visual_subject_holdout",
        "mode": args.mode,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model_config_for(model, args),
        "fixed_epochs": epochs,
        "training_subjects": sorted(training_users),
        "holdout_subjects": sorted(requested_users),
        "holdout_metrics": holdout_metrics,
    }
    torch.save(checkpoint, output / "visual_student.pt")
    summary = {
        "stage": "P87S_visual_subject_holdout",
        "status": "smoke" if args.smoke else "formal",
        "protocol": (
            "Train a fresh visual student on every subject except the explicit P87-S "
            "pseudo-Test subjects. The held labels are read exactly once after fixed-epoch "
            "training and are never used for initialization or epoch selection."
        ),
        "visual_backbone": args.backbone,
        "visual_fusion_mode": args.fusion_mode,
        "counts": {"train": len(training_indices), "holdout": len(holdout_indices)},
        "training_subjects": sorted(training_users),
        "holdout_subjects": sorted(requested_users),
        "fixed_epochs": epochs,
        "holdout_metrics": holdout_metrics,
        "student_parameters": parameter_count(model),
        "student_fp32_mib": model_size_mib(model, 4),
        "student_fp16_mib": model_size_mib(model, 2),
        "pretrained_load_audit": getattr(model, "pretrained_load_audit", None),
        "large_videomae_required_at_inference": False,
        "validation_used_during_training": False,
        "test_used_for_selection": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run_all_label_fixed_refit(
    args: argparse.Namespace,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    index_lookup: dict[str, int],
    device: torch.device,
    output: Path,
) -> None:
    epochs = int(args.all_label_fixed_epochs)
    if epochs <= 0:
        raise ValueError("all-label fixed epochs must be positive")
    if len(sample_ids) != 2914 or len(np.unique(sample_ids.astype(str))) != 2914:
        raise RuntimeError("terminal visual refit requires exactly 2914 unique labels")
    seed_all(args.seed)
    training = make_dataset(args, sample_ids.tolist(), index_lookup, augment=True)
    model = build_model(args).to(device)
    model, history, _ = train_epochs(
        model,
        make_loader(training, args, True),
        None,
        labels,
        args,
        device,
        epochs,
        early_stop=False,
    )
    write_rows(output / "training_history.csv", history)
    checkpoint = {
        "stage": "P87S_visual_all2914_refit",
        "mode": args.mode,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model_config_for(model, args),
        "fixed_epochs": epochs,
        "training_rows": len(training),
    }
    torch.save(checkpoint, output / "visual_student.pt")
    summary = {
        "stage": "P87S_visual_all2914_refit",
        "status": "smoke" if args.smoke else "formal_terminal_refit",
        "protocol": (
            "Terminal refit on all 2914 true-labeled Train rows using the epoch count "
            "frozen before Test adaptation. There is no validation loader, early stop, "
            "checkpoint selection or Test input in this stage."
        ),
        "counts": {"train": len(training), "validation": 0},
        "fixed_epochs": epochs,
        "student_parameters": parameter_count(model),
        "student_fp32_mib": model_size_mib(model, 4),
        "large_videomae_required_at_inference": False,
        "validation_used_during_training": False,
        "test_used_for_selection": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if bool(args.subject_holdout_users) != (args.subject_holdout_fixed_epochs > 0):
        raise ValueError(
            "--subject-holdout-users and --subject-holdout-fixed-epochs must be used together"
        )
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.min_epochs = 1
        if args.all_label_fixed_epochs > 0:
            args.all_label_fixed_epochs = min(args.all_label_fixed_epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
    seed_all(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, args.outer_fold, args.seed)
    sample_ids = split["sample_ids"]
    labels = split["labels"]

    reference_dataset = P86VisualPixelDataset(
        args.pixel_cache, args.teacher_features, args.teacher_logits
    )
    expected_pixel_shape = (2, args.frames, 3, args.input_resolution, args.input_resolution)
    if tuple(reference_dataset.images.shape[1:]) != expected_pixel_shape:
        raise RuntimeError(
            f"pixel cache geometry {tuple(reference_dataset.images.shape[1:])} does not match "
            f"requested {expected_pixel_shape}"
        )
    index_lookup = reference_dataset.index_lookup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_protocols = sum(
        int(value > 0)
        for value in (
            args.fixed_refit_epochs,
            args.single_fold_fixed_epochs,
            args.proxy_fixed_epochs,
            args.subject_holdout_fixed_epochs,
            args.all_label_fixed_epochs,
        )
    )
    if selected_protocols > 1:
        raise ValueError(
            "fixed-refit, permanent single-fold, proxy, subject-holdout and all-label modes are "
            "mutually exclusive"
        )
    if args.all_label_fixed_epochs > 0:
        run_all_label_fixed_refit(
            args, sample_ids, labels, index_lookup, device, output
        )
        return
    if args.subject_holdout_fixed_epochs > 0:
        run_subject_holdout_fixed_validation(
            args, split, sample_ids, labels, index_lookup, device, output
        )
        return
    if args.proxy_fixed_epochs > 0:
        run_training_only_proxy(
            args, split, sample_ids, labels, index_lookup, device, output
        )
        return
    if args.single_fold_fixed_epochs > 0:
        run_single_fold_fixed_validation(
            args, split, sample_ids, labels, index_lookup, device, output
        )
        return
    if args.fixed_refit_epochs > 0:
        run_fixed_epoch_confirmation(
            args, split, sample_ids, labels, index_lookup, device, output
        )
        return

    def ids(key: str) -> list[str]:
        return sample_ids[split[key]].tolist()

    inner_train = make_dataset(args, ids("inner_train"), index_lookup, augment=True)
    inner_dev = make_dataset(args, ids("inner_dev"), index_lookup, augment=False)
    model = build_model(args).to(device)
    model, inner_history, best_epoch = train_epochs(
        model,
        make_loader(inner_train, args, True),
        make_loader(inner_dev, args, False),
        labels[split["inner_train"]],
        args,
        device,
        args.max_epochs,
        early_stop=True,
    )
    selected_inner = next(row for row in inner_history if int(row["epoch"]) == best_epoch)
    selected_inner_metrics = {
        key.removeprefix("val_"): value
        for key, value in selected_inner.items()
        if key.startswith("val_")
    }
    selected_score = selection_score(selected_inner_metrics)
    if args.save_inner_checkpoint or args.stop_after_inner:
        inner_checkpoint = {
            "stage": "P86_visual_pixel_inner_subject_pure",
            "mode": args.mode,
            "outer_fold": args.outer_fold,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_config": model_config_for(model, args),
            "best_inner_epoch": best_epoch,
            "selected_inner_metrics": selected_inner_metrics,
            "selected_inner_score": selected_score,
            "training_scope": "inner_train_subjects_only",
            "outer_held_evaluated": False,
        }
        torch.save(inner_checkpoint, output / "inner_visual_student.pt")
    if args.stop_after_inner:
        write_rows(output / "inner_history.csv", inner_history)
        inner_summary = {
            "stage": "P86_visual_pixel_inner_anchor",
            "status": "inner_only_no_outer_evaluation",
            "mode": args.mode,
            "outer_fold": args.outer_fold,
            "visual_backbone": args.backbone,
            "visual_fusion_mode": args.fusion_mode,
            "best_inner_epoch": best_epoch,
            "selected_inner_metrics": selected_inner_metrics,
            "selected_inner_score": selected_score,
            "outer_held_evaluated": False,
            "student_parameters": parameter_count(model),
            "student_fp32_mib": model_size_mib(model, 4),
            "student_fp16_mib": model_size_mib(model, 2),
            "pretrained_load_audit": getattr(model, "pretrained_load_audit", None),
            "large_videomae_required_at_inference": False,
            "test_used_for_selection": False,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (output / "summary.json").write_text(
            json.dumps(inner_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(inner_summary, ensure_ascii=False, indent=2), flush=True)
        return
    if args.inner_gate_score > 0.0 and selected_score <= args.inner_gate_score + 1e-4:
        write_rows(output / "inner_history.csv", inner_history)
        rejected_summary = {
            "stage": "P86_visual_pixel_inner_gate",
            "status": "rejected_before_outer_held",
            "mode": args.mode,
            "outer_fold": args.outer_fold,
            "visual_backbone": args.backbone,
            "visual_fusion_mode": args.fusion_mode,
            "best_inner_epoch": best_epoch,
            "selected_inner_metrics": selected_inner_metrics,
            "selected_inner_score": selected_score,
            "required_score_strictly_above": args.inner_gate_score,
            "outer_held_evaluated": False,
            "rollback_model": "p86_visual_pixel_fold0_hybrid_v2",
            "protocol": (
                "The candidate failed the predeclared inner subject-disjoint rollback gate, so "
                "training stopped before outer-train refit and outer-held evaluation."
            ),
            "student_parameters": parameter_count(model),
            "student_fp32_mib": model_size_mib(model, 4),
            "student_fp16_mib": model_size_mib(model, 2),
            "pretrained_load_audit": getattr(model, "pretrained_load_audit", None),
            "large_videomae_required_at_inference": False,
            "test_used_for_selection": False,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (output / "summary.json").write_text(
            json.dumps(rejected_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(rejected_summary, ensure_ascii=False, indent=2), flush=True)
        return

    seed_all(args.seed + 1000 + args.outer_fold)
    outer_train = make_dataset(args, ids("outer_train"), index_lookup, augment=True)
    refit_model = build_model(args).to(device)
    refit_model, refit_history, _ = train_epochs(
        refit_model,
        make_loader(outer_train, args, True),
        None,
        labels[split["outer_train"]],
        args,
        device,
        best_epoch,
        early_stop=False,
    )
    outer_held = make_dataset(args, ids("outer_held"), index_lookup, augment=False)
    outer_metrics, prediction_rows = evaluate(
        refit_model,
        make_loader(outer_held, args, False),
        device,
        args.max_eval_batches,
    )
    write_rows(output / "outer_predictions.csv", prediction_rows)
    write_rows(output / "inner_history.csv", inner_history)
    write_rows(output / "refit_history.csv", refit_history)

    checkpoint = {
        "stage": "P86_visual_pixel_nested",
        "mode": args.mode,
        "outer_fold": args.outer_fold,
        "model_state": {key: value.detach().cpu() for key, value in refit_model.state_dict().items()},
        "model_config": model_config_for(refit_model, args),
        "best_inner_epoch": best_epoch,
        "outer_metrics": outer_metrics,
    }
    torch.save(checkpoint, output / "visual_student.pt")
    summary = {
        "stage": "P86_visual_pixel_nested_outer_fold",
        "status": "smoke" if args.smoke else "formal",
        "mode": args.mode,
        "outer_fold": args.outer_fold,
        "protocol": (
            "Inner subject-disjoint split selects a fixed epoch; a reinitialized model is refit "
            "on all outer-train subjects and outer-held subjects are evaluated exactly once."
        ),
        "representation_change_from_v1": (
            f"Use a trainable shared {args.backbone} over early/late x "
            "scene/person/workspace IR clips instead of frozen globally pooled P30 features."
        ),
        "visual_backbone": args.backbone,
        "visual_fusion_mode": args.fusion_mode,
        "counts": {
            key: int(len(split[key]))
            for key in ("inner_train", "inner_dev", "outer_train", "outer_held")
        },
        "best_inner_epoch": best_epoch,
        "outer_metrics": outer_metrics,
        "student_parameters": parameter_count(refit_model),
        "student_fp32_mib": model_size_mib(refit_model, 4),
        "student_fp16_mib": model_size_mib(refit_model, 2),
        "pretrained_load_audit": getattr(refit_model, "pretrained_load_audit", None),
        "large_videomae_required_at_inference": False,
        "large_videomae_role": (
            "training-only OOF logits, six-clip relation targets, and in feature mode aligned "
            "per-clip embeddings; none are model inputs at inference"
        ),
        "test_used_for_selection": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
