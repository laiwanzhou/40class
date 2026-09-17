from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from p86_cached_motion_data import P86CachedSequenceMotionDataset
from p86_mc3_visual_model import P86MC3VisualStudent
from train_p86_cached_motion_proxy import (
    DEFAULT_MOTION,
    DEFAULT_PIXELS,
    DEFAULT_TEACHER_FEATURES,
    DEFAULT_TEACHER_LOGITS,
    loader,
    load_npz,
    make_dataset,
    seed_all,
    to_device,
)
from train_p86_visual_student_oof import (
    class_weights,
    metric_dict,
    relation_loss,
    split_universe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast training-only screen of temporal mechanisms on cached MC3."
    )
    parser.add_argument(
        "--mechanism",
        choices=(
            "exact_time",
            "same_time_cross_view",
            "spatial_region",
            "spatiotemporal_region",
            "structured_spatial_region",
        ),
        default="exact_time",
    )
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--stage-distillation-weight", type=float, default=0.0)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def load_temporal_candidate(
    path: Path, mechanism: str
) -> tuple[P86MC3VisualStudent, dict[str, Any], list[torch.nn.Parameter]]:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    if config.get("backbone") != "mc3_18_temporal":
        raise ValueError("cached mechanism screening requires a temporal MC3 checkpoint")
    exact_time = mechanism == "exact_time"
    cross_view = mechanism == "same_time_cross_view"
    spatial_region = mechanism in {
        "spatial_region",
        "spatiotemporal_region",
        "structured_spatial_region",
    }
    region_temporal = mechanism == "spatiotemporal_region"
    structured_region = mechanism == "structured_spatial_region"
    if exact_time and config.get("exact_time_modeling", False):
        raise ValueError("source checkpoint already contains exact-time modeling")
    if cross_view and config.get("cross_view_time_modeling", False):
        raise ValueError("source checkpoint already contains same-time cross-view modeling")
    if spatial_region and config.get("spatial_region_modeling", False):
        raise ValueError("source checkpoint already contains spatial region modeling")
    if region_temporal and config.get("region_temporal_modeling", False):
        raise ValueError("source checkpoint already contains region temporal modeling")
    if structured_region and config.get("structured_region_modeling", False):
        raise ValueError("source checkpoint already contains structured region modeling")
    model = P86MC3VisualStudent(
        classes=int(config.get("classes", 40)),
        width=int(config.get("width", 512)),
        dropout=float(config.get("dropout", 0.18)),
        fusion_mode=str(config.get("fusion_mode", "gated")),
        enable_distillation_projection=bool(
            config.get("enable_distillation_projection", False)
        ),
        frames=int(config["frames"]),
        kinetics_pretrained=False,
        temporal_modeling=True,
        exact_time_modeling=exact_time,
        cross_view_time_modeling=cross_view,
        spatial_region_modeling=spatial_region,
        region_temporal_modeling=region_temporal,
        structured_region_modeling=structured_region,
    )
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if exact_time:
        prefixes = ("exact_time_projection.",)
    elif cross_view:
        prefixes = ("same_time_view_encoder.", "same_time_view_projection.")
    elif structured_region:
        prefixes = ("structured_region_projection.",)
    else:
        prefixes = (
            "spatial_region_embedding",
            "spatial_region_encoder.",
            "spatial_region_projection.",
            "region_temporal_encoder.",
        )
    expected_missing = {
        key for key in model.state_dict() if any(key.startswith(prefix) for prefix in prefixes)
    }
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"unexpected {mechanism} initialization audit: missing={missing}, "
            f"unexpected={unexpected}"
        )
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not trainable:
        raise RuntimeError(f"{mechanism} did not create trainable candidate parameters")
    return model, config, trainable


def forward(model: P86MC3VisualStudent, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    if model.spatial_region_modeling:
        return model.forward_from_backbone_region_sequence(
            batch["backbone_region_sequence"],
            batch["view_valid"],
            batch["view_quality"],
            batch["global_time_position"],
        )
    return model.forward_from_backbone_sequence(
        batch["backbone_sequence"],
        batch["view_valid"],
        batch["view_quality"],
        batch["global_time_position"],
    )


def evaluate(
    model: P86MC3VisualStudent,
    data_loader: DataLoader,
    device: torch.device,
    maximum_batches: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader):
            if maximum_batches and batch_index >= maximum_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = forward(model, batch)["logits"]
            probability = torch.softmax(logits.float(), dim=1).cpu().numpy()
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


def train(
    model: P86MC3VisualStudent,
    candidate_parameters: list[torch.nn.Parameter],
    data_loader: DataLoader,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in candidate_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        candidate_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = class_weights(labels, args.class_weight_power, device)
    history = []
    for epoch in range(1, args.epochs + 1):
        # Frozen anchor modules stay in deterministic eval mode; gradients still
        # pass through them into the selected zero-residual mechanism.
        model.eval()
        ratio = args.minimum_learning_rate / args.learning_rate + 0.5 * (
            1.0 - args.minimum_learning_rate / args.learning_rate
        ) * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1)))
        optimizer.param_groups[0]["lr"] = args.learning_rate * ratio
        totals = {
            key: 0.0
            for key in ("loss", "ce", "kd", "stage_kd", "relation", "anchor")
        }
        samples = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(data_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = forward(model, batch)
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
                    early, late = output["window_embeddings"][:, 0], output[
                        "window_embeddings"
                    ][:, 1]
                    stage_logits = torch.stack(
                        (
                            model.classifier(early),
                            model.classifier(late),
                            model.classifier(late - early),
                        ),
                        dim=1,
                    )
                    teacher_stage = torch.stack(
                        (
                            batch["teacher_early_logits"],
                            batch["teacher_late_logits"],
                            batch["teacher_temporal_delta_logits"],
                        ),
                        dim=1,
                    )
                    stage_kd = F.kl_div(
                        F.log_softmax(stage_logits / temperature, dim=-1),
                        F.softmax(teacher_stage / temperature, dim=-1),
                        reduction="batchmean",
                    ) * temperature**2 / teacher_stage.shape[1]
                else:
                    stage_kd = output["logits"].sum() * 0.0
                relation = relation_loss(
                    output["clip_embeddings"],
                    batch["teacher_features"],
                    output["clip_mask"],
                )
                anchor = F.kl_div(
                    F.log_softmax(output["logits"] / temperature, dim=1),
                    F.softmax(batch["anchor_logits"] / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature**2
                loss = (
                    ce
                    + args.distillation_weight * kd
                    + args.stage_distillation_weight * stage_kd
                    + args.relation_weight * relation
                    + args.anchor_weight * anchor
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(candidate_parameters, 2.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["label"])
            samples += count
            for key, value in {
                "loss": loss,
                "ce": ce,
                "kd": kd,
                "stage_kd": stage_kd,
                "relation": relation,
                "anchor": anchor,
            }.items():
                totals[key] += float(value.detach()) * count
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value / max(samples, 1) for key, value in totals.items()},
            "train_samples": samples,
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


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
    seed_all(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split = split_universe(load_npz(args.teacher_logits), outer_fold=0, seed=args.seed)
    train_indices = np.asarray(split["inner_train"], dtype=np.int64)
    proxy_indices = np.asarray(split["outer_held"], dtype=np.int64)
    if len(train_indices) != 1497 or len(proxy_indices) != 973:
        raise RuntimeError("P86 fixed proxy counts changed")
    full = P86CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
    )
    training = make_dataset(
        args, full, split["sample_ids"][train_indices], temporal_augment=True
    )
    proxy = make_dataset(
        args, full, split["sample_ids"][proxy_indices], temporal_augment=False
    )
    model, visual_config, candidate_parameters = load_temporal_candidate(
        args.visual_checkpoint, args.mechanism
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    baseline_metrics, baseline_rows = evaluate(
        model, loader(proxy, args, False), device, args.max_eval_batches
    )
    history = train(
        model,
        candidate_parameters,
        loader(training, args, True),
        split["labels"][train_indices],
        args,
        device,
    )
    metrics, rows = evaluate(model, loader(proxy, args, False), device, args.max_eval_batches)
    write_rows(output / "cached_visual_baseline_predictions.csv", baseline_rows)
    write_rows(output / "training_history.csv", history)
    write_rows(output / "proxy_validation_predictions.csv", rows)
    checkpoint = {
        "stage": f"P86_cached_{args.mechanism}_proxy",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "visual_config": {
            **visual_config,
            "exact_time_modeling": args.mechanism == "exact_time",
            "cross_view_time_modeling": args.mechanism == "same_time_cross_view",
            "spatial_region_modeling": args.mechanism
            in {
                "spatial_region",
                "spatiotemporal_region",
                "structured_spatial_region",
            },
            "region_temporal_modeling": args.mechanism == "spatiotemporal_region",
            "structured_region_modeling": args.mechanism
            == "structured_spatial_region",
        },
        "proxy_metrics": metrics,
    }
    torch.save(checkpoint, output / f"{args.mechanism}_student.pt")
    delta = {
        "correct": int(metrics["correct"] - baseline_metrics["correct"]),
        "accuracy_pp": 100.0 * (metrics["accuracy"] - baseline_metrics["accuracy"]),
        "macro_f1_pp": 100.0 * (metrics["macro_f1"] - baseline_metrics["macro_f1"]),
        "worst_subject_accuracy_pp": 100.0
        * (metrics["worst_subject_accuracy"] - baseline_metrics["worst_subject_accuracy"]),
    }
    summary = {
        "stage": f"P86_cached_{args.mechanism}_proxy",
        "status": "smoke" if args.smoke else "formal_proxy",
        "counts": {"train": len(training), "proxy": len(proxy), "permanent_untouched": 444},
        "cached_visual_baseline_metrics": baseline_metrics,
        "proxy_metrics": metrics,
        "delta_vs_cached_visual": delta,
        "selection_score": (
            metrics["accuracy"]
            + 0.5 * metrics["macro_f1"]
            + 0.25 * metrics["worst_subject_accuracy"]
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "large_videomae_required_at_inference": False,
        "cached_sequence_is_final_model": False,
        "accuracy_contract": (
            f"This screen only decides whether {args.mechanism} merits a raw 16x160, 16-epoch "
            "confirmation; cached weights are never deployed."
        ),
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
