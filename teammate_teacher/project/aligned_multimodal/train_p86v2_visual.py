from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from p86_mc3_visual_model import P86MC3VisualStudent
from p86v2_metrics import emission_metrics, softmax
from p86v2_protocol import (
    DEFAULT_PROTOCOL,
    assert_no_forbidden_path,
    build_split,
    load_protocol,
    read_rows,
)
from p86v2_visual_model import (
    P86V2Layer3SpatialDedupStudent,
    P86V2TemporalDedupStudent,
)
from train_p86_visual_pixel_oof import (
    batch_to_device,
    forward_model,
    make_dataset,
    make_loader,
    model_config_for,
    model_size_mib,
    parameter_count,
    train_epochs,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXELS = PROJECT_DIR / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_TEACHER_FEATURES = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR
    / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embargo-safe P86-v2 visual baseline and mechanism training"
    )
    parser.add_argument(
        "--architecture",
        choices=("p86v1_baseline", "temporal_dedup", "layer3_spatial_dedup"),
        required=True,
    )
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    parser.add_argument(
        "--unlock-confirmation",
        action="store_true",
        help="Required only after the candidate recipe is frozen in the research record.",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
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


def training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    # Keep the P86-v1 optimization recipe fixed so architecture is the only
    # difference in the first P86-v2 paired experiment.
    values = dict(vars(args))
    values.update(
        {
            "mode": "hybrid",
            "outer_fold": 0,
            "fusion_mode": "gated",
            "backbone": "mc3_18_temporal",
            "freeze_through": "layer2",
            "frames": 16,
            "input_resolution": 160,
            "pretrained_model": "unused",
            "head_learning_rate": 2e-4,
            "backbone_learning_rate": 1e-5,
            "minimum_learning_rate": 1e-5,
            "weight_decay": 0.08,
            "class_weight_power": 0.35,
            "distillation_temperature": 2.0,
            "distillation_weight": 1.0,
            "stage_distillation_weight": 0.0,
            "relation_weight": 0.2,
            "feature_weight": 0.5,
            "label_smoothing": 0.1,
            "augmentation_mode": "subject_robust",
            "min_epochs": 16,
            "patience": 0,
            "exact_time_position": False,
            "same_time_cross_view": False,
            "spatial_region_modeling": False,
            "region_local_temporal": False,
            "structured_spatial_region": False,
        }
    )
    if args.smoke:
        values["max_train_batches"] = args.max_train_batches or 2
        values["max_eval_batches"] = args.max_eval_batches or 2
    return argparse.Namespace(**values)


def build_model(architecture: str) -> nn.Module:
    common = {
        "classes": 40,
        "width": 512,
        "dropout": 0.18,
        "fusion_mode": "gated",
        "enable_distillation_projection": False,
        "frames": 16,
        "kinetics_pretrained": True,
    }
    if architecture == "p86v1_baseline":
        model: nn.Module = P86MC3VisualStudent(**common, temporal_modeling=True)
    elif architecture == "temporal_dedup":
        model = P86V2TemporalDedupStudent(**common)
    elif architecture == "layer3_spatial_dedup":
        model = P86V2Layer3SpatialDedupStudent(**common)
    else:  # pragma: no cover - argparse guards this
        raise ValueError(architecture)
    model.freeze_low_level("layer2")
    return model


def detailed_evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    maximum_batches: int,
) -> tuple[dict, dict[str, np.ndarray], list[dict[str, Any]]]:
    model.eval()
    sample_ids: list[str] = []
    users: list[str] = []
    labels: list[int] = []
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if maximum_batches and batch_index >= maximum_batches:
                break
            batch = batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = forward_model(model, batch)
            sample_ids.extend(map(str, batch["sample_id"]))
            users.extend(map(str, batch["user_id"]))
            labels.extend(batch["label"].cpu().numpy().astype(int).tolist())
            logits.append(output["logits"].float().cpu().numpy())
    logit_array = np.concatenate(logits, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)
    user_array = np.asarray(users)
    probability = softmax(logit_array)
    prediction = probability.argmax(axis=1)
    entropy = -(probability * np.log(probability.clip(1e-12))).sum(axis=1)
    rows = [
        {
            "sample_id": sample_id,
            "user_id": user,
            "label": int(label),
            "prediction": int(prediction[index]),
            "confidence": float(probability[index].max()),
            "true_probability": float(probability[index, label]),
            "entropy": float(entropy[index]),
        }
        for index, (sample_id, user, label) in enumerate(
            zip(sample_ids, users, labels, strict=True)
        )
    ]
    arrays = {
        "sample_ids": np.asarray(sample_ids),
        "users": user_array,
        "labels": label_array,
        "logits": logit_array.astype(np.float32),
    }
    return emission_metrics(logit_array, label_array, user_array), arrays, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.split == "confirmation" and not args.unlock_confirmation:
        raise RuntimeError(
            "sealed confirmation fold is locked; freeze and record the candidate first"
        )
    for path in (args.pixel_cache, args.teacher_features, args.teacher_logits):
        assert_no_forbidden_path(path)
    output = args.output_dir.resolve()
    if (output / "summary.json").exists() or (output / "visual_student.pt").exists():
        raise RuntimeError(f"refusing to overwrite an existing P86-v2 run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(args.protocol)
    rows = read_rows(args.pixel_cache / "rows.csv")
    split = build_split(rows, args.split, protocol)
    train_args = training_namespace(args)
    seed_all(args.seed)
    index_lookup = {row["sample_id"]: index for index, row in enumerate(rows)}
    train_ids = [rows[index]["sample_id"] for index in split.training_indices]
    holdout_ids = [rows[index]["sample_id"] for index in split.holdout_indices]
    training = make_dataset(train_args, train_ids, index_lookup, augment=True)
    holdout = make_dataset(train_args, holdout_ids, index_lookup, augment=False)
    labels_for_weights = np.asarray(
        [int(rows[index]["class_id"]) for index in split.training_indices], dtype=np.int64
    )
    epochs = 2 if args.smoke else int(protocol[args.split]["fixed_epochs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.architecture).to(device)
    model, history, _ = train_epochs(
        model,
        make_loader(training, train_args, True),
        None,
        labels_for_weights,
        train_args,
        device,
        epochs,
        early_stop=False,
    )
    metrics, arrays, prediction_rows = detailed_evaluate(
        model,
        make_loader(holdout, train_args, False),
        device,
        train_args.max_eval_batches,
    )
    np.savez_compressed(output / "holdout_logits.npz", **arrays)
    write_csv(output / "holdout_predictions.csv", prediction_rows)
    write_csv(output / "training_history.csv", history)

    checkpoint = {
        "stage": "P86-v2",
        "architecture": args.architecture,
        "split": args.split,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": {
            **model_config_for(model, train_args),
            "p86v2_architecture": args.architecture,
        },
        "training_subjects": split.training_subjects,
        "holdout_subjects": split.holdout_subjects,
        "embargo_subjects": tuple(protocol["embargo"]["subjects"]),
        "fixed_epochs": epochs,
    }
    torch.save(checkpoint, output / "visual_student.pt")
    checkpoint_bytes = (output / "visual_student.pt").stat().st_size
    summary = {
        "stage": "P86-v2",
        "status": "smoke" if args.smoke else "formal",
        "architecture": args.architecture,
        "split": args.split,
        "protocol_path": str(args.protocol.resolve()),
        "sample_fingerprint": split.sample_fingerprint,
        "counts": {
            "train": len(split.training_indices),
            "holdout": len(split.holdout_indices),
            "embargo_excluded": len(split.embargo_indices),
        },
        "training_subjects": split.training_subjects,
        "holdout_subjects": split.holdout_subjects,
        "embargo_subjects": protocol["embargo"]["subjects"],
        "fixed_epochs": epochs,
        "validation_used_during_training": False,
        "holdout_metrics": metrics,
        "student_parameters": parameter_count(model),
        "student_fp32_mib": model_size_mib(model, 4),
        "student_fp16_mib": model_size_mib(model, 2),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_under_100mb": checkpoint_bytes
        < int(protocol["deployment"]["checkpoint_max_bytes_exclusive"]),
        "large_teacher_required_at_inference": False,
        "embargo_used_for_training_or_metrics": False,
        "kaggle_test_used": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(train_args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
