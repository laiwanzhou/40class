from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

from build_p46_videomae_cache import restore_legacy_attention_biases
from p86_videomae_small_visual_model import (
    DEFAULT_VIDEOMAE_SMALL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    resolve_snapshot,
)
from p86_visual_pixel_data import P86VisualPixelDataset, collate_p86_pixels
from train_p46_videomae_head import l2_normalize, make_model, row_standardize
from train_p85_videomae_full40_head import sample_weights
from train_p86_visual_student_oof import split_universe


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXELS = PROJECT_DIR / "runs/p86_visual_pixel_cache_t12_v8"
DEFAULT_TEACHER_FEATURES = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86_videomae_small_frozen_probe_fold0_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen VideoMAE-Small representation probe on the fold0 inner split only."
    )
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained-model", default=DEFAULT_VIDEOMAE_SMALL)
    parser.add_argument("--outer-fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metrics(labels: np.ndarray, prediction: np.ndarray, users: np.ndarray) -> dict[str, Any]:
    per_user = {
        user: float(accuracy_score(labels[users == user], prediction[users == user]))
        for user in sorted(set(users.tolist()))
    }
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "worst_subject_accuracy": float(min(per_user.values())),
        "per_user_accuracy": per_user,
    }


def selection_score(value: dict[str, Any]) -> float:
    return (
        float(value["accuracy"])
        + 0.5 * float(value["macro_f1"])
        + 0.25 * float(value["worst_subject_accuracy"])
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, args.outer_fold, args.seed)
    sample_ids = np.asarray(split["sample_ids"]).astype(str)
    selected_ids = np.concatenate(
        (sample_ids[split["inner_train"]], sample_ids[split["inner_dev"]])
    )
    reference = P86VisualPixelDataset(
        args.pixel_cache, args.teacher_features, args.teacher_logits
    )
    expected = (2, args.frames, 3, args.resolution, args.resolution)
    if tuple(reference.images.shape[1:]) != expected:
        raise RuntimeError(
            f"pixel cache geometry {tuple(reference.images.shape[1:])} does not match {expected}"
        )
    indices = np.asarray([reference.index_lookup[value] for value in selected_ids], dtype=np.int64)
    dataset = P86VisualPixelDataset(
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        indices=indices,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_p86_pixels,
    )

    snapshot = resolve_snapshot(args.pretrained_model)
    config = VideoMAEConfig.from_pretrained(snapshot, local_files_only=True)
    config.image_size = args.resolution
    config.num_frames = args.frames
    model = VideoMAEForVideoClassification.from_pretrained(
        snapshot, config=config, local_files_only=True
    )
    load_audit = restore_legacy_attention_biases(model, snapshot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    features: list[np.ndarray] = []
    kinetics: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    users: list[str] = []
    emitted_ids: list[str] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device, non_blocking=True)
            count, windows, steps, views, height, width = images.shape
            clips = images.permute(0, 1, 3, 2, 4, 5).reshape(
                count * windows * views, steps, 1, height, width
            )
            unit = clips.float().div_(255.0).repeat(1, 1, 3, 1, 1)
            mean = unit.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
            std = unit.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                sequence = model.videomae(pixel_values=(unit - mean) / std).last_hidden_state
                embedding = model.fc_norm(sequence.mean(dim=1))
                logit = model.classifier(embedding)
            features.append(embedding.float().cpu().numpy().reshape(count, 2, 3, -1))
            kinetics.append(logit.float().cpu().numpy().reshape(count, 2, 3, -1))
            valid_masks.append(batch["view_valid"].any(dim=2).numpy())
            labels.append(batch["label"].numpy())
            users.extend(batch["user_id"])
            emitted_ids.extend(batch["sample_id"])
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(loader):
                print(
                    json.dumps(
                        {"extracted": len(emitted_ids), "total": len(dataset)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    feature_values = np.concatenate(features).astype(np.float32)
    kinetics_values = np.concatenate(kinetics).astype(np.float32)
    valid_values = np.concatenate(valid_masks).astype(bool)
    label_values = np.concatenate(labels).astype(np.int64)
    user_values = np.asarray(users).astype(str)
    emitted = np.asarray(emitted_ids).astype(str)
    if not np.array_equal(emitted, selected_ids):
        raise RuntimeError("frozen probe row order changed")
    train_count = len(split["inner_train"])
    train_index = np.arange(train_count)
    dev_index = np.arange(train_count, len(emitted))
    normalized = l2_normalize(feature_values)
    masked = normalized * valid_values[..., None]
    standardized_kinetics = row_standardize(kinetics_values.reshape(len(emitted), -1))
    matrices = {
        "six_clip_features": normalized.reshape(len(emitted), -1),
        "six_clip_features_masked": masked.reshape(len(emitted), -1),
        "kinetics_logits": standardized_kinetics,
        "features_plus_kinetics": np.concatenate(
            (normalized.reshape(len(emitted), -1), standardized_kinetics), axis=1
        ),
    }
    candidates: list[dict[str, Any]] = []
    for feature_name, values in matrices.items():
        for power in (0.35, 0.75):
            weights = sample_weights(label_values[train_index], power)
            for alpha in (300.0, 1000.0, 3000.0):
                head = make_model(alpha)
                head.fit(
                    values[train_index],
                    label_values[train_index],
                    ridge__sample_weight=weights,
                )
                prediction = head.predict(values[dev_index]).astype(np.int64)
                value_metrics = metrics(
                    label_values[dev_index], prediction, user_values[dev_index]
                )
                candidates.append(
                    {
                        "feature_set": feature_name,
                        "class_weight_power": power,
                        "alpha": alpha,
                        "selection_score": selection_score(value_metrics),
                        **value_metrics,
                    }
                )
    candidates.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    np.savez_compressed(
        output / "inner_frozen_features.npz",
        sample_ids=emitted,
        users=user_values,
        labels=label_values,
        split=np.asarray(["inner_train"] * train_count + ["inner_dev"] * len(dev_index)),
        features=feature_values.astype(np.float16),
        kinetics_logits=kinetics_values.astype(np.float16),
        view_valid=valid_values,
    )
    summary = {
        "stage": "P86_VideoMAE_Small_frozen_inner_probe",
        "protocol": (
            "Only fold0 inner-train fits low-capacity Ridge heads; only inner-dev selects among "
            "the declared feature/regularization probes. Outer-held is neither extracted nor evaluated."
        ),
        "counts": {"inner_train": int(train_count), "inner_dev": int(len(dev_index))},
        "geometry": {"frames": args.frames, "resolution": args.resolution, "clips": 6},
        "pretrained_model": args.pretrained_model,
        "pretrained_load_audit": load_audit,
        "best": candidates[0],
        "candidates": candidates,
        "outer_held_evaluated": False,
        "test_used_for_selection": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
