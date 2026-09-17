"""Frozen OpenAI CLIP semantic teacher over the P86 pixel cache.

The public CLIP backbone is frozen.  Two frames from each early/late window and
all scene/person/workspace views are embedded.  Linear heads are trained in the
three existing subject-disjoint folds; no Test rows are loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import clip
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PIXELS = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
OUTPUT = HERE / "runs/p119_clip_vitb32_semantic_teacher_v1"
A18 = PROJECT / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
FRAME_INDICES = (4, 11)
VIEW_NAMES = ("scene", "person", "workspace")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def class_names() -> list[str]:
    rows = read_rows(PROJECT / "class_mapping.csv")
    output = []
    for row in rows:
        value = row["action_name"].split("_", 1)[-1].replace("_", " ").lower()
        output.append(value)
    return output


def encode_text(model, device: torch.device) -> np.ndarray:
    prompts = []
    templates = (
        "a photo of a person {}",
        "a person is {}",
        "a grayscale video frame of a person {}",
    )
    names = class_names()
    with torch.inference_mode():
        for template in templates:
            tokens = clip.tokenize([template.format(name) for name in names]).to(device)
            feature = model.encode_text(tokens).float()
            feature = F.normalize(feature, dim=1)
            prompts.append(feature)
        output = F.normalize(torch.stack(prompts).mean(dim=0), dim=1)
    return output.cpu().numpy().astype(np.float32)


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pixel_cache / "rows.csv")
    images = np.load(args.pixel_cache / "images.npy", mmap_mode="r")
    if images.shape[:4] != (len(rows), 2, 16, 3):
        raise ValueError(f"unexpected pixel shape: {images.shape}")
    feature_path = output / "frame_features.npy"
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 2, len(FRAME_INDICES), 3, 512),
    )
    completed = np.zeros(len(rows), dtype=bool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    records: list[tuple[int, int, int, int]] = []
    for row in range(len(rows)):
        for window in range(2):
            for frame_position, frame in enumerate(FRAME_INDICES):
                for view in range(3):
                    records.append((row, window, frame_position, view))
    batch_size = args.batch_size
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            values = np.stack(
                [images[row, window, FRAME_INDICES[frame], view] for row, window, frame, view in batch_records]
            )
            tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
            tensor = tensor[:, None].repeat(1, 3, 1, 1) / 255.0
            tensor = F.interpolate(
                tensor, size=(224, 224), mode="bicubic", align_corners=False
            )
            tensor = (tensor - mean) / std
            embedding = model.encode_image(tensor).float()
            embedding = F.normalize(embedding, dim=1).cpu().numpy().astype(np.float16)
            for index, (row, window, frame, view) in enumerate(batch_records):
                features[row, window, frame, view] = embedding[index]
            if start % (batch_size * 20) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "clip_cache",
                            "encoded": min(start + len(batch_records), len(records)),
                            "total": len(records),
                        }
                    ),
                    flush=True,
                )
    features.flush()
    text_features = encode_text(model, device)
    np.save(output / "text_features.npy", text_features)
    report = {
        "stage": "P119_CLIP_ViT_B32_frozen_cache",
        "rows": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "shape": list(features.shape),
        "frame_indices": list(FRAME_INDICES),
        "views": list(VIEW_NAMES),
        "backbone": "OpenAI CLIP ViT-B/32 d05afc4; frozen",
        "test_rows_loaded": 0,
        "labels_used": False,
    }
    (output / "cache_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def descriptors(features: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    window_mean = values.mean(axis=2)  # rows, windows, views, dim
    window_std = values.std(axis=2)
    delta = window_mean[:, 1] - window_mean[:, 0]
    result = {}
    for view, name in enumerate(VIEW_NAMES):
        result[name] = np.concatenate(
            (
                window_mean[:, :, view].reshape(len(values), -1),
                window_std[:, :, view].reshape(len(values), -1),
                delta[:, view],
            ),
            axis=1,
        ).astype(np.float32)
    result["all_views"] = np.concatenate([result[name] for name in VIEW_NAMES], axis=1)
    for name, value in result.items():
        norms = np.linalg.norm(value, axis=1, keepdims=True)
        result[name] = value / np.maximum(norms, 1e-8)
    return result


def softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    scale = np.std(values, axis=1, keepdims=True)
    values = values / np.maximum(scale, 1e-6)
    values -= values.max(axis=1, keepdims=True)
    output = np.exp(values)
    return output / output.sum(axis=1, keepdims=True)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(labels)),
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def build_oof(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(args.pixel_cache / "rows.csv")
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    reference = np.load(A18)
    lookup = {value: index for index, value in enumerate(reference["sample_ids"].astype(str))}
    positions = np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)
    fold_ids = reference["fold_ids"][positions].astype(np.int64)
    if not np.array_equal(reference["labels"][positions], labels):
        raise ValueError("P119 label alignment mismatch")
    values = descriptors(np.load(args.output_dir / "frame_features.npy", mmap_mode="r"))
    variants = {}
    saved: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "labels": labels,
        "fold_ids": fold_ids,
    }
    for name, feature in values.items():
        scores = np.zeros((len(labels), 40), dtype=np.float64)
        fold_metrics = []
        for fold in sorted(set(fold_ids.tolist())):
            held = fold_ids == fold
            model = RidgeClassifier(
                alpha=args.alpha,
                class_weight="balanced",
                solver="lsqr",
                tol=1e-4,
            )
            model.fit(feature[~held], labels[~held])
            scores[held] = model.decision_function(feature[held])
            fold_metrics.append(
                {"fold": int(fold), **metrics(labels[held], scores[held].argmax(axis=1))}
            )
        probability = softmax(scores)
        prediction = probability.argmax(axis=1)
        variants[name] = {
            "feature_dim": int(feature.shape[1]),
            "metrics": metrics(labels, prediction),
            "folds": fold_metrics,
        }
        saved[f"{name}_scores"] = scores.astype(np.float32)
        saved[f"{name}_probability"] = probability.astype(np.float32)

    # Label-free zero-shot CLIP control.
    frame = np.asarray(
        np.load(args.output_dir / "frame_features.npy", mmap_mode="r"), dtype=np.float32
    )
    text = np.load(args.output_dir / "text_features.npy").astype(np.float32)
    similarities = np.einsum("nwfvd,cd->nwfvc", frame, text, optimize=True)
    zero_scores = similarities.mean(axis=(1, 2, 3)) + 0.5 * similarities.max(axis=(1, 2, 3))
    zero_prediction = zero_scores.argmax(axis=1)
    variants["zero_shot"] = {
        "feature_dim": 512,
        "metrics": metrics(labels, zero_prediction),
    }
    saved["zero_shot_scores"] = zero_scores.astype(np.float32)
    saved["zero_shot_probability"] = softmax(zero_scores).astype(np.float32)

    report = {
        "stage": "P119_CLIP_ViT_B32_semantic_teacher_OOF",
        "status": "complete",
        "protocol": {
            "backbone_frozen": True,
            "head": f"RidgeClassifier alpha={args.alpha} class_weight=balanced solver=lsqr",
            "folds": "three existing subject-disjoint folds",
            "primary_variant": "all_views",
            "held_fold_used_for_variant_or_alpha_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "variants": variants,
    }
    np.savez_compressed(args.output_dir / "oof_predictions.npz", **saved)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "cache", "oof"), default="all")
    parser.add_argument("--pixel-cache", type=Path, default=PIXELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pixel_cache = args.pixel_cache.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.stage in ("all", "cache"):
        build_cache(args)
    if args.stage in ("all", "oof"):
        build_oof(args)


if __name__ == "__main__":
    main()
