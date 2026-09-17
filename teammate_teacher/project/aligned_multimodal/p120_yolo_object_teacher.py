"""Frozen YOLO11 COCO object-evidence teacher on P86 scene/workspace frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PIXELS = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
OUTPUT = HERE / "runs/p120_yolo11n_coco_object_teacher_v1"
A18 = PROJECT / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
FRAME_INDICES = (4, 11)
VIEW_INDICES = (0, 2)
VIEW_NAMES = ("scene", "workspace")
OBJECT_CLASSES = 80


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def box_features(result) -> np.ndarray:
    output = np.zeros((3, OBJECT_CLASSES), dtype=np.float32)
    if result.boxes is None or len(result.boxes) == 0:
        return output.reshape(-1)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidence = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    height, width = result.orig_shape
    area = (
        np.maximum(xyxy[:, 2] - xyxy[:, 0], 0)
        * np.maximum(xyxy[:, 3] - xyxy[:, 1], 0)
        / max(float(height * width), 1.0)
    )
    for class_id in np.unique(classes):
        selected = classes == class_id
        output[0, class_id] = float(confidence[selected].max())
        output[1, class_id] = float(np.sum(confidence[selected] >= 0.10))
        output[2, class_id] = float(area[selected].max())
    return output.reshape(-1)


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pixel_cache / "rows.csv")
    images = np.load(args.pixel_cache / "images.npy", mmap_mode="r")
    features = np.lib.format.open_memmap(
        args.output_dir / "frame_object_features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 2, len(FRAME_INDICES), len(VIEW_INDICES), 240),
    )
    model = YOLO(args.model)
    records = [
        (row, window, frame_position, view_position)
        for row in range(len(rows))
        for window in range(2)
        for frame_position in range(len(FRAME_INDICES))
        for view_position in range(len(VIEW_INDICES))
    ]
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        batch = []
        for row, window, frame_position, view_position in batch_records:
            image = images[
                row,
                window,
                FRAME_INDICES[frame_position],
                VIEW_INDICES[view_position],
            ]
            batch.append(np.repeat(image[:, :, None], 3, axis=2))
        results = model.predict(
            source=batch,
            imgsz=args.image_size,
            conf=0.03,
            iou=0.70,
            device=0,
            half=True,
            verbose=False,
            batch=args.batch_size,
        )
        for record, result in zip(batch_records, results):
            features[record] = box_features(result).astype(np.float16)
        if start % (args.batch_size * 20) == 0:
            print(
                json.dumps(
                    {
                        "stage": "object_cache",
                        "encoded": min(start + len(batch_records), len(records)),
                        "total": len(records),
                    }
                ),
                flush=True,
            )
    features.flush()
    report = {
        "stage": "P120_YOLO11n_COCO_object_cache",
        "rows": len(rows),
        "shape": list(features.shape),
        "frame_indices": list(FRAME_INDICES),
        "views": list(VIEW_NAMES),
        "model": str(args.model),
        "model_names": model.names,
        "backbone_frozen": True,
        "labels_used": False,
        "test_rows_loaded": 0,
    }
    (args.output_dir / "cache_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def descriptors(features: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    window_mean = values.mean(axis=2)
    window_max = values.max(axis=2)
    delta = window_mean[:, 1] - window_mean[:, 0]
    output = {}
    for view, name in enumerate(VIEW_NAMES):
        output[name] = np.concatenate(
            (
                window_mean[:, :, view].reshape(len(values), -1),
                window_max[:, :, view].reshape(len(values), -1),
                delta[:, view],
            ),
            axis=1,
        )
    output["all_views"] = np.concatenate([output[name] for name in VIEW_NAMES], axis=1)
    return output


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
    values = descriptors(
        np.load(args.output_dir / "frame_object_features.npy", mmap_mode="r")
    )
    variants = {}
    saved: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "labels": labels,
        "fold_ids": fold_ids,
    }
    for name, feature in values.items():
        probability = np.zeros((len(labels), 40), dtype=np.float64)
        fold_metrics = []
        for fold in sorted(set(fold_ids.tolist())):
            held = fold_ids == fold
            model = ExtraTreesClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=12000 + int(fold),
                n_jobs=-1,
            )
            model.fit(feature[~held], labels[~held])
            probability[held] = model.predict_proba(feature[held])
            fold_metrics.append(
                {
                    "fold": int(fold),
                    **metrics(labels[held], probability[held].argmax(axis=1)),
                }
            )
        prediction = probability.argmax(axis=1)
        variants[name] = {
            "feature_dim": int(feature.shape[1]),
            "metrics": metrics(labels, prediction),
            "folds": fold_metrics,
        }
        saved[f"{name}_probability"] = probability.astype(np.float32)
    report = {
        "stage": "P120_YOLO11n_COCO_object_teacher_OOF",
        "status": "complete",
        "protocol": {
            "detector_frozen": True,
            "head": "ExtraTrees 500 depth12 leaf3 sqrt balanced",
            "primary_variant": "all_views",
            "held_fold_used_for_selection": False,
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
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=320)
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
