"""Frozen hand-object geometry teacher using YOLO11 pose and COCO objects."""

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
P29 = HERE / "runs/p29_dir_multiscale_roi_full/trial_roi_cache"
P28 = HERE / "runs/p28_adaptive_ir_pose_skeleton_full/trial_cache"
OUTPUT = HERE / "runs/p122_hand_object_relation_teacher_v1"
A18 = PROJECT / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
FRAME_POSITIONS = (4, 11)
POSE_KEYPOINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12)
OBJECT_CLASSES = (39, 40, 41, 45, 62, 63, 65, 66, 67, 71, 73, 79)
POSE_DIM = len(POSE_KEYPOINTS) * 3 + 5
OBJECT_DIM = len(OBJECT_CLASSES) * 4
RELATION_PER_OBJECT = 13
RELATION_DIM = len(OBJECT_CLASSES) * RELATION_PER_OBJECT
FRAME_DIM = POSE_DIM + OBJECT_DIM + RELATION_DIM


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def crop_bounds(
    width: int, height: int, box: np.ndarray, scale: float
) -> tuple[int, int, int, int]:
    if box.shape != (4,) or not np.isfinite(box).all():
        return 0, 0, width, height
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return 0, 0, width, height
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * scale
    left = max(0, int(round(center_x - 0.5 * side)))
    right = min(width, int(round(center_x + 0.5 * side)))
    top = max(0, int(round(center_y - 0.5 * side)))
    bottom = min(height, int(round(center_y + 0.5 * side)))
    if right <= left + 2 or bottom <= top + 2:
        return 0, 0, width, height
    return left, top, right, bottom


def project_keypoints(
    keypoints: np.ndarray, bounds: tuple[int, int, int, int]
) -> np.ndarray:
    left, top, right, bottom = bounds
    output = np.asarray(keypoints, dtype=np.float32).copy()
    output[:, 0] = (output[:, 0] - left) * 160.0 / max(right - left, 1)
    output[:, 1] = (output[:, 1] - top) * 160.0 / max(bottom - top, 1)
    return output


def distance_to_box(point: np.ndarray, box: np.ndarray) -> float:
    x, y = map(float, point)
    x1, y1, x2, y2 = map(float, box)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy) / 160.0)


def pose_vector(keypoints: np.ndarray) -> np.ndarray:
    selected = keypoints[list(POSE_KEYPOINTS)].copy()
    selected[:, :2] /= 160.0
    selected[:, :2] = np.clip(selected[:, :2], -1.0, 2.0)
    output = selected.reshape(-1).tolist()
    left_wrist = selected[5]
    right_wrist = selected[6]
    nose = selected[0]
    output.extend(
        (
            float(np.linalg.norm(left_wrist[:2] - right_wrist[:2])),
            float(np.linalg.norm(left_wrist[:2] - nose[:2])),
            float(np.linalg.norm(right_wrist[:2] - nose[:2])),
            float(left_wrist[1] - 0.5 * (selected[1, 1] + selected[2, 1])),
            float(right_wrist[1] - 0.5 * (selected[1, 1] + selected[2, 1])),
        )
    )
    return np.asarray(output, dtype=np.float32)


def best_boxes(result) -> dict[int, np.ndarray]:
    output = {}
    if result.boxes is None or len(result.boxes) == 0:
        return output
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidence = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    for class_id in OBJECT_CLASSES:
        selected = np.flatnonzero(classes == class_id)
        if len(selected):
            index = int(selected[np.argmax(confidence[selected])])
            output[class_id] = np.concatenate((boxes[index], confidence[index : index + 1]))
    return output


def frame_features(result, keypoints: np.ndarray) -> np.ndarray:
    boxes = best_boxes(result)
    pose = pose_vector(keypoints)
    objects = []
    relations = []
    wrists = (keypoints[9], keypoints[10])
    nose = keypoints[0]
    for class_id in OBJECT_CLASSES:
        value = boxes.get(class_id)
        if value is None:
            objects.extend((0.0, 0.0, 0.0, 0.0))
            relations.extend((0.0, 0.0, 2.0, 2.0, 0.0) * 2)
            relations.extend((0.0, 0.0, 2.0))
            continue
        x1, y1, x2, y2, confidence = map(float, value)
        center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / (160.0 * 160.0)
        objects.extend((confidence, area, center[0] / 160.0, center[1] / 160.0))
        box = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        for wrist in wrists:
            valid = float(wrist[2] >= 0.10)
            dx = (center[0] - wrist[0]) / 160.0 if valid else 0.0
            dy = (center[1] - wrist[1]) / 160.0 if valid else 0.0
            center_distance = float(np.hypot(dx, dy)) if valid else 2.0
            box_distance = distance_to_box(wrist[:2], box) if valid else 2.0
            inside = float(valid and x1 <= wrist[0] <= x2 and y1 <= wrist[1] <= y2)
            relations.extend((dx, dy, center_distance, box_distance, inside))
        nose_valid = float(nose[2] >= 0.10)
        relations.extend(
            (
                (center[0] - nose[0]) / 160.0 if nose_valid else 0.0,
                (center[1] - nose[1]) / 160.0 if nose_valid else 0.0,
                float(np.linalg.norm(center - nose[:2]) / 160.0) if nose_valid else 2.0,
            )
        )
    output = np.concatenate(
        (pose, np.asarray(objects, dtype=np.float32), np.asarray(relations, dtype=np.float32))
    )
    if output.shape != (FRAME_DIM,):
        raise RuntimeError(f"P122 frame feature shape changed: {output.shape}")
    return output


def trial_geometry(
    source_id: str, source_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    relative = Path(source_id).with_suffix(".npz")
    with np.load(P29 / relative, allow_pickle=False) as roi:
        roi_ids = roi["frame_ids"].astype(str)
        names = tuple(roi["region_names"].astype(str))
        boxes = roi["roi_boxes_xyxy"].astype(np.float32)
        valid = roi["roi_valid"].astype(bool)
        width = int(roi["image_width"])
        height = int(roi["image_height"])
    with np.load(P28 / relative, allow_pickle=False) as pose:
        pose_ids = pose["frame_ids"].astype(str)
        keypoints = pose["ir_keypoints_xy_conf"].astype(np.float32)
    if not np.array_equal(roi_ids, pose_ids):
        raise RuntimeError(f"P122 P28/P29 frame mismatch: {source_id}")
    workspace = names.index("hand_workspace")
    person = names.index("full_body")
    projected = np.zeros((2, len(FRAME_POSITIONS), 17, 3), dtype=np.float32)
    bounds = np.zeros((2, len(FRAME_POSITIONS), 4), dtype=np.int32)
    for window in range(2):
        for time, position in enumerate(FRAME_POSITIONS):
            frame = int(source_indices[window, position])
            box = (
                boxes[frame, workspace]
                if valid[frame, workspace]
                else boxes[frame, person]
                if valid[frame, person]
                else np.full(4, np.nan, dtype=np.float32)
            )
            value = crop_bounds(width, height, box, 1.40)
            bounds[window, time] = value
            projected[window, time] = project_keypoints(keypoints[frame], value)
    return projected, bounds


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pixel_cache / "rows.csv")
    images = np.load(args.pixel_cache / "images.npy", mmap_mode="r")
    source_indices = np.load(args.pixel_cache / "source_frame_indices.npy", mmap_mode="r")
    keypoint_cache = np.zeros((len(rows), 2, len(FRAME_POSITIONS), 17, 3), dtype=np.float32)
    for row, metadata in enumerate(rows):
        keypoint_cache[row], _ = trial_geometry(metadata["source_id"], source_indices[row])
        if row % 250 == 0:
            print(json.dumps({"stage": "geometry", "rows": row + 1}), flush=True)
    output = np.lib.format.open_memmap(
        args.output_dir / "frame_relation_features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 2, len(FRAME_POSITIONS), FRAME_DIM),
    )
    model = YOLO(args.model)
    records = [
        (row, window, time)
        for row in range(len(rows))
        for window in range(2)
        for time in range(len(FRAME_POSITIONS))
    ]
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        batch = []
        for row, window, time in batch_records:
            image = images[row, window, FRAME_POSITIONS[time], 2]
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
            row, window, time = record
            output[record] = frame_features(
                result, keypoint_cache[row, window, time]
            ).astype(np.float16)
        if start % (args.batch_size * 20) == 0:
            print(
                json.dumps(
                    {
                        "stage": "relation_cache",
                        "encoded": min(start + len(batch_records), len(records)),
                        "total": len(records),
                    }
                ),
                flush=True,
            )
    output.flush()
    report = {
        "stage": "P122_hand_object_relation_cache",
        "rows": len(rows),
        "shape": list(output.shape),
        "pose_dim": POSE_DIM,
        "object_dim": OBJECT_DIM,
        "relation_dim": RELATION_DIM,
        "object_classes": list(OBJECT_CLASSES),
        "model": str(args.model),
        "labels_used": False,
        "test_rows_loaded": 0,
    }
    (args.output_dir / "cache_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def descriptor(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=2)
    maximum = values.max(axis=2)
    delta = mean[:, 1] - mean[:, 0]
    return np.concatenate(
        (mean.reshape(len(values), -1), maximum.reshape(len(values), -1), delta),
        axis=1,
    ).astype(np.float32)


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
    raw = np.asarray(
        np.load(args.output_dir / "frame_relation_features.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    variants = {
        "pose_only": descriptor(raw[..., :POSE_DIM]),
        "object_only": descriptor(raw[..., POSE_DIM : POSE_DIM + OBJECT_DIM]),
        "relations_only": descriptor(raw[..., POSE_DIM + OBJECT_DIM :]),
        "all": descriptor(raw),
    }
    report_variants = {}
    saved: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "labels": labels,
        "fold_ids": fold_ids,
    }
    for name, feature in variants.items():
        probability = np.zeros((len(labels), 40), dtype=np.float64)
        folds = []
        for fold in sorted(set(fold_ids.tolist())):
            held = fold_ids == fold
            model = ExtraTreesClassifier(
                n_estimators=600,
                max_depth=14,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=12200 + int(fold),
                n_jobs=-1,
            )
            model.fit(feature[~held], labels[~held])
            probability[held] = model.predict_proba(feature[held])
            folds.append(
                {"fold": int(fold), **metrics(labels[held], probability[held].argmax(1))}
            )
        prediction = probability.argmax(1)
        report_variants[name] = {
            "feature_dim": int(feature.shape[1]),
            "metrics": metrics(labels, prediction),
            "folds": folds,
        }
        saved[f"{name}_probability"] = probability.astype(np.float32)
        saved[f"{name}_features"] = feature.astype(np.float16)
    report = {
        "stage": "P122_hand_object_relation_teacher_OOF",
        "status": "complete",
        "protocol": {
            "object_detector_frozen": True,
            "pose_detector_frozen": True,
            "primary_variant": "all",
            "controls": ["pose_only", "object_only", "relations_only"],
            "held_fold_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "variants": report_variants,
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
    parser.add_argument("--model", type=str, default="assets/models/yolo11n.pt")
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
