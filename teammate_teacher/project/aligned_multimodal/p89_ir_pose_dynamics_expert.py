from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from p31_skeleton_imu_preprocessing import safe_trial_path
from p89_imu_orientation_expert import signal_statistics_matrix
from p89_skeleton_identity_tracking_expert import (
    FOLDS,
    TEST_PIXEL,
    TEST_ROWS,
    TRAIN_PIXEL,
    TRAIN_ROWS,
)


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_CACHE = PROJECT_DIR / "runs/p28_adaptive_ir_pose_skeleton_full/trial_cache"
TEST_CACHE = PROJECT_DIR / "runs/p28_adaptive_ir_pose_skeleton_test/trial_cache"
OUTPUT = PROJECT_DIR / "runs/p89_ir_pose_dynamics_expert_v1"
SEED = 20260816
EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
    (0, 2), (1, 3), (2, 4),
)
PAIR_DISTANCES = (
    (9, 10), (9, 0), (10, 0), (9, 11), (10, 12), (9, 12), (10, 11),
    (9, 15), (10, 16), (7, 8), (5, 6), (11, 12), (5, 11), (6, 12),
    (0, 11), (0, 12), (15, 16),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def interpolate_pose(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(pose[:, :, :2], dtype=np.float64).copy()
    confidence = np.asarray(pose[:, :, 2], dtype=np.float64)
    mask = np.isfinite(xy).all(axis=2) & np.isfinite(confidence) & (confidence >= 0.15)
    timeline = np.arange(len(xy))
    for joint in range(17):
        valid = mask[:, joint]
        if valid.sum() >= 2:
            for axis in range(2):
                xy[~valid, joint, axis] = np.interp(
                    timeline[~valid], timeline[valid], xy[valid, joint, axis]
                )
        elif valid.sum() == 1:
            xy[:, joint] = xy[valid, joint][0]
        else:
            xy[:, joint] = 0.5
    return np.nan_to_num(xy, nan=0.5, posinf=0.5, neginf=0.5), mask


def first_difference(value: np.ndarray) -> np.ndarray:
    if len(value) < 2:
        return np.zeros_like(value)
    return np.concatenate((np.zeros_like(value[:1]), np.diff(value, axis=0)), axis=0)


def pose_signals(pose: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy, mask = interpolate_pose(pose)
    hip = 0.5 * (xy[:, 11] + xy[:, 12])
    shoulder = 0.5 * (xy[:, 5] + xy[:, 6])
    center = hip
    torso = np.linalg.norm(shoulder - hip, axis=1)
    valid_scale = np.isfinite(torso) & (torso > 1e-3)
    scale = float(np.median(torso[valid_scale])) if np.any(valid_scale) else 0.25
    normalized = (xy - center[:, None]) / max(scale, 0.05)
    local = xy - 0.5

    bone = np.stack([normalized[:, child] - normalized[:, parent] for parent, child in EDGES], axis=1)
    bone_length = np.linalg.norm(bone, axis=2)
    bone_direction = bone / np.maximum(bone_length[:, :, None], 1e-6)
    velocity = first_difference(normalized)
    acceleration = first_difference(velocity)
    speed = np.linalg.norm(velocity, axis=2)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=2)
    pair_distance = np.stack(
        [np.linalg.norm(normalized[:, left] - normalized[:, right], axis=1) for left, right in PAIR_DISTANCES],
        axis=1,
    )
    symmetry = np.stack(
        (
            speed[:, 9] - speed[:, 10],
            speed[:, 7] - speed[:, 8],
            speed[:, 13] - speed[:, 14],
            speed[:, 15] - speed[:, 16],
            bone_length[:, 0] - bone_length[:, 2],
            bone_length[:, 8] - bone_length[:, 10],
        ),
        axis=1,
    )
    box = np.asarray(boxes, dtype=np.float64)
    valid_box = np.isfinite(box[:, :4]).all(axis=1)
    box_filled = np.nan_to_num(box, nan=0.0, posinf=0.0, neginf=0.0)
    width = np.maximum(box_filled[:, 2] - box_filled[:, 0], 0.0) / 640.0
    height = np.maximum(box_filled[:, 3] - box_filled[:, 1], 0.0) / 480.0
    global_signal = np.stack(
        (
            (box_filled[:, 0] + box_filled[:, 2]) / (2.0 * 640.0) - 0.5,
            (box_filled[:, 1] + box_filled[:, 3]) / (2.0 * 480.0) - 0.5,
            width,
            height,
            width * height,
            box_filled[:, 4],
        ),
        axis=1,
    )
    global_signal = np.concatenate(
        (global_signal, first_difference(global_signal), first_difference(first_difference(global_signal))),
        axis=1,
    )
    signal = np.concatenate(
        (
            local.reshape(len(xy), -1),
            normalized.reshape(len(xy), -1),
            bone.reshape(len(xy), -1),
            bone_length,
            bone_direction.reshape(len(xy), -1),
            velocity.reshape(len(xy), -1),
            acceleration.reshape(len(xy), -1),
            speed,
            acceleration_magnitude,
            pair_distance,
            symmetry,
            global_signal,
        ),
        axis=1,
    )
    quality = np.concatenate(
        (
            mask.mean(axis=0),
            np.asarray(
                [mask.mean(), np.mean(mask[:, 9]), np.mean(mask[:, 10]), np.mean(valid_box), scale, len(xy) / 64.0],
                dtype=np.float64,
            ),
        )
    )
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0), quality


def phase_statistics(signal: np.ndarray, parts: int = 8) -> np.ndarray:
    selected = np.concatenate(
        (
            signal[:, 34:68],       # root-normalized joints
            signal[:, 100:116],     # bone lengths
            signal[:, 216:233],     # joint speed
            signal[:, 250:267],     # semantic distances
        ),
        axis=1,
    )
    blocks = []
    for part in np.array_split(selected, parts, axis=0):
        blocks.extend((part.mean(axis=0), part.std(axis=0)))
    return np.concatenate(blocks)


def feature_vector(pose: np.ndarray, boxes: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    signal, quality = pose_signals(pose, boxes)
    sampled = signal[np.asarray(chosen, dtype=np.int64)]
    blocks = (
        signal_statistics_matrix(signal).reshape(-1),
        signal_statistics_matrix(sampled).reshape(-1),
        phase_statistics(signal),
        phase_statistics(sampled),
        quality,
    )
    return np.nan_to_num(np.concatenate(blocks), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_features(
    rows_path: Path,
    pixel_cache: Path,
    pose_cache: Path,
    output_path: Path,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = read_csv(rows_path)
    indices = np.asarray(np.load(pixel_cache / "source_frame_indices.npy", mmap_mode="r"))
    if output_path.is_file() and output_path.with_suffix(".present.npy").is_file():
        return rows, np.load(output_path, mmap_mode="r"), np.load(output_path.with_suffix(".present.npy"))
    vectors: list[np.ndarray | None] = []
    present = np.zeros(len(rows), dtype=bool)
    dimension = None
    for index, row in enumerate(rows):
        path = pose_cache / safe_trial_path(row["source_id"]).with_suffix(".npz")
        if path.is_file():
            with np.load(path, allow_pickle=False) as source:
                pose = np.asarray(source["ir_keypoints_bbox_local"], dtype=np.float32)
                boxes = np.asarray(source["ir_boxes_xyxy_conf"], dtype=np.float32)
            chosen = np.asarray(indices[index], dtype=np.int64).reshape(-1)
            vector = feature_vector(pose, boxes, chosen)
            dimension = len(vector)
            vectors.append(vector)
            present[index] = True
        else:
            vectors.append(None)
        if (index + 1) % 300 == 0 or index + 1 == len(rows):
            print(f"pose features {output_path.stem} {index + 1}/{len(rows)}", flush=True)
    if dimension is None:
        raise RuntimeError("no visual pose cache rows")
    features = np.stack(
        [vector if vector is not None else np.zeros(dimension, dtype=np.float32) for vector in vectors]
    )
    np.save(output_path, features)
    np.save(output_path.with_suffix(".present.npy"), present)
    return rows, features, present


def model(name: str):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_depth=28,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=40,
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=6,
            min_child_samples=18,
            subsample=0.85,
            colsample_bytree=0.25,
            reg_alpha=1.0,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(name)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_rows, train_features, train_present = build_features(
        TRAIN_ROWS, TRAIN_PIXEL, TRAIN_CACHE, OUTPUT / "train_features.npy"
    )
    test_rows, test_features, test_present = build_features(
        TEST_ROWS, TEST_PIXEL, TEST_CACHE, OUTPUT / "test_features.npy"
    )
    labels = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in train_rows]).astype(str)
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    results = {}
    oof_probabilities = {}
    test_probabilities = {}
    for name in ("extra_trees", "lightgbm"):
        oof = np.zeros((len(labels), 40), dtype=np.float64)
        per_fold = []
        for fold in folds:
            fit = np.isin(users, fold["train_users"]) & train_present
            held = np.isin(users, fold["val_users"]) & train_present
            estimator = model(name)
            estimator.fit(train_features[fit], labels[fit])
            target = np.flatnonzero(held)
            partial = estimator.predict_proba(train_features[held])
            oof[np.ix_(target, estimator.classes_.astype(np.int64))] = partial
            per_fold.append({"fold": int(fold["fold"]), **metrics(labels[held], oof[target].argmax(1))})
        final = model(name)
        final.fit(train_features[train_present], labels[train_present])
        test_probability = np.full((len(test_rows), 40), 1.0 / 40.0, dtype=np.float64)
        partial = final.predict_proba(test_features[test_present])
        test_probability[np.ix_(np.flatnonzero(test_present), final.classes_.astype(np.int64))] = partial
        oof_probabilities[name] = oof
        test_probabilities[name] = test_probability
        results[name] = {"overall": metrics(labels[train_present], oof[train_present].argmax(1)), "folds": per_fold}
        print(name, results[name], flush=True)
    np.savez_compressed(
        OUTPUT / "oof_probabilities.npz",
        sample_ids=np.asarray([row["sample_id"] for row in train_rows]),
        labels=labels,
        present=train_present,
        **oof_probabilities,
    )
    np.savez_compressed(
        OUTPUT / "test_probabilities.npz",
        sample_ids=np.asarray([row["sample_id"] for row in test_rows]),
        present=test_present,
        **test_probabilities,
    )
    report = {
        "stage": "P89_tracked_IR_2D_pose_multistream_dynamics_v1",
        "protocol": "Frozen IR person track; bbox-local and root-normalized joints, bone vectors/directions, motion/acceleration, semantic distances, symmetry, box motion, spectral and eight-phase statistics; fixed subject-disjoint folds.",
        "feature_dimension": int(train_features.shape[1]),
        "train_present": int(train_present.sum()),
        "test_present": int(test_present.sum()),
        "models": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
