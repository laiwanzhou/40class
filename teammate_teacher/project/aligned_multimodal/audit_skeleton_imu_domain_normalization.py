from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
RESEARCH_DOCS = REPO_ROOT / "docs" / "research"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from audit_subject_generalization_ceiling import (
    build_repeat_groups,
    evaluate_cross_subject,
    evaluate_same_subject,
    gap_pp,
    load_and_align_caches,
    load_class_names,
    metrics_bundle,
    normalize_rows,
    per_class_rows,
    per_subject_rows,
    prototype_predict,
    select_ids,
    top_confusion_rows,
    write_csv,
)


DISTANCES = ("cosine", "standardized_euclidean")
GATE_DISTANCE = "standardized_euclidean"
SKELETON_VARIANTS = (
    "S0_raw",
    "S1_root_center",
    "S2_scale",
    "S3_facing",
    "S4_body_local",
    "S5_relational_motion",
)
IMU_VARIANTS = (
    "I0_raw",
    "I1_device_robust",
    "I2_magnitude",
    "I3_gravity_aligned",
    "I4_rotation_invariant",
)
VARIANT_MODALITY = {
    **{name: "Skeleton" for name in SKELETON_VARIANTS},
    **{name: "IMU" for name in IMU_VARIANTS},
}
P23_REFERENCE_FEATURE = {
    "Skeleton": "skeleton_embedding",
    "IMU": "imu_embedding",
}
ROOT = 0
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 1, 2, 3
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 4, 5, 6
SPINE, THORAX, NECK, HEAD = 7, 8, 9, 10
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 12, 13
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 14, 15, 16
ANGLE_TRIPLETS = (
    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    (THORAX, RIGHT_SHOULDER, RIGHT_ELBOW),
    (THORAX, LEFT_SHOULDER, LEFT_ELBOW),
    (THORAX, RIGHT_HIP, RIGHT_KNEE),
    (THORAX, LEFT_HIP, LEFT_KNEE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P24 no-training Skeleton and IMU domain-normalization audit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "p24_skeleton_imu_domain_normalization.json",
    )
    parser.add_argument(
        "--p22-cache-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "cache",
    )
    parser.add_argument(
        "--p23-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p23_subject_generalization_audit",
    )
    parser.add_argument(
        "--skeleton-cache-dir",
        type=Path,
        default=PROJECT_DIR / "cache" / "skeleton_raw",
    )
    parser.add_argument(
        "--imu-cache-dir",
        type=Path,
        default=PROJECT_DIR / "cache" / "imu_32",
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "subject_folds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p24_skeleton_imu_domain_normalization_audit",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RESEARCH_DOCS
        / "10_local_and_domain"
        / "25_P24Skeleton与IMU无训练域归一化审计.md",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_masks(
    signals: np.ndarray,
    time_mask: np.ndarray,
    device_mask: np.ndarray,
) -> np.ndarray:
    masked = signals.astype(np.float32) * time_mask[..., None]
    return np.concatenate(
        [
            masked.reshape(len(masked), -1),
            time_mask.astype(np.float32).reshape(len(masked), -1),
            device_mask.astype(np.float32),
        ],
        axis=1,
    )


def robust_scale_device(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    if not valid.any():
        return output
    selected = values[valid].astype(np.float64)
    center = selected.mean(axis=0)
    q25, q75 = np.percentile(selected, [25, 75], axis=0)
    scale = (q75 - q25) / 1.349
    median = np.median(selected, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(selected - median), axis=0)
    std = selected.std(axis=0)
    scale = np.where(scale > 1e-6, scale, mad_scale)
    scale = np.where(scale > 1e-6, scale, std)
    scale = np.where(scale > 1e-6, scale, 1.0)
    output[valid] = ((selected - center) / scale).astype(np.float32)
    return output


def rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-8 or target_norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    source /= source_norm
    target /= target_norm
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0:
            return np.eye(3, dtype=np.float32)
        helper = np.asarray([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            helper = np.asarray([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    axis = cross / sine
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3)
        + skew * sine
        + (skew @ skew) * (1.0 - cosine)
    )
    return rotation.astype(np.float32)


def quaternion_delta_angle(quaternion: np.ndarray) -> np.ndarray:
    output = np.zeros(len(quaternion), dtype=np.float32)
    if len(quaternion) < 2:
        return output
    normalized = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=1, keepdims=True),
        1e-8,
    )
    dots = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1))
    output[1:] = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
    return output


def load_imu_variants(
    cache_dir: Path,
    canonical_ids: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    expected_presence: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    with (cache_dir / "index.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    index_by_id = {row["sample_id"]: row for row in rows}
    if len(index_by_id) != len(rows):
        raise ValueError("Duplicate sample_id in IMU index")
    missing = set(canonical_ids.astype(str).tolist()) - set(index_by_id)
    if missing:
        raise ValueError(f"IMU index missing {len(missing)} P23 sample IDs")

    cache_indices: list[int] = []
    usable: list[bool] = []
    metadata_errors: list[str] = []
    for sample_id, label, subject, presence in zip(
        canonical_ids.astype(str),
        labels.astype(int),
        subjects.astype(str),
        expected_presence.astype(bool),
    ):
        row = index_by_id[sample_id]
        cache_indices.append(int(row["cache_index"]))
        is_usable = bool(int(row["usable"]))
        usable.append(is_usable)
        if int(row["class_id"]) != label or row["user_id"] != subject:
            metadata_errors.append(sample_id)
        if is_usable != bool(presence):
            metadata_errors.append(f"{sample_id}:presence")
    if metadata_errors:
        raise ValueError(f"IMU metadata mismatch: {metadata_errors[:5]}")

    values_file = cache_dir / "imu_float32.npy"
    time_file = cache_dir / "time_mask_uint8.npy"
    device_file = cache_dir / "device_mask_uint8.npy"
    values_mm = np.load(values_file, mmap_mode="r", allow_pickle=False)
    time_mm = np.load(time_file, mmap_mode="r", allow_pickle=False)
    device_mm = np.load(device_file, mmap_mode="r", allow_pickle=False)
    selected = np.asarray(cache_indices, dtype=np.int64)
    values = np.asarray(values_mm[selected], dtype=np.float32)
    time_mask = np.asarray(time_mm[selected], dtype=bool)
    device_mask = np.asarray(device_mm[selected], dtype=bool)
    valid = np.asarray(usable, dtype=bool)

    raw = append_masks(values, time_mask, device_mask)

    robust = values.copy()
    robust[..., :6] = 0.0
    for sample_index in range(len(values)):
        for device_index in range(values.shape[1]):
            mask = time_mask[sample_index, device_index]
            robust[sample_index, device_index, :, :6] = robust_scale_device(
                values[sample_index, device_index, :, :6],
                mask,
            )
    robust_feature = append_masks(robust, time_mask, device_mask)

    acceleration = values[..., :3]
    gyro = values[..., 3:6]
    quaternion = values[..., 6:10]
    acceleration_magnitude = np.linalg.norm(acceleration, axis=-1)
    gyro_magnitude = np.linalg.norm(gyro, axis=-1)
    rotation_angle = 2.0 * np.arccos(
        np.clip(np.abs(quaternion[..., 0]), 0.0, 1.0)
    )
    magnitude_signals = np.stack(
        [acceleration_magnitude, gyro_magnitude, rotation_angle],
        axis=-1,
    )
    magnitude_feature = append_masks(magnitude_signals, time_mask, device_mask)

    gravity_aligned = np.zeros(
        values.shape[:-1] + (7,),
        dtype=np.float32,
    )
    for sample_index in range(len(values)):
        for device_index in range(values.shape[1]):
            mask = time_mask[sample_index, device_index]
            if not mask.any():
                continue
            gravity = np.median(
                acceleration[sample_index, device_index, mask],
                axis=0,
            )
            rotation = rotation_from_to(
                gravity,
                np.asarray([0.0, 0.0, 1.0]),
            )
            gravity_aligned[sample_index, device_index, :, :3] = (
                acceleration[sample_index, device_index] @ rotation.T
            )
            gravity_aligned[sample_index, device_index, :, 3:6] = (
                gyro[sample_index, device_index] @ rotation.T
            )
            gravity_aligned[sample_index, device_index, :, 6] = rotation_angle[
                sample_index, device_index
            ]
    gravity_feature = append_masks(gravity_aligned, time_mask, device_mask)

    invariant_waveforms = np.zeros(
        values.shape[:-1] + (5,),
        dtype=np.float32,
    )
    invariant_summaries = np.zeros(
        (len(values), values.shape[1], 5, 9),
        dtype=np.float32,
    )
    for sample_index in range(len(values)):
        for device_index in range(values.shape[1]):
            mask = time_mask[sample_index, device_index]
            if not mask.any():
                continue
            acc_norm = acceleration_magnitude[sample_index, device_index]
            gyro_norm = gyro_magnitude[sample_index, device_index]
            jerk = np.zeros_like(acc_norm)
            jerk[1:] = np.linalg.norm(
                np.diff(acceleration[sample_index, device_index], axis=0),
                axis=1,
            )
            angle = rotation_angle[sample_index, device_index]
            delta_angle = quaternion_delta_angle(
                quaternion[sample_index, device_index]
            )
            signals = np.stack(
                [acc_norm, gyro_norm, jerk, angle, delta_angle],
                axis=1,
            )
            invariant_waveforms[sample_index, device_index] = signals
            for channel in range(signals.shape[1]):
                selected_values = signals[mask, channel].astype(np.float64)
                q10, q25, q75, q90 = np.percentile(
                    selected_values,
                    [10, 25, 75, 90],
                )
                invariant_summaries[sample_index, device_index, channel] = (
                    float(selected_values.mean()),
                    float(selected_values.std()),
                    float(np.median(selected_values)),
                    float(q75 - q25),
                    float(np.sqrt(np.mean(selected_values**2))),
                    float(np.mean(selected_values**2)),
                    float(selected_values.max()),
                    float(q90),
                    float(selected_values.max() - selected_values.min()),
                )
    invariant_feature = np.concatenate(
        [
            append_masks(invariant_waveforms, time_mask, device_mask),
            invariant_summaries.reshape(len(values), -1),
        ],
        axis=1,
    )

    variants = {
        "I0_raw": raw,
        "I1_device_robust": robust_feature,
        "I2_magnitude": magnitude_feature,
        "I3_gravity_aligned": gravity_feature,
        "I4_rotation_invariant": invariant_feature,
    }
    audit = {
        "cache_rows": int(len(rows)),
        "aligned_samples": int(len(canonical_ids)),
        "usable_samples": int(valid.sum()),
        "metadata_mismatches": 0,
        "source_hashes": {
            "index.csv": sha256(cache_dir / "index.csv"),
            "metadata.json": sha256(cache_dir / "metadata.json"),
            "imu_float32.npy": sha256(values_file),
            "time_mask_uint8.npy": sha256(time_file),
            "device_mask_uint8.npy": sha256(device_file),
        },
        "variant_dimensions": {
            name: int(feature.shape[1]) for name, feature in variants.items()
        },
    }
    return variants, valid, audit


def resample_pose(
    coordinates: np.ndarray,
    times: np.ndarray,
    target_frames: int,
) -> tuple[np.ndarray, float]:
    if len(coordinates) == 1:
        return (
            np.repeat(coordinates.astype(np.float32), target_frames, axis=0),
            0.0,
        )
    times = times.astype(np.float64)
    if not np.all(np.isfinite(times)) or float(times[-1] - times[0]) <= 1e-6:
        source_progress = np.linspace(0.0, 1.0, len(coordinates))
        duration = max((len(coordinates) - 1) * 0.1, 0.0)
    else:
        duration = float(times[-1] - times[0])
        source_progress = (times - times[0]) / duration
    unique_progress, unique_indices = np.unique(source_progress, return_index=True)
    coordinates = coordinates[unique_indices]
    if len(unique_progress) == 1:
        return (
            np.repeat(coordinates[:1].astype(np.float32), target_frames, axis=0),
            duration,
        )
    target = np.linspace(0.0, 1.0, target_frames)
    flattened = coordinates.reshape(len(coordinates), -1)
    output = np.stack(
        [
            np.interp(target, unique_progress, flattened[:, channel])
            for channel in range(flattened.shape[1])
        ],
        axis=1,
    ).reshape(target_frames, 17, 3)
    return output.astype(np.float32), duration


def safe_unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8 or not np.isfinite(norm):
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def skeleton_scale(centered: np.ndarray) -> float:
    shoulder = np.linalg.norm(
        centered[:, RIGHT_SHOULDER] - centered[:, LEFT_SHOULDER],
        axis=1,
    )
    torso = np.linalg.norm(centered[:, THORAX] - centered[:, ROOT], axis=1)
    values = 0.5 * (shoulder + torso)
    values = values[np.isfinite(values) & (values > 1e-6)]
    return float(np.median(values)) if len(values) else 1.0


def yaw_aligned(centered_scaled: np.ndarray) -> np.ndarray:
    shoulder_axis = (
        centered_scaled[:, RIGHT_SHOULDER]
        - centered_scaled[:, LEFT_SHOULDER]
    )
    hip_axis = centered_scaled[:, RIGHT_HIP] - centered_scaled[:, LEFT_HIP]
    axis = np.median(0.5 * (shoulder_axis + hip_axis), axis=0)
    axis[2] = 0.0
    if np.linalg.norm(axis) < 1e-8:
        return centered_scaled.copy()
    angle = float(np.arctan2(axis[1], axis[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return centered_scaled @ rotation.T


def body_local(centered_scaled: np.ndarray) -> np.ndarray:
    across = np.median(
        0.5
        * (
            centered_scaled[:, RIGHT_SHOULDER]
            - centered_scaled[:, LEFT_SHOULDER]
            + centered_scaled[:, RIGHT_HIP]
            - centered_scaled[:, LEFT_HIP]
        ),
        axis=0,
    )
    up = np.median(
        centered_scaled[:, THORAX] - centered_scaled[:, ROOT],
        axis=0,
    )
    x_axis = safe_unit(across, np.asarray([1.0, 0.0, 0.0]))
    up = up - float(np.dot(up, x_axis)) * x_axis
    z_axis = safe_unit(up, np.asarray([0.0, 0.0, 1.0]))
    if float(np.dot(z_axis, np.asarray([0.0, 0.0, 1.0]))) < 0:
        z_axis *= -1
    y_axis = safe_unit(
        np.cross(z_axis, x_axis),
        np.asarray([0.0, 1.0, 0.0]),
    )
    x_axis = safe_unit(np.cross(y_axis, z_axis), x_axis)
    basis = np.stack([x_axis, y_axis, z_axis])
    return centered_scaled @ basis.T


def angle_features(coordinates: np.ndarray) -> np.ndarray:
    features: list[np.ndarray] = []
    for first, middle, last in ANGLE_TRIPLETS:
        left = coordinates[:, first] - coordinates[:, middle]
        right = coordinates[:, last] - coordinates[:, middle]
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cosine = np.divide(
            np.sum(left * right, axis=1),
            denominator,
            out=np.zeros(len(coordinates), dtype=np.float32),
            where=denominator > 1e-8,
        )
        features.append(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)
    return np.stack(features, axis=1).astype(np.float32)


def load_skeleton_variants(
    cache_dir: Path,
    canonical_ids: np.ndarray,
    target_frames: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    metadata_file = cache_dir / "metadata.json"
    raw_file = cache_dir / "skeleton_raw_float32.npy"
    time_file = cache_dir / "frame_time_float32.npy"
    people_file = cache_dir / "person_count_uint8.npy"
    metadata = load_json(metadata_file)
    sample_ids = np.asarray(metadata["sample_ids"], dtype=str)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample_id in Skeleton cache")
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if set(canonical_ids.astype(str).tolist()) != set(index_by_id):
        raise ValueError("Skeleton cache sample_id universe differs from P23")

    raw_mm = np.load(raw_file, mmap_mode="r", allow_pickle=False)
    time_mm = np.load(time_file, mmap_mode="r", allow_pickle=False)
    people_mm = np.load(people_file, mmap_mode="r", allow_pickle=False)
    offsets = np.asarray(metadata["offsets"], dtype=np.int64)
    sampled = np.zeros((len(canonical_ids), target_frames, 17, 3), dtype=np.float32)
    durations = np.zeros(len(canonical_ids), dtype=np.float32)
    present_frames = np.zeros(len(canonical_ids), dtype=np.int64)

    for output_index, sample_id in enumerate(canonical_ids.astype(str)):
        cache_index = index_by_id[sample_id]
        offset, length = offsets[cache_index]
        frame_mask = np.asarray(
            people_mm[offset : offset + length] > 0,
            dtype=bool,
        )
        coordinates = np.asarray(
            raw_mm[offset : offset + length, :, :3],
            dtype=np.float32,
        )[frame_mask]
        times = np.asarray(
            time_mm[offset : offset + length],
            dtype=np.float32,
        )[frame_mask]
        present_frames[output_index] = len(coordinates)
        if not len(coordinates):
            continue
        sampled[output_index], durations[output_index] = resample_pose(
            coordinates,
            times,
            target_frames,
        )

    valid = present_frames > 0
    root_centered = sampled - sampled[:, :, ROOT : ROOT + 1]
    scales = np.asarray(
        [skeleton_scale(clip) for clip in root_centered],
        dtype=np.float32,
    )
    scaled = root_centered / scales[:, None, None, None]
    facing = np.stack([yaw_aligned(clip) for clip in scaled]).astype(np.float32)
    local = np.stack([body_local(clip) for clip in scaled]).astype(np.float32)

    pair_left, pair_right = np.triu_indices(17, k=1)
    distances = np.linalg.norm(
        local[:, :, pair_left] - local[:, :, pair_right],
        axis=3,
    )
    angles = np.stack([angle_features(clip) for clip in local])
    delta_time = np.divide(
        durations,
        target_frames - 1,
        out=np.ones_like(durations),
        where=durations > 1e-6,
    )
    speeds = np.linalg.norm(np.diff(local, axis=1), axis=3)
    speeds /= delta_time[:, None, None]
    relational = np.concatenate(
        [
            local.reshape(len(local), -1),
            angles.reshape(len(local), -1),
            distances.reshape(len(local), -1),
            speeds.reshape(len(local), -1),
        ],
        axis=1,
    ).astype(np.float32)

    variants = {
        "S0_raw": sampled.reshape(len(sampled), -1),
        "S1_root_center": root_centered.reshape(len(sampled), -1),
        "S2_scale": scaled.reshape(len(sampled), -1),
        "S3_facing": facing.reshape(len(sampled), -1),
        "S4_body_local": local.reshape(len(sampled), -1),
        "S5_relational_motion": relational,
    }
    audit = {
        "aligned_samples": int(len(canonical_ids)),
        "valid_samples": int(valid.sum()),
        "present_frames": {
            "minimum": int(present_frames.min()),
            "median": float(np.median(present_frames)),
            "maximum": int(present_frames.max()),
            "single_frame_samples": int((present_frames == 1).sum()),
        },
        "clip_duration_seconds": {
            "median": float(np.median(durations)),
            "maximum": float(durations.max()),
        },
        "scale": {
            "median": float(np.median(scales)),
            "minimum": float(scales.min()),
            "maximum": float(scales.max()),
        },
        "source_hashes": {
            "metadata.json": sha256(metadata_file),
            "skeleton_raw_float32.npy": sha256(raw_file),
            "frame_time_float32.npy": sha256(time_file),
            "person_count_uint8.npy": sha256(people_file),
        },
        "variant_dimensions": {
            name: int(feature.shape[1]) for name, feature in variants.items()
        },
    }
    return variants, valid, audit


def evaluate_subject_id(
    features: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    complete_groups: dict[str, list[tuple[int, int, int]]],
) -> dict[str, Any]:
    subject_names = sorted(np.unique(subjects.astype(str)).tolist())
    subject_to_id = {subject: index for index, subject in enumerate(subject_names)}
    grouped_by_class: dict[int, dict[str, list[tuple[int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for subject, groups in complete_groups.items():
        for group in groups:
            group_array = np.asarray(group)
            if not bool(valid[group_array].all()):
                continue
            class_values = np.unique(labels[group_array])
            if len(class_values) != 1:
                raise ValueError("Repeat group spans multiple action classes")
            grouped_by_class[int(class_values[0])][subject].append(group)

    true_subjects: list[int] = []
    predicted_subjects: list[int] = []
    test_actions: list[int] = []
    candidate_counts: list[int] = []
    for class_id in sorted(grouped_by_class):
        by_subject = grouped_by_class[class_id]
        if len(by_subject) < 2:
            continue
        for held_repeat in range(3):
            train_indices: list[int] = []
            train_subject_ids: list[int] = []
            test_indices: list[int] = []
            test_subject_ids: list[int] = []
            for subject in sorted(by_subject):
                subject_id = subject_to_id[subject]
                for group in by_subject[subject]:
                    test_indices.append(group[held_repeat])
                    test_subject_ids.append(subject_id)
                    for position in range(3):
                        if position != held_repeat:
                            train_indices.append(group[position])
                            train_subject_ids.append(subject_id)
            predictions, classes = prototype_predict(
                features[np.asarray(train_indices)],
                np.asarray(train_subject_ids),
                features[np.asarray(test_indices)],
                "standardized_euclidean",
            )
            true_subjects.extend(test_subject_ids)
            predicted_subjects.extend(predictions.tolist())
            test_actions.extend([class_id] * len(test_indices))
            candidate_counts.extend([len(classes)] * len(test_indices))

    truth = np.asarray(true_subjects, dtype=np.int64)
    predictions = np.asarray(predicted_subjects, dtype=np.int64)
    candidates = np.asarray(candidate_counts, dtype=np.int64)
    per_subject = {}
    for subject, subject_id in subject_to_id.items():
        mask = truth == subject_id
        if mask.any():
            per_subject[subject] = {
                "samples": int(mask.sum()),
                "accuracy": float((predictions[mask] == truth[mask]).mean()),
            }
    return {
        "samples": int(len(truth)),
        "accuracy": float((predictions == truth).mean()),
        "chance_accuracy": float(np.mean(1.0 / candidates)),
        "candidate_subjects": {
            "minimum": int(candidates.min()),
            "median": float(np.median(candidates)),
            "maximum": int(candidates.max()),
        },
        "per_subject": per_subject,
        "true_subject_ids": truth,
        "predicted_subject_ids": predictions,
        "action_ids": np.asarray(test_actions, dtype=np.int64),
    }


def load_p23_reference(
    p23_dir: Path,
    modality: str,
) -> dict[str, dict[str, np.ndarray]]:
    feature = P23_REFERENCE_FEATURE[modality]
    key = f"{feature}__{GATE_DISTANCE}"
    with np.load(p23_dir / "predictions.npz", allow_pickle=False) as archive:
        return {
            protocol: {
                field: archive[f"{key}__{protocol}__{field}"]
                for field in ("sample_ids", "labels", "subjects", "predictions")
            }
            for protocol in ("same_subject", "cross_subject_matched")
        }


def align_result(
    result: dict[str, np.ndarray],
    target_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    index_by_id = {
        sample_id: index
        for index, sample_id in enumerate(result["sample_ids"].astype(str))
    }
    if set(index_by_id) != set(target_ids.astype(str).tolist()):
        raise ValueError("Result sample_id set differs from gate reference")
    order = np.asarray([index_by_id[sample_id] for sample_id in target_ids.astype(str)])
    return {key: value[order] for key, value in result.items() if isinstance(value, np.ndarray)}


def subject_improvement_count(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> tuple[int, dict[str, Any]]:
    aligned = align_result(candidate, reference["sample_ids"])
    if not np.array_equal(aligned["labels"], reference["labels"]):
        raise ValueError("Candidate/reference labels differ")
    subjects = reference["subjects"].astype(str)
    details: dict[str, Any] = {}
    improved = 0
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        candidate_accuracy = float(
            (aligned["predictions"][mask] == aligned["labels"][mask]).mean()
        )
        reference_accuracy = float(
            (reference["predictions"][mask] == reference["labels"][mask]).mean()
        )
        delta = (candidate_accuracy - reference_accuracy) * 100.0
        improved += int(delta > 0.0)
        details[subject] = {
            "samples": int(mask.sum()),
            "reference_accuracy": reference_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "delta_pp": delta,
            "improved": bool(delta > 0.0),
        }
    return improved, details


def evaluate_variant(
    variant: str,
    features: np.ndarray,
    valid: np.ndarray,
    reference_cache: dict[str, np.ndarray],
    subject_folds: dict[str, int],
    complete_groups: dict[str, list[tuple[int, int, int]]],
    small_ids: list[int],
    hard_ids: list[int],
    class_names: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    features_by_fold = {fold: features for fold in range(3)}
    valid_by_fold = {fold: valid for fold in range(3)}
    distance_results: dict[str, Any] = {}
    class_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    small_set, hard_set = set(small_ids), set(hard_ids)

    for distance in DISTANCES:
        same = evaluate_same_subject(
            features_by_fold,
            valid_by_fold,
            reference_cache,
            subject_folds,
            complete_groups,
            distance,
        )
        cross_all = evaluate_cross_subject(
            features_by_fold,
            valid_by_fold,
            reference_cache,
            subject_folds,
            distance,
        )
        same_ids = set(same["sample_ids"].astype(str).tolist())
        cross_matched = select_ids(cross_all, same_ids)
        same_metrics = metrics_bundle(
            same["labels"], same["predictions"], small_ids, hard_ids
        )
        cross_metrics = metrics_bundle(
            cross_matched["labels"],
            cross_matched["predictions"],
            small_ids,
            hard_ids,
        )
        cross_all_metrics = metrics_bundle(
            cross_all["labels"], cross_all["predictions"], small_ids, hard_ids
        )
        distance_results[distance] = {
            "same_subject": same_metrics,
            "cross_subject_matched": cross_metrics,
            "cross_subject_all": cross_all_metrics,
            "gap_pp": {
                subset: gap_pp(same_metrics, cross_metrics, subset)
                for subset in ("all", "small", "hard")
            },
            "coverage": {
                "same_subject": same["coverage"],
                "cross_subject_matched_samples": int(len(cross_matched["labels"])),
                "cross_subject_all_samples": int(len(cross_all["labels"])),
            },
        }
        protocol_results = {
            "same_subject": same,
            "cross_subject_matched": cross_matched,
            "cross_subject_all": cross_all,
        }
        for protocol, result in protocol_results.items():
            class_rows.extend(
                per_class_rows(
                    variant,
                    distance,
                    protocol,
                    result,
                    class_names,
                    small_set,
                    hard_set,
                )
            )
            confusion_rows.extend(
                top_confusion_rows(
                    variant,
                    distance,
                    protocol,
                    result,
                    class_names,
                )
            )
            for field in ("sample_ids", "labels", "subjects", "predictions"):
                prediction_arrays[
                    f"{variant}__{distance}__{protocol}__{field}"
                ] = result[field]
        subject_rows.extend(
            per_subject_rows(
                variant,
                distance,
                same,
                cross_matched,
                cross_all,
                small_ids,
                hard_ids,
            )
        )
    return (
        distance_results,
        class_rows,
        subject_rows,
        confusion_rows,
        prediction_arrays,
    )


def gate_variant(
    variant: str,
    result: dict[str, Any],
    candidate_predictions: dict[str, np.ndarray],
    p23_reference: dict[str, dict[str, np.ndarray]],
    small_ids: list[int],
    hard_ids: list[int],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    modality = VARIANT_MODALITY[variant]
    reference_same = p23_reference["same_subject"]
    reference_cross = p23_reference["cross_subject_matched"]
    candidate_same_key = f"{variant}__{GATE_DISTANCE}__same_subject"
    candidate_cross_key = f"{variant}__{GATE_DISTANCE}__cross_subject_matched"
    candidate_same = {
        field: candidate_predictions[f"{candidate_same_key}__{field}"]
        for field in ("sample_ids", "labels", "subjects", "predictions")
    }
    candidate_cross = {
        field: candidate_predictions[f"{candidate_cross_key}__{field}"]
        for field in ("sample_ids", "labels", "subjects", "predictions")
    }
    aligned_same = align_result(candidate_same, reference_same["sample_ids"])
    aligned_cross = align_result(candidate_cross, reference_cross["sample_ids"])
    if not np.array_equal(aligned_same["labels"], reference_same["labels"]):
        raise ValueError(f"{variant} same labels differ from P23 {modality}")
    if not np.array_equal(aligned_cross["labels"], reference_cross["labels"]):
        raise ValueError(f"{variant} cross labels differ from P23 {modality}")

    reference_same_metrics = metrics_bundle(
        reference_same["labels"],
        reference_same["predictions"],
        small_ids,
        hard_ids,
    )
    reference_cross_metrics = metrics_bundle(
        reference_cross["labels"],
        reference_cross["predictions"],
        small_ids,
        hard_ids,
    )
    candidate_metrics = result[GATE_DISTANCE]
    overall_gain = (
        candidate_metrics["cross_subject_matched"]["all"]["accuracy"]
        - reference_cross_metrics["all"]["accuracy"]
    ) * 100.0
    hard_gain = (
        candidate_metrics["cross_subject_matched"]["hard"]["accuracy"]
        - reference_cross_metrics["hard"]["accuracy"]
    ) * 100.0
    same_delta = (
        candidate_metrics["same_subject"]["all"]["accuracy"]
        - reference_same_metrics["all"]["accuracy"]
    ) * 100.0
    reference_gap = (
        reference_same_metrics["all"]["accuracy"]
        - reference_cross_metrics["all"]["accuracy"]
    ) * 100.0
    gap_reduction = reference_gap - candidate_metrics["gap_pp"]["all"]
    improved_subjects, per_subject = subject_improvement_count(
        aligned_cross,
        reference_cross,
    )
    checks = {
        "cross_overall_gain": bool(
            overall_gain
            >= float(thresholds["cross_subject_overall_gain_pp_min"])
        ),
        "cross_hard_gain": bool(
            hard_gain >= float(thresholds["cross_subject_hard_gain_pp_min"])
        ),
        "gap_reduction": bool(
            gap_reduction
            >= float(thresholds["same_cross_gap_reduction_pp_min"])
        ),
        "same_subject_preserved": bool(
            same_delta
            >= float(thresholds["same_subject_accuracy_delta_pp_min"])
        ),
        "improved_subjects": bool(
            improved_subjects >= int(thresholds["improved_subjects_min"])
        ),
    }
    return {
        "variant": variant,
        "modality": modality,
        "reference_feature": P23_REFERENCE_FEATURE[modality],
        "distance": GATE_DISTANCE,
        "reference": {
            "same_accuracy": reference_same_metrics["all"]["accuracy"],
            "cross_accuracy": reference_cross_metrics["all"]["accuracy"],
            "cross_hard_accuracy": reference_cross_metrics["hard"]["accuracy"],
            "gap_pp": reference_gap,
        },
        "candidate": {
            "same_accuracy": candidate_metrics["same_subject"]["all"]["accuracy"],
            "cross_accuracy": candidate_metrics["cross_subject_matched"]["all"]["accuracy"],
            "cross_hard_accuracy": candidate_metrics["cross_subject_matched"]["hard"]["accuracy"],
            "gap_pp": candidate_metrics["gap_pp"]["all"],
        },
        "deltas": {
            "cross_overall_gain_pp": overall_gain,
            "cross_hard_gain_pp": hard_gain,
            "gap_reduction_pp": gap_reduction,
            "same_subject_accuracy_delta_pp": same_delta,
            "improved_subjects": improved_subjects,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "per_subject": per_subject,
    }


def fmt_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.2f}%"


def fmt_pp(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.2f} pp"


def render_report(
    summary: dict[str, Any],
    per_class: list[dict[str, Any]],
    per_subject: list[dict[str, Any]],
) -> str:
    lines = [
        "# 25_P24 Skeleton 与 IMU 无训练域归一化审计",
        "",
        f"**协议：** `{summary['protocol']}`",
        "",
        f"**分析代码提交：** `{summary['code_commit']}`",
        "",
        "**限制遵守：** 未训练神经网络、未运行 encoder、未修改 checkpoint、"
        "未覆盖 OOF、未启动主体对抗或对比学习。",
        "",
        "## 可解释边界",
        "",
        "P22/P23 pooled cache 不含逐帧关节和 IMU 波形，因此本实验读取已经存在的 "
        "`skeleton_raw` 与 `imu_32` cache，构造无训练显式特征。它是归一化迁移上限代理，"
        "不是“原 P23 encoder 输入归一化后”的复跑准确率。",
        "",
        "## 数据与协议",
        "",
        f"- Skeleton：{summary['input_audit']['skeleton']['aligned_samples']} 个 sample_id，"
        f"有效 {summary['input_audit']['skeleton']['valid_samples']}；present frame 中位数 "
        f"{summary['input_audit']['skeleton']['present_frames']['median']:.0f}。",
        f"- IMU：{summary['input_audit']['imu']['aligned_samples']} 个 sample_id，"
        f"可用 {summary['input_audit']['imu']['usable_samples']}；标签、subject、presence "
        "与 P22 完全一致。",
        "- 动作识别完全复用 P23 的同主体三连 2→1、跨主体 40 类 prototype、"
        "cosine 与训练侧标准化欧氏距离。",
        "- Subject-ID 使用同 action、同三连 2→1 的标准化欧氏 subject prototype；"
        "用于判断主体信息是否下降，不参与动作通过门槛。",
        "",
        "## 动作识别结果",
        "",
        "| 模态 | 步骤 | 距离 | Same | Cross | Gap | Small Cross | Hard Cross |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in SKELETON_VARIANTS + IMU_VARIANTS:
        for distance in DISTANCES:
            item = summary["results"][variant][distance]
            lines.append(
                f"| {VARIANT_MODALITY[variant]} | {variant} | {distance} | "
                f"{fmt_pct(item['same_subject']['all']['accuracy'])} | "
                f"{fmt_pct(item['cross_subject_matched']['all']['accuracy'])} | "
                f"{fmt_pp(item['gap_pp']['all'])} | "
                f"{fmt_pct(item['cross_subject_matched']['small']['accuracy'])} | "
                f"{fmt_pct(item['cross_subject_matched']['hard']['accuracy'])} |"
            )

    lines.extend(
        [
            "",
            "## 预注册通过门槛",
            "",
            "各步骤在标准化欧氏距离下与 P23 对应 Skeleton/IMU embedding 的完全相同 "
            "sample_id 比较；五项必须全部通过。",
            "",
            "| 步骤 | Cross Δ | Hard Δ | Gap缩小 | Same Δ | 改善subject | 通过 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in SKELETON_VARIANTS[1:] + IMU_VARIANTS[1:]:
        gate = summary["gates"][variant]
        deltas = gate["deltas"]
        lines.append(
            f"| {variant} | {fmt_pp(deltas['cross_overall_gain_pp'])} | "
            f"{fmt_pp(deltas['cross_hard_gain_pp'])} | "
            f"{fmt_pp(deltas['gap_reduction_pp'])} | "
            f"{fmt_pp(deltas['same_subject_accuracy_delta_pp'])} | "
            f"{deltas['improved_subjects']}/18 | {'是' if gate['passed'] else '否'} |"
        )

    lines.extend(
        [
            "",
            "## Subject-ID 可预测性",
            "",
            "| 模态 | 步骤 | Subject-ID | 随机机会 | 相对原始 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant in SKELETON_VARIANTS + IMU_VARIANTS:
        item = summary["subject_id"][variant]
        raw_variant = "S0_raw" if VARIANT_MODALITY[variant] == "Skeleton" else "I0_raw"
        raw_accuracy = summary["subject_id"][raw_variant]["accuracy"]
        lines.append(
            f"| {VARIANT_MODALITY[variant]} | {variant} | "
            f"{fmt_pct(item['accuracy'])} | {fmt_pct(item['chance_accuracy'])} | "
            f"{fmt_pp((item['accuracy'] - raw_accuracy) * 100.0)} |"
        )

    class_lookup = {
        (row["feature"], row["class_id"]): row
        for row in per_class
        if row["distance"] == GATE_DISTANCE
        and row["protocol"] == "cross_subject_matched"
    }
    class_names = summary["class_names"]
    lines.extend(
        [
            "",
            "## 每类 Cross recall（标准化欧氏）",
            "",
            "| ID | 类别 | " + " | ".join(SKELETON_VARIANTS + IMU_VARIANTS) + " |",
            "|---:|---|" + "|".join(["---:"] * (len(SKELETON_VARIANTS) + len(IMU_VARIANTS))) + "|",
        ]
    )
    for class_id in range(40):
        values = [
            fmt_pct(class_lookup[(variant, class_id)]["recall"])
            for variant in SKELETON_VARIANTS + IMU_VARIANTS
        ]
        lines.append(
            f"| {class_id} | {class_names[str(class_id)]} | "
            + " | ".join(values)
            + " |"
        )

    subject_lookup = {
        (row["feature"], row["subject"]): row
        for row in per_subject
        if row["distance"] == GATE_DISTANCE
    }
    lines.extend(
        [
            "",
            "## 每 subject Cross accuracy（标准化欧氏）",
            "",
            "### Skeleton",
            "",
            "| subject | " + " | ".join(SKELETON_VARIANTS) + " |",
            "|---|" + "|".join(["---:"] * len(SKELETON_VARIANTS)) + "|",
        ]
    )
    subjects = sorted(summary["input_audit"]["p22"]["subject_folds"])
    for subject in subjects:
        values = [
            fmt_pct(subject_lookup[(variant, subject)]["cross_matched_accuracy"])
            for variant in SKELETON_VARIANTS
        ]
        lines.append(f"| {subject} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "### IMU",
            "",
            "| subject | " + " | ".join(IMU_VARIANTS) + " |",
            "|---|" + "|".join(["---:"] * len(IMU_VARIANTS)) + "|",
        ]
    )
    for subject in subjects:
        values = [
            fmt_pct(subject_lookup[(variant, subject)]["cross_matched_accuracy"])
            for variant in IMU_VARIANTS
        ]
        lines.append(f"| {subject} | " + " | ".join(values) + " |")

    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## 最终裁决",
            "",
            f"**{decision['title']}**",
            "",
            decision["reason"],
            "",
            decision["next_recommendation"],
            "",
            "完整 JSON、每类、每 subject、混淆对、Subject-ID 与逐样本预测保存在 "
            "`aligned_multimodal/runs/p24_skeleton_imu_domain_normalization_audit/`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite P24 output: {args.output_dir}")
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite P24 report: {args.report}")

    config = load_json(args.config)
    class_names = load_class_names(args.fold_dir)
    p22_caches, p22_audit = load_and_align_caches(args.p22_cache_dir)
    reference_cache = p22_caches[0]
    canonical_ids = reference_cache["sample_ids"].astype(str)
    labels = reference_cache["labels"].astype(int)
    subjects = reference_cache["subjects"].astype(str)
    subject_folds = {
        subject: int(fold) for subject, fold in p22_audit["subject_folds"].items()
    }
    complete_groups, repeat_coverage = build_repeat_groups(
        canonical_ids,
        subjects,
        labels,
    )
    small_ids = [
        int(value)
        for value in load_json(
            PROJECT_DIR / "configs" / "p22_joint_pooled_fusion.json"
        )["evaluation"]["small_action_ids"]
    ]
    hard_ids = [
        int(value)
        for value in load_json(
            PROJECT_DIR / "configs" / "p22_joint_pooled_fusion.json"
        )["evaluation"]["hard_class_ids"]
    ]

    print("P24 loading raw Skeleton sequences", flush=True)
    skeleton_variants, skeleton_valid, skeleton_audit = load_skeleton_variants(
        args.skeleton_cache_dir,
        canonical_ids,
        int(config["skeleton"]["frames"]),
    )
    print("P24 loading aligned IMU sequences", flush=True)
    imu_variants, imu_valid, imu_audit = load_imu_variants(
        args.imu_cache_dir,
        canonical_ids,
        labels,
        subjects,
        reference_cache["presence"][:, 3],
    )
    variants = {**skeleton_variants, **imu_variants}
    valid_by_variant = {
        **{name: skeleton_valid for name in SKELETON_VARIANTS},
        **{name: imu_valid for name in IMU_VARIANTS},
    }

    results: dict[str, Any] = {}
    all_class_rows: list[dict[str, Any]] = []
    all_subject_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    all_predictions: dict[str, np.ndarray] = {}
    subject_id_results: dict[str, Any] = {}
    for index, variant in enumerate(SKELETON_VARIANTS + IMU_VARIANTS, start=1):
        print(f"[{index}/11] {variant}", flush=True)
        (
            results[variant],
            class_rows,
            subject_rows,
            confusion_rows,
            predictions,
        ) = evaluate_variant(
            variant,
            variants[variant],
            valid_by_variant[variant],
            reference_cache,
            subject_folds,
            complete_groups,
            small_ids,
            hard_ids,
            class_names,
        )
        all_class_rows.extend(class_rows)
        all_subject_rows.extend(subject_rows)
        all_confusion_rows.extend(confusion_rows)
        all_predictions.update(predictions)
        subject_id = evaluate_subject_id(
            variants[variant],
            valid_by_variant[variant],
            labels,
            subjects,
            complete_groups,
        )
        for field in ("true_subject_ids", "predicted_subject_ids", "action_ids"):
            all_predictions[f"{variant}__subject_id__{field}"] = subject_id.pop(field)
        subject_id_results[variant] = subject_id
        del variants[variant]

    p23_references = {
        modality: load_p23_reference(args.p23_dir, modality)
        for modality in ("Skeleton", "IMU")
    }
    gates: dict[str, Any] = {}
    for variant in SKELETON_VARIANTS[1:] + IMU_VARIANTS[1:]:
        gates[variant] = gate_variant(
            variant,
            results[variant],
            all_predictions,
            p23_references[VARIANT_MODALITY[variant]],
            small_ids,
            hard_ids,
            config["pass_thresholds"],
        )

    passed = [variant for variant, gate in gates.items() if gate["passed"]]
    if passed:
        decision = {
            "passed_variants": passed,
            "title": f"通过：{', '.join(passed)} 达到全部五项门槛。",
            "reason": (
                "这些单步归一化代理同时改善跨主体 overall、困难类、gap 与主体覆盖，"
                "且没有超过允许的同主体损失。"
            ),
            "next_recommendation": (
                "本次停止在审计；下一步只应把通过步骤作为唯一候选做 encoder 输入前的"
                "受控等价实验，不自动启动主体对抗或对比学习。"
            ),
        }
    else:
        decision = {
            "passed_variants": [],
            "title": "未通过：没有单一步骤同时达到全部五项门槛。",
            "reason": (
                "无训练几何/惯性归一化不足以单独消除 P23 的主体/场景迁移差距，"
                "不能据此启动更复杂训练。"
            ),
            "next_recommendation": (
                "本次停止；保留逐项结果用于判断是归一化方向无效，还是需要先拆分主体与"
                "固定场景因素，不自动启动主体对抗或对比学习。"
            ),
        }

    summary = {
        "status": "complete",
        "protocol": config["protocol"],
        "code_commit": git_commit(),
        "script_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(args.config),
        "restrictions": config["restrictions"],
        "input_audit": {
            "p22": p22_audit,
            "skeleton": skeleton_audit,
            "imu": imu_audit,
            "repeat_coverage": repeat_coverage,
            "p23_predictions_sha256": sha256(args.p23_dir / "predictions.npz"),
        },
        "protocol_detail": config,
        "class_names": {str(key): value for key, value in class_names.items()},
        "small_action_ids": small_ids,
        "hard_class_ids": hard_ids,
        "results": results,
        "gates": gates,
        "subject_id": subject_id_results,
        "per_class": all_class_rows,
        "per_subject": all_subject_rows,
        "decision": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metric_rows = []
    for variant in SKELETON_VARIANTS + IMU_VARIANTS:
        for distance in DISTANCES:
            item = results[variant][distance]
            metric_rows.append(
                {
                    "modality": VARIANT_MODALITY[variant],
                    "variant": variant,
                    "distance": distance,
                    "same_samples": item["same_subject"]["all"]["samples"],
                    "same_accuracy": item["same_subject"]["all"]["accuracy"],
                    "cross_samples": item["cross_subject_matched"]["all"]["samples"],
                    "cross_accuracy": item["cross_subject_matched"]["all"]["accuracy"],
                    "gap_pp": item["gap_pp"]["all"],
                    "same_small_accuracy": item["same_subject"]["small"]["accuracy"],
                    "cross_small_accuracy": item["cross_subject_matched"]["small"]["accuracy"],
                    "small_gap_pp": item["gap_pp"]["small"],
                    "same_hard_accuracy": item["same_subject"]["hard"]["accuracy"],
                    "cross_hard_accuracy": item["cross_subject_matched"]["hard"]["accuracy"],
                    "hard_gap_pp": item["gap_pp"]["hard"],
                }
            )
    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "per_class_recall.csv", all_class_rows)
    write_csv(args.output_dir / "per_subject_accuracy.csv", all_subject_rows)
    write_csv(args.output_dir / "top20_bidirectional_confusions.csv", all_confusion_rows)
    write_csv(
        args.output_dir / "gates.csv",
        [
            {
                "variant": variant,
                **gate["deltas"],
                **{f"check_{key}": value for key, value in gate["checks"].items()},
                "passed": gate["passed"],
            }
            for variant, gate in gates.items()
        ],
    )
    write_csv(
        args.output_dir / "subject_id.csv",
        [
            {
                "modality": VARIANT_MODALITY[variant],
                "variant": variant,
                "samples": item["samples"],
                "accuracy": item["accuracy"],
                "chance_accuracy": item["chance_accuracy"],
                "candidate_subjects_min": item["candidate_subjects"]["minimum"],
                "candidate_subjects_median": item["candidate_subjects"]["median"],
                "candidate_subjects_max": item["candidate_subjects"]["maximum"],
            }
            for variant, item in subject_id_results.items()
        ],
    )
    np.savez_compressed(args.output_dir / "predictions.npz", **all_predictions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        render_report(summary, all_class_rows, all_subject_rows),
        encoding="utf-8",
    )
    print(
        f"P24 complete: passed={passed}; output={args.output_dir}; report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
