from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


H36M_EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)


@dataclass(frozen=True)
class SkeletonSegments:
    features: torch.Tensor
    mask: torch.Tensor
    quality: torch.Tensor


def fit_skeleton_normalization(
    samples: Iterable[SkeletonSegments],
) -> tuple[np.ndarray, np.ndarray]:
    parts = [
        sample.features[sample.mask].numpy().astype(np.float64)
        for sample in samples
        if bool(sample.mask.any())
    ]
    if not parts:
        raise ValueError("no valid Skeleton segments for normalization")
    values = np.concatenate(parts, axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


def apply_skeleton_normalization(
    sample: SkeletonSegments,
    mean: np.ndarray,
    std: np.ndarray,
) -> SkeletonSegments:
    if mean.shape != (17, 6) or std.shape != (17, 6):
        raise ValueError("Skeleton normalization shape changed")
    values = sample.features.numpy().copy()
    values = (values - mean[None]) / np.maximum(std[None], 1e-6)
    values[~sample.mask.numpy()] = 0.0
    if not np.isfinite(values).all():
        raise ValueError("non-finite normalized Skeleton segments")
    return SkeletonSegments(
        features=torch.from_numpy(values.astype(np.float32)),
        mask=sample.mask.clone(),
        quality=sample.quality.clone(),
    )


def frame_bone_scale(poses: np.ndarray) -> np.ndarray:
    lengths = np.stack(
        [
            np.linalg.norm(poses[:, child] - poses[:, parent], axis=1)
            for parent, child in H36M_EDGES
        ],
        axis=1,
    )
    scale = np.median(lengths, axis=1)
    if not np.isfinite(scale).all() or bool((scale <= 1e-6).any()):
        raise ValueError("Skeleton contains an invalid H36M bone-length scale")
    return scale


def segment_local_velocity(values: np.ndarray, segments: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(values)
    same_segment = segments[1:] == segments[:-1]
    velocity[1:][same_segment] = values[1:][same_segment] - values[:-1][same_segment]
    return velocity


def resample_skeleton_segments(
    frame_ids: np.ndarray,
    retained_segments: np.ndarray,
    poses: np.ndarray,
    *,
    segment_count: int = 8,
) -> SkeletonSegments:
    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
        raise ValueError("Skeleton poses must have shape [T,17,3]")
    if len(frame_ids) != len(poses) or len(retained_segments) != len(poses):
        raise ValueError("Skeleton frame, segment, and pose lengths differ")
    if len(poses) == 0 or segment_count < 1:
        raise ValueError("Skeleton sequence and segment count must be positive")
    order = np.argsort(frame_ids)
    frames = np.asarray(frame_ids, dtype=np.float64)[order]
    segments = np.asarray(retained_segments, dtype=np.int64)[order]
    poses = np.asarray(poses, dtype=np.float64)[order]
    if len(np.unique(frames)) != len(frames):
        raise ValueError("Skeleton frame IDs must be unique")
    scales = frame_bone_scale(poses)
    normalized = poses / scales[:, None, None]
    velocity = segment_local_velocity(normalized, segments)
    source = np.concatenate((normalized, velocity), axis=2).reshape(len(poses), 102)
    target = np.linspace(frames[0], frames[-1], segment_count, dtype=np.float64)
    output = np.zeros((segment_count, 102), dtype=np.float32)
    mask = np.zeros(segment_count, dtype=bool)
    output_segment = np.full(segment_count, -1, dtype=np.int64)
    distance = np.full(segment_count, np.inf, dtype=np.float64)
    quality = np.zeros((segment_count, 4), dtype=np.float32)

    for retained_segment in pd.unique(segments):
        selected = segments == retained_segment
        positions = frames[selected]
        values = source[selected]
        segment_scales = scales[selected]
        target_indices = np.flatnonzero(
            (target >= positions[0]) & (target <= positions[-1])
        )
        if not len(target_indices):
            target_indices = np.asarray(
                [int(np.argmin(np.abs(target - np.median(positions))))]
            )
        for target_index in target_indices:
            current_distance = float(np.min(np.abs(positions - target[target_index])))
            if mask[target_index] and current_distance >= distance[target_index]:
                continue
            for channel in range(values.shape[1]):
                output[target_index, channel] = np.interp(
                    target[target_index], positions, values[:, channel]
                )
            mask[target_index] = True
            output_segment[target_index] = int(retained_segment)
            distance[target_index] = current_distance
            scale_cv = float(np.std(segment_scales) / max(np.mean(segment_scales), 1e-8))
            quality[target_index] = (
                1.0,
                float(len(positions)),
                float(retained_segment > segments.min()),
                float(1.0 / (1.0 + scale_cv)),
            )
    features = output.reshape(segment_count, 17, 6)
    features[~mask] = 0
    quality[~mask] = 0
    return SkeletonSegments(
        features=torch.from_numpy(features),
        mask=torch.from_numpy(mask),
        quality=torch.from_numpy(quality),
    )

def _read_candidate(path: Path, candidate_index: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not 0 <= candidate_index < len(payload):
        raise ValueError(f"invalid Skeleton candidate {candidate_index}: {path}")
    pose = np.asarray(payload[candidate_index].get("keypoints"), dtype=np.float64)
    if pose.shape != (17, 3) or not np.isfinite(pose).all():
        raise ValueError(f"invalid Skeleton pose: {path}")
    return pose


def load_skeleton_segments(
    clean_rows: pd.DataFrame,
    *,
    data_root: Path,
    segment_count: int = 8,
) -> SkeletonSegments:
    required = {
        "sample_id",
        "frame_id",
        "retained_segment_index",
        "skeleton_json_path",
        "candidate_index",
        "use_for_frame_training",
    }
    if not required.issubset(clean_rows.columns):
        raise ValueError(f"Skeleton clean view misses {sorted(required - set(clean_rows.columns))}")
    rows = clean_rows[clean_rows["use_for_frame_training"].astype(bool)].copy()
    rows = rows.dropna(subset=["candidate_index", "retained_segment_index"])
    rows = rows.sort_values("frame_id")
    if rows.empty or rows["sample_id"].astype(str).nunique() != 1:
        raise ValueError("Skeleton loader expects one retained trial")
    poses = np.stack(
        [
            _read_candidate(
                data_root / str(row.skeleton_json_path), int(row.candidate_index)
            )
            for row in rows.itertuples()
        ]
    )
    return resample_skeleton_segments(
        rows["frame_id"].to_numpy(dtype=np.int64),
        rows["retained_segment_index"].to_numpy(dtype=np.int64),
        poses,
        segment_count=segment_count,
    )
