from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import SequenceDatasetMixin


H36M_EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)


def frame_bone_scale(poses: np.ndarray) -> np.ndarray:
    lengths = np.stack(
        [np.linalg.norm(poses[:, child] - poses[:, parent], axis=1) for parent, child in H36M_EDGES],
        axis=1,
    )
    scale = np.median(lengths, axis=1)
    if not np.isfinite(scale).all() or (scale <= 1e-6).any():
        raise ValueError("Skeleton contains an invalid H36M bone-length scale")
    return scale


def normalize_scale(poses: np.ndarray, policy: str) -> tuple[np.ndarray, np.ndarray]:
    per_frame = frame_bone_scale(poses)
    if policy == "trial_constant":
        applied = np.full(len(poses), np.median(per_frame), dtype=np.float64)
    elif policy == "per_frame":
        applied = per_frame
    else:
        raise ValueError(f"Unsupported scale policy: {policy}")
    return poses / applied[:, None, None], applied


def segment_local_velocity(values: np.ndarray, segments: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(values)
    same_segment = segments[1:] == segments[:-1]
    velocity[1:][same_segment] = values[1:][same_segment] - values[:-1][same_segment]
    return velocity


def gap_aware_resample(
    frame_ids: np.ndarray, segments: np.ndarray, values: np.ndarray, target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frame_ids) == 0 or values.ndim != 2 or len(values) != len(frame_ids):
        raise ValueError("Expected a non-empty aligned frame/value sequence")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    order = np.argsort(frame_ids)
    frame_ids = frame_ids[order].astype(np.float64)
    segments = segments[order]
    values = values[order]
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be unique")
    target = np.linspace(frame_ids[0], frame_ids[-1], target_length, dtype=np.float64)
    output = np.zeros((target_length, values.shape[1]), dtype=np.float32)
    mask = np.zeros(target_length, dtype=bool)
    distance = np.full(target_length, np.inf, dtype=np.float64)
    for segment in pd.unique(segments):
        selected = segments == segment
        positions = frame_ids[selected]
        segment_values = values[selected]
        inside = (target >= positions[0]) & (target <= positions[-1])
        target_indices = np.flatnonzero(inside)
        if not len(target_indices):
            target_indices = np.asarray([int(np.argmin(np.abs(target - np.median(positions))))])
        for target_index in target_indices:
            current_distance = float(np.min(np.abs(positions - target[target_index])))
            if mask[target_index] and current_distance >= distance[target_index]:
                continue
            for channel in range(values.shape[1]):
                output[target_index, channel] = np.interp(
                    target[target_index], positions, segment_values[:, channel]
                )
            mask[target_index] = True
            distance[target_index] = current_distance
    return output, mask


class CleanSkeletonDataset(SequenceDatasetMixin, Dataset[dict[str, object]]):
    def __init__(
        self, clean_view: pd.DataFrame | Path, data_root: Path, users: set[str],
        scale_policy: str, sequence_length: int = 64,
    ) -> None:
        frame = (
            pd.read_csv(clean_view, encoding="utf-8-sig", dtype={"user_id": str})
            if isinstance(clean_view, Path) else clean_view.copy()
        )
        frame = frame[frame["user_id"].isin(users) & frame["use_for_frame_training"].astype(bool)].copy()
        frame = frame.sort_values(["sample_id", "frame_id"])
        self.data_root = data_root
        self.scale_policy = scale_policy
        self.sequence_length = sequence_length
        self.mean = None
        self.std = None
        self._tensor_cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        self.groups = [group.reset_index(drop=True) for _, group in frame.groupby("sample_id", sort=False)]
        if not self.groups:
            raise ValueError("No retained Skeleton trials for requested users")

    def __len__(self) -> int:
        return len(self.groups)

    def _read_candidate(self, path: Path, candidate_index: int) -> np.ndarray:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not 0 <= candidate_index < len(payload):
            raise ValueError(f"Invalid candidate {candidate_index} in {path}")
        pose = np.asarray(payload[candidate_index]["keypoints"], dtype=np.float64)
        if pose.shape != (17, 3) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid 17x3 pose in {path}")
        return pose

    def load_tensor(self, index: int, apply_normalization: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
        cached = self._tensor_cache.get(index)
        if cached is not None:
            cached_values, cached_mask, original_length = cached
            values = self.normalize(cached_values.copy(), cached_mask, apply_normalization)
            return torch.from_numpy(values), torch.from_numpy(cached_mask.copy()), original_length
        rows = self.groups[index]
        poses = np.stack([
            self._read_candidate(self.data_root / str(row.skeleton_json_path), int(row.candidate_index))
            for row in rows.itertuples()
        ])
        normalized, _ = normalize_scale(poses, self.scale_policy)
        segments = rows["retained_segment_index"].to_numpy(dtype=np.int64)
        velocity = segment_local_velocity(normalized, segments)
        features = np.concatenate([normalized, velocity], axis=2).reshape(len(rows), 102)
        values, mask = gap_aware_resample(
            rows["frame_id"].to_numpy(dtype=np.int64), segments, features, self.sequence_length
        )
        self._tensor_cache[index] = (values.copy(), mask.copy(), len(rows))
        values = self.normalize(values, mask, apply_normalization)
        return torch.from_numpy(values), torch.from_numpy(mask), len(rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        rows = self.groups[index]
        tensor, mask, original_length = self.load_tensor(index)
        return {
            "input": tensor, "temporal_mask": mask, "label": int(rows.iloc[0]["class_id"]),
            "sample_id": str(rows.iloc[0]["sample_id"]), "user_id": str(rows.iloc[0]["user_id"]),
            "length": original_length,
        }
