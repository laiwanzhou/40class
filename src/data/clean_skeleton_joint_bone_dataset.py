from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .clean_skeleton_dataset import (
    H36M_EDGES,
    CleanSkeletonDataset,
    gap_aware_resample,
    normalize_scale,
    segment_local_velocity,
)


def joint_bone_features(poses: np.ndarray, segments: np.ndarray) -> np.ndarray:
    if poses.ndim != 3 or poses.shape[1:] != (17, 3) or len(segments) != len(poses):
        raise ValueError("Expected poses [T,17,3] and aligned segment IDs")
    joint_velocity = segment_local_velocity(poses, segments)
    joint = np.concatenate([poses, joint_velocity], axis=2).reshape(len(poses), 102)
    parents = np.asarray([parent for parent, _ in H36M_EDGES], dtype=np.int64)
    children = np.asarray([child for _, child in H36M_EDGES], dtype=np.int64)
    bone_position = poses[:, children] - poses[:, parents]
    bone_velocity = joint_velocity[:, children] - joint_velocity[:, parents]
    bone = np.concatenate([bone_position, bone_velocity], axis=2).reshape(len(poses), 96)
    return np.concatenate([joint, bone], axis=1)


class CleanSkeletonJointBoneDataset(CleanSkeletonDataset):
    def __init__(
        self, clean_view: pd.DataFrame | Path, data_root: Path, users: set[str], sequence_length: int = 64,
    ) -> None:
        super().__init__(clean_view, data_root, users, "per_frame", sequence_length)
        self._joint_bone_cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    def load_tensor(self, index: int, apply_normalization: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
        cached = self._joint_bone_cache.get(index)
        if cached is None:
            rows = self.groups[index]
            poses = np.stack([
                self._read_candidate(self.data_root / str(row.skeleton_json_path), int(row.candidate_index))
                for row in rows.itertuples()
            ])
            normalized, _ = normalize_scale(poses, "per_frame")
            segments = rows["retained_segment_index"].to_numpy(dtype=np.int64)
            features = joint_bone_features(normalized, segments)
            values, mask = gap_aware_resample(
                rows["frame_id"].to_numpy(dtype=np.int64), segments, features, self.sequence_length
            )
            cached = (values.copy(), mask.copy(), len(rows))
            self._joint_bone_cache[index] = cached
        values, mask, original_length = cached
        normalized_values = self.normalize(values.copy(), mask, apply_normalization)
        return torch.from_numpy(normalized_values), torch.from_numpy(mask.copy()), original_length
