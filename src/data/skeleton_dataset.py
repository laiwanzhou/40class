from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import SequenceDatasetMixin, resample_sequence, sorted_files


class SkeletonDataset(SequenceDatasetMixin, Dataset[dict[str, object]]):
    def __init__(self, frame: pd.DataFrame, sequence_length: int = 64, joints: int = 17) -> None:
        self.frame = frame.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.joints = joints
        self.mean = None
        self.std = None

    def __len__(self) -> int:
        return len(self.frame)

    def _read_pose(self, json_path: Path, sample_id: str) -> np.ndarray:
        try:
            people = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Skeleton JSON read failed for sample_id={sample_id}: {json_path}") from exc
        if not isinstance(people, list) or not people:
            return np.zeros((self.joints, 3), dtype=np.float32)
        candidates = [person for person in people if isinstance(person, dict) and "keypoints" in person]
        if not candidates:
            return np.zeros((self.joints, 3), dtype=np.float32)
        person = max(
            candidates,
            key=lambda item: float(np.mean(item.get("keypoint_scores", [0.0]))),
        )
        pose = np.asarray(person["keypoints"], dtype=np.float32)
        if pose.shape != (self.joints, 3):
            raise ValueError(
                f"Unexpected skeleton shape {pose.shape} for sample_id={sample_id}: {json_path}"
            )
        return np.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)

    def load_tensor(self, index: int, apply_normalization: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.frame.iloc[index]
        sample_id = str(row["sample_id"])
        trial_path = Path(row["trial_path"])
        json_files = sorted_files(trial_path, {".json"}, recursive=True)
        if not json_files:
            raise FileNotFoundError(f"No skeleton JSON for sample_id={sample_id}: {trial_path}")
        poses = np.stack([self._read_pose(path, sample_id) for path in json_files])
        root = (poses[:, 11:12, :] + poses[:, 12:13, :]) * 0.5
        centered = poses - root
        scale = np.sqrt(np.mean(np.square(centered), axis=(1, 2), keepdims=True))
        centered = centered / np.maximum(scale, 1e-6)
        velocity = np.diff(centered, axis=0, prepend=centered[:1])
        features = np.concatenate([centered, velocity], axis=2).reshape(len(poses), -1)
        values = resample_sequence(features, self.sequence_length)
        mask = np.ones(self.sequence_length, dtype=bool)
        values = self.normalize(values, mask, apply_normalization)
        return torch.from_numpy(values), torch.from_numpy(mask), len(json_files)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        tensor, mask, original_length = self.load_tensor(index)
        return {
            "input": tensor,
            "temporal_mask": mask,
            "label": int(row["class_id"]),
            "sample_id": str(row["sample_id"]),
            "length": original_length,
        }
