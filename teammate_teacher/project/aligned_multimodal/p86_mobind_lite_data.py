from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from p86_cached_motion_data import (
    BOOLEAN_MOTION_FIELDS,
    MOTION_FIELDS,
    TEMPORAL_MOTION_FIELDS,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class P86MoBindMotionDataset(Dataset[dict[str, Any]]):
    """Motion-only cache reader for fast cross-modal representation learning."""

    def __init__(
        self,
        motion_cache: str | Path,
        indices: np.ndarray | list[int] | None = None,
        temporal_augment: bool = False,
        teacher_logits: str | Path | None = None,
        teacher_features: str | Path | None = None,
        imu_teacher_logits: str | Path | None = None,
        imu_event_features: str | Path | None = None,
    ) -> None:
        self.cache = Path(motion_cache).resolve()
        self.rows = read_rows(self.cache / "rows.csv")
        self.motion = {
            field: np.load(self.cache / f"{field}.npy", mmap_mode="r")
            for field in MOTION_FIELDS
        }
        completed = np.load(self.cache / "completed.npy", mmap_mode="r")
        if len(completed) != len(self.rows) or not np.asarray(completed).all():
            raise RuntimeError("P86 motion cache is incomplete")
        self.indices = (
            np.arange(len(self.rows), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.temporal_augment = bool(temporal_augment)
        self.teacher_logits_path = Path(teacher_logits).resolve() if teacher_logits else None
        self.teacher_features_path = (
            Path(teacher_features).resolve() if teacher_features else None
        )
        self.imu_teacher_logits_path = (
            Path(imu_teacher_logits).resolve() if imu_teacher_logits else None
        )
        self.imu_event_features_path = (
            Path(imu_event_features).resolve() if imu_event_features else None
        )
        self.teacher_logits = self._load_teacher_logits(self.teacher_logits_path)
        self.teacher_features = self._load_teacher_features(self.teacher_features_path)
        (
            self.imu_teacher_logits,
            self.imu_teacher_valid,
        ) = self._load_partial_imu_teacher(self.imu_teacher_logits_path)
        (
            self.imu_event_features,
            self.imu_event_valid,
        ) = self._load_partial_event_features(self.imu_event_features_path)

    def _teacher_order(self, sample_ids: np.ndarray) -> np.ndarray:
        lookup = {str(sample_id): index for index, sample_id in enumerate(sample_ids)}
        missing = [row["sample_id"] for row in self.rows if row["sample_id"] not in lookup]
        if missing:
            raise RuntimeError(f"teacher data is missing samples: {missing[:3]}")
        return np.asarray([lookup[row["sample_id"]] for row in self.rows], dtype=np.int64)

    def _load_teacher_logits(self, path: Path | None) -> np.ndarray | None:
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as data:
            order = self._teacher_order(np.asarray(data["sample_ids"]))
            return np.asarray(data["early_late_logits"], dtype=np.float32)[order]

    def _load_teacher_features(self, path: Path | None) -> np.ndarray | None:
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as data:
            order = self._teacher_order(np.asarray(data["sample_ids"]))
            features = np.asarray(data["features"], dtype=np.float32)[order]
        # View-mean preserves early/late action stages while removing camera identity.
        return features.mean(axis=2)

    def _load_partial_imu_teacher(
        self, path: Path | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if path is None:
            return None, None
        with np.load(path, allow_pickle=False) as data:
            sample_ids = np.asarray(data["sample_ids"]).astype(str)
            key = "imu_logits" if "imu_logits" in data.files else "logits"
            logits = np.asarray(data[key], dtype=np.float32)
            source_valid = (
                np.asarray(data["valid"], dtype=bool)
                if "valid" in data.files
                else np.ones(len(sample_ids), dtype=bool)
            )
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        aligned = np.zeros((len(self.rows), logits.shape[1]), dtype=np.float32)
        valid = np.zeros(len(self.rows), dtype=bool)
        for row_index, row in enumerate(self.rows):
            teacher_index = lookup.get(row["sample_id"])
            if teacher_index is not None and source_valid[teacher_index]:
                aligned[row_index] = logits[teacher_index]
                valid[row_index] = True
        return aligned, valid

    def _load_partial_event_features(
        self, path: Path | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if path is None:
            return None, None
        with np.load(path, allow_pickle=False) as data:
            sample_ids = np.asarray(data["sample_ids"]).astype(str)
            features = np.asarray(data["features"], dtype=np.float32)
            source_valid = np.asarray(data["usable"], dtype=bool)
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        aligned = np.zeros((len(self.rows), features.shape[1]), dtype=np.float32)
        valid = np.zeros(len(self.rows), dtype=bool)
        for row_index, row in enumerate(self.rows):
            feature_index = lookup.get(row["sample_id"])
            if feature_index is not None and source_valid[feature_index]:
                aligned[row_index] = features[feature_index]
                valid[row_index] = True
        return aligned, valid

    @property
    def index_lookup(self) -> dict[str, int]:
        return {row["sample_id"]: index for index, row in enumerate(self.rows)}

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(
            [int(self.rows[index]["class_id"]) for index in self.indices],
            dtype=np.int64,
        )

    @property
    def users(self) -> np.ndarray:
        return np.asarray(
            [self.rows[index]["user_id"] for index in self.indices], dtype=str
        )

    def subset(
        self, sample_ids: np.ndarray | list[str], temporal_augment: bool
    ) -> "P86MoBindMotionDataset":
        lookup = self.index_lookup
        missing = set(sample_ids) - set(lookup)
        if missing:
            raise RuntimeError(f"motion cache is missing samples: {sorted(missing)[:3]}")
        return P86MoBindMotionDataset(
            self.cache,
            indices=np.asarray([lookup[value] for value in sample_ids], dtype=np.int64),
            temporal_augment=temporal_augment,
            teacher_logits=self.teacher_logits_path,
            teacher_features=self.teacher_features_path,
            imu_teacher_logits=self.imu_teacher_logits_path,
            imu_event_features=self.imu_event_features_path,
        )

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def temporal_indices(steps: int) -> np.ndarray:
        result = np.arange(steps, dtype=np.int64)
        if np.random.random() < 0.75:
            scale = float(np.random.uniform(0.90, 1.10))
            shift = float(np.random.uniform(-0.05, 0.05) * max(steps - 1, 1))
            base = np.arange(steps, dtype=np.float32)
            positions = np.clip(
                (base - 0.5 * (steps - 1)) * scale
                + 0.5 * (steps - 1)
                + shift,
                0.0,
                float(steps - 1),
            )
            result = np.rint(positions).astype(np.int64)
        return result

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        motion = {
            field: np.asarray(self.motion[field][index]).copy()
            for field in MOTION_FIELDS
        }
        steps = motion["skeleton_features"].shape[1]
        temporal_index = (
            self.temporal_indices(steps)
            if self.temporal_augment
            else np.arange(steps, dtype=np.int64)
        )
        for field in TEMPORAL_MOTION_FIELDS:
            motion[field] = motion[field][:, temporal_index]
        result: dict[str, Any] = {
            "label": torch.tensor(int(self.rows[index]["class_id"]), dtype=torch.long),
            "sample_id": self.rows[index]["sample_id"],
            "user_id": self.rows[index]["user_id"],
            "cache_index": index,
        }
        if self.teacher_logits is not None:
            result["teacher_logits"] = torch.from_numpy(
                np.asarray(self.teacher_logits[index], dtype=np.float32).copy()
            )
        if self.teacher_features is not None:
            result["teacher_features"] = torch.from_numpy(
                np.asarray(self.teacher_features[index], dtype=np.float32).copy()
            )
        if self.imu_teacher_logits is not None:
            result["imu_teacher_logits"] = torch.from_numpy(
                np.asarray(self.imu_teacher_logits[index], dtype=np.float32).copy()
            )
            result["imu_teacher_valid"] = torch.tensor(
                bool(self.imu_teacher_valid[index]), dtype=torch.bool
            )
        if self.imu_event_features is not None:
            result["imu_event_features"] = torch.from_numpy(
                np.asarray(self.imu_event_features[index], dtype=np.float32).copy()
            )
            result["imu_event_valid"] = torch.tensor(
                bool(self.imu_event_valid[index]), dtype=torch.bool
            )
        for field, value in motion.items():
            if field in BOOLEAN_MOTION_FIELDS:
                result[field] = torch.from_numpy(value.astype(bool, copy=False))
            else:
                result[field] = torch.from_numpy(value.astype(np.float32, copy=False))
        return result


def collate_p86_mobind(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty MoBind batch")
    list_fields = {"sample_id", "user_id", "cache_index"}
    return {
        key: (
            [item[key] for item in items]
            if key in list_fields
            else torch.stack([item[key] for item in items])
        )
        for key in items[0]
    }
