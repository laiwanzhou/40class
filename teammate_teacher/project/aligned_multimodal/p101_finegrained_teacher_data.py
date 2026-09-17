"""Leakage-safe data contract for the P101 fine-grained A Teacher.

P101 joins the unpooled VideoMAEv2/InternVideo2 temporal caches to the exact
P86 part/time Skeleton and device/time IMU cache.  Only the frozen 12-subject
development allow-list is exposed.  Labels are parsed from sample ids after
the allow-list is applied; no historical 40-class prediction is loaded.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from p86_cached_motion_data import BOOLEAN_MOTION_FIELDS, MOTION_FIELDS
from p100a_global_teacher_data import (
    DEV_USER_SET,
    FOLD_ROW_COUNTS,
    FOLD_USERS,
    H3_USERS,
)


HERE = Path(__file__).resolve().parent
VMAE_CACHE = HERE / "runs/p101_large_temporal_cache_v1/videomaev2"
IV2_CACHE = HERE / "runs/p101_large_temporal_cache_v1/internvideo2"
MOTION_CACHE = HERE / "runs/p86_motion_window_cache_t16_v1"

CANONICAL_VARIANTS = {
    "V": ("visual",),
    "VS": ("visual", "skeleton"),
    "VI": ("visual", "imu"),
    "VSI": ("visual", "skeleton", "imu"),
}
SAMPLE_PATTERN = re.compile(r"^train__c(?P<class_id>\d{2})__(?P<user>user\d+)__")
TEMPORAL_MOTION_FIELDS = frozenset(MOTION_FIELDS[:10])
SKELETON_FIELDS = frozenset(MOTION_FIELDS[:6])
IMU_FIELDS = frozenset(MOTION_FIELDS[6:])


def parse_sample_id(sample_id: str) -> tuple[int, str]:
    match = SAMPLE_PATTERN.match(str(sample_id))
    if match is None:
        raise ValueError(f"unexpected sample id: {sample_id}")
    return int(match.group("class_id")), match.group("user")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_time(windows: int, steps: int) -> np.ndarray:
    """Return the exact early/late normalized coordinates used by P90/P86."""
    if windows != 2:
        raise ValueError("P101 expects two overlapping action windows")
    bounds = ((0.0, 0.70), (0.30, 1.0))
    return np.stack(
        [np.linspace(low, high, steps, dtype=np.float32) for low, high in bounds]
    )


@dataclass(frozen=True)
class P101Data:
    sample_ids: np.ndarray
    users: np.ndarray
    labels: np.ndarray
    fold_ids: np.ndarray
    visual_vmae_temporal: np.ndarray
    visual_iv2_temporal: np.ndarray
    visual_vmae_pooled: np.ndarray
    visual_iv2_pooled: np.ndarray
    visual_vmae_action: np.ndarray
    visual_iv2_action: np.ndarray
    motion: dict[str, np.ndarray]
    motion_rows: np.ndarray

    def validate(self) -> None:
        rows = len(self.sample_ids)
        if rows != 1941 or len(np.unique(self.sample_ids)) != rows:
            raise ValueError(f"P101 development rows changed: {rows}")
        if set(self.users.tolist()) != DEV_USER_SET:
            raise ValueError("P101 development subject allow-list changed")
        if set(self.users.tolist()) & H3_USERS:
            raise ValueError("H3 subject reached P101")
        counts = tuple(int((self.fold_ids == fold).sum()) for fold in range(4))
        if counts != FOLD_ROW_COUNTS:
            raise ValueError(f"P101 outer fold counts changed: {counts}")
        for fold, held_users in enumerate(FOLD_USERS):
            if set(self.users[self.fold_ids == fold].tolist()) != set(held_users):
                raise ValueError(f"P101 fold {fold} subject membership changed")
        expected = {
            "visual_vmae_temporal": (rows, 2, 3, 8, 768),
            "visual_iv2_temporal": (rows, 2, 3, 8, 768),
            "visual_vmae_pooled": (rows, 2, 3, 768),
            "visual_iv2_pooled": (rows, 2, 3, 768),
            "visual_vmae_action": (rows, 2, 3, 710),
            "visual_iv2_action": (rows, 2, 3, 400),
        }
        for name, shape in expected.items():
            values = getattr(self, name)
            if values.shape != shape:
                raise ValueError(f"{name} has {values.shape}, expected {shape}")
        parsed = [parse_sample_id(sample_id) for sample_id in self.sample_ids]
        if not np.array_equal(
            np.asarray([value[0] for value in parsed], dtype=np.int64), self.labels
        ):
            raise ValueError("P101 labels were not parsed from sample ids")
        if not np.array_equal(
            np.asarray([value[1] for value in parsed], dtype=str), self.users
        ):
            raise ValueError("P101 users were not parsed from sample ids")
        if set(self.motion) != set(MOTION_FIELDS):
            raise ValueError("P101 motion field contract changed")
        if self.motion_rows.shape != (rows,):
            raise ValueError("P101 motion row alignment changed")

    @property
    def skeleton_available(self) -> np.ndarray:
        mask = self.motion["skeleton_joint_mask"][self.motion_rows]
        return np.asarray(mask).reshape(len(self.sample_ids), -1).any(axis=1)

    @property
    def imu_available(self) -> np.ndarray:
        mask = self.motion["imu_bin_mask"][self.motion_rows]
        return np.asarray(mask).reshape(len(self.sample_ids), -1).any(axis=1)

    def indices_for_fold(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        if fold not in range(4):
            raise ValueError(f"invalid outer fold: {fold}")
        train = np.flatnonzero(self.fold_ids != fold).astype(np.int64)
        held = np.flatnonzero(self.fold_ids == fold).astype(np.int64)
        if set(self.users[train]) & set(self.users[held]):
            raise RuntimeError("P101 outer subject leakage")
        return train, held

    def summary(self) -> dict[str, object]:
        return {
            "stage": "P101_FINEGRAINED_A_TEACHER",
            "rows": len(self.sample_ids),
            "subjects": sorted(set(self.users.tolist())),
            "folds": [
                {
                    "fold": fold,
                    "held_users": list(FOLD_USERS[fold]),
                    "rows": int((self.fold_ids == fold).sum()),
                }
                for fold in range(4)
            ],
            "h3_rows": 0,
            "historical_40class_predictions_loaded": False,
            "skeleton_available": float(self.skeleton_available.mean()),
            "imu_available": float(self.imu_available.mean()),
        }


def _load_complete_array(cache: Path, name: str) -> np.ndarray:
    completed = np.load(cache / "completed.npy", mmap_mode="r")
    if not np.asarray(completed).all():
        raise RuntimeError(f"incomplete P101 cache: {cache}")
    return np.load(cache / f"{name}.npy", mmap_mode="r")


@lru_cache(maxsize=1)
def load_p101_data() -> P101Data:
    vmae_rows = read_rows(VMAE_CACHE / "rows.csv")
    iv2_rows = read_rows(IV2_CACHE / "rows.csv")
    sample_ids = np.asarray([row["sample_id"] for row in vmae_rows], dtype=str)
    if [row["sample_id"] for row in iv2_rows] != sample_ids.tolist():
        raise RuntimeError("P101 visual cache row orders differ")
    parsed = [parse_sample_id(sample_id) for sample_id in sample_ids]
    users = np.asarray([value[1] for value in parsed], dtype=str)
    labels = np.asarray([value[0] for value in parsed], dtype=np.int64)
    fold_lookup = {
        user: fold for fold, held_users in enumerate(FOLD_USERS) for user in held_users
    }
    fold_ids = np.asarray([fold_lookup[user] for user in users], dtype=np.int64)

    motion_rows_csv = read_rows(MOTION_CACHE / "rows.csv")
    motion_ids = [row["sample_id"] for row in motion_rows_csv]
    motion_lookup = {sample_id: index for index, sample_id in enumerate(motion_ids)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in motion_lookup]
    if missing:
        raise KeyError(f"P86 motion cache misses P101 rows: {missing[:3]}")
    motion_rows = np.asarray(
        [motion_lookup[sample_id] for sample_id in sample_ids], dtype=np.int64
    )
    motion = {
        field: np.load(MOTION_CACHE / f"{field}.npy", mmap_mode="r")
        for field in MOTION_FIELDS
    }
    motion_completed = np.load(MOTION_CACHE / "completed.npy", mmap_mode="r")
    if not np.asarray(motion_completed[motion_rows]).all():
        raise RuntimeError("P86 motion rows needed by P101 are incomplete")

    data = P101Data(
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=fold_ids,
        visual_vmae_temporal=_load_complete_array(VMAE_CACHE, "temporal_tokens"),
        visual_iv2_temporal=_load_complete_array(IV2_CACHE, "temporal_tokens"),
        visual_vmae_pooled=_load_complete_array(VMAE_CACHE, "pooled_features"),
        visual_iv2_pooled=_load_complete_array(IV2_CACHE, "pooled_features"),
        visual_vmae_action=_load_complete_array(VMAE_CACHE, "action_logits"),
        visual_iv2_action=_load_complete_array(IV2_CACHE, "action_logits"),
        motion=motion,
        motion_rows=motion_rows,
    )
    data.validate()
    return data


def _row_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=-1, keepdims=True)
    std = np.maximum(values.std(axis=-1, keepdims=True), 1e-4)
    return (values - mean) / std


class P101Dataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data: P101Data,
        indices: np.ndarray,
        modalities: tuple[str, ...],
        skeleton_source: np.ndarray | None = None,
        imu_source: np.ndarray | None = None,
        zero_modalities: Iterable[str] = (),
        reverse_modalities: Iterable[str] = (),
        sample_weights: np.ndarray | None = None,
        nested_vs_error: np.ndarray | None = None,
        negative_imu_source: np.ndarray | None = None,
    ) -> None:
        if "visual" not in modalities:
            raise ValueError("P101 A Teacher always contains Visual")
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.modalities = tuple(modalities)
        identity = np.arange(len(data.sample_ids), dtype=np.int64)
        self.skeleton_source = identity if skeleton_source is None else np.asarray(skeleton_source, dtype=np.int64)
        self.imu_source = identity if imu_source is None else np.asarray(imu_source, dtype=np.int64)
        self.zero_modalities = frozenset(zero_modalities)
        self.reverse_modalities = frozenset(reverse_modalities)
        self.sample_weights = sample_weights
        self.nested_vs_error = nested_vs_error
        self.negative_imu_source = negative_imu_source

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _tensor(values: np.ndarray, boolean: bool = False) -> torch.Tensor:
        copied = np.asarray(values).copy()
        if boolean:
            return torch.from_numpy(copied.astype(bool, copy=False))
        return torch.from_numpy(copied.astype(np.float32, copy=False))

    def _motion(
        self,
        item: dict[str, torch.Tensor],
        dev_row: int,
        modality: str,
        prefix: str = "",
        source_override: np.ndarray | None = None,
    ) -> None:
        if modality == "skeleton":
            fields = SKELETON_FIELDS
            source_map = self.skeleton_source
        elif modality == "imu":
            fields = IMU_FIELDS
            source_map = self.imu_source
        else:
            raise ValueError(modality)
        if source_override is not None:
            source_map = source_override
        source = int(source_map[dev_row])
        cache_row = int(self.data.motion_rows[source])
        reverse = modality in self.reverse_modalities
        zero = modality in self.zero_modalities
        for field in fields:
            values = self.data.motion[field][cache_row]
            if reverse and field in TEMPORAL_MOTION_FIELDS:
                values = np.asarray(values)[:, ::-1]
            tensor = self._tensor(values, field in BOOLEAN_MOTION_FIELDS)
            if zero and (field in BOOLEAN_MOTION_FIELDS or field.endswith("quality")):
                tensor = torch.zeros_like(tensor)
            item[prefix + field] = tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        item: dict[str, torch.Tensor] = {
            "row": torch.tensor(row, dtype=torch.long),
            "label": torch.tensor(int(self.data.labels[row]), dtype=torch.long),
            "weight": torch.tensor(
                1.0 if self.sample_weights is None else float(self.sample_weights[row]),
                dtype=torch.float32,
            ),
            "nested_vs_error": torch.tensor(
                0.0 if self.nested_vs_error is None else float(self.nested_vs_error[row]),
                dtype=torch.float32,
            ),
            "visual_vmae_temporal": self._tensor(self.data.visual_vmae_temporal[row]),
            "visual_iv2_temporal": self._tensor(self.data.visual_iv2_temporal[row]),
            "visual_vmae_pooled": self._tensor(self.data.visual_vmae_pooled[row]),
            "visual_iv2_pooled": self._tensor(self.data.visual_iv2_pooled[row]),
            "visual_vmae_action": self._tensor(_row_standardize(self.data.visual_vmae_action[row])),
            "visual_iv2_action": self._tensor(_row_standardize(self.data.visual_iv2_action[row])),
            "visual_time": self._tensor(canonical_time(2, 8)),
            "motion_time": self._tensor(canonical_time(2, 16)),
        }
        if "skeleton" in self.modalities:
            self._motion(item, row, "skeleton")
        if "imu" in self.modalities:
            self._motion(item, row, "imu")
            if self.negative_imu_source is not None:
                source = int(self.negative_imu_source[row])
                self._motion(
                    item,
                    row,
                    "imu",
                    prefix="negative_",
                    source_override=self.negative_imu_source,
                )
                item["negative_imu_row"] = torch.tensor(source, dtype=torch.long)
        return item


def class_user_sample_weights(data: P101Data, indices: np.ndarray) -> np.ndarray:
    labels = data.labels[indices]
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    class_weight = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    users, user_counts = np.unique(data.users[indices], return_counts=True)
    user_weight = {
        user: 1.0 / np.sqrt(float(count))
        for user, count in zip(users, user_counts, strict=True)
    }
    result = np.ones(len(data.sample_ids), dtype=np.float32)
    result[indices] = np.asarray(
        [class_weight[data.labels[row]] * user_weight[data.users[row]] for row in indices],
        dtype=np.float32,
    )
    result[indices] /= result[indices].mean()
    return result


def within_subject_permutation(
    data: P101Data, indices: np.ndarray, modality: str, seed: int
) -> np.ndarray:
    if modality not in {"skeleton", "imu"}:
        raise ValueError(modality)
    output = np.arange(len(data.sample_ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    available = data.skeleton_available if modality == "skeleton" else data.imu_available
    for user in np.unique(data.users[indices]):
        user_rows = indices[data.users[indices] == user]
        for present in (False, True):
            group = user_rows[available[user_rows] == present]
            if len(group) > 1:
                permuted = rng.permutation(group)
                if np.array_equal(permuted, group):
                    permuted = np.roll(permuted, 1)
                output[group] = permuted
    return output


def within_subject_wrong_label_source(
    data: P101Data, indices: np.ndarray, seed: int
) -> np.ndarray:
    """Choose a same-subject, different-label IMU negative for every train row."""
    output = np.arange(len(data.sample_ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    available = data.imu_available
    for row in indices:
        candidates = indices[
            (data.users[indices] == data.users[row])
            & (data.labels[indices] != data.labels[row])
            & (available[indices] == available[row])
        ]
        if len(candidates):
            output[row] = int(rng.choice(candidates))
    return output
