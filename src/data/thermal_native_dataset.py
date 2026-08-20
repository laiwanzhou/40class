from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


SEALED_USER_IDS = frozenset({"user4", "user17", "user23", "user24"})
QUALITY_NAMES = (
    "directory_present",
    "decodable_frame_fraction",
    "distinct_frame_ratio",
    "unique_sampled_source_ratio",
    "duration_support",
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def normalized_frame_indices(frame_count: int, *, num_segments: int = 16) -> tuple[int, ...]:
    """Map uniform normalized-time targets to nearest Thermal-native indices."""
    if frame_count < 1 or num_segments < 1:
        raise ValueError("frame_count and num_segments must be positive")
    if num_segments == 1:
        return (0,)
    if frame_count == 1:
        return (0,) * num_segments
    scale = (frame_count - 1) / (num_segments - 1)
    return tuple(min(frame_count - 1, math.floor(index * scale + 0.5)) for index in range(num_segments))


def source_uniqueness_mask(indices: Sequence[int]) -> torch.Tensor:
    seen: set[int] = set()
    mask: list[bool] = []
    for index in indices:
        mask.append(index not in seen)
        seen.add(index)
    return torch.tensor(mask, dtype=torch.bool)


@dataclass(frozen=True)
class ThermalTrialRecord:
    sample_id: str
    class_id: int
    action_name: str
    user_id: str
    trial_id: str
    thermal_dir: Path | None
    directory_present: bool
    usable: bool
    file_count: int
    decodable_frame_count: int
    distinct_frame_ratio: float

    @classmethod
    def from_audit(cls, row: Mapping[str, Any], data_root: Path) -> "ThermalTrialRecord":
        forbidden = {
            "ir_frame_indices",
            "ir_bbox",
            "motion_peak_indices",
            "paired_frame_position",
        }
        overlap = forbidden.intersection(row)
        if overlap:
            raise ValueError(f"Cross-modal or motion sampling fields are forbidden: {sorted(overlap)}")
        user_id = str(row["user_id"])
        if user_id in SEALED_USER_IDS:
            raise ValueError(f"Refusing sealed user in Thermal development dataset: {user_id}")
        class_dir = f"{int(row['class_id'])}_{row['action_name']}"
        path = data_root / "Thermal" / class_dir / user_id / str(row["trial_id"])
        return cls(
            sample_id=str(row["sample_id"]),
            class_id=int(row["class_id"]),
            action_name=str(row["action_name"]),
            user_id=user_id,
            trial_id=str(row["trial_id"]),
            thermal_dir=path if bool(row["directory_present"]) else None,
            directory_present=bool(row["directory_present"]),
            usable=bool(row["usable"]),
            file_count=int(row["file_count"]),
            decodable_frame_count=int(row["decodable_frame_count"]),
            distinct_frame_ratio=float(row["distinct_frame_ratio"] or 0.0),
        )


class ThermalNativeDataset(Dataset[dict[str, Any]]):
    """Canonical trial dataset using only Thermal-native time and full frames."""

    def __init__(
        self,
        records: Sequence[ThermalTrialRecord],
        *,
        training: bool,
        num_segments: int = 16,
        resize_short_side: int = 256,
        crop_size: int = 224,
        seed: int = 20260715,
        route: str = "full_frame",
    ) -> None:
        if route != "full_frame":
            raise ValueError("The first T1-B primary/control dataset route must be full_frame")
        if num_segments != 16 or resize_short_side != 256 or crop_size != 224:
            raise ValueError("Frozen T1-B input contract is 16 segments, resize 256, crop 224")
        if any(record.user_id in SEALED_USER_IDS for record in records):
            raise ValueError("Thermal development dataset may not contain sealed users")
        sample_ids = [record.sample_id for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Duplicate canonical sample_id in Thermal dataset")
        self.records = tuple(records)
        self.training = training
        self.num_segments = num_segments
        self.resize_short_side = resize_short_side
        self.crop_size = crop_size
        self.seed = seed
        self.route = route
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _frame_paths(self, record: ThermalTrialRecord) -> tuple[Path, ...]:
        if record.thermal_dir is None or not record.thermal_dir.is_dir():
            return ()
        return tuple(
            sorted(
                (
                    path
                    for path in record.thermal_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=_natural_key,
            )
        )

    def _spatial_parameters(self, sample_id: str, resized_width: int, resized_height: int) -> tuple[int, int, bool]:
        if resized_width < self.crop_size or resized_height < self.crop_size:
            raise ValueError("Resized Thermal frame is smaller than crop contract")
        if not self.training:
            return (
                (resized_height - self.crop_size) // 2,
                (resized_width - self.crop_size) // 2,
                False,
            )
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{sample_id}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        top = rng.randint(0, resized_height - self.crop_size)
        left = rng.randint(0, resized_width - self.crop_size)
        return top, left, rng.random() < 0.5

    def _load_clip(self, record: ThermalTrialRecord, paths: Sequence[Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = normalized_frame_indices(len(paths), num_segments=self.num_segments)
        images: list[Image.Image] = []
        for index in indices:
            with Image.open(paths[index]) as source:
                images.append(source.convert("RGB"))
        resized = [TF.resize(image, self.resize_short_side, antialias=True) for image in images]
        width, height = resized[0].size
        if any(image.size != (width, height) for image in resized):
            raise ValueError(f"Thermal trial changes resolution: {record.sample_id}")
        top, left, flip = self._spatial_parameters(record.sample_id, width, height)
        frames: list[torch.Tensor] = []
        for image in resized:
            cropped = TF.crop(image, top, left, self.crop_size, self.crop_size)
            if flip:
                cropped = TF.hflip(cropped)
            tensor = TF.normalize(TF.pil_to_tensor(cropped).float().div_(255.0), IMAGENET_MEAN, IMAGENET_STD)
            frames.append(tensor)
        return torch.stack(frames), torch.tensor(indices, dtype=torch.long), source_uniqueness_mask(indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        started = time.perf_counter()
        record = self.records[index]
        paths = self._frame_paths(record)
        available = bool(record.usable and paths)
        if available:
            clips, source_indices, unique_mask = self._load_clip(record, paths)
        else:
            clips = torch.zeros(self.num_segments, 3, self.crop_size, self.crop_size)
            source_indices = torch.full((self.num_segments,), -1, dtype=torch.long)
            unique_mask = torch.zeros(self.num_segments, dtype=torch.bool)
        decoded_fraction = (
            min(1.0, record.decodable_frame_count / record.file_count)
            if record.file_count
            else 0.0
        )
        quality = torch.tensor(
            [
                float(record.directory_present),
                decoded_fraction,
                record.distinct_frame_ratio,
                float(unique_mask.float().mean()),
                min(record.decodable_frame_count / self.num_segments, 1.0),
            ],
            dtype=torch.float32,
        )
        return {
            "clips": clips,
            "labels": torch.tensor(record.class_id, dtype=torch.long),
            "sample_id": record.sample_id,
            "user_id": record.user_id,
            "quality": quality,
            "quality_mask": torch.ones(len(QUALITY_NAMES), dtype=torch.bool),
            "availability": torch.tensor(available, dtype=torch.bool),
            "source_indices": source_indices,
            "source_unique_mask": unique_mask,
            "num_frames": torch.tensor(record.decodable_frame_count, dtype=torch.long),
            "route": self.route,
            "fallback_reason": "not_applicable_primary_full_frame",
            "preprocessing_ms": torch.tensor((time.perf_counter() - started) * 1000.0),
        }


def collate_thermal_trials(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty Thermal batch")
    tensor_keys = (
        "clips",
        "labels",
        "quality",
        "quality_mask",
        "availability",
        "source_indices",
        "source_unique_mask",
        "num_frames",
        "preprocessing_ms",
    )
    batch = {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    batch.update(
        {
            "sample_ids": tuple(str(item["sample_id"]) for item in items),
            "user_ids": tuple(str(item["user_id"]) for item in items),
            "routes": tuple(str(item["route"]) for item in items),
            "fallback_reasons": tuple(str(item["fallback_reason"]) for item in items),
        }
    )
    return batch


def load_development_records(
    audit_path: Path,
    split_path: Path,
    data_root: Path,
) -> tuple[list[ThermalTrialRecord], list[ThermalTrialRecord]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    rows = audit["thermal_data_audit"]["canonical_trial_records"]
    train_users = set(split["train_user_ids"])
    validation_users = set(split["validation_user_ids"])
    if train_users & validation_users or (train_users | validation_users) & SEALED_USER_IDS:
        raise ValueError("Development split overlaps validation or sealed users")
    train = [ThermalTrialRecord.from_audit(row, data_root) for row in rows if row["user_id"] in train_users]
    validation = [
        ThermalTrialRecord.from_audit(row, data_root)
        for row in rows
        if row["user_id"] in validation_users
    ]
    if {record.user_id for record in train} != train_users:
        raise ValueError("Thermal audit does not cover the exact train12 users")
    if {record.user_id for record in validation} != validation_users:
        raise ValueError("Thermal audit does not cover the exact validation users")
    return train, validation
