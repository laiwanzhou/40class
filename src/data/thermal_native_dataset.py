from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.audit_thermal_v2_inputs import build_quality_vector, decode_rgb, load_jsonl, trial_path
from scripts.build_thermal_v2_pose_cache import pose_key
from src.data.thermal_v2_features import signed_grayscale_differences
from src.data.thermal_v2_inventory import image_files
from src.data.thermal_v2_sampling import normalized_window_indices, uniqueness_mask


def _resize_short(image: np.ndarray, short_side: int = 176) -> np.ndarray:
    height, width = image.shape[:2]
    scale = short_side / min(height, width)
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _geometry(rng: random.Random, training: bool) -> tuple[float, float, float, float, bool]:
    if not training:
        return 1.0, 1.0, 0.5, 0.5, False
    return (
        rng.uniform(0.8, 1.0),
        rng.uniform(0.9, 1.1),
        rng.random(),
        rng.random(),
        rng.random() < 0.5,
    )


def _transform_image(
    image: np.ndarray,
    *,
    bbox: Sequence[int] | None,
    geometry: tuple[float, float, float, float, bool],
    mean: torch.Tensor,
    std: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    if bbox is not None:
        x1, y1, x2, y2 = (int(value) for value in bbox)
        image = image[y1:y2, x1:x2]
    image = _resize_short(image)
    height, width = image.shape[:2]
    scale, ratio, y_position, x_position, flip = geometry
    if training:
        crop_area = height * width * scale
        crop_width = min(width, max(1, int(round(math.sqrt(crop_area * ratio)))))
        crop_height = min(height, max(1, int(round(math.sqrt(crop_area / ratio)))))
        top = int(round((height - crop_height) * y_position))
        left = int(round((width - crop_width) * x_position))
    else:
        crop_height = crop_width = min(160, height, width)
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
    image = image[top : top + crop_height, left : left + crop_width]
    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    if flip:
        image = np.ascontiguousarray(image[:, ::-1])
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - mean[:, None, None]) / std[:, None, None]


class ThermalNativeDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        data_root: Path,
        context_path: Path,
        normalization_path: Path,
        pose_cache_path: Path,
        partition: str,
        training: bool,
        seed: int = 20260715,
    ) -> None:
        if partition not in {"train12", "val_user6_user7"}:
            raise ValueError("partition must be train12 or val_user6_user7")
        self.data_root = data_root.resolve()
        self.records = [
            row for row in load_jsonl(context_path.resolve()) if row["development_split"] == partition
        ]
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        self.mean = torch.tensor(normalization["rgb_mean"], dtype=torch.float32)
        self.std = torch.tensor(normalization["rgb_std"], dtype=torch.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or bool((self.std <= 0).any()):
            raise ValueError("invalid train12 RGB normalization")
        with np.load(pose_cache_path, allow_pickle=False) as payload:
            keys = payload["keys"].astype(str)
            poses = payload["pose"].astype(np.float32)
            valid = payload["valid"].astype(bool)
        if poses.shape != (len(keys), 56) or valid.shape != (len(keys),):
            raise ValueError("invalid pose cache shapes")
        self.pose_lookup = {
            key: (torch.from_numpy(pose.copy()), bool(is_valid))
            for key, pose, is_valid in zip(keys, poses, valid, strict=True)
        }
        self._verify_pose_completeness()

    def _verify_pose_completeness(self) -> None:
        missing: list[str] = []
        for record in self.records:
            if not record.get("usable", False):
                continue
            files = image_files(trial_path(self.data_root, record))
            for index in {i for window in normalized_window_indices(len(files)) for i in window}:
                key = pose_key(str(record["sample_id"]), index)
                if key not in self.pose_lookup:
                    missing.append(key)
                    if len(missing) == 3:
                        break
        if missing:
            raise ValueError("pose cache missing required keys: " + ", ".join(missing))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, sample_id: str, view: str) -> random.Random:
        value = f"{self.seed}|{self.epoch}|{sample_id}|{view}".encode("utf-8")
        return random.Random(int(hashlib.sha256(value).hexdigest()[:16], 16))

    def _empty(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "full_rgb": torch.zeros(3, 3, 16, 160, 160),
            "crop_rgb": torch.zeros(3, 3, 16, 160, 160),
            "motion": torch.zeros(3, 16, 1, 160, 160),
            "pose": torch.zeros(3, 16, 56),
            "pose_mask": torch.zeros(3, 16, dtype=torch.bool),
            "window_mask": torch.zeros(3, dtype=torch.bool),
            "uniqueness_mask": torch.zeros(3, 16, dtype=torch.bool),
            "availability": torch.zeros(4, dtype=torch.bool),
            "quality": torch.zeros(8),
            "label": torch.tensor(int(record["class_id"]), dtype=torch.long),
            "loss_eligible": torch.tensor(False),
            "sample_id": str(record["sample_id"]),
            "user_id": str(record["user_id"]),
        }

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        if not record.get("usable", False):
            return self._empty(record)
        files = image_files(trial_path(self.data_root, record))
        windows = normalized_window_indices(len(files))
        unique_indices = sorted({index for window in windows for index in window})
        decoded = {index: decode_rgb(files[index]) for index in unique_indices}
        full_geometry = _geometry(self._rng(str(record["sample_id"]), "full"), self.training)
        crop_geometry = _geometry(self._rng(str(record["sample_id"]), "crop"), self.training)
        full_by_index = {
            index: _transform_image(
                decoded[index], bbox=None, geometry=full_geometry, mean=self.mean,
                std=self.std, training=self.training
            )
            for index in unique_indices
        }
        crop_available = bool(record.get("context_available", False))
        crop_by_index = (
            {
                index: _transform_image(
                    decoded[index], bbox=record["bbox_xyxy"], geometry=crop_geometry,
                    mean=self.mean, std=self.std, training=self.training
                )
                for index in unique_indices
            }
            if crop_available
            else {}
        )
        full_windows, crop_windows, motion_windows = [], [], []
        pose_windows, pose_masks, unique_masks = [], [], []
        flat_indices: list[int] = []
        flat_pose_mask: list[bool] = []
        for window in windows:
            full = torch.stack([full_by_index[index] for index in window])
            full_windows.append(full.permute(1, 0, 2, 3))
            crop = (
                torch.stack([crop_by_index[index] for index in window])
                if crop_available else torch.zeros(16, 3, 160, 160)
            )
            crop_windows.append(crop.permute(1, 0, 2, 3))
            motion_windows.append(signed_grayscale_differences(full))
            entries = [self.pose_lookup[pose_key(str(record["sample_id"]), index)] for index in window]
            pose_windows.append(torch.stack([entry[0] for entry in entries]))
            mask = torch.tensor([entry[1] for entry in entries], dtype=torch.bool)
            pose_masks.append(mask)
            unique_masks.append(torch.tensor(uniqueness_mask(window), dtype=torch.bool))
            flat_indices.extend(window)
            flat_pose_mask.extend(mask.tolist())
        pose_available = any(flat_pose_mask)
        return {
            "full_rgb": torch.stack(full_windows),
            "crop_rgb": torch.stack(crop_windows),
            "motion": torch.stack(motion_windows),
            "pose": torch.stack(pose_windows),
            "pose_mask": torch.stack(pose_masks),
            "window_mask": torch.ones(3, dtype=torch.bool),
            "uniqueness_mask": torch.stack(unique_masks),
            "availability": torch.tensor([True, crop_available, True, pose_available]),
            "quality": torch.tensor(
                build_quality_vector(record, flat_indices, flat_pose_mask), dtype=torch.float32
            ),
            "label": torch.tensor(int(record["class_id"]), dtype=torch.long),
            "loss_eligible": torch.tensor(True),
            "sample_id": str(record["sample_id"]),
            "user_id": str(record["user_id"]),
        }
