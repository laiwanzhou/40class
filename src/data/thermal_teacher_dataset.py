from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from scripts.audit_thermal_v2_inputs import decode_rgb, load_jsonl, trial_path
from src.data.thermal_native_dataset import _geometry, _transform_image
from src.data.thermal_v2_inventory import image_files
from src.data.thermal_v2_sampling import normalized_window_indices


class ThermalTeacherDataset(Dataset[dict[str, Any]]):
    """Thermal v2 raster-only dataset for training-only video teachers."""

    def __init__(
        self,
        *,
        data_root: Path,
        context_path: Path,
        normalization_path: Path,
        partition: str,
        training: bool,
        seed: int = 20260715,
    ) -> None:
        if partition not in {"train12", "val_user6_user7"}:
            raise ValueError("partition must be train12 or val_user6_user7")
        self.data_root = data_root.resolve()
        self.records = [
            row for row in load_jsonl(context_path.resolve())
            if row["development_split"] == partition
        ]
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        self.mean = torch.tensor(normalization["rgb_mean"], dtype=torch.float32)
        self.std = torch.tensor(normalization["rgb_std"], dtype=torch.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or bool((self.std <= 0).any()):
            raise ValueError("invalid train12 RGB normalization")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, sample_id: str, view: str) -> random.Random:
        raw = f"{self.seed}|{self.epoch}|{sample_id}|{view}".encode("utf-8")
        return random.Random(int(hashlib.sha256(raw).hexdigest()[:16], 16))

    @staticmethod
    def _empty(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "full_rgb": torch.zeros(3, 3, 16, 160, 160),
            "crop_rgb": torch.zeros(3, 3, 16, 160, 160),
            "window_mask": torch.zeros(3, dtype=torch.bool),
            "availability": torch.zeros(2, dtype=torch.bool),
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
        indices = sorted({index for window in windows for index in window})
        decoded = {index: decode_rgb(files[index]) for index in indices}
        full_geometry = _geometry(self._rng(str(record["sample_id"]), "full"), self.training)
        crop_geometry = _geometry(self._rng(str(record["sample_id"]), "crop"), self.training)
        full = {
            index: _transform_image(
                decoded[index], bbox=None, geometry=full_geometry, mean=self.mean,
                std=self.std, training=self.training,
            )
            for index in indices
        }
        crop_available = bool(record.get("context_available", False))
        crop = {
            index: _transform_image(
                decoded[index], bbox=record["bbox_xyxy"], geometry=crop_geometry,
                mean=self.mean, std=self.std, training=self.training,
            )
            for index in indices
        } if crop_available else {}

        full_windows = [
            torch.stack([full[index] for index in window], dim=1) for window in windows
        ]
        crop_windows = [
            torch.stack([crop[index] for index in window], dim=1)
            if crop_available else torch.zeros(3, 16, 160, 160)
            for window in windows
        ]
        return {
            "full_rgb": torch.stack(full_windows),
            "crop_rgb": torch.stack(crop_windows),
            "window_mask": torch.ones(3, dtype=torch.bool),
            "availability": torch.tensor([True, crop_available], dtype=torch.bool),
            "label": torch.tensor(int(record["class_id"]), dtype=torch.long),
            "loss_eligible": torch.tensor(True),
            "sample_id": str(record["sample_id"]),
            "user_id": str(record["user_id"]),
        }
