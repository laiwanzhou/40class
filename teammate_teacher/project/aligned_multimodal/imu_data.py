from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DEVICES = ("WTC", "WTLA", "WTRA", "WTLL", "WTRL")
DEVICE_TO_INDEX = {name: index for index, name in enumerate(DEVICES)}
CHANNEL_NAMES = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "relative_quat_w",
    "relative_quat_x",
    "relative_quat_y",
    "relative_quat_z",
)
CHANNEL_GROUPS = {
    "accgyro": tuple(range(6)),
    "accgyroquat": tuple(range(10)),
}


@dataclass(frozen=True)
class IMUIndexRow:
    cache_index: int
    split: str
    sample_id: str
    class_id: int
    user_id: str
    trial_id: str
    source_sample_id: str
    source_path: str
    usable: bool
    device_count: int
    duration_seconds: float


def read_index(path: Path) -> list[IMUIndexRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        IMUIndexRow(
            cache_index=int(row["cache_index"]),
            split=row["split"],
            sample_id=row["sample_id"],
            class_id=int(row["class_id"]),
            user_id=row["user_id"],
            trial_id=row["trial_id"],
            source_sample_id=row["source_sample_id"],
            source_path=row["source_path"],
            usable=bool(int(row["usable"])),
            device_count=int(row["device_count"]),
            duration_seconds=float(row["duration_seconds"]),
        )
        for row in rows
    ]


def load_cache_metadata(cache_dir: Path) -> dict[str, object]:
    return json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))


def compute_channel_normalizer(
    values: np.ndarray,
    time_mask: np.ndarray,
    indices: np.ndarray,
    channel_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(values[indices])[..., channel_indices]
    mask = np.asarray(time_mask[indices], dtype=np.float64)[..., None]
    count = np.maximum(mask.sum(axis=(0, 1, 2)), 1.0)
    mean = (selected * mask).sum(axis=(0, 1, 2)) / count
    variance = (((selected - mean) ** 2) * mask).sum(axis=(0, 1, 2)) / count
    std = np.sqrt(np.maximum(variance, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class IMUDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        rows: list[IMUIndexRow],
        channel_group: str,
        mean: np.ndarray,
        std: np.ndarray,
        training: bool,
        device_dropout: float = 0.0,
        noise_std: float = 0.0,
        forced_drop_device: int | None = None,
    ) -> None:
        if channel_group not in CHANNEL_GROUPS:
            raise ValueError(f"Unknown IMU channel group: {channel_group}")
        self.cache_dir = Path(cache_dir).resolve()
        self.rows = rows
        self.channel_indices = CHANNEL_GROUPS[channel_group]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.training = training
        self.device_dropout = float(device_dropout)
        self.noise_std = float(noise_std)
        self.forced_drop_device = forced_drop_device
        self._arrays: dict[str, np.ndarray] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def _array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            filenames = {
                "values": "imu_float32.npy",
                "time_mask": "time_mask_uint8.npy",
                "device_mask": "device_mask_uint8.npy",
            }
            self._arrays[name] = np.load(
                self.cache_dir / filenames[name], mmap_mode="r", allow_pickle=False
            )
        return self._arrays[name]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[index]
        values = np.asarray(
            self._array("values")[row.cache_index][..., self.channel_indices],
            dtype=np.float32,
        ).copy()
        time_mask = np.asarray(
            self._array("time_mask")[row.cache_index], dtype=np.float32
        ).copy()
        device_mask = np.asarray(
            self._array("device_mask")[row.cache_index], dtype=np.float32
        ).copy()

        values = (values - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)
        values *= time_mask[..., None]

        if self.forced_drop_device is not None:
            device_mask[self.forced_drop_device] = 0.0
            time_mask[self.forced_drop_device] = 0.0
            values[self.forced_drop_device] = 0.0

        if self.training and self.device_dropout > 0:
            present = np.flatnonzero(device_mask > 0)
            if len(present) > 1 and torch.rand(1).item() < self.device_dropout:
                drop_index = int(present[int(torch.randint(len(present), (1,)).item())])
                device_mask[drop_index] = 0.0
                time_mask[drop_index] = 0.0
                values[drop_index] = 0.0

        if self.training and self.noise_std > 0:
            noise = torch.randn(values.shape).numpy().astype(np.float32) * self.noise_std
            values += noise * time_mask[..., None]

        return {
            "imu": torch.from_numpy(values).permute(0, 2, 1),
            "time_mask": torch.from_numpy(time_mask),
            "device_mask": torch.from_numpy(device_mask),
            "label": row.class_id,
            "sample_id": row.sample_id,
        }

