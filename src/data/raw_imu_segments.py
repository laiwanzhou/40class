from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import torch


DEVICE_ROLES = ("WTRA", "WTLA", "WTC", "WTRL", "WTLL")
SENSOR_COLUMNS = (
    "加速度X(g)", "加速度Y(g)", "加速度Z(g)",
    "角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)",
    "角度X(°)", "角度Y(°)", "角度Z(°)",
    "磁场X(uT)", "磁场Y(uT)", "磁场Z(uT)",
    "四元数0()", "四元数1()", "四元数2()", "四元数3()",
)


@dataclass(frozen=True)
class IMUSegments:
    features: torch.Tensor
    role_mask: torch.Tensor
    quality: torch.Tensor


def _role(device_name: str) -> str:
    match = re.match(r"([A-Za-z]+)", device_name)
    return match.group(1).upper() if match else device_name.upper()


def _read_trial(path: Path) -> pd.DataFrame:
    csv_files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".csv")
    if len(csv_files) != 2:
        raise ValueError(f"expected two IMU CSV files in {path}; got {len(csv_files)}")
    tables: list[pd.DataFrame] = []
    required = {"时间", "设备名称", *SENSOR_COLUMNS}
    for csv_path in csv_files:
        table = pd.read_csv(csv_path)
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"IMU CSV misses {sorted(missing)}: {csv_path}")
        tables.append(table.dropna(how="all"))
    combined = pd.concat(tables, ignore_index=True)
    combined["_role"] = combined["设备名称"].astype(str).map(_role)
    combined["_time"] = pd.to_datetime(combined["时间"], errors="coerce")
    return combined


def load_raw_imu_segments(path: Path, segment_count: int = 8) -> IMUSegments:
    if segment_count < 1:
        raise ValueError("IMU segment count must be positive")
    combined = _read_trial(path)
    features = np.zeros((segment_count, len(DEVICE_ROLES), len(SENSOR_COLUMNS)), dtype=np.float32)
    role_mask = np.zeros((segment_count, len(DEVICE_ROLES)), dtype=bool)
    quality = np.zeros((segment_count, len(DEVICE_ROLES), 3), dtype=np.float32)
    valid_times = combined["_time"].dropna()
    global_span = (
        max((valid_times.max() - valid_times.min()).total_seconds(), 1e-6)
        if len(valid_times)
        else 1.0
    )
    for role_index, role in enumerate(DEVICE_ROLES):
        group = combined[combined["_role"].eq(role)].copy()
        group = group.sort_values("_time", kind="stable", na_position="last")
        if group.empty:
            continue
        raw = group.loc[:, SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        finite_fraction = np.isfinite(raw).mean(axis=1)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        times = group["_time"]
        if times.notna().sum() >= 2 and times.max() > times.min():
            elapsed = (times - times.min()).dt.total_seconds().to_numpy(dtype=np.float64)
            observed_span = max(float(np.nanmax(elapsed)), 1e-6)
            positions = np.nan_to_num(elapsed / observed_span, nan=0.0)
            role_span_fraction = min(1.0, observed_span / global_span)
        else:
            positions = np.linspace(0.0, 1.0, len(group), dtype=np.float64)
            role_span_fraction = 0.0 if len(group) == 1 else 1.0
        bins = np.minimum(segment_count - 1, np.floor(positions * segment_count).astype(np.int64))
        for segment in range(segment_count):
            selected = bins == segment
            if not selected.any():
                continue
            features[segment, role_index] = raw[selected].mean(axis=0).astype(np.float32)
            role_mask[segment, role_index] = True
            quality[segment, role_index] = (
                float(selected.sum()),
                float(role_span_fraction),
                float(finite_fraction[selected].mean()),
            )
    return IMUSegments(
        features=torch.from_numpy(features),
        role_mask=torch.from_numpy(role_mask),
        quality=torch.from_numpy(quality),
    )

def fit_imu_normalization(
    samples: Iterable[IMUSegments],
) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[np.ndarray]] = [[] for _ in DEVICE_ROLES]
    for sample in samples:
        for role_index in range(len(DEVICE_ROLES)):
            selected = sample.role_mask[:, role_index].numpy()
            if selected.any():
                values[role_index].append(
                    sample.features[:, role_index].numpy()[selected].astype(np.float64)
                )
    mean = np.zeros((len(DEVICE_ROLES), len(SENSOR_COLUMNS)), dtype=np.float32)
    std = np.ones_like(mean)
    for role_index, parts in enumerate(values):
        if not parts:
            continue
        stacked = np.concatenate(parts, axis=0)
        mean[role_index] = stacked.mean(axis=0).astype(np.float32)
        std[role_index] = np.maximum(stacked.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


def apply_imu_normalization(
    sample: IMUSegments,
    mean: np.ndarray,
    std: np.ndarray,
) -> IMUSegments:
    expected = (len(DEVICE_ROLES), len(SENSOR_COLUMNS))
    if mean.shape != expected or std.shape != expected:
        raise ValueError("IMU normalization shape changed")
    values = sample.features.numpy().copy()
    values = (values - mean[None]) / np.maximum(std[None], 1e-6)
    values[~sample.role_mask.numpy()] = 0.0
    if not np.isfinite(values).all():
        raise ValueError("non-finite normalized IMU segments")
    return IMUSegments(
        features=torch.from_numpy(values.astype(np.float32)),
        role_mask=sample.role_mask.clone(),
        quality=sample.quality.clone(),
    )
