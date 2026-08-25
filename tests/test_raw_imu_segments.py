from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.raw_imu_segments import (
    DEVICE_ROLES,
    SENSOR_COLUMNS,
    apply_imu_normalization,
    fit_imu_normalization,
    load_raw_imu_segments,
)


def write_imu_csv(
    path: Path,
    *,
    role: str,
    values: list[float],
    reverse_rows: bool = False,
) -> None:
    rows = []
    for index, value in enumerate(values):
        row = {
            "时间": f"2026-01-01 00:00:{index:02d}.000",
            "设备名称": f"{role}-device",
        }
        row.update({column: value + offset for offset, column in enumerate(SENSOR_COLUMNS)})
        rows.append(row)
    if reverse_rows:
        rows.reverse()
    pd.DataFrame(rows).to_csv(path, index=False)


def test_imu_loader_preserves_device_roles_and_masks_missing_roles(tmp_path: Path) -> None:
    write_imu_csv(tmp_path / "a.csv", role="WTRA", values=[3.0, 1.0, 2.0], reverse_rows=True)
    write_imu_csv(tmp_path / "b.csv", role="WTLL", values=[1.0, 2.0, 3.0])

    result = load_raw_imu_segments(tmp_path, segment_count=8)

    assert result.features.shape == (8, 5, 16)
    assert result.role_mask[:, DEVICE_ROLES.index("WTRA")].any()
    assert result.role_mask[:, DEVICE_ROLES.index("WTLL")].any()
    for role in ("WTLA", "WTC", "WTRL"):
        role_index = DEVICE_ROLES.index(role)
        assert not result.role_mask[:, role_index].any()
        assert torch.count_nonzero(result.features[:, role_index]) == 0


def test_imu_segment_values_retain_temporal_direction(tmp_path: Path) -> None:
    write_imu_csv(
        tmp_path / "a.csv", role="WTRA", values=[float(value) for value in range(16)]
    )
    write_imu_csv(
        tmp_path / "b.csv", role="WTLL", values=[float(value) for value in range(16)]
    )

    result = load_raw_imu_segments(tmp_path, segment_count=8)
    role = DEVICE_ROLES.index("WTRA")

    assert result.features[0, role, 0] < result.features[-1, role, 0]
    assert result.role_mask[:, role].all()


def test_imu_normalization_uses_only_available_roles(tmp_path: Path) -> None:
    write_imu_csv(tmp_path / "a.csv", role="WTRA", values=[1.0] * 16)
    write_imu_csv(tmp_path / "b.csv", role="WTLL", values=[3.0] * 16)
    sample = load_raw_imu_segments(tmp_path, segment_count=8)

    mean, std = fit_imu_normalization([sample])
    normalized = apply_imu_normalization(sample, mean, std)

    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert torch.isfinite(normalized.features).all()
    assert torch.count_nonzero(normalized.features[~normalized.role_mask]) == 0
