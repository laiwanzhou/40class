from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from build_p86_motion_window_cache import (  # noqa: E402
    aggregate_imu,
    bin_assignments,
    p20_device_statistics,
)


def test_bin_assignments_share_duplicate_visual_frames_without_leakage() -> None:
    slots, points = bin_assignments(
        np.asarray([0.0, 0.0, 1.0, 2.0]),
        np.asarray([-0.1, 0.0, 0.4, 0.6, 1.6, 2.0, 2.1]),
    )
    assert slots.tolist() == [0, 0, 1, 2]
    assert points.tolist() == [-1, 0, 0, 1, 2, 2, -1]


def test_p20_statistics_match_expected_shape_and_constant_signal() -> None:
    values = np.ones((5, 6), dtype=np.float32)
    statistics = p20_device_statistics(values)
    assert statistics.shape == (48,)
    first_channel = statistics[:8]
    np.testing.assert_allclose(first_channel, [1, 0, 1, 1, 1, 0, 0, 0])


def test_aggregate_imu_keeps_raw_and_compensated_channels_on_visual_grid() -> None:
    frame_times = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    source_indices = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    # One device has three identity-quaternion samples; the other four are absent.
    values = np.zeros((3, 10), dtype=np.float32)
    values[:, 0] = [1.0, 2.0, 3.0]
    values[:, 6] = 1.0
    times = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    offsets = np.asarray([0, 3, 3, 3, 3, 3], dtype=np.int64)
    result = aggregate_imu(
        frame_times, source_indices, values, times, offsets, points_per_bin=2
    )
    assert result["sequences"].shape == (2, 3, 5, 2, 16)
    assert result["statistics"].shape == (2, 3, 5, 52)
    assert result["global_statistics"].shape == (5, 48)
    assert result["global_mask"][0, 0] == 1
    assert result["global_mask"][1:, 0].sum() == 0
    # Identity orientation means raw and compensated acceleration are identical.
    first = result["sequences"][0, 0, 0, 0]
    np.testing.assert_allclose(first[:6], first[6:12])
    assert first[12] == 1.0
