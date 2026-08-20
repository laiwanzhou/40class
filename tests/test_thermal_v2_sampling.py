from __future__ import annotations

import pytest

from src.data.thermal_v2_sampling import (
    DEFAULT_WINDOWS,
    normalized_probe_indices,
    normalized_window_indices,
    uniqueness_mask,
)


def test_three_windows_have_fixed_endpoints_and_lengths() -> None:
    indices = normalized_window_indices(101)

    assert DEFAULT_WINDOWS == ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))
    assert len(indices) == 3
    assert all(len(window) == 16 for window in indices)
    assert indices[0][0] == 0
    assert indices[0][-1] == 50
    assert indices[1][0] == 25
    assert indices[1][-1] == 75
    assert indices[2][0] == 50
    assert indices[2][-1] == 100
    assert all(left <= right for window in indices for left, right in zip(window, window[1:]))


@pytest.mark.parametrize("frame_count", (1, 2, 8, 16, 101))
def test_window_indices_remain_in_thermal_bounds(frame_count: int) -> None:
    indices = normalized_window_indices(frame_count)

    assert all(0 <= index < frame_count for window in indices for index in window)


def test_singleton_repeats_are_explicit() -> None:
    window = normalized_window_indices(1)[0]

    assert window == (0,) * 16
    assert uniqueness_mask(window) == (True,) + (False,) * 15


def test_half_up_rounding_does_not_use_bankers_rounding() -> None:
    indices = normalized_window_indices(3, windows=((0.0, 1.0),), frames_per_window=5)

    assert indices == ((0, 1, 1, 2, 2),)


def test_eight_probe_indices_are_uniform_and_deduplicated() -> None:
    assert normalized_probe_indices(101) == (0, 14, 29, 43, 57, 71, 86, 100)
    assert normalized_probe_indices(2) == (0, 1)
    assert normalized_probe_indices(1) == (0,)


@pytest.mark.parametrize("frame_count", (0, -1))
def test_non_positive_frame_count_is_rejected(frame_count: int) -> None:
    with pytest.raises(ValueError, match="frame_count must be positive"):
        normalized_window_indices(frame_count)
