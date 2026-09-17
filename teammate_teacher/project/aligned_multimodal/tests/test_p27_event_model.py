from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p27_data import align_frame_maps
from p27_model import P27EventModel, fp16_size_mib, parameter_count


def test_full_frame_key_is_authoritative() -> None:
    maps = {
        "depth": {
            "100_00000001": Path("depth-1"),
            "200_00000002": Path("depth-2"),
        },
        "ir": {
            "100_00000001": Path("ir-1"),
            "999_00000002": Path("ir-2"),
        },
        "skeleton": {
            "100_00000001": Path("skeleton-1"),
            "888_00000002": Path("skeleton-2"),
        },
    }
    aligned, keys, mode = align_frame_maps(maps)
    assert aligned is maps
    assert keys == ["100_00000001"]
    assert mode == "full"


def test_unique_counter_fallback_repairs_schema_mismatch() -> None:
    maps = {
        "depth": {
            "00000001": Path("depth-1"),
            "00000002": Path("depth-2"),
        },
        "ir": {
            "00000001": Path("ir-1"),
            "00000002": Path("ir-2"),
        },
        "skeleton": {
            "123_00000001": Path("skeleton-1"),
            "456_00000002": Path("skeleton-2"),
        },
    }
    aligned, keys, mode = align_frame_maps(maps)
    assert keys == ["00000001", "00000002"]
    assert mode == "counter_fallback"
    assert aligned["skeleton"]["00000001"] == Path("skeleton-1")


def test_duplicate_terminal_counter_never_silently_collapses() -> None:
    maps = {
        "depth": {"00000001": Path("depth")},
        "ir": {"00000001": Path("ir")},
        "skeleton": {
            "123_00000001": Path("skeleton-a"),
            "456_00000001": Path("skeleton-b"),
        },
    }
    _, keys, mode = align_frame_maps(maps)
    assert keys == []
    assert mode == "ambiguous_counter"


def test_model_budget_and_imu_attention_window() -> None:
    model = P27EventModel(torch.zeros(10), torch.ones(10))
    assert parameter_count(model) == 12_593_249
    assert fp16_size_mib(model) < 40.0
    assert tuple(model.imu_window_mask.shape) == (12, 60)
    for query_time in range(12):
        allowed = (~model.imu_window_mask[query_time]).reshape(12, 5).any(dim=1)
        expected = torch.zeros(12, dtype=torch.bool)
        expected[max(0, query_time - 1) : min(12, query_time + 2)] = True
        torch.testing.assert_close(allowed.cpu(), expected)
