from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_depth_geometry_descriptor import (  # noqa: E402
    DEPTH_STATISTICS,
    TEMPORAL_STATISTICS,
    depth_color_to_rank,
    roi_depth_statistics,
    save_descriptor_artifact,
    temporal_summary,
)
from p99_depth_geometry_expert import recipe_matrices  # noqa: E402


def test_jet_hue_becomes_monotonic_depth_rank() -> None:
    hsv = np.asarray(
        [[[0, 255, 255], [30, 255, 255], [60, 255, 255], [90, 255, 255], [120, 255, 255], [0, 0, 0]]],
        dtype=np.uint8,
    )
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rank, valid = depth_color_to_rank(bgr)
    np.testing.assert_allclose(rank[0, :5], np.linspace(0.0, 1.0, 5), atol=1e-6)
    assert valid[0, :5].all()
    assert not valid[0, 5]
    assert rank[0, 5] == 0.0


def test_roi_depth_statistics_tracks_missing_pixels_and_quantiles() -> None:
    rank = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    value = roi_depth_statistics(rank, valid, np.asarray([0, 0, 3, 3]), True)
    assert value.shape == (len(DEPTH_STATISTICS),)
    assert value[0] == 15 / 16
    assert value[3] <= value[4] <= value[5]
    np.testing.assert_array_equal(
        roi_depth_statistics(rank, valid, np.full(4, np.nan), False),
        np.zeros(len(DEPTH_STATISTICS)),
    )


def test_temporal_summary_has_fixed_finite_schema_and_phase_signal() -> None:
    values = np.asarray([[0.0, 1.0], [1.0, 1.0], [4.0, 1.0]], dtype=np.float32)
    summary, names = temporal_summary(values, ["moving", "constant"])
    assert summary.shape == (len(TEMPORAL_STATISTICS) * 2,)
    assert len(names) == len(summary)
    assert np.isfinite(summary).all()
    assert summary[names.index("late_mean_moving")] > summary[names.index("early_mean_moving")]
    assert summary[names.index("linear_slope_constant")] == 0.0
    singleton, _ = temporal_summary(np.asarray([[2.0, 3.0]]), ["a", "b"])
    assert np.isfinite(singleton).all()


def test_descriptor_artifact_is_label_free_and_recipe_alignment_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "descriptor.npz"
    arrays = {
        "sample_ids": np.asarray(["b", "a"]),
        "users": np.asarray(["user2", "user1"]),
        "geometry": np.asarray([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32),
        "geometry_feature_names": np.asarray(["g0", "g1"]),
        "depth_surface": np.asarray([[7.0], [5.0]], dtype=np.float32),
        "depth_surface_feature_names": np.asarray(["d0"]),
    }
    save_descriptor_artifact(path, arrays)
    with np.load(path, allow_pickle=False) as source:
        assert "labels" not in source.files
    matrices = recipe_matrices(
        arrays["sample_ids"],
        {"geometry": arrays["geometry"], "depth_surface": arrays["depth_surface"]},
        np.asarray(["a", "b"]),
        {
            "geometry_only": {"groups": ["geometry"]},
            "depth_geometry": {"groups": ["geometry", "depth_surface"]},
        },
    )
    np.testing.assert_array_equal(matrices["geometry_only"], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        matrices["depth_geometry"], [[1.0, 2.0, 5.0], [3.0, 4.0, 7.0]]
    )
