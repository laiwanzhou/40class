from pathlib import Path

import numpy as np

from scripts.experiments.pose_skeleton_matching_audit.audit_core import (
    decode_jet_relative_depth,
    frame_key,
    left_right_swap,
    map_skeleton_to_yolo_order,
    normalize_pose,
    percentile_auc,
    retrieval_query_is_valid,
)


def test_frame_key_accepts_all_three_modalities() -> None:
    suffix = "2025-05-08_11-06-39.616_00000087"
    assert frame_key(Path(f"IR_{suffix}.png")) == suffix
    assert frame_key(Path(f"Depth_{suffix}_Color.png")) == suffix
    assert frame_key(Path(f"Color_{suffix}.json")) == suffix


def test_depth_decoder_is_monotonic_across_jet_hues_and_masks_black() -> None:
    rgb = np.array([[[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 0]]])
    decoded = decode_jet_relative_depth(rgb)[0]
    assert np.all(np.diff(decoded[:5]) > 0)
    assert np.isnan(decoded[-1])


def test_normalization_does_not_create_missing_joints() -> None:
    points = np.zeros((1, 17, 2), dtype=float)
    points[0, 5:7] = [[-1, 1], [1, 1]]
    points[0, 11:13] = [[-1, 0], [1, 0]]
    mask = np.ones((1, 17), dtype=bool)
    mask[0, 9] = False
    normalized, result_mask, _, _ = normalize_pose(points, mask)
    assert not result_mask[0, 9]
    assert np.isnan(normalized[0, 9]).all()


def test_left_right_swap_is_an_involution() -> None:
    values = np.arange(17)[None, :, None]
    assert np.array_equal(left_right_swap(left_right_swap(values)), values)


def test_h36m_skeleton_mapping_keeps_only_twelve_true_common_joints() -> None:
    skeleton = np.arange(17 * 3, dtype=float).reshape(17, 3)
    mapped = map_skeleton_to_yolo_order(skeleton)
    assert np.isfinite(mapped).all(axis=1).sum() == 12
    assert np.array_equal(mapped[5], skeleton[11])
    assert np.array_equal(mapped[12], skeleton[1])
    assert np.isnan(mapped[:5]).all()


def test_percentile_auc_has_expected_extremes() -> None:
    assert percentile_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert percentile_auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0


def test_retrieval_query_requires_explicit_observation_floor() -> None:
    mask = np.zeros((32, 17), dtype=bool)
    mask[:4, 5:17] = True
    assert retrieval_query_is_valid(mask, 48)
    mask[3, 16] = False
    assert not retrieval_query_is_valid(mask, 48)
