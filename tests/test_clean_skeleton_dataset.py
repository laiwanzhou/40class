from __future__ import annotations

import numpy as np

from src.data.clean_skeleton_dataset import gap_aware_resample, normalize_scale, segment_local_velocity


def simple_pose(scale: float) -> np.ndarray:
    pose = np.zeros((17, 3), dtype=np.float64)
    pose[:, 0] = np.arange(17) * scale
    pose[:, 2] = np.arange(17) * scale * 0.5
    return pose


def test_c0_uses_one_trial_constant_scale() -> None:
    poses = np.stack([simple_pose(1.0), simple_pose(2.0)])
    _, scales = normalize_scale(poses, "trial_constant")

    assert scales[0] == scales[1]


def test_c1_uses_same_estimator_per_frame() -> None:
    poses = np.stack([simple_pose(1.0), simple_pose(2.0)])
    _, scales = normalize_scale(poses, "per_frame")

    assert np.isclose(scales[1], 2.0 * scales[0])


def test_velocity_resets_at_each_segment_start() -> None:
    values = np.asarray([[1.0], [3.0], [10.0], [14.0]])
    velocity = segment_local_velocity(values, np.asarray([0, 0, 1, 1]))

    assert velocity[:, 0].tolist() == [0.0, 2.0, 0.0, 4.0]


def test_gap_aware_resample_does_not_interpolate_across_gap() -> None:
    frame_ids = np.asarray([0, 1, 8, 9])
    segments = np.asarray([0, 0, 1, 1])
    values = np.asarray([[0.0], [1.0], [8.0], [9.0]])

    output, mask = gap_aware_resample(frame_ids, segments, values, target_length=10)

    assert mask.tolist() == [True, True, False, False, False, False, False, False, True, True]
    assert np.all(output[~mask] == 0.0)


def test_singleton_segment_is_kept_in_nearest_output_slot() -> None:
    output, mask = gap_aware_resample(
        np.asarray([0, 5, 10]), np.asarray([0, 1, 2]), np.asarray([[1.0], [2.0], [3.0]]), 5
    )

    assert mask.sum() == 3
    assert 2.0 in output[mask, 0]
