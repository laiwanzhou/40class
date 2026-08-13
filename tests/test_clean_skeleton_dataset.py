from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path

from src.data.clean_skeleton_dataset import (
    CleanSkeletonGraphDataset,
    gap_aware_resample,
    gap_aware_resample_with_segments,
    normalize_scale,
    segment_local_velocity,
)


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


def test_resample_preserves_explicit_segment_identity() -> None:
    _, mask, output_segments = gap_aware_resample_with_segments(
        np.asarray([0, 1, 2, 3]),
        np.asarray([7, 7, 9, 9]),
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        target_length=4,
    )

    assert mask.tolist() == [True, True, True, True]
    assert output_segments.tolist() == [7, 7, 9, 9]


def test_graph_dataset_returns_joint_features_and_segment_ids(tmp_path: Path) -> None:
    rows = []
    for frame_id, segment in [(0, 0), (1, 0), (2, 1), (3, 1)]:
        path = tmp_path / f"frame_{frame_id}.json"
        path.write_text(json.dumps([{"keypoints": simple_pose(1.0).tolist()}]), encoding="utf-8")
        rows.append({
            "sample_id": "trial", "user_id": "user1", "class_id": 0, "frame_id": frame_id,
            "retained_segment_index": segment, "candidate_index": 0,
            "skeleton_json_path": path.name, "use_for_frame_training": True,
        })
    dataset = CleanSkeletonGraphDataset(
        pd.DataFrame(rows), tmp_path, {"user1"}, "per_frame", sequence_length=4
    )

    item = dataset[0]

    assert item["input"]["features"].shape == (4, 17, 6)
    assert item["input"]["segment_ids"].tolist() == [0, 0, 1, 1]
