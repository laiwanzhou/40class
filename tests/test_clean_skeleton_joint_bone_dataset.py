from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_skeleton_joint_bone_dataset import (
    CleanSkeletonJointBoneDataset,
    joint_bone_features,
)


def test_joint_bone_features_have_frozen_198d_order() -> None:
    poses = np.zeros((2, 17, 3), dtype=np.float64)
    poses[0, :, 0] = np.arange(17)
    poses[1, :, 0] = 2 * np.arange(17)
    features = joint_bone_features(poses, np.asarray([0, 0]))

    assert features.shape == (2, 198)
    joint = features.reshape(2, -1)[:, :102].reshape(2, 17, 6)
    bone = features[:, 102:].reshape(2, 16, 6)
    assert np.array_equal(joint[:, :, :3], poses)
    assert np.all(joint[0, :, 3:] == 0.0)
    assert np.array_equal(joint[1, :, 3:], poses[1] - poses[0])
    assert bone[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert bone[1, 0].tolist() == [2.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_bone_velocity_resets_at_segment_boundary() -> None:
    poses = np.zeros((3, 17, 3), dtype=np.float64)
    poses[:, :, 0] = np.asarray([1.0, 2.0, 20.0])[:, None] * np.arange(17)[None, :]
    features = joint_bone_features(poses, np.asarray([0, 0, 1]))
    bones = features[:, 102:].reshape(3, 16, 6)

    assert np.any(bones[1, :, 3:] != 0.0)
    assert np.all(bones[2, :, 3:] == 0.0)


def test_joint_bone_dataset_resamples_combined_features_without_gap_bridge(tmp_path: Path) -> None:
    rows = []
    for frame_id, segment, multiplier in [(0, 0, 1.0), (1, 0, 1.1), (8, 1, 2.0), (9, 1, 2.1)]:
        pose = np.zeros((17, 3), dtype=np.float64)
        pose[:, 0] = np.arange(17) * multiplier
        pose[:, 2] = np.arange(17) * multiplier * 0.5
        path = tmp_path / f"frame_{frame_id}.json"
        path.write_text(json.dumps([{"keypoints": pose.tolist()}]), encoding="utf-8")
        rows.append({
            "sample_id": "trial", "user_id": "user1", "class_id": 0, "frame_id": frame_id,
            "retained_segment_index": segment, "candidate_index": 0,
            "skeleton_json_path": path.name, "use_for_frame_training": True,
        })
    dataset = CleanSkeletonJointBoneDataset(
        pd.DataFrame(rows), tmp_path, {"user1"}, sequence_length=10
    )

    tensor, mask, _ = dataset.load_tensor(0, apply_normalization=False)

    assert tensor.shape == (10, 198)
    assert mask.tolist() == [True, True, False, False, False, False, False, False, True, True]
    assert np.all(tensor.numpy()[~mask.numpy()] == 0.0)
