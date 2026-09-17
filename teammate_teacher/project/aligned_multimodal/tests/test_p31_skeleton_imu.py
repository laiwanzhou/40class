from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p31_skeleton_imu_model import P31SkeletonIMUPartEncoders, model_size_mib
from p31_skeleton_imu_preprocessing import (
    assign_points_to_frame_intervals,
    build_skeleton_features,
    parse_imu_timestamp,
)


def test_midpoint_interval_assignment_keeps_every_point_once() -> None:
    frames = np.asarray((10.0, 10.1, 10.2, 10.3))
    points = np.asarray((9.8, 10.01, 10.049, 10.051, 10.25, 10.6))
    assigned = assign_points_to_frame_intervals(points, frames)
    assert assigned.tolist() == [0, 0, 0, 1, 3, 3]
    counts = np.bincount(assigned, minlength=len(frames))
    assert int(counts.sum()) == len(points)


def test_legacy_non_zero_padded_imu_timestamp_is_valid() -> None:
    assert parse_imu_timestamp("2025-5-7 16:42:19.85") == parse_imu_timestamp(
        "2025-05-07 16:42:19.850"
    )


def test_skeleton_features_are_translation_and_scale_invariant() -> None:
    time_steps = 5
    joint = np.arange(17, dtype=np.float32)
    xyz = np.stack((joint * 0.02, joint * -0.01, joint * 0.03), axis=1)
    raw = np.repeat(xyz[None], time_steps, axis=0)
    raw[:, 13, 0] += np.linspace(0.0, 0.2, time_steps)
    raw = np.concatenate((raw, np.ones((time_steps, 17, 1), np.float32)), axis=2)
    times = np.arange(time_steps, dtype=np.float64) * 0.1
    first = build_skeleton_features(raw, times)
    transformed = raw.copy()
    transformed[..., :3] = transformed[..., :3] * 2.5 + np.asarray((5.0, -2.0, 8.0))
    second = build_skeleton_features(transformed, times)
    assert np.allclose(first["features"], second["features"], atol=2e-5)
    assert np.allclose(first["relations"], second["relations"], atol=2e-5)


def synthetic_batch() -> dict[str, torch.Tensor]:
    batch_size, time_steps, points = 2, 4, 7
    frame_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]], dtype=torch.bool
    )
    skeleton_features = torch.randn(batch_size, time_steps, 17, 13)
    skeleton_joint_mask = frame_mask[:, :, None].expand(-1, -1, 17).clone()
    skeleton_feature_mask = skeleton_joint_mask[..., None].expand(-1, -1, -1, 13).clone()
    skeleton_relations = torch.randn(batch_size, time_steps, 18)
    skeleton_relation_mask = frame_mask[:, :, None].expand(-1, -1, 18).clone()
    skeleton_frame_quality = frame_mask.float()

    imu_values = torch.randn(batch_size, 5, points, 10)
    quaternion = imu_values[..., 6:10]
    imu_values[..., 6:10] = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    imu_point_mask = torch.zeros(batch_size, 5, points, dtype=torch.bool)
    imu_point_mask[0, :, :6] = True
    imu_point_mask[1, 1:3, :4] = True
    imu_device_mask = imu_point_mask.any(dim=2)
    imu_frame_index = torch.full((batch_size, 5, points), -1, dtype=torch.long)
    imu_frame_index[0, :, :6] = torch.tensor((0, 0, 1, 2, 3, 3))
    imu_frame_index[1, 1:3, :4] = torch.tensor((0, 0, 1, 1))
    imu_times = torch.zeros(batch_size, 5, points)
    imu_times[0, :, :6] = torch.tensor((0.00, 0.03, 0.10, 0.20, 0.29, 0.31))
    imu_times[1, 1:3, :4] = torch.tensor((0.00, 0.04, 0.09, 0.11))
    frame_times = torch.tensor(
        [[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.0, 0.0]], dtype=torch.float32
    )
    return {
        "frame_mask": frame_mask,
        "frame_time_seconds": frame_times,
        "skeleton_features": skeleton_features,
        "skeleton_feature_mask": skeleton_feature_mask,
        "skeleton_joint_mask": skeleton_joint_mask,
        "skeleton_relations": skeleton_relations,
        "skeleton_relation_mask": skeleton_relation_mask,
        "skeleton_frame_quality": skeleton_frame_quality,
        "imu_values": imu_values,
        "imu_time_seconds": imu_times,
        "imu_frame_index": imu_frame_index,
        "imu_point_mask": imu_point_mask,
        "imu_device_mask": imu_device_mask,
    }


def test_variable_length_part_encoders_masks_counts_and_gradients() -> None:
    model = P31SkeletonIMUPartEncoders(dropout=0.0)
    batch = synthetic_batch()
    output = model(batch)
    assert output["skeleton_part_tokens"].shape == (2, 4, 8, 256)
    assert output["imu_part_tokens"].shape == (2, 4, 8, 256)
    assert output["imu_device_tokens"].shape == (2, 4, 5, 256)
    assert torch.count_nonzero(output["skeleton_part_tokens"][1, 2:]) == 0
    assert torch.count_nonzero(output["imu_part_tokens"][1, 2:]) == 0
    # No head-mounted IMU exists; head remains explicitly absent, not fake stillness.
    assert not output["imu_part_mask"][:, :, 1].any()
    assert int(output["imu_interval_count"].sum().item()) == int(
        batch["imu_point_mask"].sum().item()
    )
    loss = output["skeleton_part_tokens"].square().mean()
    loss = loss + output["imu_part_tokens"].square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model_size_mib(model) < 2.0
