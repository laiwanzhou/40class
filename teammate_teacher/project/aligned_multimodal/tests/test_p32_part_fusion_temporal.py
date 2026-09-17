from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p30_shared_dir_roi_model import PYRAMID_FEATURE_DIM
from p32_part_fusion_temporal_model import (
    P32PartFusionTemporalModel,
    model_size_mib,
    visual_part_region_prior,
)


def synthetic_batch() -> dict[str, torch.Tensor]:
    batch_size, time_steps, points = 2, 5, 7
    frame_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    roi_valid = frame_mask[:, :, None].expand(-1, -1, 7).clone()
    features = torch.randn(batch_size, time_steps, 2, 7, PYRAMID_FEATURE_DIM)
    skeleton_joint_mask = frame_mask[:, :, None].expand(-1, -1, 17).clone()
    skeleton_feature_mask = skeleton_joint_mask[..., None].expand(-1, -1, -1, 13).clone()
    skeleton_relations = torch.randn(batch_size, time_steps, 18)
    skeleton_relation_mask = frame_mask[:, :, None].expand(-1, -1, 18).clone()

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
    imu_frame_index[0, :, :6] = torch.tensor((0, 0, 1, 2, 3, 4))
    imu_frame_index[1, 1:3, :4] = torch.tensor((0, 0, 1, 2))
    imu_time = torch.zeros(batch_size, 5, points)
    imu_time[0, :, :6] = torch.tensor((0.0, 0.03, 0.1, 0.2, 0.3, 0.4))
    imu_time[1, 1:3, :4] = torch.tensor((0.0, 0.03, 0.1, 0.2))
    frame_time = torch.tensor(
        [[0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.0, 0.0]]
    )
    time_position = torch.tensor(
        [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.5, 1.0, 0.0, 0.0]]
    )
    return {
        "features": features,
        "roi_quality": roi_valid.float() * 0.8,
        "roi_valid": roi_valid,
        "roi_source": roi_valid.long(),
        "roi_clipped_ratio": torch.zeros(batch_size, time_steps, 7),
        "pose_quality_factor": frame_mask.float(),
        "frame_mask": frame_mask,
        "frame_time_seconds": frame_time,
        "time_position": time_position,
        "skeleton_features": torch.randn(batch_size, time_steps, 17, 13),
        "skeleton_feature_mask": skeleton_feature_mask,
        "skeleton_joint_mask": skeleton_joint_mask,
        "skeleton_relations": skeleton_relations,
        "skeleton_relation_mask": skeleton_relation_mask,
        "skeleton_frame_quality": frame_mask.float(),
        "imu_values": imu_values,
        "imu_time_seconds": imu_time,
        "imu_frame_index": imu_frame_index,
        "imu_point_mask": imu_point_mask,
        "imu_device_mask": imu_device_mask,
    }


def test_visual_semantic_prior_does_not_invent_local_leg_or_head_roi() -> None:
    allowed = visual_part_region_prior()
    assert allowed.shape == (8, 7)
    assert allowed[1].nonzero().flatten().tolist() == [0, 6]
    assert allowed[5].nonzero().flatten().tolist() == [0, 6]
    assert allowed[6].nonzero().flatten().tolist() == [0, 6]
    assert set(allowed[7].nonzero().flatten().tolist()) == {0, 1, 2, 3, 4, 5}


def test_steps_13_14_variable_sequence_masks_gates_and_gradients() -> None:
    model = P32PartFusionTemporalModel(dropout=0.0, modality_dropout=0.0)
    batch = synthetic_batch()
    output = model(batch)
    assert output["fused_part_tokens"].shape == (2, 5, 8, 256)
    assert output["temporal_sequence"].shape == (2, 5, 256)
    assert output["trial_embedding"].shape == (2, 384)
    assert output["modality_gate"].shape == (2, 5, 8, 3)
    assert torch.count_nonzero(output["fused_part_tokens"][1, 3:]) == 0
    assert torch.count_nonzero(output["temporal_sequence"][1, 3:]) == 0
    expected_gate_sum = output["fused_part_mask"].float()
    assert torch.allclose(output["modality_gate"].sum(dim=3), expected_gate_sum, atol=1e-6)
    assert int(output["imu_interval_count"].sum().item()) == int(
        batch["imu_point_mask"].sum().item()
    )
    loss = output["trial_embedding"].square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model_size_mib(model) < 20.0
