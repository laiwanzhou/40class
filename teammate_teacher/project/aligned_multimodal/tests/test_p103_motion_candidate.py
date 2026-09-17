from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p103_motion_candidate_model import MotionCandidateConfig, P103MotionCandidateTeacher


def synthetic_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(21)
    skeleton_mask = torch.ones(batch_size, 2, 16, 17, dtype=torch.bool)
    imu_mask = torch.ones(batch_size, 2, 16, 5, dtype=torch.bool)
    return {
        "vmae_features": torch.randn(batch_size, 6, 768, generator=generator),
        "vmae_actions": torch.randn(batch_size, 6, 710, generator=generator),
        "vjepa_features": torch.randn(batch_size, 16, 1024, generator=generator),
        "vjepa_actions": torch.randn(batch_size, 16, 174, generator=generator),
        "candidate_ids": torch.tensor([[0, 1, 2, -1]] * batch_size),
        "a_context": torch.zeros(batch_size, 4, 7),
        "motion_time": torch.stack(
            (
                torch.linspace(0.0, 0.7, 16),
                torch.linspace(0.3, 1.0, 16),
            )
        ).repeat(batch_size, 1, 1),
        "skeleton_scale": torch.ones(batch_size),
        "imu_scale": torch.ones(batch_size),
        "skeleton_features": torch.randn(
            batch_size, 2, 16, 17, 13, generator=generator
        ),
        "skeleton_feature_mask": skeleton_mask[..., None].expand(-1, -1, -1, -1, 13),
        "skeleton_joint_mask": skeleton_mask,
        "skeleton_relations": torch.randn(batch_size, 2, 16, 18, generator=generator),
        "skeleton_relation_mask": torch.ones(batch_size, 2, 16, 18, dtype=torch.bool),
        "skeleton_frame_quality": torch.ones(batch_size, 2, 16),
        "imu_sequences": torch.randn(batch_size, 2, 16, 5, 4, 16, generator=generator),
        "imu_sequence_mask": torch.ones(batch_size, 2, 16, 5, 4, dtype=torch.bool),
        "imu_bin_statistics": torch.randn(batch_size, 2, 16, 5, 52, generator=generator),
        "imu_bin_mask": imu_mask,
        "imu_global_statistics": torch.randn(batch_size, 5, 48, generator=generator),
        "imu_global_mask": torch.ones(batch_size, 5, 2),
    }


def test_motion_candidate_is_staged_and_candidate_conditioned() -> None:
    torch.manual_seed(22)
    model = P103MotionCandidateTeacher(MotionCandidateConfig(dropout=0.0)).eval()
    output = model(synthetic_batch(1), return_attention=True)
    assert output["candidate_scores"].shape == (1, 4)
    assert output["local_candidate_evidence"].shape == (1, 4, 128)
    assert output["skeleton_candidate_evidence"].shape == (1, 4, 128)
    assert output["imu_candidate_evidence"].shape == (1, 4, 128)
    assert output["attention_groups"].shape == (1, 4, 27)
    assert output["candidate_scores"][0, 3] < -1000
    assert not torch.allclose(
        output["skeleton_candidate_evidence"][0, 0],
        output["skeleton_candidate_evidence"][0, 1],
    )
    assert not hasattr(model, "classifier")


def test_zero_motion_removes_sample_specific_motion_content() -> None:
    torch.manual_seed(23)
    model = P103MotionCandidateTeacher(MotionCandidateConfig(dropout=0.0)).eval()
    batch = synthetic_batch(2)
    for name in ("vmae_features", "vmae_actions", "vjepa_features", "vjepa_actions"):
        batch[name][1] = batch[name][0]
    batch["skeleton_scale"].zero_()
    batch["imu_scale"].zero_()
    output = model(batch)
    assert torch.allclose(
        output["candidate_scores"][0], output["candidate_scores"][1], atol=1e-5
    )


def test_negative_motion_prefix_changes_motion_not_local_query() -> None:
    torch.manual_seed(24)
    model = P103MotionCandidateTeacher(MotionCandidateConfig(dropout=0.0)).eval()
    batch = synthetic_batch(1)
    for name in (
        "skeleton_features",
        "skeleton_feature_mask",
        "skeleton_joint_mask",
        "skeleton_relations",
        "skeleton_relation_mask",
        "skeleton_frame_quality",
        "imu_sequences",
        "imu_sequence_mask",
        "imu_bin_statistics",
        "imu_bin_mask",
        "imu_global_statistics",
        "imu_global_mask",
    ):
        value = batch[name]
        batch["negative_" + name] = value.flip(2) if value.ndim >= 3 else value.clone()
    aligned = model(batch)
    negative = model(batch, motion_prefix="negative_")
    assert torch.allclose(
        aligned["local_candidate_evidence"], negative["local_candidate_evidence"]
    )
    assert not torch.allclose(
        aligned["skeleton_candidate_evidence"], negative["skeleton_candidate_evidence"]
    )
