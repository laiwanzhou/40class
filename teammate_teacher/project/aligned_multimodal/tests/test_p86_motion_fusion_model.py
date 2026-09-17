from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p86_motion_fusion_model import P86AlignedMotionResidual  # noqa: E402
from train_p86_mobind_fusion_proxy import loader as fusion_loader  # noqa: E402


def motion_batch(batch: int, steps: int, present: bool) -> dict[str, torch.Tensor]:
    skeleton_joint_mask = torch.full((batch, 2, steps, 17), present)
    skeleton_feature_mask = skeleton_joint_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 13)
    skeleton_relation_mask = torch.full((batch, 2, steps, 18), present)
    imu_bin_mask = torch.full((batch, 2, steps, 5), present)
    imu_sequence_mask = imu_bin_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
    return {
        "skeleton_features": torch.randn(batch, 2, steps, 17, 13),
        "skeleton_feature_mask": skeleton_feature_mask,
        "skeleton_joint_mask": skeleton_joint_mask,
        "skeleton_relations": torch.randn(batch, 2, steps, 18),
        "skeleton_relation_mask": skeleton_relation_mask,
        "skeleton_frame_quality": torch.ones(batch, 2, steps),
        "imu_sequences": torch.randn(batch, 2, steps, 5, 4, 16),
        "imu_sequence_mask": imu_sequence_mask,
        "imu_bin_statistics": torch.randn(batch, 2, steps, 5, 52),
        "imu_bin_mask": imu_bin_mask,
        "imu_global_statistics": torch.randn(batch, 5, 48),
        "imu_global_mask": torch.tensor([1.0, 1.0]).view(1, 1, 2).expand(batch, 5, 2),
    }


def test_all_label_fusion_loader_keeps_terminal_partial_batch() -> None:
    args = SimpleNamespace(
        batch_size=64,
        workers=0,
        final_refit=False,
        all_label_refit=True,
        subject_holdout_users=[],
    )
    data = fusion_loader(list(range(2914)), args, shuffle=True)
    assert not data.drop_last
    assert len(data) == 46


def test_missing_motion_is_exact_visual_fallback() -> None:
    module = P86AlignedMotionResidual(dropout=0.0).eval()
    visual = torch.randn(2, 2, 3, 4, 512)
    fused, audit = module(
        visual, torch.rand(2, 2, 4), motion_batch(2, 4, present=False)
    )
    assert torch.equal(fused, visual)
    assert not audit["motion_available"].any()


def test_present_motion_has_small_trainable_residual_and_bounded_size() -> None:
    module = P86AlignedMotionResidual(dropout=0.0)
    visual = torch.randn(2, 2, 3, 4, 512, requires_grad=True)
    fused, audit = module(
        visual, torch.rand(2, 2, 4), motion_batch(2, 4, present=True)
    )
    assert fused.shape == visual.shape
    assert audit["motion_attention"].shape == (2 * 2 * 4, 4, 3, 13)
    assert 0 < float(audit["motion_residual_strength"].detach()) <= 0.25
    assert sum(parameter.numel() for parameter in module.parameters()) < 1_000_000
    fused.square().mean().backward()
    assert module.imu_point_encoder[1].weight.grad is not None
    assert module.skeleton_joint_encoder[1].weight.grad is not None
