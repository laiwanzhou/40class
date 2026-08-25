from __future__ import annotations

import torch

from src.models.body_motion_segment_encoder import BodyMotionSegmentEncoder


def inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(17)
    skeleton = torch.randn(batch, 8, 17, 6)
    imu = torch.randn(batch, 8, 5, 16)
    skeleton_mask = torch.ones(batch, 8, dtype=torch.bool)
    imu_mask = torch.ones(batch, 8, 5, dtype=torch.bool)
    return skeleton, imu, skeleton_mask, imu_mask


def test_body_encoder_supports_skeleton_only_imu_only_and_both() -> None:
    model = BodyMotionSegmentEncoder(output_dim=32, heads=4)
    skeleton, imu, skeleton_mask, imu_mask = inputs()

    both = model(skeleton, imu, skeleton_mask, imu_mask)
    skeleton_only = model(
        skeleton, imu, skeleton_mask, torch.zeros_like(imu_mask)
    )
    imu_only = model(
        skeleton, imu, torch.zeros_like(skeleton_mask), imu_mask
    )

    expected = (2, 8, 1, 32)
    assert both.tokens.shape == skeleton_only.tokens.shape == imu_only.tokens.shape == expected
    assert both.mask.all() and skeleton_only.mask.all() and imu_only.mask.all()


def test_unavailable_body_tokens_are_exactly_zero() -> None:
    model = BodyMotionSegmentEncoder(output_dim=32, heads=4)
    skeleton, imu, skeleton_mask, imu_mask = inputs()

    result = model(
        skeleton,
        imu,
        torch.zeros_like(skeleton_mask),
        torch.zeros_like(imu_mask),
    )

    assert not result.mask.any()
    assert torch.count_nonzero(result.tokens) == 0


def test_both_body_paths_receive_gradient_when_available() -> None:
    model = BodyMotionSegmentEncoder(output_dim=32, heads=4)
    skeleton, imu, skeleton_mask, imu_mask = inputs()

    result = model(skeleton, imu, skeleton_mask, imu_mask)
    result.tokens.sum().backward()

    assert model.skeleton_projection.weight.grad is not None
    assert model.skeleton_projection.weight.grad.abs().sum() > 0
    assert model.imu_projection.weight.grad is not None
    assert model.imu_projection.weight.grad.abs().sum() > 0


def test_missing_imu_roles_do_not_change_role_weight_normalization() -> None:
    model = BodyMotionSegmentEncoder(output_dim=32, heads=4)
    skeleton, imu, skeleton_mask, imu_mask = inputs(batch=1)
    imu_mask[:, :, 1:4] = False

    result = model(skeleton, imu, skeleton_mask, imu_mask)

    imu_weights = result.quality[:, :, 0, 2:7]
    assert torch.allclose(imu_weights.sum(dim=2), torch.ones(1, 8))
    assert torch.count_nonzero(imu_weights[:, :, 1:4]) == 0
