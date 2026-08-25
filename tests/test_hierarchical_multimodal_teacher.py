from __future__ import annotations

import torch
from torch import nn

from src.models.body_motion_segment_encoder import BodyMotionSegmentEncoder
from src.models.hierarchical_action_query_fusion import HierarchicalActionQueryFusion
from src.models.hierarchical_multimodal_teacher import (
    GroupDropout,
    HierarchicalMultimodalTeacher,
)
from src.models.structured_ir_depth_visual_encoder import (
    StructuredIRDepthVisualEncoder,
)
from src.training.hierarchical_multimodal_losses import hierarchical_teacher_loss


class TinyTemporalBackbone(nn.Module):
    def __init__(self, dim: int = 12) -> None:
        super().__init__()
        self.embed_dim = dim
        self.proj = nn.Conv3d(3, dim, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.tail = nn.Linear(dim, dim)

    def encode_prefix(self, clips: torch.Tensor) -> torch.Tensor:
        return self.proj(clips).mean(dim=(3, 4)).transpose(1, 2)[:, :, None]

    def encode_tail(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tail(tokens.mean(dim=2))


def tiny_teacher() -> HierarchicalMultimodalTeacher:
    dim = 32
    return HierarchicalMultimodalTeacher(
        visual_encoder=StructuredIRDepthVisualEncoder(
            backbone=TinyTemporalBackbone(), output_dim=dim
        ),
        body_encoder=BodyMotionSegmentEncoder(output_dim=dim, heads=4),
        fusion=HierarchicalActionQueryFusion(
            dim=dim, classes=40, heads=4, layers=2
        ),
        dim=dim,
        classes=40,
    )


def complete_batch(batch: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    return {
        "visual": torch.randn(batch, 2, 4, 3, 16, 4, 4),
        "visual_view_availability": torch.ones(batch, 2, 4, dtype=torch.bool),
        "skeleton": torch.randn(batch, 8, 17, 6),
        "skeleton_mask": torch.ones(batch, 8, dtype=torch.bool),
        "imu": torch.randn(batch, 8, 5, 16),
        "imu_role_mask": torch.ones(batch, 8, 5, dtype=torch.bool),
        "availability": torch.ones(batch, 4, dtype=torch.bool),
    }


def test_teacher_auxiliary_heads_prevent_group_starvation() -> None:
    model = tiny_teacher()
    batch = complete_batch()

    output = model(batch, dropout_policy=GroupDropout.disabled())
    losses = hierarchical_teacher_loss(
        output, torch.tensor([0, 1]), epoch=1, natural_pattern=True
    )
    losses["loss"].backward()

    assert model.visual_encoder.context_router.weight.grad.abs().sum() > 0
    assert model.visual_encoder.wrist_router.weight.grad.abs().sum() > 0
    assert model.body_encoder.skeleton_projection.weight.grad.abs().sum() > 0
    assert model.body_encoder.imu_projection.weight.grad.abs().sum() > 0


def test_group_dropout_never_removes_every_usable_group() -> None:
    model = tiny_teacher()
    model.train()
    policy = GroupDropout(context=1.0, wrist=1.0, body=1.0, visual=1.0)

    output = model(complete_batch(batch=8), dropout_policy=policy)

    assert output["effective_group_mask"].any(dim=1).all()


def test_teacher_supports_visual_only_and_body_only() -> None:
    model = tiny_teacher()
    batch = complete_batch()
    visual_only = {key: value.clone() for key, value in batch.items()}
    visual_only["skeleton_mask"].zero_()
    visual_only["imu_role_mask"].zero_()
    body_only = {key: value.clone() for key, value in batch.items()}
    body_only["visual_view_availability"].zero_()

    visual_output = model(visual_only, dropout_policy=GroupDropout.disabled())
    body_output = model(body_only, dropout_policy=GroupDropout.disabled())

    assert visual_output["core_available"].all()
    assert body_output["core_available"].all()
    assert torch.isfinite(visual_output["logits"]).all()
    assert torch.isfinite(body_output["logits"]).all()


def test_explicit_empty_enabled_modalities_disables_every_group() -> None:
    model = tiny_teacher()

    output = model(
        complete_batch(batch=1),
        dropout_policy=GroupDropout.disabled(),
        enabled_modalities=(),
    )

    assert not output["core_available"].any()
    assert not output["effective_group_mask"].any()


def test_no_core_rows_are_excluded_from_teacher_loss() -> None:
    model = tiny_teacher()
    batch = complete_batch(batch=1)
    batch["visual_view_availability"].zero_()
    batch["skeleton_mask"].zero_()
    batch["imu_role_mask"].zero_()

    output = model(batch, dropout_policy=GroupDropout.disabled())
    losses = hierarchical_teacher_loss(
        output, torch.tensor([3]), epoch=1, natural_pattern=True
    )

    assert not output["core_available"].any()
    assert losses["supervised_rows"].item() == 0
    assert losses["loss"].item() == 0.0


def test_entropy_floor_is_disabled_after_epoch_two() -> None:
    model = tiny_teacher()
    output = model(complete_batch(), dropout_policy=GroupDropout.disabled())

    early = hierarchical_teacher_loss(
        output, torch.tensor([0, 1]), epoch=1, natural_pattern=True
    )
    late = hierarchical_teacher_loss(
        output, torch.tensor([0, 1]), epoch=3, natural_pattern=True
    )

    assert early["entropy_floor"].item() >= 0
    assert late["entropy_floor"].item() == 0.0
