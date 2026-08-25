from __future__ import annotations

import torch
from torch import nn

from src.models.structured_ir_depth_visual_encoder import (
    StructuredIRDepthVisualEncoder,
    VideoMAESegmentBackboneAdapter,
)


class TinyTemporalBackbone(nn.Module):
    def __init__(self, dim: int = 12) -> None:
        super().__init__()
        self.embed_dim = dim
        self.proj = nn.Conv3d(3, dim, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.tail = nn.Linear(dim, dim)

    def encode_prefix(self, clips: torch.Tensor) -> torch.Tensor:
        values = self.proj(clips).mean(dim=(3, 4)).transpose(1, 2)
        return values[:, :, None]

    def encode_tail(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tail(tokens.mean(dim=2))


class TinyPatchEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(3, dim, kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.proj(clips).flatten(2).transpose(1, 2)


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.linear(tokens)


class TinyVideoMAE(nn.Module):
    def __init__(self, dim: int = 12) -> None:
        super().__init__()
        self.embed_dim = dim
        self.patch_embed = TinyPatchEmbed(dim)
        self.pos_embed = torch.zeros(1, 32, dim)
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList(TinyBlock(dim) for _ in range(4))
        self.fc_norm = nn.LayerNorm(dim)


def inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    ir = torch.randn(batch, 4, 3, 16, 4, 4)
    depth = torch.randn_like(ir)
    availability = torch.ones(batch, 2, 4, dtype=torch.bool)
    return ir, depth, availability


def test_visual_encoder_keeps_context_and_wrist_in_training_graph() -> None:
    model = StructuredIRDepthVisualEncoder(
        backbone=TinyTemporalBackbone(dim=12), output_dim=32
    )
    ir, depth, availability = inputs()

    result = model(ir=ir, depth=depth, availability=availability)
    result.tokens.sum().backward()

    assert result.tokens.shape == (2, 8, 2, 32)
    assert result.mask.shape == (2, 8, 2)
    assert model.context_router.weight.grad is not None
    assert model.context_router.weight.grad.abs().sum() > 0
    assert model.wrist_router.weight.grad is not None
    assert model.wrist_router.weight.grad.abs().sum() > 0


def test_zero_depth_adapter_recovers_ir_tokens_exactly() -> None:
    torch.manual_seed(11)
    model = StructuredIRDepthVisualEncoder(
        backbone=TinyTemporalBackbone(dim=12), output_dim=32
    )
    ir, first_depth, availability = inputs()
    second_depth = torch.randn_like(first_depth) * 20

    first = model(ir=ir, depth=first_depth, availability=availability)
    second = model(ir=ir, depth=second_depth, availability=availability)

    assert torch.equal(first.tokens, second.tokens)
    assert torch.count_nonzero(first.quality[..., 0]) == 0


def test_visual_encoder_masks_unavailable_context_without_masking_wrist() -> None:
    model = StructuredIRDepthVisualEncoder(
        backbone=TinyTemporalBackbone(dim=12), output_dim=32
    )
    ir, depth, availability = inputs(batch=1)
    availability[:, :, :2] = False

    result = model(ir=ir, depth=depth, availability=availability)

    assert not result.mask[:, :, 0].any()
    assert result.mask[:, :, 1].all()
    assert torch.count_nonzero(result.tokens[:, :, 0]) == 0


def test_visual_encoder_uses_single_available_wrist_exactly() -> None:
    model = StructuredIRDepthVisualEncoder(
        backbone=TinyTemporalBackbone(dim=12), output_dim=32
    )
    ir, depth, availability = inputs(batch=1)
    availability[:, :, 3] = False

    result = model(ir=ir, depth=depth, availability=availability)

    assert result.mask[:, :, 1].all()
    assert torch.allclose(result.quality[:, :, 1, 2], torch.ones(1, 8))
    assert torch.count_nonzero(result.quality[:, :, 1, 3]) == 0


def test_videomae_adapter_preserves_segments_and_freezes_prefix() -> None:
    backbone = TinyVideoMAE(dim=12)
    adapter = VideoMAESegmentBackboneAdapter(
        backbone=backbone, frozen_prefix_blocks=2, segment_count=8
    )

    prefix = adapter.encode_prefix(torch.randn(2, 3, 16, 4, 4))
    output = adapter.encode_tail(prefix)

    assert prefix.shape == (2, 8, 4, 12)
    assert output.shape == (2, 8, 12)
    assert all(not parameter.requires_grad for parameter in backbone.patch_embed.parameters())
    assert all(
        not parameter.requires_grad
        for block in backbone.blocks[:2]
        for parameter in block.parameters()
    )
    assert all(
        parameter.requires_grad
        for block in backbone.blocks[2:]
        for parameter in block.parameters()
    )
