from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p86_mobind_lite_model import P86SeparateMotionEncoder  # noqa: E402
from p93_temporal_mobind_model import (  # noqa: E402
    P93TemporalMoBindStudent,
    P93TemporalMoBindV2Student,
    P93TemporalCrossAttentionPoolStudent,
)
from p93_spatial_mobind_model import P93SpatialCrossAttentionPoolStudent  # noqa: E402


class DummyVisual(nn.Module):
    temporal_modeling = True

    def __init__(self) -> None:
        super().__init__()
        self.temporal_fusion = nn.Sequential(
            nn.LayerNorm(512 * 4), nn.Linear(512 * 4, 512), nn.GELU()
        )

    @staticmethod
    def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(sequence.dtype).unsqueeze(-1)
        return (sequence * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def pool_temporal_sequence(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, windows, views, steps, width = sequence.shape
        values = sequence.reshape(batch * windows * views, steps, width)
        valid = mask.reshape(batch * windows * views, steps)
        whole = self._masked_mean(values, valid)
        midpoint = max(steps // 2, 1)
        early = self._masked_mean(values[:, :midpoint], valid[:, :midpoint])
        late = self._masked_mean(values[:, midpoint:], valid[:, midpoint:])
        ordered = self.temporal_fusion(
            torch.cat((whole, early, late, late - early), dim=1)
        )
        return (whole + ordered).reshape(batch, windows, views, width)


class DummyResidual(nn.Module):
    def __init__(self, motion_width: int = 8) -> None:
        super().__init__()
        self.reliability_groups = 1
        self.visual_query = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, motion_width)
        )
        self.reliability = nn.Sequential(
            nn.LayerNorm(motion_width * 4), nn.Linear(motion_width * 4, 1)
        )
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(motion_width * 4), nn.Linear(motion_width * 4, 512)
        )
        self.residual_logit = nn.Parameter(torch.tensor(0.0))

    def strength(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_logit)


class DummySeparateResidual(DummyResidual):
    def __init__(self, motion_width: int = 8) -> None:
        super().__init__(motion_width)
        self.modality = "separate"
        self.encoder = P86SeparateMotionEncoder(
            nn.Identity(), nn.Identity(), width=motion_width
        )


def test_missing_motion_is_exact_prepool_visual_fallback() -> None:
    model = P93TemporalMoBindStudent(DummyVisual(), DummyResidual(), 1).eval()
    visual = torch.randn(2, 2, 3, 4, 512)
    visual_mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    motion = torch.randn(2, 2, 4, 10, 8)
    motion_mask = torch.zeros(2, 2, 4, 10, dtype=torch.bool)

    fused, audit = model._fuse_temporal_tokens(
        visual, visual_mask, motion, motion_mask, None
    )

    assert torch.equal(fused, visual)
    assert not audit["motion_time_available"].any()
    assert not audit["motion_temporal_attention"].any()


def test_temporal_attention_cannot_read_outside_registered_radius() -> None:
    model = P93TemporalMoBindStudent(DummyVisual(), DummyResidual(), 1).eval()
    visual = torch.randn(1, 2, 3, 5, 512)
    visual_mask = torch.ones(1, 2, 3, 5, dtype=torch.bool)
    motion = torch.randn(1, 2, 5, 10, 8)
    motion_mask = torch.ones(1, 2, 5, 10, dtype=torch.bool)

    _, audit = model._fuse_temporal_tokens(
        visual, visual_mask, motion, motion_mask, None
    )
    temporal_attention = audit["motion_temporal_attention"]
    query = torch.arange(5)[:, None]
    memory = torch.arange(5)[None, :]
    outside = (query - memory).abs() > 1

    assert torch.equal(
        temporal_attention[..., outside],
        torch.zeros_like(temporal_attention[..., outside]),
    )
    assert torch.allclose(
        temporal_attention.sum(dim=-1),
        torch.ones_like(temporal_attention.sum(dim=-1)),
    )


def test_v2_zero_initialization_is_exact_p86_temporal_fallback() -> None:
    model = P93TemporalMoBindV2Student(
        DummyVisual(), DummySeparateResidual(), temporal_radius=1
    ).eval()
    visual = torch.randn(2, 2, 3, 4, 512)
    visual_mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    motion = torch.randn(2, 2, 4, 10, 8)
    motion_mask = torch.ones(2, 2, 4, 10, dtype=torch.bool)

    fused, audit = model._fuse_temporal_tokens(
        visual, visual_mask, motion, motion_mask
    )

    assert torch.equal(fused, visual)
    assert not audit["temporal_residual_rms"].any()
    assert torch.allclose(
        audit["temporal_modality_weight"].sum(dim=-1),
        torch.ones_like(audit["temporal_modality_weight"][..., 0]),
    )


def test_v2_missing_motion_remains_exact_after_projection_changes() -> None:
    model = P93TemporalMoBindV2Student(
        DummyVisual(), DummySeparateResidual(), temporal_radius=1
    ).eval()
    with torch.no_grad():
        for projection in (
            model.skeleton_temporal_projection,
            model.imu_temporal_projection,
        ):
            projection[4].weight.normal_(std=0.1)
            projection[4].bias.fill_(0.2)
    visual = torch.randn(1, 2, 3, 4, 512)
    visual_mask = torch.ones(1, 2, 3, 4, dtype=torch.bool)
    motion = torch.randn(1, 2, 4, 10, 8)
    motion_mask = torch.zeros(1, 2, 4, 10, dtype=torch.bool)

    fused, audit = model._fuse_temporal_tokens(
        visual, visual_mask, motion, motion_mask
    )

    assert torch.equal(fused, visual)
    assert not audit["temporal_motion_available"].any()
    assert not audit["temporal_modality_weight"].any()


def test_v2_fixed_budget_bounds_residual_and_zero_ablation() -> None:
    model = P93TemporalMoBindV2Student(
        DummyVisual(),
        DummySeparateResidual(),
        temporal_radius=0,
        temporal_budget=0.10,
    ).eval()
    with torch.no_grad():
        for projection in (
            model.skeleton_temporal_projection,
            model.imu_temporal_projection,
        ):
            projection[4].weight.normal_(std=2.0)
            projection[4].bias.fill_(2.0)
    visual = torch.randn(2, 2, 3, 4, 512)
    visual_mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    motion = torch.randn(2, 2, 4, 10, 8)
    motion_mask = torch.ones(2, 2, 4, 10, dtype=torch.bool)

    fused, _ = model._fuse_temporal_tokens(
        visual, visual_mask, motion, motion_mask
    )
    assert float((fused - visual).abs().max().detach()) <= 0.100001

    model.set_temporal_ablation("zero")
    zero_fused, _ = model._fuse_temporal_tokens(
        visual, visual_mask, motion, motion_mask
    )
    assert torch.equal(zero_fused, visual)


def test_v2_modalities_have_independent_five_part_attention() -> None:
    model = P93TemporalMoBindV2Student(
        DummyVisual(), DummySeparateResidual(), temporal_radius=1
    ).eval()
    visual = torch.randn(1, 2, 3, 5, 512)
    visual_mask = torch.ones(1, 2, 3, 5, dtype=torch.bool)
    motion = torch.randn(1, 2, 5, 10, 8)
    motion_mask = torch.ones(1, 2, 5, 10, dtype=torch.bool)

    _, audit = model._fuse_temporal_tokens(
        visual, visual_mask, motion, motion_mask
    )

    assert audit["temporal_skeleton_part_attention"].shape[-1] == 5
    assert audit["temporal_imu_part_attention"].shape[-1] == 5
    query = torch.arange(5)[:, None]
    memory = torch.arange(5)[None, :]
    outside = (query - memory).abs() > 1
    for key in (
        "temporal_skeleton_time_attention",
        "temporal_imu_time_attention",
    ):
        attention = audit[key]
        assert torch.equal(
            attention[..., outside], torch.zeros_like(attention[..., outside])
        )


def test_v3_zero_initialization_is_exact_p86_pooling_fallback() -> None:
    visual_module = DummyVisual()
    model = P93TemporalCrossAttentionPoolStudent(
        visual_module,
        DummySeparateResidual(),
        temporal_radius=1,
    ).eval()
    visual = torch.randn(2, 2, 3, 5, 512)
    visual_mask = torch.ones(2, 2, 3, 5, dtype=torch.bool)
    motion = torch.randn(2, 2, 5, 10, 8)
    motion_mask = torch.ones(2, 2, 5, 10, dtype=torch.bool)

    clips, audit = model._cross_modal_pool(
        visual, visual_mask, motion, motion_mask
    )
    baseline = visual_module.pool_temporal_sequence(visual, visual_mask)

    assert torch.allclose(clips, baseline, atol=1e-6, rtol=0.0)
    assert not audit["temporal_pool_logit"].any()
    assert torch.allclose(
        audit["temporal_pool_attention"].sum(dim=-1),
        torch.ones_like(audit["temporal_pool_attention"][..., 0]),
    )


def test_v3_local_cross_attention_cannot_read_outside_radius() -> None:
    model = P93TemporalCrossAttentionPoolStudent(
        DummyVisual(), DummySeparateResidual(), temporal_radius=1
    ).eval()
    visual = torch.randn(1, 2, 3, 5, 512)
    visual_mask = torch.ones(1, 2, 3, 5, dtype=torch.bool)
    motion = torch.randn(1, 2, 5, 10, 8)
    motion_mask = torch.ones(1, 2, 5, 10, dtype=torch.bool)

    _, audit = model._cross_modal_pool(
        visual, visual_mask, motion, motion_mask
    )
    query = torch.arange(5)[:, None]
    memory = torch.arange(5)[None, :]
    outside = (query - memory).abs() > 1
    for key in (
        "temporal_skeleton_cross_attention",
        "temporal_imu_cross_attention",
    ):
        attention = audit[key]
        time_attention = attention.sum(dim=-1)
        assert torch.equal(
            time_attention[..., outside],
            torch.zeros_like(time_attention[..., outside]),
        )
        assert torch.allclose(
            attention.sum(dim=(-1, -2)),
            torch.ones_like(attention[..., 0, 0]),
        )


def test_v3_missing_motion_and_zero_ablation_recover_p86_pooling() -> None:
    visual_module = DummyVisual()
    model = P93TemporalCrossAttentionPoolStudent(
        visual_module, DummySeparateResidual(), temporal_radius=1
    ).eval()
    with torch.no_grad():
        for scorer in (
            model.skeleton_cross_attention,
            model.imu_cross_attention,
        ):
            scorer.compatibility[4].weight.normal_(std=0.2)
            scorer.compatibility[4].bias.fill_(0.3)
    visual = torch.randn(2, 2, 3, 5, 512)
    visual_mask = torch.ones(2, 2, 3, 5, dtype=torch.bool)
    motion = torch.randn(2, 2, 5, 10, 8)
    baseline = visual_module.pool_temporal_sequence(visual, visual_mask)

    missing_clips, missing_audit = model._cross_modal_pool(
        visual,
        visual_mask,
        motion,
        torch.zeros(2, 2, 5, 10, dtype=torch.bool),
    )
    assert torch.allclose(missing_clips, baseline, atol=1e-6, rtol=0.0)
    assert not missing_audit["temporal_motion_available"].any()

    model.set_temporal_ablation("zero")
    zero_clips, _ = model._cross_modal_pool(
        visual,
        visual_mask,
        motion,
        torch.ones(2, 2, 5, 10, dtype=torch.bool),
    )
    assert torch.allclose(zero_clips, baseline, atol=1e-6, rtol=0.0)


def test_v4_zero_initialization_is_exact_compact_spatial_fallback() -> None:
    model = P93SpatialCrossAttentionPoolStudent(
        DummyVisual(), DummySeparateResidual(), spatial_grid=5
    ).eval()
    compact = torch.randn(2, 2, 3, 4, 512)
    regions = torch.randn(2, 2, 3, 4, 25, 512)
    visual_mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    motion = torch.randn(2, 2, 4, 10, 8)
    motion_mask = torch.ones(2, 2, 4, 10, dtype=torch.bool)

    corrected, audit = model._condition_spatial_pool(
        compact, regions, visual_mask, motion, motion_mask
    )

    assert torch.equal(corrected, compact)
    assert not audit["spatial_pool_logit"].any()
    assert torch.allclose(
        audit["spatial_pool_attention"].sum(dim=-1),
        torch.ones_like(audit["spatial_pool_attention"][..., 0]),
    )
    assert audit["spatial_skeleton_part_attention"].shape[-2:] == (25, 5)
    assert audit["spatial_imu_part_attention"].shape[-2:] == (25, 5)


def test_v4_missing_motion_and_zero_ablation_recover_compact_sequence() -> None:
    model = P93SpatialCrossAttentionPoolStudent(
        DummyVisual(), DummySeparateResidual(), spatial_grid=5
    ).eval()
    with torch.no_grad():
        for scorer in (
            model.skeleton_spatial_scorer,
            model.imu_spatial_scorer,
        ):
            scorer.compatibility[4].weight.normal_(std=0.3)
            scorer.compatibility[4].bias.fill_(0.2)
    compact = torch.randn(2, 2, 3, 4, 512)
    regions = torch.randn(2, 2, 3, 4, 25, 512)
    visual_mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    motion = torch.randn(2, 2, 4, 10, 8)

    missing, audit = model._condition_spatial_pool(
        compact,
        regions,
        visual_mask,
        motion,
        torch.zeros(2, 2, 4, 10, dtype=torch.bool),
    )
    assert torch.equal(missing, compact)
    assert not audit["spatial_motion_available"].any()

    model.set_spatial_ablation("zero")
    zero, _ = model._condition_spatial_pool(
        compact,
        regions,
        visual_mask,
        motion,
        torch.ones(2, 2, 4, 10, dtype=torch.bool),
    )
    assert torch.equal(zero, compact)
