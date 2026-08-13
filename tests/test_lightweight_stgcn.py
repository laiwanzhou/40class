from __future__ import annotations

import math

import torch

from src.models.lightweight_stgcn import (
    LightweightSTGCN,
    SegmentAwareTemporalConv,
    h36m_normalized_adjacency,
)


def test_h36m_adjacency_is_fixed_symmetric_and_normalized() -> None:
    adjacency = h36m_normalized_adjacency()

    assert adjacency.shape == (17, 17)
    assert not adjacency.requires_grad
    assert torch.allclose(adjacency, adjacency.T)
    assert torch.all(torch.diag(adjacency) > 0)
    assert torch.linalg.eigvalsh(adjacency).abs().max() <= 1.00001


def test_model_returns_required_shapes() -> None:
    model = LightweightSTGCN(channels=(32, 48, 64), embedding_dim=128, num_classes=40)
    features = torch.randn(3, 64, 17, 6)
    segment_ids = torch.zeros(3, 64, dtype=torch.long)
    mask = torch.ones(3, 64, dtype=torch.bool)

    output = model({"features": features, "segment_ids": segment_ids}, temporal_mask=mask)

    assert output["sequence_features"].shape == (3, 64, 17, 64)
    assert output["embedding"].shape == (3, 128)
    assert output["logits"].shape == (3, 40)


def test_v1_uses_frozen_temporal_contract_without_adaptive_graph_parameters() -> None:
    model = LightweightSTGCN(channels=(32, 48, 64))

    assert [block.temporal.dilation for block in model.blocks] == [1, 2, 4]
    assert [block.temporal.kernel_size for block in model.blocks] == [5, 5, 5]
    assert 1 + sum((block.temporal.kernel_size - 1) * block.temporal.dilation for block in model.blocks) == 29
    assert all("adjacency" not in name for name, _ in model.named_parameters())
    assert len([name for name, _ in model.named_buffers() if name.endswith("adjacency")]) == 3


def test_temporal_initialization_uses_kernel_times_input_channels() -> None:
    layer = SegmentAwareTemporalConv(32, 64, kernel_size=5)
    expected_bound = 1.0 / math.sqrt(5 * 32)

    assert layer.initialization_bound == expected_bound
    assert layer.weight.abs().max() <= expected_bound


def test_temporal_conv_does_not_cross_segment_boundary() -> None:
    torch.manual_seed(7)
    layer = SegmentAwareTemporalConv(4, 5, kernel_size=3, dilation=2)
    inputs = torch.randn(1, 8, 2, 4)
    segment_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    mask = torch.ones(1, 8, dtype=torch.bool)
    changed = inputs.clone()
    changed[:, 4:] += 1000.0

    baseline = layer(inputs, segment_ids, mask)
    perturbed = layer(changed, segment_ids, mask)

    assert torch.allclose(baseline[:, :4], perturbed[:, :4], atol=1e-5, rtol=1e-5)


def test_masked_frames_cannot_affect_valid_sequence_features() -> None:
    torch.manual_seed(11)
    model = LightweightSTGCN(channels=(24, 32), embedding_dim=128, num_classes=40).eval()
    features = torch.randn(1, 12, 17, 6)
    mask = torch.tensor([[True] * 5 + [False] * 3 + [True] * 4])
    segment_ids = torch.tensor([[0] * 5 + [-1] * 3 + [1] * 4])
    changed = features.clone()
    changed[:, 5:8] = 10000.0

    with torch.no_grad():
        baseline = model({"features": features, "segment_ids": segment_ids}, temporal_mask=mask)
        perturbed = model({"features": changed, "segment_ids": segment_ids}, temporal_mask=mask)

    assert torch.allclose(baseline["sequence_features"], perturbed["sequence_features"], atol=1e-5)
    assert torch.allclose(baseline["embedding"], perturbed["embedding"], atol=1e-5)
    assert torch.all(baseline["sequence_features"][:, 5:8] == 0)


def test_model_stays_within_lightweight_parameter_budget() -> None:
    model = LightweightSTGCN(channels=(32, 48, 64), embedding_dim=128, num_classes=40)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 60_920
    assert "adjacency" not in dict(model.named_parameters())
