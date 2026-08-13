from __future__ import annotations

import torch
from torch import nn

from src.models.segment_aware_tcn import (
    SegmentAwareConv1d,
    SegmentAwareTemporalClassifier,
)
from src.models.tcn import TemporalClassifier


def test_segment_aware_conv_matches_conv1d_for_one_fully_valid_segment() -> None:
    torch.manual_seed(10)
    original = nn.Conv1d(5, 7, kernel_size=5, padding=4, dilation=2).eval()
    candidate = SegmentAwareConv1d(5, 7, kernel_size=5, padding=4, dilation=2).eval()
    candidate.weight.data.copy_(original.weight.data)
    candidate.bias.data.copy_(original.bias.data)
    inputs = torch.randn(3, 5, 23)
    mask = torch.ones(3, 23, dtype=torch.bool)
    segments = torch.zeros(3, 23, dtype=torch.long)

    expected = original(inputs)
    actual = candidate(inputs, segments, mask)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_segment_aware_conv_blocks_cross_segment_sources_without_gap() -> None:
    torch.manual_seed(11)
    conv = SegmentAwareConv1d(3, 4, kernel_size=5, padding=2).eval()
    inputs = torch.randn(1, 3, 10)
    changed = inputs.clone()
    changed[:, :, 5:] += 1000.0
    mask = torch.ones(1, 10, dtype=torch.bool)
    segments = torch.tensor([[0] * 5 + [1] * 5])

    baseline = conv(inputs, segments, mask)
    perturbed = conv(changed, segments, mask)

    assert torch.allclose(baseline[:, :, :5], perturbed[:, :, :5], atol=1e-5, rtol=1e-5)


def test_segment_aware_model_isolates_segments_across_all_layers() -> None:
    torch.manual_seed(12)
    model = SegmentAwareTemporalClassifier(input_features=6, channels=(8, 12), dropout=0.0).eval()
    features = torch.randn(1, 14, 6)
    changed = features.clone()
    changed[:, 8:] += 500.0
    mask = torch.tensor([[True] * 5 + [False] * 3 + [True] * 6])
    segments = torch.tensor([[0] * 5 + [-1] * 3 + [1] * 6])

    with torch.no_grad():
        baseline = model.encode({"features": features, "segment_ids": segments}, mask)
        perturbed = model.encode({"features": changed, "segment_ids": segments}, mask)

    assert torch.allclose(baseline[:, :5], perturbed[:, :5], atol=1e-5, rtol=1e-5)
    assert torch.all(baseline[:, 5:8] == 0)


def test_full_t1_matches_original_tcn_for_one_segment_after_state_copy() -> None:
    torch.manual_seed(13)
    original = TemporalClassifier(input_features=10, channels=(12, 16), dropout=0.0).eval()
    candidate = SegmentAwareTemporalClassifier(input_features=10, channels=(12, 16), dropout=0.0).eval()
    candidate.load_from_temporal_classifier(original)
    features = torch.randn(2, 19, 10)
    mask = torch.ones(2, 19, dtype=torch.bool)
    segments = torch.zeros(2, 19, dtype=torch.long)

    with torch.no_grad():
        expected = original(features, temporal_mask=mask)
        actual = candidate({"features": features, "segment_ids": segments}, temporal_mask=mask)

    assert torch.allclose(actual["embedding"], expected["embedding"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual["logits"], expected["logits"], atol=1e-6, rtol=1e-5)


def test_t1_parameter_count_matches_c1() -> None:
    original = TemporalClassifier(input_features=102, channels=(64, 128))
    candidate = SegmentAwareTemporalClassifier(input_features=102, channels=(64, 128))

    assert sum(parameter.numel() for parameter in candidate.parameters()) == 172_776
    assert sum(parameter.numel() for parameter in candidate.parameters()) == sum(
        parameter.numel() for parameter in original.parameters()
    )
