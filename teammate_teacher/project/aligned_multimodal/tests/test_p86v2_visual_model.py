from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p86_mc3_visual_model import P86MC3VisualStudent
from p86v2_visual_model import (
    P86V2Layer3SpatialDedupStudent,
    P86V2TemporalDedupStudent,
)


def test_unique_times_start_at_p86v1_function() -> None:
    torch.manual_seed(7)
    baseline = P86MC3VisualStudent(
        frames=4, kinetics_pretrained=False, temporal_modeling=True
    ).eval()
    candidate = P86V2TemporalDedupStudent(
        frames=4, kinetics_pretrained=False
    ).eval()
    missing, unexpected = candidate.load_state_dict(baseline.state_dict(), strict=False)
    assert missing == [
        "continuous_time_projection.weight",
        "continuous_time_projection.bias",
    ]
    assert unexpected == []
    sequence = torch.randn(2, 2, 3, 4, 512)
    valid = torch.ones(2, 2, 4, 3, dtype=torch.bool)
    quality = torch.ones(2, 2, 4, 3)
    source_time = torch.tensor(
        [[[0.0, 0.2, 0.4, 0.7], [0.3, 0.6, 0.8, 1.0]]] * 2
    )
    with torch.inference_mode():
        expected = baseline.forward_from_backbone_sequence(sequence, valid, quality)
        actual = candidate.forward_from_backbone_sequence(
            sequence, valid, quality, source_time
        )
    torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)


def test_duplicate_source_times_are_masked() -> None:
    model = P86V2TemporalDedupStudent(frames=4, kinetics_pretrained=False).eval()
    source_time = torch.tensor([[[0.0, 0.0, 0.4, 0.7], [0.3, 0.6, 0.6, 1.0]]])
    expected = torch.tensor([[[True, False, True, True], [True, True, False, True]]])
    assert torch.equal(model.unique_time_mask(source_time), expected)
    sequence = torch.randn(1, 2, 3, 4, 512)
    valid = torch.ones(1, 2, 4, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 4, 3)
    with torch.inference_mode():
        output = model.forward_from_backbone_sequence(
            sequence, valid, quality, source_time
        )
    assert torch.isfinite(output["logits"]).all()


def test_spatial_branch_is_zero_residual_and_budgeted() -> None:
    model = P86V2Layer3SpatialDedupStudent(frames=4, kinetics_pretrained=False)
    assert torch.count_nonzero(model.layer3_spatial_projection[3].weight) == 0
    assert torch.count_nonzero(model.layer3_spatial_projection[3].bias) == 0
    assert sum(parameter.numel() for parameter in model.parameters()) < 18_500_000


def test_candidate_only_layers_do_not_shift_paired_rng_stream() -> None:
    torch.manual_seed(29)
    P86MC3VisualStudent(frames=4, kinetics_pretrained=False, temporal_modeling=True)
    expected = torch.random.get_rng_state()
    torch.manual_seed(29)
    P86V2TemporalDedupStudent(frames=4, kinetics_pretrained=False)
    actual = torch.random.get_rng_state()
    assert torch.equal(actual, expected)
