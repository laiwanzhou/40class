from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.diagnostics.block_attribution import (
    ReadOnlyResidualAttributor,
    classify_segment_source,
    tensor_attribution,
)
from src.diagnostics.activation_trace import state_dict_digest


class _Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False),
            nn.Conv2d(3, 6, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(6, 3, 1, bias=False),
        )
        self.gamma = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.m(value)


def test_block_attributor_decomposes_branch_without_changing_output_or_state() -> None:
    torch.manual_seed(3)
    residual = _Residual().eval()
    inputs = torch.randn(16, 3, 8, 8)
    digest_before = state_dict_digest(residual.state_dict())

    with torch.inference_mode():
        baseline = residual(inputs)
        with ReadOnlyResidualAttributor(residual) as tracer:
            traced = residual(inputs)
        report = tracer.report()

    assert torch.equal(traced, baseline)
    assert report["input"]["shape"] == [16, 3, 8, 8]
    assert report["project_conv_bn"]["shape"] == [16, 3, 8, 8]
    assert report["residual_identity_max_abs_error"] < 1e-6
    assert report["raw_project_to_actual_branch_max_abs_error"] < 1e-6
    assert len(report["skip_branch_cosine"]["per_segment"]) == 16
    assert state_dict_digest(residual.state_dict()) == digest_before


def test_tensor_attribution_reports_channel_spatial_and_segment_concentration() -> None:
    values = torch.zeros(16, 4, 10, 10)
    values[7, 2, 4, 5] = 20.0

    report = tensor_attribution(values)

    assert report["segment_energy"]["top1_share"] == 1.0
    assert report["segment_energy"]["source"] == "single_segment"
    assert report["per_segment"][7]["channel_energy"]["top1_share"] == 1.0
    assert report["per_segment"][7]["spatial_energy"]["top1pct_share"] == 1.0


def test_segment_source_thresholds_are_frozen() -> None:
    assert classify_segment_source([0.6, *([0.4 / 15] * 15)]) == "single_segment"
    assert classify_segment_source([0.3, 0.25, 0.2, *([0.25 / 13] * 13)]) == "few_segments"
    assert classify_segment_source([1.0 / 16] * 16) == "distributed_sequence"


def test_t1b3_entrypoint_has_no_training_or_checkpoint_write_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/diagnose_thermal_t1b3_block_attribution.py"
    ).read_text(encoding="utf-8")

    assert "build_optimizer" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "model.train(" not in source
    assert "torch.save(" not in source


def test_committed_t1b3_report_is_read_only_and_uses_full_spike_cohort() -> None:
    report = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "reports/thermal_t1b3_block_attribution.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "block_level_attribution_complete_training_still_stopped"
    assert not any(report["safety_boundary"].values())
    assert report["cohort"]["spike_count"] == 23
    assert report["cohort"]["class36_count"] == 12
    assert report["control_matching"]["tier_counts"] == {
        "same_user_same_class": 23
    }
    assert report["conclusion"]["primary_mechanism"] == (
        "finetuning_induced_convolution_branch_amplification_with_extreme_tail"
    )
    assert report["conclusion"]["pretrained_vs_epoch16"] == (
        "absent_in_pretrained_emerges_after_finetuning"
    )
    assert all(
        (
            report["integrity"]["all_hooked_logits_exact"],
            report["integrity"]["all_hooked_embeddings_exact"],
            report["integrity"]["pretrained_state_unchanged"],
            report["integrity"]["epoch16_state_unchanged"],
            report["integrity"]["checkpoint_file_unchanged"],
        )
    )
    assert report["decision"]["single_variable_experiment_authorized"] is False
