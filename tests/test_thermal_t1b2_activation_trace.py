from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.diagnostics.activation_trace import (
    ReadOnlyActivationTracer,
    first_amplification,
    state_dict_digest,
)


class _TraceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.block1 = nn.Sequential(nn.Linear(4, 2), nn.BatchNorm1d(2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block1(self.block0(inputs))


def test_read_only_hooks_preserve_logits_and_complete_state() -> None:
    torch.manual_seed(5)
    model = _TraceModel().eval()
    inputs = torch.randn(3, 4)
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    digest_before = state_dict_digest(model.state_dict())

    with torch.inference_mode():
        baseline = model(inputs)
        with ReadOnlyActivationTracer(model, ("block0", "block1")) as tracer:
            traced = model(inputs)

    assert torch.equal(baseline, traced)
    assert [row["point"] for row in tracer.records] == ["block0", "block1"]
    assert all(row["finite"] for row in tracer.records)
    assert state_dict_digest(model.state_dict()) == digest_before
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])


def test_repeated_module_calls_receive_stable_call_suffixes() -> None:
    shared = nn.ReLU()

    class _Repeated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = shared

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.shared(self.shared(value))

    model = _Repeated().eval()
    with torch.inference_mode(), ReadOnlyActivationTracer(model, ("shared",)) as tracer:
        model(torch.tensor([-1.0, 2.0]))

    assert [row["point"] for row in tracer.records] == ["shared#0", "shared#1"]


def test_first_amplification_uses_ordered_rms_ratio() -> None:
    control = [
        {"point": "input", "rms": 2.0},
        {"point": "stage0.block0", "rms": 3.0},
        {"point": "stage0.block1", "rms": 4.0},
    ]
    spike = [
        {"point": "input", "rms": 2.2},
        {"point": "stage0.block0", "rms": 12.0},
        {"point": "stage0.block1", "rms": 48.0},
    ]

    located = first_amplification(spike, control, ratio_threshold=10.0)

    assert located["point"] == "stage0.block1"
    assert located["ratio"] == 12.0
    assert located["previous_point"] == "stage0.block0"
    assert located["previous_ratio"] == 4.0


def test_t1b2_entrypoint_contains_no_training_or_checkpoint_write_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root / "scripts/diagnose_thermal_t1b2_activation_trace.py"
    ).read_text(encoding="utf-8")

    assert "build_optimizer" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "model.train(" not in source
    assert "torch.save(" not in source


def test_committed_t1b2_report_proves_read_only_trace_and_stops() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (project_root / "reports/thermal_t1b2_activation_trace.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "first_abnormal_amplification_localized_training_still_stopped"
    assert not any(report["safety_boundary"].values())
    assert report["checkpoint"]["sha256"] == (
        "ca9c11c0f4d50f67c89da52284f726085dfeb3d1578621438872b9255a05d827"
    )
    assert report["precision_agreement"] == {
        "first_point_by_precision": {
            "fp32": "spatial.backbone.stages.2.5",
            "bfloat16": "spatial.backbone.stages.2.5",
        },
        "same_first_point": True,
    }
    assert all(
        (
            report["integrity"]["all_hooked_logits_exact"],
            report["integrity"]["all_hooked_embeddings_exact"],
            report["integrity"]["state_dict_digest_unchanged"],
            report["integrity"]["every_state_tensor_unchanged"],
            report["integrity"]["checkpoint_file_unchanged"],
        )
    )
    assert all(pair["same_user"] and pair["same_class"] for pair in report["matched_pairs"])
    assert report["decision"]["training_authorized"] is False
