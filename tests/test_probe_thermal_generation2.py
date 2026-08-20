from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.probe_thermal_generation2 import (
    ProvenancePolicyError,
    canonicalize_model_batch,
    deployment_ledger,
    scan_student_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/thermal_generation2_environment_probe.json"


class ProvenanceModel(nn.Module):
    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        self.initialization_provenance = {
            "student": "random" if not pretrained else "pretrained",
            "pretrained_student_weights": pretrained,
        }


def test_provenance_scanner_rejects_pretrained_student_and_teacher_asset() -> None:
    scan_student_provenance(ProvenanceModel(pretrained=False), deployment_files=[])

    with pytest.raises(ProvenancePolicyError, match="pretrained"):
        scan_student_provenance(ProvenanceModel(pretrained=True), deployment_files=[])
    with pytest.raises(ProvenancePolicyError, match="teacher"):
        scan_student_provenance(
            ProvenanceModel(pretrained=False),
            deployment_files=[Path("artifacts/teacher_logits.npz")],
        )


def test_deployment_ledger_deduplicates_identical_files(tmp_path: Path) -> None:
    first = tmp_path / "one.bin"
    second = tmp_path / "two.bin"
    first.write_bytes(b"same")
    second.write_bytes(b"same")

    ledger = deployment_ledger([first, second])

    assert ledger["unique_bytes"] == 4
    assert ledger["unique_sha256_count"] == 1
    assert len(ledger["items"]) == 2


def test_model_batch_adapter_only_moves_raster_channel_before_time() -> None:
    raw = {
        "full_rgb": torch.zeros(2, 3, 16, 3, 8, 8),
        "crop_rgb": torch.zeros(2, 3, 16, 3, 8, 8),
        "motion": torch.zeros(2, 3, 16, 1, 8, 8),
        "pose": torch.zeros(2, 3, 16, 56),
    }

    adapted = canonicalize_model_batch(raw)

    assert adapted["full_rgb"].shape == (2, 3, 3, 16, 8, 8)
    assert adapted["crop_rgb"].shape == (2, 3, 3, 16, 8, 8)
    assert adapted["motion"] is raw["motion"]
    assert adapted["pose"] is raw["pose"]


def test_generated_probe_report_passes_all_frozen_gates() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))

    assert payload["stage"] == "A2"
    assert payload["status"] == "passed"
    assert payload["zero_formal_training"] is True
    assert payload["hardware"]["gpu_name"] == "NVIDIA GeForce RTX 5060 Laptop GPU"
    assert payload["input_contract"]["real_thermal_trials"] >= 1
    assert payload["input_contract"]["forbidden_evidence_read"] is False
    assert set(payload["models"]) == {"b_x3d_xs", "a_multistream"}
    for model in payload["models"].values():
        assert model["fp32_inference"]["output_shape"][1] == 40
        assert model["fp32_inference"]["finite"] is True
        assert model["bfloat16_train_smoke"]["finite"] is True
        assert model["bfloat16_train_smoke"]["optimizer_steps"] == 1
        assert model["bfloat16_train_smoke"]["effective_batch_trials"] == 8
        assert model["bfloat16_train_smoke"]["peak_allocated_mib"] < 7300
        assert model["deployment"]["complete_package_bytes"] < 95_000_000
        assert model["deployment"]["pretrained_student_weights"] is False
        assert model["deployment"]["teacher_assets"] == []
    assert payload["projected_training_time"]["b_x3d_xs"]["basis"] == "measured"
    assert payload["projected_training_time"]["a_direct"]["basis"] == "measured"
    assert payload["projected_training_time"]["c1_r2plus1d18"]["basis"] == "heuristic"
    assert payload["measurement_scope"]["latency"] == "model_only_preprocessed_tensors"
    assert payload["measurement_scope"]["online_yolo_latency_measured"] is False
    assert payload["a3_prerequisites"] == {
        "route_b_report_verified": False,
        "full_pose_cache_present": False,
        "a_direct_training_authorized": False,
    }
    assert payload["next_action"] == "route_b_and_pose_cache_before_a3_authorization"
    assert len(payload["provenance"]["repository_head_at_probe"]) == 40
    assert len(payload["provenance"]["probe_script_sha256"]) == 64
    assert len(payload["provenance"]["config_sha256"]) == 2
    assert all(len(value) == 64 for value in payload["provenance"]["config_sha256"].values())
