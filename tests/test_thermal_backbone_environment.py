from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from scripts.probe_thermal_backbones import (
    DeploymentAsset,
    build_deployment_ledger,
    require_complete_pretrained_load,
    require_file_sha256,
    validate_candidate,
)
from src.models.thermal_tsm import MobileNetV3SmallTSM, TemporalShift


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_temporal_shift_moves_channels_without_wrapping() -> None:
    values = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8, 1, 1)

    shifted = TemporalShift(num_segments=4, fold_div=4)(values)

    assert torch.equal(shifted[:, :-1, :2], values[:, 1:, :2])
    assert torch.count_nonzero(shifted[:, -1, :2]) == 0
    assert torch.equal(shifted[:, 1:, 2:4], values[:, :-1, 2:4])
    assert torch.count_nonzero(shifted[:, 0, 2:4]) == 0
    assert torch.equal(shifted[:, :, 4:], values[:, :, 4:])


def test_temporal_shift_never_crosses_trial_boundaries() -> None:
    first = torch.ones(1, 4, 8, 1, 1)
    second = torch.full((1, 4, 8, 1, 1), 100.0)
    shifted = TemporalShift(num_segments=4, fold_div=4)(
        torch.cat([first, second], dim=0)
    )

    assert shifted[0].max().item() <= 1.0
    assert shifted[1, :, 4:].min().item() == 100.0
    assert shifted[1, :-1, :2].min().item() == 100.0


def test_temporal_shift_rejects_wrong_segment_count() -> None:
    with pytest.raises(ValueError, match="num_segments"):
        TemporalShift(num_segments=16)(torch.zeros(2, 15, 8, 2, 2))


def test_strict_pretrained_load_rejects_partial_state() -> None:
    module = nn.Linear(3, 2)
    partial = {"weight": module.weight.detach().clone()}

    with pytest.raises(ValueError, match="incomplete pretrained"):
        require_complete_pretrained_load(module, partial)


def test_weight_hash_gate_rejects_replaced_cache_file(tmp_path: Path) -> None:
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"replaced")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_file_sha256(weight, "0" * 64)


def test_candidate_validation_rejects_missing_provenance_and_bad_output() -> None:
    with pytest.raises(ValueError, match="provenance"):
        validate_candidate(
            provenance={"source_url": "", "license": "", "weight_sha256": ""},
            logits=torch.zeros(2, 40),
            num_classes=40,
            package_bytes=1,
        )

    with pytest.raises(ValueError, match="finite"):
        validate_candidate(
            provenance={
                "source_url": "https://example.invalid/source",
                "license": "BSD-3-Clause",
                "weight_sha256": "a" * 64,
            },
            logits=torch.full((2, 40), float("nan")),
            num_classes=40,
            package_bytes=1,
        )


def test_candidate_validation_enforces_head_and_package_ceiling() -> None:
    provenance = {
        "source_url": "https://example.invalid/source",
        "license": "BSD-3-Clause",
        "weight_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="40-class"):
        validate_candidate(provenance, torch.zeros(2, 39), 39, 1)
    with pytest.raises(ValueError, match="95,000,000"):
        validate_candidate(provenance, torch.zeros(2, 40), 40, 95_000_000)


def test_deployment_ledger_counts_identical_asset_once(tmp_path: Path) -> None:
    weight = tmp_path / "shared.pt"
    weight.write_bytes(b"shared-weight")
    duplicate_reference = DeploymentAsset.from_file("shared-yolo-again", weight)
    ledger = build_deployment_ledger(
        [
            DeploymentAsset.from_file("shared-yolo", weight),
            duplicate_reference,
            DeploymentAsset(
                name="thermal",
                identity="thermal-state",
                serialized_bytes=23,
                sha256="b" * 64,
            ),
        ]
    )

    assert ledger["unique_asset_count"] == 2
    assert ledger["deduplicated_reference_count"] == 1
    assert ledger["total_serialized_bytes"] == weight.stat().st_size + 23


def test_iformer_s_headroom_gate_is_strict() -> None:
    provenance = {
        "source_url": "https://github.com/sail-sg/iFormer",
        "license": "Apache-2.0",
        "weight_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="headroom"):
        validate_candidate(
            provenance,
            torch.zeros(2, 40),
            40,
            93_000_001,
            minimum_headroom_bytes=2_000_000,
        )


def test_mobilenet_tsm_forward_contract_without_downloading_weights() -> None:
    model = MobileNetV3SmallTSM(weights=None, num_classes=40, num_segments=16)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 16, 3, 224, 224))

    assert logits.shape == (2, 40)
    assert torch.isfinite(logits).all()


def test_only_eligible_mobilenet_candidate_has_a_t1a_config() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs/experiments/thermal_mobilenetv3_tsm_train12_val2.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["status"] == "eligible_after_human_approval"
    assert config["backbone"]["pretrained_weights"] == "IMAGENET1K_V1"
    assert config["temporal"]["num_segments"] == 16
    assert config["temporal"]["fold_div"] == 8
    assert config["num_classes"] == 40
    assert config["training_authorized"] is False
    assert not (
        PROJECT_ROOT / "configs/experiments/thermal_iformer_t_tsm_train12_val2.yaml"
    ).exists()
    assert not (
        PROJECT_ROOT / "configs/experiments/thermal_iformer_s_tsm_train12_val2.yaml"
    ).exists()


def test_thermal_split_sidecar_freezes_original_split_and_metric_labels() -> None:
    split_path = (
        PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json"
    )
    sidecar = json.loads(
        (
            PROJECT_ROOT / "reports/thermal_train12_val2_split_sidecar_audit.json"
        ).read_text(encoding="utf-8")
    )

    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == sidecar[
        "source_split_sha256"
    ]
    assert sidecar["source_split_modified"] is False
    assert sidecar["pair_audit"]["candidate_two_user_pairs"] == 91
    assert sidecar["pair_audit"]["qualifying_pairs"] == [["user6", "user7"]]
    assert sidecar["t1b_checkpoint_rule"]["macro_f1_labels"] == list(range(40))
    assert len(sidecar["low_support_class_ids"]) == 10
    assert sidecar["reporting_rules"]["final_candidate_retention_basis"] == (
        "shared_train14_oof"
    )
    assert sidecar["training_performed"] is False
