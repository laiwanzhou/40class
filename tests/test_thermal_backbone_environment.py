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
    IFORMER_CHECKPOINT_SHA256,
    IFORMER_REPOSITORY,
    IFORMER_REVISION,
    build_deployment_ledger,
    load_official_iformer_checkpoint,
    require_complete_pretrained_load,
    require_file_sha256,
    validate_candidate,
)
from src.models.thermal_tsm import IFormerTSM, MobileNetV3SmallTSM, TemporalShift


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeIFormer(nn.Module):
    def __init__(self, *, emits_tuple: bool = False) -> None:
        super().__init__()
        self.use_bn = True
        self.last_proj = False
        self.downsample_layers = nn.ModuleList(
            [
                nn.Conv2d(3, 8, 3, stride=2, padding=1),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.Conv2d(16, 24, 3, stride=2, padding=1),
                nn.Conv2d(24, 32, 3, stride=2, padding=1),
            ]
        )
        first_stage = _TupleStage() if emits_tuple else nn.Identity()
        self.stages = nn.ModuleList(
            [first_stage, nn.Identity(), nn.Identity(), nn.Identity()]
        )
        self.classifier = nn.Linear(32, 40)


class _TupleStage(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = torch.ones(1, device=x.device, dtype=x.dtype)
        return x, auxiliary


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


def test_official_iformer_checkpoint_requires_hash_and_model_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "iFormer_t.pth"
    torch.save({"model": {"weight": torch.ones(2, 2)}, "optimizer": {}}, checkpoint)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    state = load_official_iformer_checkpoint(checkpoint, expected)

    assert list(state) == ["weight"]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_official_iformer_checkpoint(checkpoint, "0" * 64)

    invalid = tmp_path / "invalid.pth"
    torch.save({"optimizer": {}}, invalid)
    invalid_hash = hashlib.sha256(invalid.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="model state"):
        load_official_iformer_checkpoint(invalid, invalid_hash)


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
        "source_url": IFORMER_REPOSITORY,
        "license": "MIT",
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


def test_iformer_tsm_forward_contract_without_downloading_weights() -> None:
    model = IFormerTSM(
        backbone=_FakeIFormer(),
        num_classes=40,
        num_segments=16,
        fold_div=8,
        shift_before_stages=(0, 1, 2, 3),
    ).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 16, 3, 224, 224))

    assert logits.shape == (2, 40)
    assert torch.isfinite(logits).all()


def test_iformer_tsm_preserves_official_tuple_stage_state() -> None:
    model = IFormerTSM(
        backbone=_FakeIFormer(emits_tuple=True),
        num_classes=40,
        num_segments=16,
        fold_div=8,
    ).eval()

    with torch.inference_mode():
        logits = model(torch.zeros(2, 16, 3, 224, 224))

    assert logits.shape == (2, 40)
    assert torch.isfinite(logits).all()


def test_iformer_source_identity_is_the_mobile_iclr_family() -> None:
    assert IFORMER_REPOSITORY == "https://github.com/ChuanyangZheng/iFormer"
    assert IFORMER_REVISION == "2a87540fcb345afe9d950a58d0eb3873b938c3dc"
    assert IFORMER_CHECKPOINT_SHA256 == (
        "7cbd778e3604694eb1a0becbf2e6a22798586f6bb46610a5c22b39880efb967e"
    )


def test_qualified_iformer_t_is_paused_only_on_t1b_branch() -> None:
    mobile_config_path = (
        PROJECT_ROOT
        / "configs/experiments/thermal_mobilenetv3_tsm_train12_val2.yaml"
    )
    mobile_config = yaml.safe_load(mobile_config_path.read_text(encoding="utf-8"))

    assert mobile_config["status"] == "eligible_after_human_approval"
    assert mobile_config["backbone"]["pretrained_weights"] == "IMAGENET1K_V1"

    iformer_config_path = (
        PROJECT_ROOT / "configs/experiments/thermal_iformer_t_tsm_train12_val2.yaml"
    )
    iformer_config = yaml.safe_load(iformer_config_path.read_text(encoding="utf-8"))
    assert iformer_config["stage"] == "thermal_t1b_development"
    assert (
        iformer_config["status"]
        == "t1b2_zero_training_trace_complete_waiting_human_decision"
    )
    assert iformer_config["training_authorized"] is False
    assert iformer_config["resume_requires_new_human_approval"] is True
    assert iformer_config["t1b1"]["classifier_bn_root_cause_confirmed"] is False
    assert iformer_config["t1b1"]["bn_free_short_run_authorized"] is False
    assert iformer_config["t1b2"]["state_dict_or_bn_buffer_updated"] is False
    assert iformer_config["t1b2"]["epoch18_resume_authorized"] is False
    assert iformer_config["t1b2"]["head_change_authorized"] is False
    assert iformer_config["num_classes"] == 40
    assert iformer_config["backbone"]["family"] == "ChuanyangZheng_iFormer_t"
    assert iformer_config["backbone"]["source_revision"] == IFORMER_REVISION
    assert iformer_config["backbone"]["weight_sha256"] == IFORMER_CHECKPOINT_SHA256
    assert iformer_config["backbone"]["strict_pretrained_load"] is True
    assert iformer_config["temporal"]["num_segments"] == 16
    assert iformer_config["temporal"]["fold_div"] == 8
    iformer_s_config_path = (
        PROJECT_ROOT / "configs/experiments/thermal_iformer_s_tsm_train12_val2.yaml"
    )
    iformer_s_config = yaml.safe_load(
        iformer_s_config_path.read_text(encoding="utf-8")
    )
    assert iformer_s_config["status"] == "conditional_not_authorized"
    assert iformer_s_config["training_authorized"] is False
    assert iformer_s_config["backbone"]["family"] == "ChuanyangZheng_iFormer_s"


def test_corrective_probe_report_freezes_iformer_identity_and_no_training() -> None:
    report = json.loads(
        (PROJECT_ROOT / "reports/thermal_backbone_environment_probe.json").read_text(
            encoding="utf-8"
        )
    )
    candidate = report["candidates"]["iformer_t_tsm"]

    assert report["scientific_baseline_commit"] == (
        "c42bb43091c79903e5fde5655c2846c87305895a"
    )
    assert report["training_performed"] is False
    assert report["labels_or_evidence_read"] is False
    assert candidate["qualification"] == "eligible_primary_t1a_after_human_approval"
    assert candidate["pretrained_loading_audit_passed"] is True
    assert candidate["random_initialization_fallback_allowed"] is False
    assert candidate["provenance"]["source_url"] == IFORMER_REPOSITORY
    assert candidate["provenance"]["official_constructor_auto_downloads_weights"] is False
    assert candidate["provenance"]["missing_keys"] == []
    assert candidate["provenance"]["unexpected_keys"] == []
    assert candidate["forward"]["output_shape"] == [2, 40]


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
