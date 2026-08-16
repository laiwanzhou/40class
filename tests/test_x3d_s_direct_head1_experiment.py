from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = ROOT / "configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_direct_head1.yaml"
REFERENCE_CONFIG = ROOT / "configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_partial2.yaml"


def _reporter():
    try:
        return importlib.import_module(
            "scripts.report_x3d_s_train12_val2_user6_user7_direct_head1"
        )
    except ModuleNotFoundError:
        pytest.fail("Direct-Head reporter has not been implemented")


def _metrics(*, accuracy: float, macro_f1: float, worst: float) -> dict:
    return {
        "sample_count": 4,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "worst_user_accuracy": worst,
        "worst_user_id": "user6",
        "per_user_accuracy": {"user6": 0.5, "user7": 0.5},
        "duration_buckets": {
            "<=13": {"sample_count": 1, "accuracy": 1.0, "macro_f1": 0.025},
            "14-32": {"sample_count": 1, "accuracy": 0.0, "macro_f1": 0.0},
            "33-64": {"sample_count": 1, "accuracy": 1.0, "macro_f1": 0.025},
            ">64": {"sample_count": 1, "accuracy": 0.0, "macro_f1": 0.0},
        },
    }


def _write_predictions(path: Path, predictions: list[int]) -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    logits = np.full((4, 40), -8.0, dtype=np.float32)
    logits[np.arange(4), np.asarray(predictions)] = -0.01
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        labels=labels,
        logits=logits,
        user_ids=np.asarray(["user6", "user6", "user7", "user7"]),
        num_frames=np.asarray([10, 20, 40, 80], dtype=np.int64),
    )


def test_candidate_changes_only_the_composite_head() -> None:
    candidate = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    reference = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))

    assert candidate["head_type"] == "direct"
    assert candidate["embedding_dim"] == 2048
    candidate.pop("head_type")
    candidate["embedding_dim"] = 256
    assert candidate == reference
    assert "backbone_block_lrs" not in candidate["optimizer"]
    assert candidate["optimizer"]["backbone_lr"] == pytest.approx(3e-5)
    assert candidate["optimizer"]["head_lr"] == pytest.approx(3e-4)
    assert candidate["training"]["warmup_epochs"] == 2
    assert candidate["training"]["scheduler_horizon_epochs"] == 20
    assert candidate["temporal"]["train_views_per_window"] == 1
    assert candidate["temporal"]["max_clips"] == 8
    assert candidate["output_root"] == "outputs/x3d_s_ir_context_train12_val2_dev"


def test_direct_head_decision_contract_is_three_way() -> None:
    reporter = _reporter()
    reference = {
        "accuracy": 0.5324675324675324,
        "macro_f1": 0.42157462519220035,
        "worst_user_accuracy": 0.5323383084577115,
    }

    assert reporter.evaluate_direct_head_candidate(
        {**reference, "accuracy": reference["accuracy"] - 0.020001}, reference
    ) == "human_review_regression"
    assert reporter.evaluate_direct_head_candidate(
        {**reference, "accuracy": 0.56}, reference
    ) == "preferred"
    assert reporter.evaluate_direct_head_candidate(reference, reference) == "non_winning_ablation"


def test_direct_classifier_relative_l2_drift_uses_same_shape_initialization() -> None:
    reporter = _reporter()
    initial = {
        "classifier.weight": torch.tensor([[1.0, 2.0]]),
        "classifier.bias": torch.tensor([1.0]),
    }
    trained = {
        "classifier.weight": torch.tensor([[2.0, 2.0]]),
        "classifier.bias": torch.tensor([1.0]),
    }

    assert reporter._relative_l2_drift(initial, trained, prefix="classifier.") == pytest.approx(
        1.0 / np.sqrt(6.0)
    )


def test_build_report_emits_matched_direct_head_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reporter = _reporter()
    candidate_dir = tmp_path / "candidate"
    reference_dir = tmp_path / "reference"
    candidate_dir.mkdir()
    reference_dir.mkdir()
    _write_predictions(candidate_dir / "val_predictions_best_accuracy.npz", [0, 1, 2, 4])
    _write_predictions(reference_dir / "val_predictions_best_accuracy.npz", [0, 4, 2, 3])

    candidate_metrics = _metrics(accuracy=0.75, macro_f1=0.075, worst=0.5)
    reference_metrics = _metrics(accuracy=0.75, macro_f1=0.075, worst=0.5)
    standalone = {
        "schema_version": 1,
        "role": "train12_val2_matched_reference_report",
        "run_id": candidate_dir.name,
        "metrics": candidate_metrics,
        "training_log_at_selected_epoch": {
            "train_accuracy": 0.95,
            "train_macro_f1": 0.90,
            "train_minus_val_accuracy": 0.20,
        },
        "resources": {"checkpoint_bytes": 101, "prediction_archive_bytes": 102,
                      "peak_cuda_memory_bytes": 103,
                      "ir_route_serialized_weight_subtotal": 104},
    }
    reference_report = {
        "role": "train12_val2_matched_reference_report",
        "run_id": reference_dir.name,
        "metrics": reference_metrics,
        "training_log_at_selected_epoch": {
            "train_accuracy": 0.90,
            "train_macro_f1": 0.80,
            "train_minus_val_accuracy": 0.15,
        },
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference_report), encoding="utf-8")
    (candidate_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "head_type": "direct",
                "embedding_dim": 2048,
                "custom_head_parameter_count": 81960,
                "checkpoint_bytes": {"best_accuracy": 101},
                "prediction_archive_bytes": {"best_accuracy": 102},
                "peak_cuda_memory_bytes": 103,
                "ir_route_serialized_weight_subtotal": 104,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(reporter, "build_standalone_report", lambda _: standalone)
    monkeypatch.setattr(reporter, "_direct_classifier_drift", lambda _: 0.25)

    report = reporter.build_report(
        candidate_dir,
        reference_report_path=reference_path,
        reference_run_directory=reference_dir,
    )

    assert report["head_replacement"] == "composite_projector_and_classifier_to_direct_classifier"
    assert report["model_contract"] == {
        "head_type": "direct",
        "embedding_dim": 2048,
        "custom_head_parameter_count": 81960,
    }
    assert report["train_to_validation_gap"]["accuracy"] == pytest.approx(0.20)
    assert report["train_to_validation_gap"]["macro_f1"] == pytest.approx(0.825)
    assert report["train_to_validation_gap_delta"]["accuracy"] == pytest.approx(0.05)
    assert set(report["per_user_delta"]) == {"user6", "user7"}
    assert set(report["per_user_delta"]["user6"]) == {"accuracy", "macro_f1"}
    assert set(report["duration_delta"]["<=13"]) == {"accuracy", "macro_f1"}
    assert report["matched_prediction_diagnostics"]["prediction_disagreement_count"] == 2
    assert report["matched_prediction_diagnostics"]["candidate_confidence"]["nll"] >= 0.0
    assert report["direct_classifier_relative_l2_drift_vs_initialization"] == 0.25
    assert report["resources"]["checkpoint_bytes"] == 101
    assert report["resources"]["prediction_archive_bytes"] == 102
    assert report["resources"]["peak_cuda_memory_bytes"] == 103
    assert report["resources"]["ir_route_serialized_weight_subtotal"] == 104


def test_preregistration_binds_frozen_reference_and_rejects_historical_split() -> None:
    path = ROOT / "reports/x3d_s_train12_val2_user6_user7_direct_head1_preregistration.json"
    preregistration = json.loads(path.read_text(encoding="utf-8"))

    assert preregistration["run_id"] == (
        "x3d_s_ir_context_train12_val2_user6_user7_direct_head1_seed20260715"
    )
    assert preregistration["expected_custom_head_parameter_count"] == 81960
    assert preregistration["human_review_accuracy_floor"] == pytest.approx(
        0.5124675324675324
    )
    assert preregistration["matched_reference"]["validation_user_ids"] == ["user6", "user7"]
    assert preregistration["historical_user21_user22_report_is_matched_reference"] is False
    assert set(preregistration["decision_contract"]) == {
        "human_review_regression",
        "preferred",
        "non_winning_ablation",
    }
