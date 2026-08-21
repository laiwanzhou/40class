from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.report_thermal_a_direct import build_a_direct_report


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/thermal_a_multistream_direct_train12_val2.json"


def test_report_recomputes_fixed_metrics_and_records_policy() -> None:
    labels = np.asarray([0, 1, 0, 1])
    logits = np.zeros((4, 40), np.float32)
    logits[np.arange(4), [0, 1, 1, 1]] = 3.0
    archive = {
        "sample_ids": np.asarray(["a", "b", "c", "d"]),
        "users": np.asarray(["user6", "user6", "user7", "user7"]),
        "labels": labels,
        "logits": logits,
        "predictions": logits.argmax(1),
        "availability": np.ones((4, 4), np.bool_),
        "quality": np.ones((4, 8), np.float32),
        "stream_norms": np.arange(16, dtype=np.float32).reshape(4, 4),
    }
    manifest = {
        "status": "completed_50_epoch_hard_stop",
        "experiment_id": "thermal_a_multistream_direct_train12_val2",
        "selected_epoch": 4,
        "epochs_completed": 50,
        "teacher_logits_loaded": False,
        "checkpoint_bytes": 123,
        "checkpoint_sha256": "a" * 64,
        "cuda_peak_allocated_mib": 12.0,
        "cuda_peak_reserved_mib": 15.0,
        "total_seconds": 20.0,
    }
    a2 = {
        "models": {
            "a_multistream": {
                "parameters": 100,
                "deployment": {"complete_package_bytes": 1000},
                "fp32_inference": {"latency": {"median_ms_per_trial": 2.0, "p95_ms_per_trial": 3.0}},
            }
        }
    }

    report = build_a_direct_report(manifest=manifest, archive=archive, a2_report=a2)

    assert report["metrics"]["combined"]["accuracy"] == 0.75
    assert set(report["metrics"]["per_user"]) == {"user6", "user7"}
    assert len(report["metrics"]["combined"]["per_class_recall"]) == 40
    assert report["policy"]["teacher_logits_loaded"] is False
    assert report["policy"]["route_b_report_verified"] is False
    assert report["policy"]["route_b_prerequisite_waived_by_user"] is True
    assert report["deployment"]["under_95000000_bytes"] is True


def test_committed_a3_report_is_complete_and_stops_before_a4() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))

    combined = payload["metrics"]["combined"]
    assert payload["stage"] == "A3"
    assert payload["status"] == "completed_stopped_before_a4"
    assert payload["epochs_completed"] == 50
    assert payload["selected_epoch"] == 38
    assert combined["accuracy"] == 0.27320954907161804
    assert combined["macro_f1"] == 0.1896036550647166
    assert combined["worst_user_accuracy"] == 0.22564102564102564
    assert len(combined["per_class_recall"]) == 40
    assert combined["zero_recall_classes"] == 19
    assert set(payload["metrics"]["per_user"]) == {"user6", "user7"}
    assert payload["deployment"]["checkpoint_bytes"] == 15_069_955
    assert payload["deployment"]["under_95000000_bytes"] is True
    assert payload["policy"]["teacher_logits_loaded"] is False
    assert payload["policy"]["heldout4_labels_read"] is False
    assert payload["policy"]["competition_test_read"] is False
    assert payload["policy"]["quarantined_evidence_read"] is False
    assert payload["decision_gate"]["passed"] is False
    assert payload["next_action"] == "stop_for_human_review_before_A4"
