from __future__ import annotations

import numpy as np

from scripts.report_thermal_a_direct import build_a_direct_report


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
