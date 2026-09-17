from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_vt1_h2_confirmation import build_targets, gate_side  # noqa: E402


def test_joint_gate_requires_significance_and_cross_user_direction() -> None:
    gate = {
        "minimum_net": 1,
        "minimum_positive_users": 3,
        "minimum_nonnegative_users": 4,
        "maximum_worst_user_regression_pp": 1.0,
        "maximum_paired_exact_pvalue": 0.05,
    }
    passed = gate_side(
        {"net": 5, "mcnemar_exact_pvalue": 0.04},
        {"u1": 1, "u2": 1, "u3": 1, "u4": 0, "u5": -1},
        0.60,
        0.595,
        gate,
    )
    assert passed["passed"] is True
    failed = gate_side(
        {"net": 5, "mcnemar_exact_pvalue": 0.06},
        {"u1": 1, "u2": 1, "u3": 1, "u4": 0, "u5": -1},
        0.60,
        0.595,
        gate,
    )
    assert failed["passed"] is False
    assert failed["checks"]["paired_exact"] is False


def test_h2_targets_do_not_copy_evaluation_labels(tmp_path: Path) -> None:
    summary = tmp_path / "teacher_summary.json"
    summary.write_text(json.dumps({"stage": "P99_VT1_H2_confirmation"}), encoding="utf-8")
    predictions = tmp_path / "teacher_predictions.npz"
    probability = np.full((2, 40), 1e-4, dtype=np.float32)
    probability[0, 7] = 1.0
    probability[1, 3] = 1.0
    probability /= probability.sum(axis=1, keepdims=True)
    np.savez_compressed(
        predictions,
        sample_ids=np.asarray(["b", "a"]),
        users=np.asarray(["user2", "user1"]),
        labels=np.asarray([9, 8]),
        anchor_probability=probability,
        direct_probability=probability,
    )
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "subject_holdout_predictions.csv").write_text(
        "sample_id,user_id,label,prediction\na,user1,8,3\nb,user2,9,7\n", encoding="utf-8"
    )
    checkpoint = base_dir / "unified_student.pt"
    checkpoint.write_bytes(b"test")
    config = {"base_checkpoint": str(checkpoint)}
    manifest = build_targets(config, tmp_path / "output", summary, predictions)
    assert manifest["source_contains_evaluation_labels"] is True
    assert manifest["labels_read_or_written_to_targets"] is False
    for target in manifest["targets"].values():
        with np.load(target["path"], allow_pickle=False) as written:
            assert "labels" not in written.files
            np.testing.assert_array_equal(written["sample_ids"], ["a", "b"])
            np.testing.assert_array_equal(written["emission_prediction"], [3, 7])
