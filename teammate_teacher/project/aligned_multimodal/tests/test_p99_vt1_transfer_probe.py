from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_transfer_probe import NUM_CLASSES, write_target  # noqa: E402
from p99_vt1_transfer_probe import build_target  # noqa: E402


def test_build_target_aligns_rows_without_copying_labels(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.npz"
    ids = np.asarray(["a", "b"])
    users = np.asarray(["user1", "user2"])
    write_target(anchor, ids, users, np.full((2, NUM_CLASSES), 1 / NUM_CLASSES), "anchor")

    probability = np.full((2, NUM_CLASSES), 1e-4, dtype=np.float32)
    probability[0, 7] = 1.0
    probability[1, 3] = 1.0
    probability /= probability.sum(axis=1, keepdims=True)
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        sample_ids=np.asarray(["b", "a"]),
        users=np.asarray(["user2", "user1"]),
        labels=np.asarray([9, 8]),
        direct_probability=probability,
    )
    teacher_summary = tmp_path / "summary.json"
    teacher_summary.write_text(
        json.dumps({"stage": "P99_VT1_H1", "student_gate": {"passed": True}}),
        encoding="utf-8",
    )
    config = {
        "teacher_summary": str(teacher_summary),
        "teacher_predictions": str(predictions),
        "teacher_probability_key": "direct_probability",
        "anchor_control_target": str(anchor),
    }
    manifest = build_target(config, tmp_path / "output")
    assert manifest["source_contains_evaluation_labels"] is True
    assert manifest["labels_read_or_written_to_target"] is False
    with np.load(manifest["target"]["path"], allow_pickle=False) as target:
        assert "labels" not in target.files
        np.testing.assert_array_equal(target["sample_ids"], ids)
        np.testing.assert_array_equal(target["emission_prediction"], [3, 7])
