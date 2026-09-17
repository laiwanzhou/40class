from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_transfer_probe import NUM_CLASSES, write_target  # noqa: E402
from p99_visual_transfer_probe import (  # noqa: E402
    build_targets,
    focus_group_transfer_audit,
    paired_exact_pvalue,
    probability_key,
)


def test_probability_key_is_stable() -> None:
    assert probability_key("internvideo2") == "internvideo2_direct_probability"


def test_paired_and_focus_group_audits() -> None:
    labels = np.asarray([0] * 8 + [1] * 3)
    control = np.asarray([1] * 5 + [0] * 3 + [1] * 3)
    candidate = np.asarray([0] * 5 + [1] * 3 + [1] * 3)
    assert paired_exact_pvalue(labels, control, candidate) == 0.7265625
    audit = focus_group_transfer_audit(labels, control, candidate, {"zero": [0]})
    assert audit["zero"] == {
        "rows": 8,
        "anchor_correct": 3,
        "student_correct": 5,
        "rescue": 5,
        "harm": 3,
    }


def test_build_targets_reorders_rows_and_never_copies_labels(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.npz"
    ids = np.asarray(["a", "b"])
    users = np.asarray(["user1", "user2"])
    probability = np.full((2, NUM_CLASSES), 1.0 / NUM_CLASSES)
    write_target(anchor, ids, users, probability, "anchor")
    visual = tmp_path / "visual.npz"
    visual_probability = np.full((2, NUM_CLASSES), 1e-4, dtype=np.float32)
    visual_probability[0, 7] = 1.0
    visual_probability[1, 3] = 1.0
    visual_probability /= visual_probability.sum(axis=1, keepdims=True)
    np.savez_compressed(
        visual,
        sample_ids=np.asarray(["b", "a"]),
        users=np.asarray(["user2", "user1"]),
        labels=np.asarray([9, 8]),
        expert_direct_probability=visual_probability,
    )
    config = {
        "visual_predictions": str(visual),
        "anchor_control_target": str(anchor),
        "targets": ["expert"],
    }
    manifest = build_targets(config, tmp_path / "output")
    assert manifest["source_contains_evaluation_labels"] is True
    assert manifest["labels_read_or_written_to_targets"] is False
    with np.load(manifest["targets"]["expert"]["path"], allow_pickle=False) as target:
        assert "labels" not in target.files
        np.testing.assert_array_equal(target["sample_ids"], ids)
        np.testing.assert_array_equal(target["emission_prediction"], [3, 7])
