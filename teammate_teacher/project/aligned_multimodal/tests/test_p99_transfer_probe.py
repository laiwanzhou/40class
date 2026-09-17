from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_transfer_probe import (  # noqa: E402
    NUM_CLASSES,
    normalize_probability,
    probability_metrics,
    rescue_harm,
    write_target,
)


def test_write_target_excludes_ground_truth(tmp_path: Path) -> None:
    ids = np.asarray(["a", "b"])
    users = np.asarray(["user1", "user2"])
    probability = np.full((2, NUM_CLASSES), 1.0 / NUM_CLASSES)
    path = tmp_path / "target.npz"
    result = write_target(path, ids, users, probability, "unit_test")
    assert result["rows"] == 2
    with np.load(path, allow_pickle=False) as source:
        assert "labels" not in source.files
        assert source["target_mask"].all()
        np.testing.assert_allclose(source["emission_probability"].sum(axis=1), 1.0)


def test_probability_metrics_and_rescue_harm() -> None:
    labels = np.arange(NUM_CLASSES)
    probability = np.full((NUM_CLASSES, NUM_CLASSES), 1e-3)
    probability[np.arange(NUM_CLASSES), labels] = 1.0
    value = probability_metrics(probability, labels)
    assert value["correct"] == NUM_CLASSES
    assert value["top5"] == 1.0
    base = labels.copy()
    base[:2] = 3
    candidate = labels.copy()
    candidate[2] = 4
    assert rescue_harm(labels, base, candidate) == {
        "rescue": 2,
        "harm": 1,
        "net": 1,
        "changed": 3,
    }


def test_probability_normalization_rejects_negative_values() -> None:
    values = np.ones((3, NUM_CLASSES))
    values[0, 0] = -1.0
    try:
        normalize_probability(values)
    except ValueError as error:
        assert "invalid" in str(error)
    else:
        raise AssertionError("negative target probability was accepted")
