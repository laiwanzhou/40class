from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ALIGNED = HERE.parent
if str(ALIGNED) not in sys.path:
    sys.path.insert(0, str(ALIGNED))

from a18_full_teacher_data import (  # noqa: E402
    A18_ROWS,
    A18_SOURCE_USERS,
    A18_SOURCE_USER_SET,
    load_a18_data,
)
from audit_a18_teacher import confusion_rows, true_rank  # noqa: E402


def test_a18_full_refit_contract_has_no_split() -> None:
    data = load_a18_data()
    assert len(data.sample_ids) == A18_ROWS == 2914
    assert len(A18_SOURCE_USERS) == len(A18_SOURCE_USER_SET) == 18
    assert set(data.users.tolist()) == A18_SOURCE_USER_SET
    assert np.all(data.fold_ids == -1)
    assert np.array_equal(data.full_indices(), np.arange(A18_ROWS))
    summary = data.summary()
    assert summary["split"] == "none_full_refit"
    assert summary["validation_rows"] == 0
    assert summary["h3_rows_selected"] == 0
    assert summary["historical_40class_expert_probability_loaded"] is False


def test_a18_recipe_is_single_and_frozen() -> None:
    config = json.loads(
        (ALIGNED / "configs/a18_full_teacher.json").read_text(encoding="utf-8")
    )
    assert config["variant"] == "VS"
    assert config["training"]["epochs"] == 24
    assert config["training"]["seed"] == 20260823
    assert config["session"] == {
        "gap_seconds": 30.0,
        "transition_weight": 0.3,
        "trigram_backoff": 1.0,
        "beam_width": 50,
        "posterior_temperature": 1.0,
        "selection": "single frozen P102-validated recipe; no A18 grid or held-label selection",
    }


def test_fresh_error_helpers_use_only_given_predictions() -> None:
    probability = np.asarray(
        [
            [0.1, 0.6, 0.2, 0.1],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([2, 1, 3], dtype=np.int64)
    assert true_rank(probability, labels).tolist() == [2, 2, 1]
    names = {value: str(value) for value in range(4)}
    directed, pairs = confusion_rows(
        labels,
        probability.argmax(axis=1),
        np.asarray(["u1", "u2", "u3"]),
        names,
    )
    assert [(row["true_class"], row["predicted_class"]) for row in directed] == [
        (2, 1),
        (1, 2),
    ]
    assert pairs[0]["class_a"] == 1
    assert pairs[0]["class_b"] == 2
    assert pairs[0]["bidirectional_errors"] == 2
