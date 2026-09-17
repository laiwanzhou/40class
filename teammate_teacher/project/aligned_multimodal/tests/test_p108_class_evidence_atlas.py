from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_p108_class_evidence_atlas import (  # noqa: E402
    A9_USER_SET,
    BLOCK_MODALITY,
    CANDIDATES,
    EXPECTED_TOP10,
    FULL_OUTER_FOLDS,
    SEMANTIC_RULES,
    build_full_fold_ids,
    candidate_modalities,
    evidence_strength,
    select_confusers,
    sum_binary,
)


def test_frozen_scope_and_candidate_panel() -> None:
    assert set(SEMANTIC_RULES) == set(EXPECTED_TOP10)
    assert len(EXPECTED_TOP10) == 10
    assert len(CANDIDATES) == 15
    assert set(value for candidate in CANDIDATES for value in candidate) == set(
        BLOCK_MODALITY
    )
    assert candidate_modalities(("VLIT", "SWT", "IAPD")) == (
        "Visual",
        "Skeleton",
        "IMU",
    )


def test_full_fold_partition_has_three_a9_per_fold() -> None:
    users = np.asarray([value for fold in FULL_OUTER_FOLDS for value in fold])
    fold_ids = build_full_fold_ids(users)
    assert sorted(np.bincount(fold_ids).tolist()) == [6, 6, 6]
    for fold in range(3):
        held = set(users[fold_ids == fold].tolist())
        assert len(held & A9_USER_SET) == 3
        assert len(held - A9_USER_SET) == 3


def test_confuser_selection_ignores_unselected_rows() -> None:
    labels = np.asarray([8, 8, 8, 8, 8, 8])
    prediction = np.asarray([10, 10, 9, 8, 1, 1])
    probability = np.full((6, 40), 1e-4, dtype=np.float64)
    probability[:, 8] = 0.2
    probability[:, 10] = 0.3
    probability[:, 9] = 0.1
    probability[:, 1] = 0.4
    users = np.asarray(["a", "b", "c", "d", "held", "held"])
    selected = np.asarray([True, True, True, True, False, False])
    confusers, audit = select_confusers(
        8, labels, prediction, probability, users, selected
    )
    assert confusers[:2] == [10, 9]
    assert next(row for row in audit if row["class_id"] == 1)["error_rows"] == 0


def test_binary_aggregation_and_strength_are_frozen() -> None:
    combined = sum_binary(
        [
            {"tp": 3, "fp": 1, "fn": 1, "tn": 5},
            {"tp": 2, "fp": 0, "fn": 2, "tn": 6},
        ]
    )
    assert combined["tp"] == 5
    assert combined["fp"] == 1
    assert combined["fn"] == 3
    assert np.isclose(combined["f1"], 10 / 14)
    assert evidence_strength(2, 2, 0.8, 0.6, 0.5, 0.5) == "STRONG"
    assert evidence_strength(1, 2, 0.65, 0.60, 0.55, 0.2) == "MODERATE"
    assert evidence_strength(1, 1, 0.7, 0.72, 0.4, 0.8) == "WEAK/UNRESOLVED"
