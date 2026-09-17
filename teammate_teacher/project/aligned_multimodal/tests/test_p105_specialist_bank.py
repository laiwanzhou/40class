from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from validate_p105_specialist_bank_oof import (
    SPECIALISTS,
    exact_edge_oracle_assignments,
    route_assignments,
    stable_specialist,
)


def test_top3_router_resolves_overlap_by_alternate_probability() -> None:
    probability = np.zeros((2, 40), dtype=np.float64)
    probability[0, [7, 8, 37]] = [0.60, 0.25, 0.15]
    probability[1, [8, 9, 7]] = [0.60, 0.25, 0.15]

    routes = route_assignments(
        probability,
        list(SPECIALISTS),
        {value["family_key"] for value in SPECIALISTS},
    )

    assert routes.tolist() == ["7__8", "8__9"]


def test_top3_router_never_uses_disabled_specialist() -> None:
    probability = np.zeros((1, 40), dtype=np.float64)
    probability[0, [7, 8, 37]] = [0.60, 0.25, 0.15]

    routes = route_assignments(probability, list(SPECIALISTS), {"7__37"})

    assert routes.tolist() == ["7__37"]


def test_exact_edge_oracle_only_routes_locked_a_errors() -> None:
    labels = np.asarray([3, 7, 8, 8, 9])
    a_prediction = np.asarray([5, 37, 8, 9, 7])

    routes = exact_edge_oracle_assignments(labels, a_prediction, list(SPECIALISTS))

    assert routes.tolist() == ["3__5", "7__37", "", "8__9", ""]


def test_stability_gate_requires_fold_and_subject_support() -> None:
    aggregate = {
        "net": 5,
        "aligned_minus_shuffle_correct": 2,
        "aligned_minus_zero_correct": 7,
        "per_fold_net": {"0": 2, "1": 0, "2": 3, "3": 0},
        "per_subject": {"positive_subjects": 3, "negative_subjects": 2},
    }

    assert stable_specialist(aggregate) is True
    aggregate["per_fold_net"]["2"] = -3
    assert stable_specialist(aggregate) is False
