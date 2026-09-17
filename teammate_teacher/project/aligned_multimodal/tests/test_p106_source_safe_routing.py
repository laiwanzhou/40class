from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from run_p106_source_safe_routing import (
    fit_call_model,
    resolve_calls,
    routing_features,
)


def test_candidate_requires_both_family_classes_in_top5_and_disagreement() -> None:
    probability = np.full((3, 40), 1e-6, dtype=np.float64)
    probability[0, [3, 5, 8, 9, 10]] = [0.35, 0.20, 0.19, 0.14, 0.12]
    probability[1, [3, 8, 9, 10, 11, 5]] = [0.35, 0.20, 0.19, 0.14, 0.11, 0.01]
    probability[2, [3, 5, 8, 9, 10]] = [0.35, 0.20, 0.19, 0.14, 0.12]
    probability /= probability.sum(axis=1, keepdims=True)
    specialist = np.asarray([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])

    features, candidate, a_prediction, specialist_prediction = routing_features(
        probability, specialist, [3, 5]
    )

    assert features.shape == (3, 15)
    assert candidate.tolist() == [True, False, False]
    assert a_prediction.tolist() == [3, 3, 3]
    assert specialist_prediction.tolist() == [5, 5, 3]


def test_call_model_abstains_without_both_decisive_outcomes() -> None:
    features = np.arange(60, dtype=np.float64).reshape(4, 15)
    candidate = np.ones(4, dtype=bool)
    labels = np.asarray([5, 5, 5, 5])
    a_prediction = np.asarray([3, 3, 3, 3])
    specialist_prediction = np.asarray([5, 5, 5, 5])

    scaler, model, audit = fit_call_model(
        features, candidate, labels, a_prediction, specialist_prediction
    )

    assert scaler is None
    assert model is None
    assert audit["authorized"] is False
    assert audit["rescue_examples"] == 4
    assert audit["harm_examples"] == 0


def test_overlapping_calls_choose_highest_probability_and_roster_tie_order() -> None:
    roster = (
        {"family_key": "first"},
        {"family_key": "second"},
    )
    call_masks = {
        "first": np.asarray([True, True, False]),
        "second": np.asarray([True, True, True]),
    }
    probabilities = {
        "first": np.asarray([0.8, 0.7, 0.0]),
        "second": np.asarray([0.9, 0.7, 0.6]),
    }

    routes = resolve_calls(call_masks, probabilities, roster)

    assert routes.tolist() == ["second", "first", "second"]
