from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from analyze_p107_expanded_oracle_bank import (
    SPECIALISTS,
    family_masks,
    family_triggers,
    simulate_oracles,
    system_metrics,
    topk_structure,
)


def test_union_oracle_deduplicates_shared_rescue_and_forced_call_can_harm() -> None:
    labels = np.asarray([7, 7, 24, 3])
    a_prediction = np.asarray([8, 37, 26, 3])
    eligible = family_masks(labels)
    predictions = np.zeros((len(labels), len(SPECIALISTS)), dtype=np.int64)
    predictions[0, 1] = 7
    predictions[0, 2] = 7
    predictions[1, 1] = 8
    predictions[1, 2] = 7
    predictions[2, 4] = 26
    predictions[2, 7] = 24
    predictions[3, 0] = 5

    result = simulate_oracles(labels, a_prediction, eligible, predictions)

    assert result["union"]["selected"].tolist() == [1, 2, 7, -1]
    assert system_metrics(labels, a_prediction, result["union"]["prediction"])["net"] == 3
    forced = system_metrics(labels, a_prediction, result["forced"]["prediction"])
    assert forced["rescue"] == 3
    assert forced["harm"] == 1
    assert forced["net"] == 2


def test_exact_edge_uses_truth_and_a_prediction_not_membership_only() -> None:
    labels = np.asarray([7, 7, 24, 3])
    a_prediction = np.asarray([8, 37, 26, 3])
    eligible = family_masks(labels)
    predictions = np.zeros((len(labels), len(SPECIALISTS)), dtype=np.int64)
    predictions[0, 1] = 7
    predictions[0, 2] = 7
    predictions[1, 1] = 8
    predictions[1, 2] = 7
    predictions[2, 4] = 26
    predictions[2, 7] = 24
    predictions[3, 0] = 5

    result = simulate_oracles(labels, a_prediction, eligible, predictions)

    assert result["exact"]["selected"].tolist() == [2, 1, 4, -1]
    assert system_metrics(labels, a_prediction, result["exact"]["prediction"])["net"] == 1


def test_top3_and_top5_family_triggers_are_unarbitrated() -> None:
    probability = np.full((2, 40), 1e-8, dtype=np.float64)
    probability[0, [7, 8, 37, 9, 3]] = [0.40, 0.25, 0.20, 0.10, 0.05]
    probability[1, [24, 26, 27, 7, 8]] = [0.40, 0.25, 0.20, 0.10, 0.05]
    probability /= probability.sum(axis=1, keepdims=True)
    order, ranks = topk_structure(probability)

    top3, top5 = family_triggers(order, ranks)

    assert top3[0, 1] and top3[0, 2]
    assert top5[0, 1] and top5[0, 2] and top5[0, 3]
    assert top3[1, 4] and top3[1, 7]
    assert top5[1, 4] and top5[1, 7]
