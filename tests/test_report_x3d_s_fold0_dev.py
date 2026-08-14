from __future__ import annotations

import pytest

from scripts.report_x3d_s_fold0_dev import evaluate_candidate


BASELINE = {
    "accuracy": 0.57125,
    "macro_f1": 0.486918,
    "worst_user_accuracy": 0.533835,
}


def test_decision_rule_requires_all_three_targets() -> None:
    assert evaluate_candidate(
        {
            "accuracy": 0.63,
            "macro_f1": 0.52,
            "worst_user_accuracy": 0.5338,
        },
        BASELINE,
    ) == "target_met"

    assert evaluate_candidate(
        {
            "accuracy": 0.64,
            "macro_f1": 0.519,
            "worst_user_accuracy": 0.55,
        },
        BASELINE,
    ) == "continue_stage_a"

    assert evaluate_candidate(
        {
            "accuracy": 0.64,
            "macro_f1": 0.54,
            "worst_user_accuracy": 0.53,
        },
        BASELINE,
    ) == "continue_stage_a"


@pytest.mark.parametrize("accuracy", [0.55, 0.55124])
def test_accuracy_regression_greater_than_two_points_requires_review(accuracy: float) -> None:
    assert evaluate_candidate(
        {
            "accuracy": accuracy,
            "macro_f1": 0.60,
            "worst_user_accuracy": 0.60,
        },
        BASELINE,
    ) == "human_review_regression"


def test_exactly_two_point_regression_does_not_trigger_review() -> None:
    assert evaluate_candidate(
        {
            "accuracy": BASELINE["accuracy"] - 0.02,
            "macro_f1": 0.40,
            "worst_user_accuracy": 0.40,
        },
        BASELINE,
    ) == "continue_stage_a"
