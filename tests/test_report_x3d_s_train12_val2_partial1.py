from __future__ import annotations

from scripts.report_x3d_s_train12_val2_partial1 import evaluate_matched_candidate


REFERENCE = {
    "accuracy": 0.5524691358024691,
    "macro_f1": 0.41865149855281436,
    "worst_user_accuracy": 0.45864661654135336,
}


def test_greater_than_two_point_accuracy_regression_requires_review() -> None:
    metrics = dict(REFERENCE)
    metrics["accuracy"] = REFERENCE["accuracy"] - 0.020001

    assert evaluate_matched_candidate(metrics, REFERENCE) == "human_review_regression"


def test_candidate_is_preferred_only_when_all_metrics_are_no_worse() -> None:
    metrics = {
        "accuracy": REFERENCE["accuracy"] + 0.001,
        "macro_f1": REFERENCE["macro_f1"],
        "worst_user_accuracy": REFERENCE["worst_user_accuracy"],
    }
    assert evaluate_matched_candidate(metrics, REFERENCE) == "preferred"

    metrics["worst_user_accuracy"] -= 0.001
    assert evaluate_matched_candidate(metrics, REFERENCE) == "not_preferred"


def test_exactly_two_point_accuracy_regression_is_not_manual_review() -> None:
    metrics = dict(REFERENCE)
    metrics["accuracy"] = REFERENCE["accuracy"] - 0.02

    assert evaluate_matched_candidate(metrics, REFERENCE) == "not_preferred"
