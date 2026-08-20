from scripts.report_x3d_s_ir_depth4 import decision


def test_accuracy_at_point63_is_only_eligible_for_manual_stability_review() -> None:
    assert (
        decision(
            {
                "accuracy": 0.63,
                "macro_f1": 0.52,
                "worst_user_accuracy": 0.58,
            }
        )
        == "eligible_for_manual_stability_review"
    )


def test_improvement_below_point63_does_not_unlock_stability_work() -> None:
    assert (
        decision(
            {
                "accuracy": 0.60,
                "macro_f1": 0.50,
                "worst_user_accuracy": 0.55,
            }
        )
        == "improved_but_below_stability_gate"
    )


def test_more_than_two_point_regression_requires_human_review() -> None:
    assert (
        decision(
            {
                "accuracy": 0.51,
                "macro_f1": 0.40,
                "worst_user_accuracy": 0.49,
            }
        )
        == "human_review_regression"
    )
