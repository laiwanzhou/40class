from scripts.report_x3d_s_single13_fixed_context import decision


def test_fixed_context_result_is_preferred_spatial_candidate() -> None:
    assert (
        decision(
            {
                "accuracy": 0.5376623376623376,
                "macro_f1": 0.4275141328945328,
                "worst_user_accuracy": 0.5174129353233831,
            }
        )
        == "preferred_spatial_candidate"
    )


def test_fixed_context_accuracy_regression_requires_human_review() -> None:
    assert (
        decision(
            {
                "accuracy": 0.50,
                "macro_f1": 0.42,
                "worst_user_accuracy": 0.50,
            }
        )
        == "human_review_regression"
    )
