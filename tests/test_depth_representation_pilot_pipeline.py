from __future__ import annotations

from scripts.run_depth_representation_pilot_pipeline import select_representation


def test_selection_filters_meaningful_accuracy_loss_before_macro_f1() -> None:
    rows = [
        {
            "depth_representation": "raw",
            "accuracy": 0.50,
            "macro_f1": 0.45,
            "small_action_macro_f1": 0.40,
            "generalization_gap": 0.20,
            "worst_user_macro_f1": 0.20,
        },
        {
            "depth_representation": "relative",
            "accuracy": 0.48,
            "macro_f1": 0.46,
            "small_action_macro_f1": 0.42,
            "generalization_gap": 0.18,
            "worst_user_macro_f1": 0.22,
        },
        {
            "depth_representation": "raw+relative",
            "accuracy": 0.47,
            "macro_f1": 0.60,
            "small_action_macro_f1": 0.55,
            "generalization_gap": 0.10,
            "worst_user_macro_f1": 0.30,
        },
    ]
    selected, annotated = select_representation(rows, 0.02)
    assert selected == "relative"
    assert [row["accuracy_eligible"] for row in annotated] == [True, True, False]


def test_selection_uses_small_action_f1_as_first_tie_break() -> None:
    rows = [
        {
            "depth_representation": representation,
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "small_action_macro_f1": small_f1,
            "generalization_gap": 0.2,
            "worst_user_macro_f1": 0.2,
        }
        for representation, small_f1 in (("raw", 0.3), ("relative", 0.35))
    ]
    selected, _ = select_representation(rows, 0.02)
    assert selected == "relative"
