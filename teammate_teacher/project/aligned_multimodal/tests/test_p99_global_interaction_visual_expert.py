from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_global_interaction_visual_expert import (  # noqa: E402
    build_selection_gate,
    paired_change,
)


def test_paired_change_reports_exact_discordance() -> None:
    labels = np.asarray([0, 0, 1, 1])
    base = np.asarray([0, 1, 1, 0])
    candidate = np.asarray([0, 0, 0, 0])
    result = paired_change(labels, base, candidate)
    assert result["rescue"] == 1
    assert result["harm"] == 1
    assert result["net"] == 0
    assert result["mcnemar_exact_pvalue"] == 1.0


def test_selection_gate_enforces_user_risk_and_interventions() -> None:
    config = {
        "selection_gate": {
            "minimum_net_gain": 1,
            "minimum_positive_users": 1,
            "minimum_nonnegative_users": 1,
            "maximum_worst_user_accuracy_drop_pp": 1.0,
        }
    }
    labels = np.asarray([0, 1, 0, 1])
    users = np.asarray(["a", "a", "b", "b"])
    global_prediction = np.asarray([1, 1, 0, 0])
    direct_prediction = np.asarray([0, 1, 0, 0])
    gate = build_selection_gate(
        config,
        labels,
        users,
        direct_prediction,
        global_prediction,
        direct_correct=3,
        zero_correct=2,
        shuffle_correct=2,
    )
    assert gate["passed"] is True
    assert gate["positive_users"] == 1
    assert gate["nonnegative_users"] == 2


def test_selection_gate_rejects_worst_user_reversal() -> None:
    config = {
        "selection_gate": {
            "minimum_net_gain": 0,
            "minimum_positive_users": 0,
            "minimum_nonnegative_users": 1,
            "maximum_worst_user_accuracy_drop_pp": 1.0,
        }
    }
    labels = np.asarray([0, 1, 0, 1])
    users = np.asarray(["a", "a", "b", "b"])
    global_prediction = np.asarray([0, 1, 1, 1])
    direct_prediction = np.asarray([0, 0, 0, 1])
    gate = build_selection_gate(
        config,
        labels,
        users,
        direct_prediction,
        global_prediction,
        direct_correct=3,
        zero_correct=2,
        shuffle_correct=2,
    )
    assert gate["checks"]["worst_user_risk"] is False
    assert gate["passed"] is False
