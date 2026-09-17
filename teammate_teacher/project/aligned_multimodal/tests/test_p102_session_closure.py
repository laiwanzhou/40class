from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from audit_p102_session_closure import (
    classification_metrics,
    comparison,
    exact_mcnemar_p,
    log_softmax,
    select_final_baseline,
    true_rank,
)


def test_log_softmax_is_normalized() -> None:
    logits = np.asarray([[2.0, 1.0, -1.0], [-2.0, 4.0, 0.5]])
    log_probability = log_softmax(logits)
    assert np.allclose(np.exp(log_probability).sum(axis=1), 1.0)


def test_true_rank_and_metrics() -> None:
    probability = np.asarray(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.1, 0.8, 0.1]], dtype=np.float64
    )
    labels = np.asarray([0, 1, 2])
    users = np.asarray(["u1", "u1", "u2"])
    assert true_rank(probability, labels).tolist() == [1, 2, 2]
    metrics = classification_metrics(probability, labels, users)
    assert metrics["top1_correct"] == 1
    assert metrics["top3_correct"] == 3
    assert metrics["worst_subject"]["user"] == "u2"


def test_comparison_counts_rescue_and_harm() -> None:
    labels = np.asarray([0, 1, 2, 0])
    users = np.asarray(["u1", "u1", "u2", "u2"])
    raw = np.eye(3)[np.asarray([1, 1, 2, 0])]
    candidate = np.eye(3)[np.asarray([0, 1, 0, 0])]
    result = comparison(labels, users, raw, candidate)
    assert result["rescue"] == 1
    assert result["harm"] == 1
    assert result["net"] == 0
    assert exact_mcnemar_p(1, 1) == 1.0


def test_final_selection_uses_frozen_metric_order() -> None:
    systems = {
        "VS_session": {
            "metrics": {"top1_correct": 10, "macro_f1": 0.7, "worst_subject": {"top1": 0.5}, "top5_correct": 12},
            "vs_raw": {"per_subject": {"u1": {"net": 1}, "u2": {"net": -1}}},
        },
        "F3_session": {
            "metrics": {"top1_correct": 11, "macro_f1": 0.6, "worst_subject": {"top1": 0.4}, "top5_correct": 11},
            "vs_raw": {"per_subject": {"u1": {"net": 0}, "u2": {"net": 0}}},
        },
    }
    assert select_final_baseline(systems) == "F3_session"
