from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_x3d_s_train_vs_oof import (
    deterministic_eval_manifest,
    population_metrics,
    summarize_fold_metrics,
)


def test_deterministic_eval_manifest_routes_only_requested_users_to_val() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "user_id": ["u1", "u2", "heldout"],
            "split": ["train", "train", "val"],
        }
    )
    result = deterministic_eval_manifest(frame, ("u1", "u2"))
    assert result["sample_id"].tolist() == ["a", "b"]
    assert set(result["split"]) == {"val"}
    with pytest.raises(ValueError, match="missing requested users"):
        deterministic_eval_manifest(frame, ("u1", "unknown"))


def test_fold_summary_distinguishes_training_log_train_eval_and_outer_val() -> None:
    summary = summarize_fold_metrics(
        fold=0,
        training_log={"train_accuracy": 0.90, "train_macro_f1": 0.88},
        train_eval={"accuracy": 0.95, "macro_f1": 0.93, "worst_user_accuracy": 0.91},
        outer_val={"accuracy": 0.57, "macro_f1": 0.49, "worst_user_accuracy": 0.53},
        train_count=10,
        val_count=5,
    )
    assert summary["training_log_train_accuracy"] == 0.90
    assert summary["deterministic_train_eval_accuracy"] == 0.95
    assert summary["formal_outer_val_accuracy"] == 0.57
    assert np.isclose(summary["train_to_val_accuracy_gap"], 0.38)
    assert np.isclose(summary["train_to_val_macro_f1_gap"], 0.44)
    assert np.isclose(summary["train_to_val_worst_user_accuracy_gap"], 0.38)


def test_population_metrics_recomputes_macro_f1_after_concatenation() -> None:
    archives = [
        {
            "labels": np.asarray([0, 0, 1]),
            "logits": np.asarray([[2, 0], [2, 0], [2, 0]]),
            "user_ids": np.asarray(["u1", "u1", "u1"]),
        },
        {
            "labels": np.asarray([1]),
            "logits": np.asarray([[0, 2]]),
            "user_ids": np.asarray(["u2"]),
        },
    ]
    metrics = population_metrics(archives)
    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.75
    assert np.isclose(metrics["macro_f1"], (0.8 + 2 / 3) / 2 / 20)
    assert metrics["worst_user_accuracy"] == 2 / 3
