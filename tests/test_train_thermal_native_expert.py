from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.train_thermal_native_expert import (
    T1BRecipe,
    checkpoint_rank,
    development_membership,
    masked_trial_cross_entropy,
    thermal_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_t1b_recipe() -> None:
    recipe = T1BRecipe()

    assert recipe.epochs == 30
    assert recipe.batch_size == 4
    assert recipe.gradient_accumulation == 4
    assert recipe.backbone_lr == 3e-5
    assert recipe.head_lr == 3e-4
    assert recipe.label_smoothing == 0.1
    assert recipe.seed == 20260715
    assert recipe.num_workers == 0


def test_loss_masks_unavailable_trials_and_is_finite() -> None:
    logits = torch.zeros(3, 40, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    availability = torch.tensor([True, False, True])

    loss = masked_trial_cross_entropy(logits, labels, availability, label_smoothing=0.1)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert torch.count_nonzero(logits.grad[[0, 2]]) > 0


def test_checkpoint_order_is_macro_then_accuracy_then_worst_user_then_epoch() -> None:
    base = {"macro_f1": 0.3, "accuracy": 0.4, "worst_user_accuracy": 0.2}

    assert checkpoint_rank({**base, "macro_f1": 0.31}, 9) > checkpoint_rank(base, 1)
    assert checkpoint_rank({**base, "accuracy": 0.41}, 9) > checkpoint_rank(base, 1)
    assert checkpoint_rank({**base, "worst_user_accuracy": 0.21}, 9) > checkpoint_rank(base, 1)
    assert checkpoint_rank(base, 1) > checkpoint_rank(base, 2)


def test_metrics_fix_macro_labels_to_all_40_and_report_each_user() -> None:
    labels = np.asarray([0, 1, 0, 1])
    predictions = np.asarray([0, 1, 1, 1])
    users = np.asarray(["user6", "user6", "user7", "user7"])

    metrics = thermal_metrics(labels, predictions, users)

    assert metrics["per_class_support"][:2] == [2, 2]
    assert len(metrics["per_class_f1"]) == 40
    assert set(metrics["per_user"]) == {"user6", "user7"}
    assert metrics["worst_user_accuracy"] == min(
        row["accuracy"] for row in metrics["per_user"].values()
    )


def test_development_membership_is_exact_and_never_contains_sealed_users() -> None:
    split = json.loads(
        (PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json").read_text(
            encoding="utf-8"
        )
    )

    train, validation = development_membership(split, audit)

    assert {row["user_id"] for row in train} == set(split["train_user_ids"])
    assert {row["user_id"] for row in validation} == {"user6", "user7"}
    assert len(train) == 2039
    assert len(validation) == 388
    assert sum(bool(row["usable"]) for row in train) == 1922
    assert sum(bool(row["usable"]) for row in validation) == 377
    assert not ({"user4", "user17", "user23", "user24"} & {row["user_id"] for row in train + validation})
