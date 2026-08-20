from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from src.train_skeleton_c1_balanced_seed_ensemble_strict_oof import (
    balanced_sample_weights,
    make_balanced_sampler,
)


def test_e2_config_changes_only_the_training_sampler_contract() -> None:
    e1 = yaml.safe_load(
        Path("configs/experiments/skeleton_c1_seed_ensemble_strict_oof.yaml").read_text(encoding="utf-8")
    )
    e2 = yaml.safe_load(
        Path("configs/experiments/skeleton_c1_balanced_seed_ensemble_strict_oof.yaml").read_text(
            encoding="utf-8"
        )
    )
    ignored = {"output_root", "report_dir", "sampling"}

    for key, value in e1.items():
        if key not in ignored:
            assert e2[key] == value
    assert e2["sampling"] == {
        "strategy": "inverse_sqrt_class_frequency",
        "replacement": True,
        "num_samples": "train_scope_size",
    }


def test_balanced_sample_weights_use_train_scope_inverse_sqrt_frequency() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 2, 2], dtype=np.int64)

    weights, counts = balanced_sample_weights(labels, num_classes=4)

    np.testing.assert_array_equal(counts, np.asarray([4, 1, 2, 0]))
    np.testing.assert_allclose(
        weights,
        np.asarray([0.5, 0.5, 0.5, 0.5, 1.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]),
    )


def test_balanced_sampler_is_seed_reproducible_and_draws_scope_size() -> None:
    weights = np.asarray([0.5, 0.5, 1.0, 1.0], dtype=np.float64)

    first = list(make_balanced_sampler(weights, seed=123))
    second = list(make_balanced_sampler(weights, seed=123))
    different = list(make_balanced_sampler(weights, seed=124))

    assert first == second
    assert len(first) == len(weights)
    assert first != different
    assert all(0 <= index < len(weights) for index in first)


def test_balanced_sample_weights_reject_invalid_labels() -> None:
    for labels in (
        np.asarray([], dtype=np.int64),
        np.asarray([0, -1], dtype=np.int64),
        np.asarray([0, 4], dtype=np.int64),
    ):
        with np.testing.assert_raises(ValueError):
            balanced_sample_weights(labels, num_classes=4)

    with np.testing.assert_raises(ValueError):
        make_balanced_sampler(np.asarray([1.0, 0.0]), seed=1)
