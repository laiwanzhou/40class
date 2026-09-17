from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from audit_p86_teacher_mechanisms import (
    feature_probe_sets,
    fixed_model_perturbations,
    mean_impute_blocks,
)


def test_feature_probe_shapes() -> None:
    features = np.random.default_rng(1).normal(size=(5, 2, 3, 1024)).astype(np.float32)
    probes = feature_probe_sets(features)
    assert probes["early_late_all_views"].shape == (5, 6144)
    assert probes["early_all_views"].shape == (5, 3072)
    assert probes["scene_early_late"].shape == (5, 2048)
    assert probes["scene_person_early_late"].shape == (5, 4096)
    assert np.isfinite(np.concatenate(tuple(probes.values()), axis=1)).all()


def test_mean_imputation_is_fold_local_and_block_exact() -> None:
    values = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    mean = np.full((2, 3, 4), -7.0, dtype=np.float32)
    output = mean_impute_blocks(values, mean, views=(1,))
    assert np.all(output[:, :, 1] == -7.0)
    assert np.array_equal(output[:, :, 0], values[:, :, 0])
    assert np.array_equal(output[:, :, 2], values[:, :, 2])


def test_fixed_perturbations_preserve_shape_and_baseline() -> None:
    values = np.random.default_rng(2).normal(size=(3, 2, 3, 1024)).astype(np.float32)
    mean = values.mean(axis=0)
    conditions = fixed_model_perturbations(values, mean)
    assert set(conditions) == {
        "baseline",
        "drop_scene",
        "drop_person",
        "drop_workspace",
        "drop_early",
        "drop_late",
        "swap_early_late",
        "collapse_early_late",
        "swap_person_workspace",
        "collapse_view_identity",
    }
    assert np.array_equal(conditions["baseline"], values)
    assert np.array_equal(conditions["swap_early_late"][:, 0], values[:, 1])
    assert np.array_equal(conditions["swap_person_workspace"][:, :, 1], values[:, :, 2])
    assert all(condition.shape == values.shape for condition in conditions.values())
