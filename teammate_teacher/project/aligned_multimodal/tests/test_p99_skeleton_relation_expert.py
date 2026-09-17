from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_skeleton_relation_expert import (  # noqa: E402
    full_probability,
    masked_robust_statistics,
    relation_descriptor,
)


def test_masked_statistics_ignore_invalid_values() -> None:
    values = np.asarray([[[1.0], [100.0], [3.0]]])
    valid = np.asarray([[[True], [False], [True]]])
    statistics = masked_robust_statistics(values, valid)
    assert statistics.shape == (1, 15)
    assert statistics[0, 0] == 2.0
    assert np.isfinite(statistics).all()


def test_relation_descriptor_is_finite_and_hand_focused() -> None:
    skeleton = np.ones((2, 2, 16, 17, 13), dtype=np.float32)
    feature_mask = np.ones_like(skeleton, dtype=bool)
    relations = np.ones((2, 2, 16, 18), dtype=np.float32)
    relation_mask = np.ones_like(relations, dtype=bool)
    quality = np.ones((2, 2, 16), dtype=np.float32)
    descriptor, audit = relation_descriptor(
        skeleton,
        feature_mask,
        relations,
        relation_mask,
        quality,
        [0, 8, 10, 11, 12, 13, 14, 15, 16],
        [0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16, 17],
    )
    assert descriptor.shape == (2, 4313)
    assert audit["arm_signal_count"] == 108
    assert audit["relation_signal_count"] == 15
    assert audit["all_finite"] is True


def test_global_context_is_a_structural_addition() -> None:
    skeleton = np.ones((1, 2, 16, 17, 13), dtype=np.float32)
    feature_mask = np.ones_like(skeleton, dtype=bool)
    relations = np.ones((1, 2, 16, 18), dtype=np.float32)
    relation_mask = np.ones_like(relations, dtype=bool)
    quality = np.ones((1, 2, 16), dtype=np.float32)
    descriptor, audit = relation_descriptor(
        skeleton,
        feature_mask,
        relations,
        relation_mask,
        quality,
        [0, 8, 10, 11, 12, 13, 14, 15, 16],
        [0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16, 17],
        include_global_context=True,
    )
    assert descriptor.shape == (1, 6548)
    assert audit["include_global_context"] is True
    assert audit["global_context_signal_count"] == 71


def test_full_probability_restores_missing_classes() -> None:
    class Dummy:
        classes_ = np.asarray([1, 7])

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.25, 0.75]]), (len(values), 1))

    probability = full_probability(Dummy(), np.zeros((3, 2)))
    assert probability.shape == (3, 40)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    np.testing.assert_array_equal(probability.argmax(axis=1), [7, 7, 7])
