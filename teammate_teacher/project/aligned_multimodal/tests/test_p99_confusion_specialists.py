from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_confusion_specialists import full_probability, prior_probability  # noqa: E402


def test_prior_is_restricted_to_frozen_candidate_classes() -> None:
    probability = prior_probability(np.asarray([21, 21, 22]), [21, 22], 2)
    assert probability.shape == (2, 40)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert np.all(probability[:, 21] > probability[:, 22])
    assert np.all(probability[:, 0] < 1e-8)


def test_full_probability_preserves_original_class_ids() -> None:
    class Dummy:
        classes_ = np.asarray([19, 24, 26])

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.1, 0.2, 0.7]]), (len(values), 1))

    probability = full_probability(Dummy(), np.zeros((3, 2)))
    np.testing.assert_array_equal(probability.argmax(axis=1), [26, 26, 26])
