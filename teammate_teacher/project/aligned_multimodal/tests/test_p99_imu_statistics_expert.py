from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_imu_statistics_expert import (  # noqa: E402
    DEVICE_COUNT,
    DEVICE_FEATURES,
    PAIR_AND_DURATION_FEATURES,
    full_probability,
    non_spectral_feature_indices,
)


def test_non_spectral_selection_preserves_phase_pairs_and_duration() -> None:
    selected = non_spectral_feature_indices()
    assert len(selected) == 4502
    assert len(np.unique(selected)) == len(selected)
    assert selected.min() == 0
    assert selected.max() == DEVICE_COUNT * DEVICE_FEATURES + PAIR_AND_DURATION_FEATURES - 1
    # Four spectral statistics are removed from each of 36 signals x 5 devices.
    assert 5222 - len(selected) == 4 * 36 * 5


def test_full_feature_contract_has_expected_spectral_delta() -> None:
    selected = non_spectral_feature_indices()
    full = np.arange(5222)
    removed = np.setdiff1d(full, selected)
    assert len(removed) == 720
    # The pairwise coordination and duration tail is never removed.
    assert np.all(np.isin(np.arange(5070, 5222), selected))


def test_full_probability_restores_missing_classes() -> None:
    class Dummy:
        classes_ = np.asarray([2, 9])

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.1, 0.9]]), (len(values), 1))

    probability = full_probability(Dummy(), np.zeros((2, 3)))
    assert probability.shape == (2, 40)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    np.testing.assert_array_equal(probability.argmax(axis=1), [9, 9])
