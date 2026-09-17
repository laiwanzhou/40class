from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from train_p102_candidate_reranker_oof import (
    masked_moments,
    point_features,
    visual_summary,
)


def test_visual_summary_retains_region_and_signed_time_information() -> None:
    temporal = np.zeros((2, 2, 3, 8, 4), dtype=np.float32)
    action = np.zeros((2, 2, 3, 5), dtype=np.float32)
    temporal[0, 1] = 1.0
    summary = visual_summary(temporal, action)
    assert summary.shape == (2, 4 * 3 * 4 + 2 * 3 * 5)
    assert not np.allclose(summary[0], summary[1])


def test_masked_moments_ignore_unavailable_values() -> None:
    values = np.asarray([[[1.0], [100.0], [3.0]]])
    mask = np.asarray([[1, 0, 1]])
    mean, std = masked_moments(values, mask, axes=(1,))
    assert np.allclose(mean, [[2.0]])
    assert np.allclose(std, [[1.0]])


def test_point_features_only_emit_deployment_candidates() -> None:
    probability = np.asarray([[0.6, 0.3, 0.1]])
    candidates = np.asarray([[0, 2, -1]])
    embeddings = {name: np.asarray([[0.5, -0.5]]) for name in ("videomae", "internvideo", "skeleton", "imu")}
    prototypes = {
        name: (np.zeros((40, 2), dtype=np.float32), np.ones((40, 2), dtype=np.float32))
        for name in embeddings
    }
    matrix, rows, classes, slices = point_features(
        np.asarray([0]), candidates, probability, embeddings, prototypes, np.ones(40)
    )
    assert matrix.shape[0] == 2
    assert rows.tolist() == [0, 0]
    assert classes.tolist() == [0, 2]
    assert 1 not in classes
    assert set(slices) == set(embeddings)

