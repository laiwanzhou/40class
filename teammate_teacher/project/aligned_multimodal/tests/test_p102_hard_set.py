from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from analyze_p102_hard_set import (
    CandidateRecipe,
    build_candidate_ids,
    confusion_neighbors,
)


def test_confusion_neighbors_require_cross_subject_stability() -> None:
    labels = np.asarray([1, 1, 2, 2, 3])
    prediction = np.asarray([0, 0, 0, 0, 0])
    users = np.asarray(["u1", "u2", "u1", "u1", "u3"])
    recipe = CandidateRecipe(2, 5, 2, min_neighbor_subjects=2, min_neighbor_count=2)
    neighbors, records = confusion_neighbors(
        labels, prediction, users, np.arange(len(labels)), recipe
    )
    assert neighbors[0] == [1]
    assert records[0]["neighbor_true"] == 1


def test_candidate_builder_never_inserts_oracle_label() -> None:
    probability = np.asarray([[0.6, 0.3, 0.1, 0.0]])
    neighbors = {0: [2], 1: [], 2: [], 3: []}
    recipe = CandidateRecipe(base_k=2, max_size=3, neighbors_per_anchor=1)
    candidates = build_candidate_ids(probability, neighbors, recipe)
    assert candidates.tolist() == [[0, 1, 2]]
    # An unrelated true label cannot appear unless it is in deployment-visible
    # Top-K or the source confusion graph.
    assert 3 not in candidates[0]


def test_candidate_builder_respects_full_base_capacity() -> None:
    probability = np.asarray([[0.6, 0.3, 0.1]])
    neighbors = {0: [2], 1: [2], 2: []}
    recipe = CandidateRecipe(base_k=2, max_size=2, neighbors_per_anchor=2)
    assert build_candidate_ids(probability, neighbors, recipe).tolist() == [[0, 1]]
