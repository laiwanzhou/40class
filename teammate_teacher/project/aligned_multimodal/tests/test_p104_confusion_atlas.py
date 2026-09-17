from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from audit_p104_confusion_atlas import connected_components, select_pair_families


def test_pair_family_selection_uses_only_source_rows() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2])
    prediction = np.asarray([1, 1, 0, 0, 1, 0, 3, 3, 3, 3, 3, 3])
    users = np.asarray(["a", "b", "a", "b", "c", "c", "held"] * 1 + ["held"] * 5)
    probability = np.full((len(labels), 4), 0.01, dtype=np.float64)
    probability[np.arange(len(labels)), prediction] = 0.97
    probability /= probability.sum(axis=1, keepdims=True)
    source = np.arange(len(labels)) < 6

    selected, eligible, _ = select_pair_families(
        labels,
        prediction,
        probability,
        users,
        source,
        min_errors=5,
        min_subjects=3,
        max_families=8,
    )

    assert [value["family_key"] for value in selected] == ["0__1"]
    assert eligible[0]["error_rows"] == 6
    assert eligible[0]["subjects"] == 3
    assert all(value["family_key"] != "2__3" for value in eligible)


def test_pair_family_ranking_and_cluster_diagnostic() -> None:
    edges = [
        {"family_key": "8__10", "classes": [8, 10], "error_rows": 12},
        {"family_key": "8__9", "classes": [8, 9], "error_rows": 8},
        {"family_key": "21__22", "classes": [21, 22], "error_rows": 7},
    ]
    clusters = connected_components(edges)

    assert clusters[0]["classes"] == [8, 9, 10]
    assert clusters[0]["benchmark_candidate"] is True
    assert clusters[1]["classes"] == [21, 22]
    assert clusters[1]["benchmark_candidate"] is False
