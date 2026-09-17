from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ALIGNED = HERE.parent
if str(ALIGNED) not in sys.path:
    sys.path.insert(0, str(ALIGNED))

from audit_p110_p89_temporal_matching import (  # noqa: E402
    DEFAULT_P89,
    FAMILIES,
    load_p89,
    ordered_distance,
    predict_from_distance,
    set_distance,
    uniform_indices,
)


def test_p89_canonical_contract() -> None:
    p89 = load_p89(DEFAULT_P89)
    assert len(p89["labels"]) == 2470
    assert int(np.sum(p89["labels"] == p89["prediction"])) == 2117
    assert int(np.sum(p89["labels"] != p89["prediction"])) == 353


def test_uniform_indices_include_sequence_endpoints() -> None:
    indices = uniform_indices(32, steps=12)
    assert indices.shape == (12,)
    assert indices[0] == 0
    assert indices[-1] == 31
    assert np.all(indices[1:] >= indices[:-1])


def test_ordered_and_set_matching_prefer_identical_sequence() -> None:
    first = np.eye(4, dtype=np.float32)
    reverse = first[::-1].copy()
    query = first[None]
    reference = np.stack((first, reverse))

    ordered = ordered_distance(query, reference, device="cpu")
    unordered = set_distance(query, reference, device="cpu")

    assert ordered.shape == (1, 2)
    assert ordered[0, 0] < ordered[0, 1]
    assert np.isclose(unordered[0, 0], unordered[0, 1], atol=1e-6)


def test_predict_from_distance_honors_per_query_source_mask() -> None:
    distance = np.asarray(
        [
            [0.0, 0.9, 0.2, 0.3],
            [0.9, 0.0, 0.4, 0.1],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
    masks = [
        np.asarray([False, True, True, True]),
        np.asarray([True, False, True, True]),
    ]
    prediction, margin, scores = predict_from_distance(distance, labels, (1, 2), masks)
    assert prediction.tolist() == [2, 2]
    assert margin.shape == (2,)
    assert scores.shape == (2, 2)


def test_frozen_families_have_only_documented_overlap() -> None:
    memberships: dict[int, list[str]] = {}
    for family_name, classes in FAMILIES.items():
        for class_id in classes:
            memberships.setdefault(class_id, []).append(family_name)
    overlaps = {key: value for key, value in memberships.items() if len(value) > 1}
    assert set(overlaps) == {14}
