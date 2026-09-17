from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_depth_oof_expert import (  # noqa: E402
    NUM_CLASSES,
    change_audit,
    class_sample_weights,
    feature_families,
    fit_temperature,
    metrics,
    softmax,
)
from p99_depth_anchor_teacher import (  # noqa: E402
    anchor_probability,
    fit_weight,
    geometric_pool,
)


def test_feature_families_keep_depth_views_separate() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(7, 3, 8)).astype(np.float32)
    action = rng.normal(size=(7, 3, 11)).astype(np.float32)
    result = feature_families(features, action)
    assert result["roi_all"].shape == (7, 24)
    assert result["roi_mean"].shape == (7, 8)
    assert result["k710_logits"].shape == (7, 33)
    assert result["roi_all_plus_k710"].shape == (7, 57)


def test_temperature_and_probabilities_are_finite() -> None:
    rng = np.random.default_rng(5)
    scores = rng.normal(size=(80, NUM_CLASSES))
    labels = np.arange(80) % NUM_CLASSES
    temperature = fit_temperature(scores, labels)
    probability = softmax(scores / temperature)
    assert 0.04 < temperature < 21.0
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_metrics_and_change_audit_use_full_40_class_contract() -> None:
    labels = np.arange(NUM_CLASSES)
    logits = np.full((NUM_CLASSES, NUM_CLASSES), -3.0)
    logits[np.arange(NUM_CLASSES), labels] = 3.0
    value = metrics(logits, labels)
    assert value["correct"] == NUM_CLASSES
    assert value["top5"] == 1.0
    assert np.asarray(value["confusion_matrix"]).shape == (NUM_CLASSES, NUM_CLASSES)
    base = labels.copy()
    candidate = labels.copy()
    candidate[0] = 1
    audit = change_audit(labels, base, candidate)
    assert audit == {
        "rescue": 0,
        "harm": 1,
        "net": -1,
        "changed": 1,
        "oracle_union_correct": NUM_CLASSES,
    }


def test_class_weights_are_mean_normalized() -> None:
    labels = np.repeat(np.arange(NUM_CLASSES), np.arange(1, NUM_CLASSES + 1))
    weights = class_sample_weights(labels, 0.75)
    assert weights.shape == labels.shape
    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] > weights[-1]


def test_geometric_pool_has_exact_anchor_and_depth_endpoints() -> None:
    anchor = anchor_probability(np.asarray([0, 3]), 0.94)
    depth = anchor_probability(np.asarray([2, 3]), 0.70)
    np.testing.assert_allclose(geometric_pool(anchor, depth, 0.0), anchor)
    np.testing.assert_allclose(geometric_pool(anchor, depth, 1.0), depth)


def test_fit_weight_prefers_informative_depth_and_rejects_wrong_depth() -> None:
    labels = np.asarray([0, 1, 2, 3] * 10)
    anchor = anchor_probability(np.zeros(len(labels), dtype=np.int64), 0.55)
    good = anchor_probability(labels, 0.85)
    bad = anchor_probability((labels + 1) % NUM_CLASSES, 0.85)
    good_weight = fit_weight(anchor, good, labels, (0.0, 1.0))
    bad_weight = fit_weight(anchor, bad, labels, (0.0, 1.0))
    assert good_weight > 0.5
    assert bad_weight < 0.3
    assert good_weight > bad_weight
