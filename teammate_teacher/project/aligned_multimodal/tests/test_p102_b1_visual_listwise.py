from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from audit_p87_sequence_decoder import DecoderConfig, RecordingMetadata
from audit_p102_session_closure import comparison
from train_p102_b1_visual_listwise_session_oof import (
    candidate_local_probability,
    candidate_targets,
    fit_listwise_directions,
    listwise_scores,
    session_decode_candidate_probability,
)


def test_listwise_mask_excludes_padding_and_finds_target() -> None:
    candidates = np.asarray([[2, 0, -1], [1, 2, 0]], dtype=np.int64)
    labels = np.asarray([0, 1], dtype=np.int64)
    valid, target = candidate_targets(candidates, labels, np.asarray([0, 1]))
    assert valid.tolist() == [[True, True, False], [True, True, True]]
    assert target.tolist() == [1, 0]
    score = listwise_scores(
        torch.zeros((2, 2), dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.float64),
        torch.as_tensor(candidates),
        torch.full((2, 3), 1.0 / 3.0, dtype=torch.float64),
    )
    assert score[0, 2].item() < -1e8


def test_zero_direction_exactly_recovers_candidate_restricted_a() -> None:
    embeddings = np.asarray([[1.0, -1.0], [0.5, 0.25]])
    direction = np.zeros((2, 4), dtype=np.float64)
    candidates = np.asarray([[0, 2, -1], [1, 3, -1]])
    a = np.asarray([[0.5, 0.2, 0.2, 0.1], [0.1, 0.5, 0.1, 0.3]])
    result = candidate_local_probability(
        embeddings, direction, candidates, a, np.asarray([True, True])
    )
    assert np.allclose(result[0, [0, 2]], [5.0 / 7.0, 2.0 / 7.0])
    assert np.allclose(result[1, [1, 3]], [0.625, 0.375])
    assert np.all(result[[0, 1], [1, 0]] == 0.0)
    assert result.argmax(axis=1).tolist() == a.argmax(axis=1).tolist()


def test_listwise_visual_direction_learns_discriminative_signal() -> None:
    embeddings = np.asarray(
        [[-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    candidates = np.tile(np.asarray([[0, 1]], dtype=np.int64), (6, 1))
    probability = np.full((6, 40), 1e-6, dtype=np.float64)
    probability[:, :2] = 0.5 - 19e-6
    direction, audit = fit_listwise_directions(
        embeddings,
        candidates,
        probability,
        labels,
        np.arange(6),
        seed=7,
        l2=0.01,
        max_iter=50,
    )
    result = candidate_local_probability(
        embeddings, direction, candidates, probability, np.ones(6, dtype=bool)
    )
    assert audit["final_unweighted_ce"] < audit["initial_unweighted_ce"]
    assert np.array_equal(result.argmax(axis=1), labels)


def test_session_decode_masks_held_labels_and_never_adds_non_candidate() -> None:
    labels = np.asarray([0, 1, 2, 2], dtype=np.int64)
    source = np.asarray([True, True, False, False])
    held = ~source
    local = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    candidates = np.asarray([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64)
    metadata = RecordingMetadata(
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        users=np.asarray(["u1", "u1", "x", "x"]),
        dates=np.asarray(["2026-01-01"] * 4),
        starts=np.asarray([0.0, 10.0, 0.0, 10.0]),
    )
    result, audit = session_decode_candidate_probability(
        local,
        candidates,
        labels,
        source,
        held,
        metadata,
        DecoderConfig(30.0, 0.3, 1.0, 10),
    )
    assert audit["transition_fit_labels_masked_outside_source"] == 2
    assert audit["final_mass_outside_candidate"] == 0.0
    assert np.all(result[held, 2] == 0.0)
    assert np.allclose(result[held].sum(axis=1), 1.0)


def test_gate_uses_comparison_net_field() -> None:
    labels = np.asarray([0, 1])
    users = np.asarray(["u", "u"])
    a = np.eye(2)[[1, 1]]
    b = np.eye(2)[[0, 0]]
    result = comparison(labels, users, a, b)
    assert result["net"] == 0
