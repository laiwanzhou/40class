from __future__ import annotations

import numpy as np

from audit_p87_sequence_decoder import RecordingMetadata
from p88_sequence_decoder import (
    RepeatConfig,
    _contained_short_sessions,
    _pair_score,
    _repeat_groups,
)


def test_contained_sessions_split_short_inside_medium_block() -> None:
    metadata = RecordingMetadata(
        sample_ids=np.asarray([f"s{i}" for i in range(6)]),
        users=np.asarray(["hidden"] * 6),
        dates=np.asarray(["2026-01-01"] * 6),
        starts=np.asarray([0.0, 10.0, 70.0, 80.0, 250.0, 260.0]),
    )
    blocks = _contained_short_sessions(
        np.arange(6), metadata, short_gap_seconds=30.0, medium_gap_seconds=120.0
    )
    assert [[values.tolist() for values in block] for block in blocks] == [
        [[0, 1], [2, 3]],
        [[4, 5]],
    ]


def test_repeat_groups_require_probability_and_path_agreement() -> None:
    sessions = [np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4, 5])]
    probability = np.asarray(
        [
            [0.90, 0.08, 0.02],
            [0.02, 0.90, 0.08],
            [0.85, 0.10, 0.05],
            [0.04, 0.88, 0.08],
            [0.02, 0.08, 0.90],
            [0.88, 0.08, 0.04],
        ]
    )
    log_probability = np.log(probability)
    prediction = probability.argmax(axis=1)
    config = RepeatConfig(180.0, 0.5, 0.8, 0.8, 3)
    groups = _repeat_groups(sessions, log_probability, prediction, config)
    assert len(groups) == 1
    assert [session.tolist() for session in groups[0]] == [[0, 1], [2, 3]]


def test_pair_score_is_one_for_identical_inputs() -> None:
    probability = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    similarity, overlap = _pair_score(
        probability, probability, np.asarray([0, 1]), np.asarray([0, 1])
    )
    assert np.isclose(similarity, 1.0)
    assert overlap == 1.0
