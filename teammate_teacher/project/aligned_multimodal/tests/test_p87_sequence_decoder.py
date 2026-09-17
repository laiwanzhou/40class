from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from audit_p87_sequence_decoder import (
    RecordingMetadata,
    build_sessions,
    decode_unique_beam,
    decode_unique_beam_posterior,
    fit_transition_model,
)


def test_sessions_split_on_gap_and_support_anonymous_dates() -> None:
    metadata = RecordingMetadata(
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        users=np.asarray(["u1", "u2", "u1", "u1"]),
        dates=np.asarray(["day", "day", "day", "day"]),
        starts=np.asarray([1.0, 2.0, 10.0, np.nan]),
    )
    known = build_sessions(np.arange(4), metadata, 5.0, grouping="known_user")
    anonymous = build_sessions(np.arange(4), metadata, 5.0, grouping="anonymous_date")
    assert [session.tolist() for session in known] == [[0], [2], [1]]
    assert [session.tolist() for session in anonymous] == [[0, 1], [2]]


def test_transition_model_is_finite_and_normalized() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 3])
    sessions = [np.asarray([0, 1, 2]), np.asarray([3, 4, 5])]
    model = fit_transition_model(
        labels, sessions, num_classes=4, trigram_backoff=2.0
    )
    assert np.isfinite(model.start_log_probability).all()
    assert np.isfinite(model.bigram_log_probability).all()
    assert np.isfinite(model.trigram_log_probability).all()
    assert np.allclose(np.exp(model.bigram_log_probability).sum(axis=1), 1.0)
    assert np.allclose(np.exp(model.trigram_log_probability).sum(axis=2), 1.0)


def test_unique_beam_prevents_duplicate_labels() -> None:
    labels = np.asarray([0, 1, 2])
    model = fit_transition_model(
        labels, [np.arange(3)], num_classes=3, trigram_backoff=2.0
    )
    emission = np.log(
        np.asarray(
            [
                [0.90, 0.08, 0.02],
                [0.80, 0.19, 0.01],
                [0.05, 0.10, 0.85],
            ]
        )
    )
    prediction = decode_unique_beam(
        emission, model, transition_weight=0.0, beam_width=20
    )
    assert prediction.tolist() == [0, 1, 2]
    assert len(set(prediction.tolist())) == len(prediction)


def test_beam_posterior_is_normalized_and_matches_map_path() -> None:
    labels = np.asarray([0, 1, 2])
    model = fit_transition_model(
        labels, [np.arange(3)], num_classes=3, trigram_backoff=2.0
    )
    emission = np.log(
        np.asarray(
            [
                [0.55, 0.40, 0.05],
                [0.45, 0.50, 0.05],
                [0.05, 0.10, 0.85],
            ]
        )
    )
    posterior = decode_unique_beam_posterior(
        emission,
        model,
        transition_weight=0.0,
        beam_width=20,
        posterior_temperature=1.0,
    )
    prediction = decode_unique_beam(
        emission, model, transition_weight=0.0, beam_width=20
    )
    assert np.allclose(posterior.path_probability.sum(), 1.0)
    assert np.allclose(posterior.marginals.sum(axis=1), 1.0)
    assert posterior.paths[0].tolist() == prediction.tolist()
    assert posterior.marginals.shape == emission.shape


def test_beam_posterior_temperature_must_be_positive() -> None:
    model = fit_transition_model(
        np.asarray([0]), [np.asarray([0])], num_classes=2, trigram_backoff=2.0
    )
    try:
        decode_unique_beam_posterior(
            np.log(np.asarray([[0.6, 0.4]])),
            model,
            transition_weight=0.0,
            beam_width=2,
            posterior_temperature=0.0,
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected a ValueError for non-positive temperature")
