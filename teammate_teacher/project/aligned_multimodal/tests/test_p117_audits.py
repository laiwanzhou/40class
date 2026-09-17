from __future__ import annotations

import numpy as np

from audit_p117_identifier_leakage_ceiling import (
    parse_identifier_labels,
    required_coverage,
)
from p117_transductive_multicandidate_router import session_context


def test_identifier_control_and_required_coverage() -> None:
    values = parse_identifier_labels(
        np.asarray(["train__c00__user1__x", "train__c39__user9__y"])
    )
    assert values.tolist() == [0, 39]
    assert np.isclose(required_coverage(0.85, 0.97), 0.8)


def test_session_context_is_label_free_and_finite() -> None:
    safe = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float64,
    )
    output = session_context([np.asarray([0, 1, 2])], safe, candidate)
    assert output.shape == (3, 14)
    assert np.isfinite(output).all()
    assert np.all(output[:, 0] == 3)
