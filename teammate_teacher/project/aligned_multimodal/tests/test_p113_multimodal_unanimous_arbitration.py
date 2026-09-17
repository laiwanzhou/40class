from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from audit_p113_multimodal_unanimous_arbitration import (  # noqa: E402
    transition,
    unanimous_prediction,
)


def test_unanimous_prediction_only_changes_four_way_agreement() -> None:
    base = np.asarray([1, 1, 1, 1])
    predictions = np.asarray(
        [
            [2, 2, 2, 1],
            [2, 2, 1, 1],
            [2, 2, 2, 1],
            [2, 1, 2, 1],
        ]
    )
    candidate, changed = unanimous_prediction(predictions, base)
    assert candidate.tolist() == [2, 1, 1, 1]
    assert changed.tolist() == [True, False, False, False]


def test_transition_separates_rescue_and_harm() -> None:
    labels = np.asarray([2, 1, 2, 1])
    base = np.asarray([1, 1, 1, 1])
    candidate = np.asarray([2, 2, 1, 1])
    result = transition(labels, base, candidate)
    assert result == {"routes": 4, "changes": 2, "rescue": 1, "harm": 1, "net": 0}
