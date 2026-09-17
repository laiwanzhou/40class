from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_pretrained_skeleton_experts import (  # noqa: E402
    hdgcn_matrix,
    motionbert_matrix,
    selection_gate,
)


def test_pretrained_skeleton_matrices_have_frozen_shapes() -> None:
    motionbert = motionbert_matrix(np.ones((3, 9216), dtype=np.float32))
    hdgcn = hdgcn_matrix(np.ones((3, 6, 256), dtype=np.float32))
    assert motionbert.shape == (3, 9216)
    assert hdgcn.shape == (3, 1536)
    np.testing.assert_allclose(
        np.linalg.norm(hdgcn.reshape(3, 6, 256), axis=-1), 1.0, atol=1e-6
    )


def test_selection_gate_requires_rescue_in_every_user() -> None:
    config = {
        "selection_gate": {
            "minimum_accuracy": 0.45,
            "minimum_anchor_rescue": 3,
            "minimum_users_with_rescue": 3,
        }
    }
    result = {
        "metrics": {"accuracy": 0.5, "correct": 50},
        "zero_metrics": {"correct": 10},
        "shuffle_metrics": {"correct": 9},
        "vs_anchor": {"rescue": 4},
        "extended_audit": {
            "per_user_change": {
                "a": {"rescue": 1},
                "b": {"rescue": 1},
                "c": {"rescue": 2},
                "d": {"rescue": 0},
            }
        },
    }
    gate = selection_gate(config, result)
    assert gate["users_with_rescue"] == 3
    assert gate["checks"]["minimum_users_with_rescue"] is True
    assert gate["checks"]["every_user_rescue"] is False
    assert gate["passed"] is False
