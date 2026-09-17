from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_depth_anchor_teacher import anchor_probability, geometric_pool  # noqa: E402
from p99_depth_oof_expert import canonical_hash  # noqa: E402
from p99_visual_anchor_teacher import validate_frozen_inputs  # noqa: E402


def fixtures() -> tuple[dict, dict, dict]:
    config = {"visual_expert": "internvideo2", "anchor_confidence": 0.94}
    visual = {"experts": {"internvideo2": {"alpha": 3.0}}}
    v0 = {
        "stage": "P99_V0_H1_visual_single_expert_pool",
        "config_sha256": canonical_hash(visual),
        "experts": {"internvideo2": {}},
    }
    return config, visual, v0


def test_zero_visual_weight_exactly_recovers_anchor() -> None:
    anchor = anchor_probability(np.asarray([3, 7]), 0.94)
    visual = np.full((2, 40), 1.0 / 40)
    np.testing.assert_allclose(geometric_pool(anchor, visual, 0.0), anchor)


def test_h1_validation_freezes_v0_identity() -> None:
    config, visual, v0 = fixtures()
    expected = canonical_hash(
        {"fusion": config, "visual": visual, "visual_expert": "internvideo2"}
    )
    assert validate_frozen_inputs("h1", config, visual, v0, None) == expected


def test_h2_requires_matching_frozen_h1_summary() -> None:
    config, visual, v0 = fixtures()
    combined = validate_frozen_inputs("h1", config, visual, v0, None)
    h1 = {
        "stage": "P99_VT1_H1",
        "config_sha256": combined,
        "selected_visual_expert": "internvideo2",
    }
    assert validate_frozen_inputs("h2_confirmation", config, visual, v0, h1) == combined
    with pytest.raises(ValueError, match="requires"):
        validate_frozen_inputs("h2_confirmation", config, visual, v0, None)
