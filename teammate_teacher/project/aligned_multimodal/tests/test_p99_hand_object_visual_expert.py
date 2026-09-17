from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_hand_object_visual_expert import (  # noqa: E402
    hand_feature_families,
    load_v0_control,
)


def test_hand_feature_families_keep_frozen_semantics() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(4, 2, 3, 768)).astype(np.float32)
    families = hand_feature_families(source)
    assert families["hand_all"].shape == (4, 4608)
    assert families["mean_windows"].shape == (4, 2304)
    assert families["interaction_only"].shape == (4, 1536)
    assert np.isfinite(families["hand_all"]).all()
    np.testing.assert_allclose(
        np.linalg.norm(families["interaction_only"].reshape(4, 2, 768), axis=-1),
        1.0,
        rtol=1e-5,
        atol=1e-5,
    )


def test_hand_feature_families_reject_wrong_layout() -> None:
    with pytest.raises(ValueError, match="full/peak"):
        hand_feature_families(np.zeros((2, 3, 768), dtype=np.float32))


def test_v0_control_requires_p99_stage(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"stage": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="P99-V0"):
        load_v0_control(path)


def test_v0_control_extracts_only_frozen_comparison(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {
        "stage": "P99_V0_H1_visual_single_expert_pool",
        "experts": {
            "videomaev2_early_late": {
                "metrics": {"correct": 10, "top5": 0.8},
                "vs_anchor": {"rescue": 3, "harm": 4},
                "extended_audit": {"per_user_change": {"u": {"rescue": 1}}},
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    control = load_v0_control(path)
    assert control == {
        "name": "videomaev2_early_late",
        "correct": 10,
        "top5": 0.8,
        "rescue": 3,
        "harm": 4,
        "per_user": {"u": {"rescue": 1}},
    }
