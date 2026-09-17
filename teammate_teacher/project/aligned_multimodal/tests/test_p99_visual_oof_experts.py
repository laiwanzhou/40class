from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p99_visual_oof_experts import (  # noqa: E402
    focus_group_audit,
    internvideo_matrix,
    pairwise_expert_audit,
    videomae_matrix,
    vjepa_dense24_matrix,
)


def test_visual_feature_recipes_have_frozen_dimensions_and_are_finite() -> None:
    rng = np.random.default_rng(7)
    vmae = rng.normal(size=(3, 2, 3, 768)).astype(np.float32)
    k400 = rng.normal(size=(3, 2, 3, 400)).astype(np.float32)
    dense = rng.normal(size=(3, 24, 1024)).astype(np.float32)
    ssv2 = rng.normal(size=(3, 24, 174)).astype(np.float32)
    matrices = (
        videomae_matrix(vmae),
        internvideo_matrix(vmae, k400),
        vjepa_dense24_matrix(dense, ssv2),
    )
    assert matrices[0].shape == (3, 4608)
    assert matrices[1].shape == (3, 7008)
    assert matrices[2].shape == (3, 9584)
    assert all(np.isfinite(value).all() for value in matrices)


def test_focus_group_audit_counts_rescue_harm_and_top5() -> None:
    labels = np.asarray([1, 2, 3, 1])
    anchor = np.asarray([0, 2, 3, 1])
    probability = np.full((4, 40), 1e-5, dtype=np.float64)
    probability[0, 1] = 1.0
    probability[1, 0] = 1.0
    probability[2, 3] = 1.0
    probability[3, 2] = 1.0
    for row in (1, 3):
        probability[row, labels[row]] = 0.0
        probability[row, 4:9] = 0.5
    value = focus_group_audit(labels, anchor, probability, [1, 2])
    assert value["rows"] == 3
    assert value["anchor_correct"] == 2
    assert value["expert_correct"] == 1
    assert value["rescue"] == 1
    assert value["harm"] == 2
    assert value["expert_top5_correct"] == 1


def test_pairwise_expert_audit_separates_unique_anchor_rescues() -> None:
    labels = np.asarray([0, 1, 2, 3])
    anchor = np.asarray([9, 1, 9, 9])
    first = np.full((4, 40), 1e-6)
    second = np.full((4, 40), 1e-6)
    first[np.arange(4), [0, 8, 2, 8]] = 1.0
    second[np.arange(4), [0, 1, 8, 3]] = 1.0
    value = pairwise_expert_audit(labels, anchor, first, second)
    assert value["anchor_rescue_overlap"] == 1
    assert value["first_unique_anchor_rescue"] == 1
    assert value["second_unique_anchor_rescue"] == 1
    assert value["anchor_rescue_union"] == 3
