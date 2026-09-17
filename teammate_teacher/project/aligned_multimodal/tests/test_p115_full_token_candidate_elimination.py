from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p115_full_token_candidate_elimination import (  # noqa: E402
    FullTokenCandidateScorer,
    PooledCandidateScorer,
    TokenGrid,
    anchored_topk,
    build_protocol,
    conformal_lower_threshold,
    inner_user_groups,
    recursive_eliminate,
)


def test_full_token_scorer_attends_every_spatial_temporal_token_before_score() -> None:
    grid = TokenGrid(windows=2, times=3, views=2, height=2, width=2, channels=8)
    model = FullTokenCandidateScorer(grid=grid, width=12, heads=3, classes=4, dropout=0.0)
    tokens = torch.randn((1, *grid.shape), requires_grad=True)
    candidates = torch.tensor([[1, 3]])
    logits, attention = model(tokens, candidates, return_attention=True)
    assert logits.shape == (1, 2)
    assert attention is not None
    assert attention.shape == (1, 3, 2, grid.token_count)
    assert torch.all(attention > 0)
    logits.sum().backward()
    gradient = tokens.grad.reshape(grid.token_count, grid.channels).abs().sum(dim=1)
    assert torch.all(gradient > 0)
    assert not any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.modules())


def test_pooled_control_cannot_claim_token_attention() -> None:
    model = PooledCandidateScorer(input_dim=12, width=8, classes=4, dropout=0.0)
    logits, attention = model(torch.randn(2, 12), torch.tensor([[0, 2], [1, 3]]))
    assert logits.shape == (2, 2)
    assert attention is None
    with pytest.raises(ValueError, match="no token attention"):
        model(torch.randn(1, 12), return_attention=True)


def test_anchored_topk_places_frozen_p89_first_without_duplicates() -> None:
    probability = np.asarray([[0.40, 0.35, 0.20, 0.05], [0.50, 0.20, 0.15, 0.15]])
    anchor = np.asarray([2, 0])
    result = anchored_topk(probability, anchor, 3)
    assert result.tolist() == [[2, 0, 1], [0, 1, 2]]
    assert all(len(set(row)) == 3 for row in result.tolist())


def test_fixed_conformal_cutoff_is_conservative_under_strict_elimination() -> None:
    values = np.arange(1, 202, dtype=np.float64)
    threshold = conformal_lower_threshold(values, alpha=0.01)
    killed = int(np.sum(values < threshold))
    assert threshold == 2.0
    assert killed / len(values) < 0.01


def test_recursive_elimination_protects_p89_anchor_and_two_candidate_floor() -> None:
    candidates = np.asarray([8, 7, 10, 14, 6])
    scores = np.asarray([0.01, 0.02, 0.03, 0.90, 0.80])
    survivors, eliminated = recursive_eliminate(
        candidates, scores, threshold=0.50, anchor=8, minimum_survivors=2
    )
    assert 8 in survivors
    assert len(survivors) == 3
    assert survivors.tolist() == [8, 14, 6]
    assert eliminated.tolist() == [7, 10]


def test_inner_user_groups_are_disjoint_and_complete() -> None:
    users = np.asarray([f"user{index}" for index in range(1, 13)])
    groups = inner_user_groups(users, count=3)
    flattened = [user for group in groups for user in group]
    assert len(groups) == 3
    assert len(flattened) == len(set(flattened)) == len(users)
    assert set(flattened) == set(users.tolist())


def test_p89_anchor_remains_frozen() -> None:
    protocol = build_protocol()
    assert len(protocol.ids) == 2470
    assert int(np.sum(protocol.safe == protocol.labels)) == 2117
