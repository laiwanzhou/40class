from __future__ import annotations

import torch

from src.models.hierarchical_action_query_fusion import (
    HierarchicalActionQueryFusion,
)
from src.models.multimodal_token_contract import GroupTokens


def group_tokens(
    *, batch: int, streams: int, dim: int, available: bool = True
) -> GroupTokens:
    torch.manual_seed(31 + streams)
    tokens = torch.randn(batch, 8, streams, dim, requires_grad=True)
    mask = torch.full((batch, 8, streams), available, dtype=torch.bool)
    if not available:
        tokens = torch.zeros(batch, 8, streams, dim, requires_grad=True)
    quality = torch.zeros(batch, 8, streams, 2)
    return GroupTokens(
        tokens=tokens,
        mask=mask,
        quality=quality,
        quality_mask=torch.zeros_like(quality, dtype=torch.bool),
    )


def test_action_queries_never_attend_to_masked_body_group() -> None:
    model = HierarchicalActionQueryFusion(dim=32, classes=40, heads=4, layers=2)
    visual = group_tokens(batch=2, streams=2, dim=32)
    body = group_tokens(batch=2, streams=1, dim=32, available=False)

    output = model(visual=visual, body=body)

    assert torch.count_nonzero(output["group_attention"][:, :, 2]) == 0
    assert output["logits"].shape == (2, 40)
    assert output["action_features"].shape == (2, 40, 32)


def test_all_available_groups_receive_gradient() -> None:
    model = HierarchicalActionQueryFusion(dim=32, classes=40, heads=4, layers=2)
    visual = group_tokens(batch=2, streams=2, dim=32)
    body = group_tokens(batch=2, streams=1, dim=32)

    output = model(visual=visual, body=body)
    output["logits"].sum().backward()

    assert visual.tokens.grad is not None
    assert visual.tokens.grad[:, :, 0].abs().sum() > 0
    assert visual.tokens.grad[:, :, 1].abs().sum() > 0
    assert body.tokens.grad is not None
    assert body.tokens.grad.abs().sum() > 0


def test_attention_sums_to_one_over_available_tokens() -> None:
    model = HierarchicalActionQueryFusion(dim=32, classes=40, heads=4, layers=2)
    visual = group_tokens(batch=1, streams=2, dim=32)
    body = group_tokens(batch=1, streams=1, dim=32)

    output = model(visual=visual, body=body)

    assert torch.allclose(
        output["group_attention"].sum(dim=2), torch.ones(1, 40), atol=1e-6
    )
    assert torch.allclose(
        output["segment_attention"].sum(dim=2), torch.ones(1, 40), atol=1e-6
    )


def test_all_missing_groups_return_zero_attention_and_core_unavailable() -> None:
    model = HierarchicalActionQueryFusion(dim=32, classes=40, heads=4, layers=2)
    visual = group_tokens(batch=1, streams=2, dim=32, available=False)
    body = group_tokens(batch=1, streams=1, dim=32, available=False)

    output = model(visual=visual, body=body)

    assert not output["core_available"].any()
    assert torch.count_nonzero(output["group_attention"]) == 0
    assert torch.count_nonzero(output["segment_attention"]) == 0
    assert torch.isfinite(output["logits"]).all()
