from __future__ import annotations

import torch

from p88_depth_residual_model import (
    P88DepthResidual,
    assemble_depth_residual_features,
)


def test_depth_residual_is_exactly_zero_at_initialisation() -> None:
    model = P88DepthResidual(input_width=1548, hidden_width=32)
    features = torch.randn(4, 1548)
    assert torch.equal(model(features), torch.zeros(4, 40))


def test_depth_feature_contract() -> None:
    anchor = torch.randn(3, 512)
    depth = torch.randn(3, 512)
    validity = torch.randn(3, 12)
    result = assemble_depth_residual_features(anchor, depth, validity)
    assert result.shape == (3, 1548)
