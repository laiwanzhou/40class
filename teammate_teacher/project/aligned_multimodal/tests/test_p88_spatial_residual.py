from __future__ import annotations

import torch

from p88_spatial_residual_model import P88Layer3SpatialResidual


def test_spatial_residual_is_zero_initialised() -> None:
    model = P88Layer3SpatialResidual(width=32)
    descriptor = torch.randn(2, 2, 3, 3, 4, 256)
    quality = torch.ones(2, 2, 3)
    delta, auxiliary = model(descriptor, quality)
    assert torch.equal(delta, torch.zeros(2, 40))
    assert torch.equal(auxiliary, torch.zeros(2, 40))
