from __future__ import annotations

import numpy as np
import torch

from src.models.target16_linear_residual import Target16LinearResidual


def test_zero_initialization_exactly_reproduces_b2_target_logits() -> None:
    target_ids = np.arange(16)
    model = Target16LinearResidual(192, target_ids)
    embeddings = torch.randn(3, 192)
    base_logits = torch.randn(3, 40)
    output = model(embeddings, base_logits)
    assert torch.equal(output["logits"], base_logits[:, :16])
    assert torch.count_nonzero(output["delta_logits"]) == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == 3088


def test_only_residual_head_receives_gradients() -> None:
    model = Target16LinearResidual(192, np.arange(16))
    embeddings = torch.randn(4, 192)
    base_logits = torch.randn(4, 40)
    loss = torch.nn.functional.cross_entropy(model(embeddings, base_logits)["logits"], torch.arange(4))
    loss.backward()
    assert model.residual_head.weight.grad is not None
    assert model.residual_head.bias.grad is not None
