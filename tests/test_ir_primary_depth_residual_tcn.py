from __future__ import annotations

import torch

from src.models.ir_primary_depth_residual_tcn import IRPrimaryDepthResidualTCN


def inputs() -> tuple[torch.Tensor, ...]:
    ir = torch.randn(2, 4, 4, 1, 64, 64)
    depth = torch.randn(2, 4, 2, 3, 64, 64)
    ir_valid = torch.ones(2, 4, 4, dtype=torch.bool)
    depth_valid = torch.ones(2, 4, 2, dtype=torch.bool)
    confidence = torch.ones(2, 4, 4)
    temporal = torch.ones(2, 4, dtype=torch.bool)
    return ir, depth, ir_valid, depth_valid, confidence, temporal


def test_forward_shapes_and_ir_biased_depth_gate() -> None:
    model = IRPrimaryDepthResidualTCN(pretrained=False, channels=32, embedding_dim=32)
    output = model(*inputs())
    assert output["logits"].shape == (2, 40)
    assert output["route_logits"].shape == (2, 2)
    assert output["ir_roi_attention"].shape == (2, 4, 3)
    assert torch.allclose(output["depth_gate"], torch.full((2, 4), 0.1), atol=1e-5)


def test_all_invalid_local_views_have_zero_attention() -> None:
    values = list(inputs())
    values[2][:, :, 1:] = False
    values[3][:, :, 1] = False
    values[4][:, :, 1:] = 0.0
    model = IRPrimaryDepthResidualTCN(pretrained=False, channels=32, embedding_dim=32)
    output = model(*values)
    assert torch.equal(output["ir_roi_attention"], torch.zeros_like(output["ir_roi_attention"]))
    assert torch.equal(output["depth_relation_gate"], torch.zeros_like(output["depth_relation_gate"]))
    assert torch.isfinite(output["logits"]).all()
