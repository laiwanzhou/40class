from __future__ import annotations

import torch

from src.data.ir_primary_full_sequence_dataset import QUALITY_NAMES
from src.models.expert_contract import ExpertOutput, VisualExpertForward
from src.models.ir_primary_depth_residual_tcn import IRPrimaryDepthResidualTCN


def inputs(frames: int = 4) -> tuple[torch.Tensor, ...]:
    ir = torch.randn(2, frames, 4, 1, 64, 64)
    depth = torch.randn(2, frames, 2, 3, 64, 64)
    depth_pixel_valid = torch.ones(2, frames, 2, 1, 64, 64, dtype=torch.bool)
    view_valid = torch.ones(2, frames, 6, dtype=torch.bool)
    reliability = torch.ones(2, frames, 6)
    temporal = torch.ones(2, frames, dtype=torch.bool)
    quality = torch.ones(2, len(QUALITY_NAMES))
    quality_mask = torch.ones_like(quality, dtype=torch.bool)
    return ir, depth, depth_pixel_valid, view_valid, reliability, temporal, quality, quality_mask


def model(*, activation_checkpointing: bool = False) -> IRPrimaryDepthResidualTCN:
    return IRPrimaryDepthResidualTCN(
        pretrained=False,
        channels=32,
        embedding_dim=32,
        dropout=0.0,
        spatial_view_chunk_size=8,
        activation_checkpointing=activation_checkpointing,
    )


def test_forward_uses_stable_expert_contract() -> None:
    network = model().eval()
    output = network(*inputs())
    assert isinstance(output, VisualExpertForward)
    assert isinstance(output.expert, ExpertOutput)
    assert output.expert.main_logits.shape == (2, 40)
    assert output.small_gate_logits.shape == (2, 2)
    assert output.expert.embedding.shape == (2, 32)
    assert output.expert.quality.shape == (2, len(QUALITY_NAMES))
    assert output.expert.availability.shape == (2, 1)
    assert output.ir_roi_attention.shape == (2, 4, 3)


def test_all_invalid_local_views_have_zero_attention() -> None:
    values = list(inputs())
    values[3][:, :, 1:4] = False
    values[3][:, :, 5] = False
    values[4][:, :, 1:4] = 0.0
    values[4][:, :, 5] = 0.0
    network = model().eval()
    output = network(*values)
    assert torch.equal(output.ir_roi_attention, torch.zeros_like(output.ir_roi_attention))
    assert torch.equal(output.depth_relation_gate, torch.zeros_like(output.depth_relation_gate))
    assert torch.isfinite(output.expert.main_logits).all()


def test_padding_does_not_change_valid_sequence_prediction() -> None:
    torch.manual_seed(7)
    network = model().eval()
    base = [
        value[:1].clone() if index in {6, 7} else value[:1, :3].clone()
        for index, value in enumerate(inputs(frames=3))
    ]
    with torch.inference_mode():
        expected = network(*base).expert.main_logits
    padded: list[torch.Tensor] = []
    for index, value in enumerate(base):
        if index in {6, 7}:
            padded.append(value)
            continue
        shape = list(value.shape)
        shape[1] = 6
        fill = False if value.dtype == torch.bool else 0.0
        target = torch.full(shape, fill, dtype=value.dtype)
        target[:, :3] = value
        padded.append(target)
    with torch.inference_mode():
        actual = network(*padded).expert.main_logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_activation_checkpointing_preserves_spatial_gradients() -> None:
    network = model(activation_checkpointing=True).train()
    values = tuple(value[:1, :2] if index not in {6, 7} else value[:1] for index, value in enumerate(inputs()))
    output = network(*values)
    (output.expert.main_logits.sum() + output.small_gate_logits.sum()).backward()
    assert network.ir_encoder[0][0][0].weight.grad is not None
    assert network.depth_encoder.features[0].weight.grad is not None
    assert int(network.ir_encoder[0][0][1].num_batches_tracked) == 1
    assert int(network.depth_encoder.features[1].num_batches_tracked) == 1


def test_bfloat16_autocast_accepts_valid_view_scatter() -> None:
    network = model().eval()
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.inference_mode():
        output = network(*inputs(frames=2))
    assert output.expert.main_logits.dtype == torch.bfloat16
    assert torch.isfinite(output.expert.main_logits).all()
