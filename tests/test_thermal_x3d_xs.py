from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from src.models.thermal_x3d_xs import (
    RandomInitializationPolicyError,
    ThermalX3DXSBaseline,
    build_thermal_x3d_xs_backbone,
)


class RecordingBackbone(nn.Module):
    feature_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls.append(inputs)
        means = inputs.mean(dim=(1, 2, 3, 4))
        return means[:, None].repeat(1, self.feature_dim)


def test_pretrained_policy_rejects_before_any_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *args, **kwargs: pytest.fail("pretrained download must never be called"),
    )

    with pytest.raises(RandomInitializationPolicyError, match="pretrained"):
        build_thermal_x3d_xs_backbone(pretrained=True)


def test_builder_has_random_provenance_and_no_kinetics_projection() -> None:
    backbone = build_thermal_x3d_xs_backbone(pretrained=False)

    assert backbone.feature_dim == 2048
    assert backbone.initialization_provenance == {
        "source": "pytorchvideo.models.x3d.create_x3d",
        "topology": "x3d_xs",
        "pretrained": False,
        "input_clip_length": 16,
        "input_crop_size": 160,
    }
    assert backbone.network.blocks[-1].proj is None
    assert sum(parameter.numel() for parameter in backbone.parameters()) < 3_100_000


def test_real_backbone_forward_returns_pooled_feature() -> None:
    backbone = build_thermal_x3d_xs_backbone(pretrained=False).eval()

    with torch.inference_mode():
        output = backbone(torch.zeros(1, 3, 16, 160, 160))

    assert output.shape == (1, 2048)
    assert torch.isfinite(output).all()


def test_baseline_encodes_windows_sequentially_and_masks_mean() -> None:
    backbone = RecordingBackbone()
    model = ThermalX3DXSBaseline(backbone=backbone, num_classes=40, dropout=0.0)
    full_rgb = torch.stack(
        (
            torch.ones(2, 3, 4, 8, 8),
            torch.full((2, 3, 4, 8, 8), 2.0),
            torch.full((2, 3, 4, 8, 8), 9.0),
        ),
        dim=1,
    )
    window_mask = torch.tensor([[True, True, False], [True, False, False]])
    availability = torch.tensor([[True, False, True, False], [True, True, True, True]])
    quality = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    output = model(
        full_rgb,
        window_mask=window_mask,
        availability=availability,
        quality=quality,
    )

    assert len(backbone.calls) == 3
    assert all(call.shape == (2, 3, 4, 8, 8) for call in backbone.calls)
    assert output["embedding"][0].tolist() == pytest.approx([1.5] * 4)
    assert output["embedding"][1].tolist() == pytest.approx([1.0] * 4)
    assert output["logits"].shape == (2, 40)
    assert torch.isfinite(output["logits"]).all()
    assert output["availability"] is availability
    assert output["quality"] is quality


def test_baseline_state_dict_is_well_below_deployment_budget() -> None:
    model = ThermalX3DXSBaseline(num_classes=40)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    assert buffer.tell() < 20_000_000
