from __future__ import annotations

import io

import torch
from torch import nn

from src.models.thermal_multistream import ThermalMultiStreamStudent


class TinyRasterBackbone(nn.Module):
    feature_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        means = inputs.mean(dim=(1, 2, 3, 4), keepdim=False) * self.scale
        return means[:, None].repeat(1, self.feature_dim)


def inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "full_rgb": torch.ones(batch, 3, 3, 4, 32, 32),
        "crop_rgb": torch.full((batch, 3, 3, 4, 32, 32), 2.0),
        "motion": torch.randn(batch, 3, 4, 1, 32, 32),
        "pose": torch.randn(batch, 3, 4, 56),
        "window_mask": torch.ones(batch, 3, dtype=torch.bool),
        "pose_mask": torch.ones(batch, 3, 4, dtype=torch.bool),
        "availability": torch.ones(batch, 4, dtype=torch.bool),
        "quality": torch.full((batch, 8), 0.5),
    }


def test_exact_motion_and_pose_topology() -> None:
    model = ThermalMultiStreamStudent(raster_backbone=TinyRasterBackbone())

    assert model.motion_encoder.conv1.kernel_size == (3, 5, 5)
    assert model.motion_encoder.conv1.stride == (1, 2, 2)
    assert model.motion_encoder.conv2.kernel_size == (3, 3, 3)
    assert model.motion_encoder.conv2.stride == (2, 2, 2)
    assert model.motion_encoder.conv3.in_channels == 32
    assert model.motion_encoder.conv3.out_channels == 64
    assert model.pose_encoder.input_projection.in_channels == 56
    assert model.pose_encoder.input_projection.out_channels == 128
    assert [block.depthwise.dilation for block in model.pose_encoder.blocks] == [(1,), (2,)]
    assert all(block.depthwise.groups == 128 for block in model.pose_encoder.blocks)
    assert model.fusion_norm.normalized_shape == (780,)
    assert model.classifier.in_features == 780
    assert model.classifier.out_features == 40


def test_full_and_crop_share_one_raster_encoder_and_projection() -> None:
    backbone = TinyRasterBackbone()
    model = ThermalMultiStreamStudent(raster_backbone=backbone).eval()

    with torch.inference_mode():
        output = model(**inputs(batch=1))

    assert model.raster_encoder is backbone
    assert backbone.calls == 6
    assert output["logits"].shape == (1, 40)
    assert output["embedding"].shape == (1, 780)


def test_unavailable_crop_zeroes_only_crop_projection() -> None:
    model = ThermalMultiStreamStudent(raster_backbone=TinyRasterBackbone()).eval()
    batch = inputs()
    batch["availability"][0, 1:] = False
    batch["pose_mask"][0] = False
    captured: list[torch.Tensor] = []
    handle = model.fusion_norm.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )

    with torch.inference_mode():
        output = model(**batch)
    handle.remove()

    fusion_input = captured[0]
    assert torch.count_nonzero(fusion_input[0, :256]) > 0
    assert torch.count_nonzero(fusion_input[0, 256:512]) == 0
    assert torch.count_nonzero(fusion_input[0, 512:768]) == 0
    assert torch.count_nonzero(fusion_input[1, 256:512]) > 0
    assert torch.isfinite(output["logits"]).all()
    assert output["availability"] is batch["availability"]
    assert output["quality"] is batch["quality"]


def test_missing_auxiliary_streams_retain_finite_full_frame_path() -> None:
    model = ThermalMultiStreamStudent(raster_backbone=TinyRasterBackbone()).eval()
    batch = inputs(batch=1)
    batch["availability"][:] = torch.tensor([True, False, False, False])
    batch["pose_mask"][:] = False
    batch["crop_rgb"][:] = 0
    batch["motion"][:] = 0
    batch["pose"][:] = 0

    with torch.inference_mode():
        output = model(**batch)

    assert torch.isfinite(output["embedding"]).all()
    assert torch.isfinite(output["logits"]).all()
    assert torch.count_nonzero(output["embedding"][:, :256]) > 0


def test_real_student_fits_parameter_and_state_dict_limits() -> None:
    model = ThermalMultiStreamStudent()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000_000
    assert buffer.tell() < 45_000_000
