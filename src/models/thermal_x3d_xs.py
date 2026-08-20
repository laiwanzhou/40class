from __future__ import annotations

from typing import Any

import torch
from torch import nn


class RandomInitializationPolicyError(ValueError):
    """Raised when a deployable generation-2 student requests pretrained weights."""


class ThermalX3DXSBackbone(nn.Module):
    feature_dim = 2048

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network
        self.initialization_provenance: dict[str, Any] = {
            "source": "pytorchvideo.models.x3d.create_x3d",
            "topology": "x3d_xs",
            "pretrained": False,
            "input_clip_length": 16,
            "input_crop_size": 160,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1:] != (3, 16, 160, 160):
            raise ValueError("X3D-XS input must have shape [B,3,16,160,160]")
        features = self.network(inputs)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError("X3D-XS pooled feature contract changed")
        return features


def build_thermal_x3d_xs_backbone(*, pretrained: bool = False) -> ThermalX3DXSBackbone:
    if pretrained:
        raise RandomInitializationPolicyError(
            "pretrained Thermal generation-2 student weights are forbidden"
        )

    from pytorchvideo.models.x3d import create_x3d

    network = create_x3d(
        input_channel=3,
        input_clip_length=16,
        input_crop_size=160,
        model_num_class=400,
        width_factor=2.0,
        depth_factor=2.2,
    )
    head = network.blocks[-1]
    if not isinstance(head.proj, nn.Linear) or head.proj.out_features != 400:
        raise RuntimeError("unexpected PyTorchVideo X3D Kinetics projection")
    head.proj = None
    head.activation = None
    return ThermalX3DXSBackbone(network)


def _masked_window_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class ThermalX3DXSBaseline(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module | None = None,
        num_classes: int = 40,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone or build_thermal_x3d_xs_backbone(pretrained=False)
        feature_dim = int(getattr(self.backbone, "feature_dim", 0))
        if feature_dim < 1:
            raise ValueError("backbone must declare a positive feature_dim")
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        full_rgb: torch.Tensor,
        *,
        window_mask: torch.Tensor,
        availability: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if full_rgb.ndim != 6:
            raise ValueError("full_rgb must have shape [B,W,C,T,H,W]")
        batch, windows = full_rgb.shape[:2]
        if window_mask.shape != (batch, windows):
            raise ValueError("window_mask must have shape [B,W]")
        if availability.shape != (batch, 4) or quality.shape != (batch, 8):
            raise ValueError("availability/quality must have shapes [B,4] and [B,8]")
        window_features = torch.stack(
            [self.backbone(full_rgb[:, index]) for index in range(windows)], dim=1
        )
        embedding = _masked_window_mean(window_features, window_mask)
        logits = self.classifier(self.dropout(embedding))
        return {
            "logits": logits,
            "embedding": embedding,
            "availability": availability,
            "quality": quality,
        }
