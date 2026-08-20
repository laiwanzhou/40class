from __future__ import annotations

import torch
from torch import nn

from src.models.thermal_x3d_xs import build_thermal_x3d_xs_backbone


def _masked_mean(features: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    if dim != 1 or features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError("masked mean expects features [B,T,F] and mask [B,T]")
    weights = mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    return (features * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class MotionEncoder(nn.Module):
    feature_dim = 128

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)
        )
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.activation = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Linear(64, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[2] != 1:
            raise ValueError("motion window must have shape [B,T,1,H,W]")
        encoded = inputs.permute(0, 2, 1, 3, 4)
        encoded = self.activation(self.bn1(self.conv1(encoded)))
        encoded = self.activation(self.bn2(self.conv2(encoded)))
        encoded = self.activation(self.bn3(self.conv3(encoded)))
        return self.projection(self.pool(encoded).flatten(1))


class DepthwiseTemporalResidual(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.depthwise(inputs))
        output = self.activation(self.norm(residual) + inputs)
        return output * mask[:, None].to(dtype=output.dtype)


class PoseEncoder(nn.Module):
    feature_dim = 128

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(56, 128, kernel_size=1)
        self.blocks = nn.ModuleList(
            [DepthwiseTemporalResidual(128, dilation=1), DepthwiseTemporalResidual(128, dilation=2)]
        )
        self.projection = nn.Sequential(
            nn.Linear(128, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[2] != 56:
            raise ValueError("pose window must have shape [B,T,56]")
        if mask.shape != inputs.shape[:2]:
            raise ValueError("pose mask must have shape [B,T]")
        encoded = self.input_projection(inputs.transpose(1, 2))
        encoded = encoded * mask[:, None].to(dtype=encoded.dtype)
        for block in self.blocks:
            encoded = block(encoded, mask)
        pooled = _masked_mean(encoded.transpose(1, 2), mask, dim=1)
        return self.projection(pooled)


class ThermalMultiStreamStudent(nn.Module):
    fusion_dim = 780

    def __init__(
        self,
        *,
        raster_backbone: nn.Module | None = None,
        num_classes: int = 40,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.raster_encoder = raster_backbone or build_thermal_x3d_xs_backbone(
            pretrained=False
        )
        raster_dim = int(getattr(self.raster_encoder, "feature_dim", 0))
        if raster_dim < 1:
            raise ValueError("raster backbone must declare a positive feature_dim")
        self.raster_projection = nn.Sequential(
            nn.Linear(raster_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.motion_encoder = MotionEncoder()
        self.pose_encoder = PoseEncoder()
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.fusion_dim, num_classes)
        self.initialization_provenance = {
            "student": "random",
            "pretrained_student_weights": False,
            "raster": getattr(self.raster_encoder, "initialization_provenance", None),
        }

    def _encode_raster(self, inputs: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        window_features = torch.stack(
            [self.raster_encoder(inputs[:, index]) for index in range(inputs.shape[1])],
            dim=1,
        )
        return self.raster_projection(_masked_mean(window_features, window_mask, dim=1))

    def _encode_motion(self, inputs: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        window_features = torch.stack(
            [self.motion_encoder(inputs[:, index]) for index in range(inputs.shape[1])],
            dim=1,
        )
        return _masked_mean(window_features, window_mask, dim=1)

    def _encode_pose(
        self, inputs: torch.Tensor, pose_mask: torch.Tensor, window_mask: torch.Tensor
    ) -> torch.Tensor:
        window_features = torch.stack(
            [
                self.pose_encoder(inputs[:, index], pose_mask[:, index])
                for index in range(inputs.shape[1])
            ],
            dim=1,
        )
        valid_windows = window_mask & pose_mask.any(dim=2)
        return _masked_mean(window_features, valid_windows, dim=1)

    def forward(
        self,
        *,
        full_rgb: torch.Tensor,
        crop_rgb: torch.Tensor,
        motion: torch.Tensor,
        pose: torch.Tensor,
        window_mask: torch.Tensor,
        pose_mask: torch.Tensor,
        availability: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if full_rgb.ndim != 6 or crop_rgb.shape != full_rgb.shape:
            raise ValueError("full_rgb and crop_rgb must share shape [B,W,C,T,H,W]")
        batch, windows = full_rgb.shape[:2]
        if motion.ndim != 6 or motion.shape[:2] != (batch, windows):
            raise ValueError("motion must have shape [B,W,T,1,H,W]")
        if pose.shape[:3] != (batch, windows, full_rgb.shape[3]) or pose.shape[3] != 56:
            raise ValueError("pose must have shape [B,W,T,56]")
        if window_mask.shape != (batch, windows):
            raise ValueError("window_mask must have shape [B,W]")
        if pose_mask.shape != pose.shape[:3]:
            raise ValueError("pose_mask must have shape [B,W,T]")
        if availability.shape != (batch, 4) or quality.shape != (batch, 8):
            raise ValueError("availability/quality must have shapes [B,4] and [B,8]")

        available = availability.to(device=full_rgb.device, dtype=full_rgb.dtype)
        full_feature = self._encode_raster(full_rgb, window_mask) * available[:, 0:1]
        crop_feature = self._encode_raster(crop_rgb, window_mask) * available[:, 1:2]
        motion_feature = self._encode_motion(motion, window_mask) * available[:, 2:3]
        pose_feature = self._encode_pose(pose, pose_mask, window_mask) * available[:, 3:4]
        fusion_input = torch.cat(
            (
                full_feature,
                crop_feature,
                motion_feature,
                pose_feature,
                available,
                quality.to(device=full_rgb.device, dtype=full_rgb.dtype),
            ),
            dim=1,
        )
        if fusion_input.shape != (batch, self.fusion_dim):
            raise RuntimeError("Thermal fusion contract must produce 780 values")
        embedding = self.fusion_norm(fusion_input)
        logits = self.classifier(self.dropout(embedding))
        return {
            "logits": logits,
            "embedding": embedding,
            "availability": availability,
            "quality": quality,
        }
