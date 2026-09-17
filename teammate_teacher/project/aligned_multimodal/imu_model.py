from __future__ import annotations

import torch
from torch import nn


TINY_IMU_DEVICES = 5
TINY_IMU_FEATURES_PER_DEVICE = 48
TINY_IMU_MASKS_PER_DEVICE = 2


class DeviceAwareIMUStudent(nn.Module):
    """Fixed P20 statistical IMU student with an explicit pooled-feature API."""

    def __init__(
        self,
        dropout: float = 0.15,
        num_devices: int = TINY_IMU_DEVICES,
        input_dim: int = TINY_IMU_FEATURES_PER_DEVICE + TINY_IMU_MASKS_PER_DEVICE,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.num_devices = int(num_devices)
        self.device_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.device_embedding = nn.Parameter(torch.zeros(self.num_devices, 64))
        nn.init.normal_(self.device_embedding, std=0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_devices * 64),
            nn.Linear(self.num_devices * 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def encode_pooled(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.device_encoder(inputs)
        encoded = encoded + self.device_embedding.unsqueeze(0)
        return encoded.flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_pooled(inputs))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class IMUTemporalModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_devices: int = 5,
        hidden_channels: int = 64,
        num_classes: int = 40,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_devices = num_devices
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels + 1, hidden_channels, 5, padding=2, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(hidden_channels, 1, dropout),
            ResidualTemporalBlock(hidden_channels, 2, dropout),
            ResidualTemporalBlock(hidden_channels, 4, dropout),
        )
        self.device_embedding = nn.Parameter(torch.zeros(num_devices, hidden_channels * 2))
        self.missing_embedding = nn.Parameter(torch.zeros(num_devices, hidden_channels * 2))
        nn.init.normal_(self.device_embedding, std=0.02)
        nn.init.normal_(self.missing_embedding, std=0.02)
        feature_dim = num_devices * hidden_channels * 2 + num_devices
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        device_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, devices, _, steps = imu.shape
        mask = time_mask.reshape(batch * devices, 1, steps)
        inputs = imu.reshape(batch * devices, imu.shape[2], steps)
        encoded = self.temporal(self.stem(torch.cat([inputs, mask], dim=1)))
        denominator = mask.sum(dim=2).clamp_min(1.0)
        average = (encoded * mask).sum(dim=2) / denominator
        maximum = encoded.masked_fill(mask == 0, -1e4).amax(dim=2)
        maximum = torch.where(mask.sum(dim=2) > 0, maximum, torch.zeros_like(maximum))
        features = torch.cat([average, maximum], dim=1).reshape(batch, devices, -1)
        features = features + self.device_embedding.unsqueeze(0)
        present = device_mask.unsqueeze(-1)
        features = present * features + (1.0 - present) * self.missing_embedding.unsqueeze(0)
        return self.classifier(torch.cat([features.flatten(1), device_mask], dim=1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fp32_size_mb(model: nn.Module) -> float:
    return parameter_count(model) * 4 / (1024**2)
