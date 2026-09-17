from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class ThermalMobileNet(nn.Module):
    """共享二维编码器 + 无参数时间统计，输入形状为 B,T,C,H,W。"""

    def __init__(self, num_classes: int = 40, dropout: float = 0.2) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.feature_dim = 576
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim * 3, num_classes),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = clip.shape
        frames = clip.reshape(batch_size * time_steps, channels, height, width)
        features = self.features(frames)
        features = features.mean(dim=(-2, -1))
        features = features.reshape(batch_size, time_steps, self.feature_dim)

        temporal_mean = features.mean(dim=1)
        temporal_max = features.amax(dim=1)
        if time_steps > 1:
            temporal_delta = (features[:, 1:] - features[:, :-1]).abs().mean(dim=1)
        else:
            temporal_delta = torch.zeros_like(temporal_mean)
        pooled = torch.cat([temporal_mean, temporal_max, temporal_delta], dim=1)
        return self.classifier(pooled)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fp32_size_mb(model: nn.Module) -> float:
    return parameter_count(model) * 4 / (1024**2)
