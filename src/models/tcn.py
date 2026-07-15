from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Identity() if input_channels == output_channels else nn.Conv1d(input_channels, output_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs) + self.skip(inputs)


class TemporalClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        embedding_dim: int = 128,
        num_classes: int = 40,
        channels: tuple[int, ...] = (64, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_features
        for index, channel in enumerate(channels):
            blocks.append(ResidualTemporalBlock(current, channel, 2**index, dropout))
            current = channel
        self.encoder = nn.Sequential(*blocks)
        self.projection = nn.Sequential(nn.Linear(current, embedding_dim), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor, temporal_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded = self.encoder(inputs.transpose(1, 2)).transpose(1, 2)
        if temporal_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weights = temporal_mask.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        embedding = self.projection(pooled)
        return {"embedding": embedding, "logits": self.classifier(embedding)}
