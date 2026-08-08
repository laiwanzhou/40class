from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MaskedDilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (inputs + self.layers(inputs)) * mask


class TemporalBranch(nn.Module):
    def __init__(self, channels: int, dilations: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        self.blocks = nn.ModuleList(
            MaskedDilatedResidualBlock(channels, dilation, dropout) for dilation in self.dilations
        )
        self.attention = nn.Conv1d(channels, 1, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + 4 * sum(self.dilations)

    def forward(self, inputs: torch.Tensor, temporal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = temporal_mask.unsqueeze(1).to(inputs.dtype)
        features = inputs * mask
        for block in self.blocks:
            features = block(features, mask)
        logits = self.attention(features).squeeze(1)
        logits = logits.masked_fill(~temporal_mask, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1).masked_fill(~temporal_mask, 0.0)
        attended = (features * attention.unsqueeze(1)).sum(dim=2)
        denominator = temporal_mask.sum(dim=1, keepdim=True).clamp_min(1).to(features.dtype)
        mean = features.sum(dim=2) / denominator
        maximum = features.masked_fill(~temporal_mask.unsqueeze(1), torch.finfo(features.dtype).min).max(dim=2).values
        return torch.cat((attended, mean, maximum), dim=1), attention


class FullSequenceMultiScaleTCN(nn.Module):
    def __init__(
        self,
        frame_feature_dim: int = 128,
        channels: int = 128,
        embedding_dim: int = 256,
        num_classes: int = 40,
        short_dilations: Sequence[int] = (1, 2, 4),
        long_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(frame_feature_dim, channels), nn.GELU(), nn.Dropout(dropout),
        )
        self.short_branch = TemporalBranch(channels, short_dilations, dropout)
        self.long_branch = TemporalBranch(channels, long_dilations, dropout)
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(channels * 6),
            nn.Linear(channels * 6, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, frame_features: torch.Tensor, temporal_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if frame_features.ndim != 3:
            raise ValueError("Frame features must have shape [B, T, F].")
        if temporal_mask.shape != frame_features.shape[:2]:
            raise ValueError("Temporal mask does not align with frame features.")
        if not temporal_mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one valid frame.")
        projected = self.input_projection(frame_features).transpose(1, 2)
        short, short_attention = self.short_branch(projected, temporal_mask)
        long, long_attention = self.long_branch(projected, temporal_mask)
        embedding = self.sequence_projection(torch.cat((short, long), dim=1))
        return {
            "embedding": embedding,
            "logits": self.classifier(embedding),
            "short_attention": short_attention,
            "long_attention": long_attention,
        }
