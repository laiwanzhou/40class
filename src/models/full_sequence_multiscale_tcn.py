from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time step, excluding padded-time statistics."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs.transpose(1, 2)).transpose(1, 2)


class MaskedDilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = ChannelLayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = ChannelLayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.activation(self.norm1(self.conv1(inputs)))) * mask
        features = self.dropout(self.activation(self.norm2(self.conv2(features)))) * mask
        return (inputs + features) * mask


class TemporalBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        dilations: Sequence[int],
        dropout: float,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        self.blocks = nn.ModuleList(
            MaskedDilatedResidualBlock(channels, dilation, dropout) for dilation in self.dilations
        )
        self.activation_checkpointing = bool(activation_checkpointing)
        self.attention = nn.Conv1d(channels, 1, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + 4 * sum(self.dilations)

    def forward(self, inputs: torch.Tensor, temporal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = temporal_mask.unsqueeze(1).to(inputs.dtype)
        features = inputs * mask
        for block in self.blocks:
            if self.activation_checkpointing and self.training and features.requires_grad:
                features = checkpoint(block, features, mask, use_reentrant=False)
            else:
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
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(frame_feature_dim, channels), nn.GELU(), nn.Dropout(dropout),
        )
        self.short_branch = TemporalBranch(channels, short_dilations, dropout, activation_checkpointing)
        self.long_branch = TemporalBranch(channels, long_dilations, dropout, activation_checkpointing)
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
