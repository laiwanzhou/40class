from __future__ import annotations

import math

import torch
from torch import nn

from src.data.clean_skeleton_dataset import H36M_EDGES


def h36m_normalized_adjacency() -> torch.Tensor:
    adjacency = torch.eye(17, dtype=torch.float32)
    for parent, child in H36M_EDGES:
        adjacency[parent, child] = 1.0
        adjacency[child, parent] = 1.0
    degree = adjacency.sum(dim=1)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class FixedGraphConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.register_buffer("adjacency", h36m_normalized_adjacency(), persistent=True)
        self.projection = nn.Linear(input_channels, output_channels, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("vw,btwc->btvc", self.adjacency, inputs)
        return self.projection(aggregated)


class SegmentAwareTemporalConv(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int = 5, dilation: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.initialization_bound = 1.0 / math.sqrt(kernel_size * input_channels)
        self.weight = nn.Parameter(torch.empty(kernel_size, input_channels, output_channels))
        self.bias = nn.Parameter(torch.zeros(output_channels))
        nn.init.uniform_(self.weight, -self.initialization_bound, self.initialization_bound)

    def forward(
        self, inputs: torch.Tensor, segment_ids: torch.Tensor, temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 4 or segment_ids.shape != inputs.shape[:2]:
            raise ValueError("Expected features [B,T,V,C] and segment IDs [B,T]")
        if temporal_mask.shape != inputs.shape[:2]:
            raise ValueError("temporal_mask must match [B,T]")
        output = inputs.new_zeros((*inputs.shape[:3], self.weight.shape[2]))
        radius = self.kernel_size // 2
        time_steps = inputs.shape[1]
        for kernel_index, position in enumerate(range(-radius, radius + 1)):
            offset = position * self.dilation
            shifted = torch.roll(inputs, shifts=-offset, dims=1)
            shifted_segments = torch.roll(segment_ids, shifts=-offset, dims=1)
            shifted_mask = torch.roll(temporal_mask, shifts=-offset, dims=1)
            boundary = torch.ones_like(temporal_mask)
            if abs(offset) >= time_steps:
                boundary.fill_(False)
            elif offset > 0:
                boundary[:, time_steps - offset:] = False
            elif offset < 0:
                boundary[:, :-offset] = False
            valid = (
                boundary & temporal_mask & shifted_mask & (segment_ids == shifted_segments)
                & (segment_ids >= 0)
            )
            contribution = torch.einsum("btvc,co->btvo", shifted, self.weight[kernel_index])
            output = output + contribution * valid[:, :, None, None].to(output.dtype)
        return (output + self.bias) * temporal_mask[:, :, None, None].to(output.dtype)


class SpatialTemporalBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, dilation: int, dropout: float,
        temporal_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.spatial = FixedGraphConv(input_channels, output_channels)
        self.spatial_norm = nn.LayerNorm(output_channels)
        self.temporal = SegmentAwareTemporalConv(
            output_channels, output_channels, kernel_size=temporal_kernel_size, dilation=dilation
        )
        self.temporal_norm = nn.LayerNorm(output_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if input_channels == output_channels
            else nn.Linear(input_channels, output_channels, bias=False)
        )

    def forward(
        self, inputs: torch.Tensor, segment_ids: torch.Tensor, temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = temporal_mask[:, :, None, None].to(inputs.dtype)
        residual = self.residual(inputs) * mask
        spatial = self.activation(self.spatial_norm(self.spatial(inputs * mask)))
        temporal = self.temporal(spatial, segment_ids, temporal_mask)
        return self.dropout(self.activation(self.temporal_norm(temporal) + residual)) * mask


class LightweightSTGCN(nn.Module):
    def __init__(
        self, input_channels: int = 6, channels: tuple[int, ...] = (32, 48, 64),
        embedding_dim: int = 128, num_classes: int = 40, dropout: float = 0.2,
        temporal_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("At least one ST-GCN channel is required")
        blocks = []
        current = input_channels
        for index, channel in enumerate(channels):
            blocks.append(SpatialTemporalBlock(
                current, channel, 2**index, dropout, temporal_kernel_size=temporal_kernel_size
            ))
            current = channel
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Sequential(nn.Linear(current, embedding_dim), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, inputs: dict[str, torch.Tensor], temporal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = inputs["features"]
        segment_ids = inputs["segment_ids"]
        if features.ndim != 4 or features.shape[2:] != (17, 6):
            raise ValueError("Expected Skeleton features [B,T,17,6]")
        if temporal_mask is None:
            temporal_mask = segment_ids >= 0
        temporal_mask = temporal_mask.bool() & (segment_ids >= 0)
        encoded = features
        for block in self.blocks:
            encoded = block(encoded, segment_ids, temporal_mask)
        weights = temporal_mask[:, :, None, None].to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=(1, 2))
        denominator = weights.sum(dim=1).clamp_min(1.0) * encoded.shape[2]
        pooled = pooled / denominator.squeeze(-1)
        embedding = self.projection(pooled)
        return {
            "sequence_features": encoded,
            "embedding": embedding,
            "logits": self.classifier(embedding),
        }
