from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


H36M_PARENTS = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


def normalized_adjacency() -> torch.Tensor:
    adjacency = np.eye(17, dtype=np.float32)
    for child, parent in enumerate(H36M_PARENTS):
        adjacency[child, parent] = 1.0
        adjacency[parent, child] = 1.0
    degree = adjacency.sum(axis=1, keepdims=True)
    return torch.from_numpy(adjacency / np.maximum(degree, 1.0))


class GraphTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.self_project = nn.Linear(width, width)
        self.neighbor_project = nn.Linear(width, width, bias=False)
        self.spatial_norm = nn.LayerNorm(width)
        padding = 2 * dilation
        self.temporal = nn.Sequential(
            nn.Conv2d(
                width,
                width,
                kernel_size=(5, 1),
                padding=(padding, 0),
                dilation=(dilation, 1),
                groups=width,
                bias=False,
            ),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1, bias=False),
        )
        self.temporal_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor = torch.einsum("vw,btwd->btvd", adjacency, source)
        spatial = self.self_project(source) + self.neighbor_project(neighbor)
        source = self.spatial_norm(source + self.dropout(torch.nn.functional.gelu(spatial)))
        temporal = self.temporal(source.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.temporal_norm(source + self.dropout(temporal))


class TemporalAttentionPool(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        attended = torch.sum(sequence * weight.unsqueeze(-1), dim=1)
        maximum = sequence.amax(dim=1)
        return torch.cat([attended, maximum], dim=1)


class P27TemporalSkeleton(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        num_frames: int = 32,
        graph_width: int = 96,
        temporal_width: int = 192,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        dropout: float = 0.25,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.register_buffer("adjacency", normalized_adjacency())
        self.input_project = nn.Sequential(
            nn.Linear(input_dim, graph_width),
            nn.LayerNorm(graph_width),
            nn.GELU(),
        )
        self.joint_position = nn.Parameter(
            torch.randn(1, 1, 17, graph_width) / math.sqrt(graph_width)
        )
        self.graph_blocks = nn.ModuleList(
            [
                GraphTemporalBlock(graph_width, dilation, dropout * 0.5)
                for dilation in (1, 2, 4)
            ]
        )
        self.frame_project = nn.Sequential(
            nn.Linear(graph_width * 5, temporal_width),
            nn.LayerNorm(temporal_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_position = nn.Parameter(
            torch.randn(1, num_frames, temporal_width) / math.sqrt(temporal_width)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_width,
            nhead=transformer_heads,
            dim_feedforward=temporal_width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.temporal_norm = nn.LayerNorm(temporal_width)
        self.pool = TemporalAttentionPool(temporal_width)
        self.embedding = nn.Sequential(
            nn.Linear(temporal_width * 2, temporal_width),
            nn.LayerNorm(temporal_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(temporal_width, num_classes)

    def encode(self, skeleton: torch.Tensor) -> torch.Tensor:
        source = self.input_project(skeleton) + self.joint_position
        for block in self.graph_blocks:
            source = block(source, self.adjacency)
        global_mean = source.mean(dim=2)
        global_maximum = source.amax(dim=2)
        head = source[:, :, (8, 9, 10)].mean(dim=2)
        left_arm = source[:, :, (11, 12, 13)].mean(dim=2)
        right_arm = source[:, :, (14, 15, 16)].mean(dim=2)
        frames = self.frame_project(
            torch.cat(
                [global_mean, global_maximum, head, left_arm, right_arm], dim=-1
            )
        )
        frames = frames + self.temporal_position[:, : frames.shape[1]]
        frames = self.temporal_norm(self.temporal_encoder(frames))
        return self.embedding(self.pool(frames))

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(skeleton))
