from __future__ import annotations

import torch
from torch import nn


class P88DepthResidual(nn.Module):
    """Small zero-initialised correction on top of a frozen P87 anchor."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int = 128,
        dropout: float = 0.20,
        residual_scale: float = 0.50,
    ) -> None:
        super().__init__()
        if input_width < 1 or hidden_width < 1 or residual_scale <= 0:
            raise ValueError("invalid P88 residual geometry")
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.residual_scale = float(residual_scale)
        self.encoder = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_width, 40)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_width:
            raise ValueError(
                f"expected [B,{self.input_width}] features, got {tuple(features.shape)}"
            )
        return self.residual_scale * self.classifier(self.encoder(features))


def assemble_depth_residual_features(
    anchor_embedding: torch.Tensor,
    depth_embedding: torch.Tensor,
    depth_valid_statistics: torch.Tensor,
) -> torch.Tensor:
    if anchor_embedding.shape != depth_embedding.shape:
        raise ValueError("P87 and Depth embedding shapes differ")
    if anchor_embedding.ndim != 2 or anchor_embedding.shape[1] != 512:
        raise ValueError("P88 expects 512-wide P87 embeddings")
    if depth_valid_statistics.ndim != 2:
        raise ValueError("Depth validity statistics must be a matrix")
    return torch.cat(
        (
            depth_embedding,
            depth_embedding - anchor_embedding,
            depth_embedding * anchor_embedding,
            depth_valid_statistics,
        ),
        dim=1,
    )
