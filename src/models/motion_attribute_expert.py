from __future__ import annotations

import torch
from torch import nn


class MultiScaleTemporalBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        kernels: tuple[int, int, int] = (3, 5, 9),
        dilations: tuple[int, int, int] = (1, 2, 4),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    input_dim,
                    output_dim,
                    kernel,
                    padding=dilation * (kernel - 1) // 2,
                    dilation=dilation,
                    bias=False,
                )
                for kernel, dilation in zip(kernels, dilations, strict=True)
            ]
        )
        self.residual = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim, bias=False)
        )
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values * mask[:, :, None].to(values.dtype)
        temporal = values.transpose(1, 2)
        encoded = torch.stack([branch(temporal) for branch in self.branches]).mean(0)
        encoded = encoded.transpose(1, 2)
        output = self.dropout(
            self.activation(self.norm(encoded + self.residual(values)))
        )
        return output * mask[:, :, None].to(output.dtype)


class MotionAttributeExpert(nn.Module):
    def __init__(
        self,
        *,
        channels: tuple[int, int, int] = (128, 192, 256),
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(17 * 6, channels[0])
        self.input_norm = nn.LayerNorm(channels[0])
        self.blocks = nn.ModuleList(
            (
                MultiScaleTemporalBlock(channels[0], channels[0], dropout=dropout),
                MultiScaleTemporalBlock(channels[0], channels[1], dropout=dropout),
                MultiScaleTemporalBlock(channels[1], channels[2], dropout=dropout),
            )
        )
        self.embedding = nn.Sequential(
            nn.Linear(channels[2], embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.family_head = nn.Linear(embedding_dim, 6)
        self.attribute_head = nn.Linear(embedding_dim, 16)
        self.action_head = nn.Linear(embedding_dim, 40)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4 or features.shape[1:] != (96, 17, 6):
            raise ValueError("motion expert features must be [B,96,17,6]")
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("motion expert mask must be bool [B,96]")
        values = features.flatten(2)
        values = torch.nn.functional.gelu(
            self.input_norm(self.input_projection(values))
        )
        values = values * mask[:, :, None].to(values.dtype)
        for block in self.blocks:
            values = block(values, mask)
        weights = mask[:, :, None].to(values.dtype)
        pooled = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        available = mask.any(dim=1)
        embedding = self.embedding(pooled) * available[:, None].to(pooled.dtype)
        family_logits = self.family_head(embedding) * available[:, None].to(
            embedding.dtype
        )
        attribute_predictions = self.attribute_head(embedding) * available[
            :, None
        ].to(embedding.dtype)
        action_logits = self.action_head(embedding) * available[:, None].to(
            embedding.dtype
        )
        return {
            "embedding": embedding,
            "family_logits": family_logits,
            "attribute_predictions": attribute_predictions,
            "action_logits": action_logits,
            "available": available,
        }
