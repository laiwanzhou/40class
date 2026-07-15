from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class VisualBaseline(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 40, dropout: float = 0.2) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(nn.Linear(feature_dim, embedding_dim), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor, temporal_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, frames, channels, height, width = inputs.shape
        features = self.backbone(inputs.reshape(batch * frames, channels, height, width))
        features = features.reshape(batch, frames, -1)
        if temporal_mask is None:
            pooled = features.mean(dim=1)
        else:
            weights = temporal_mask.to(features.dtype).unsqueeze(-1)
            pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        embedding = self.projection(pooled)
        return {"embedding": embedding, "logits": self.classifier(embedding)}
