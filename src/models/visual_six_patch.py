from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class VisualSixPatch(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 40,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.patch_scorer = nn.Linear(feature_dim, 1)
        self.fusion_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        inputs: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 6 or inputs.shape[2] != 7:
            raise ValueError(f"Expected input [B,T,7,C,H,W], got {tuple(inputs.shape)}")
        batch, frames, views, channels, height, width = inputs.shape
        features = self.backbone(
            inputs.reshape(batch * frames * views, channels, height, width)
        ).reshape(batch, frames, views, -1)
        global_features = features[:, :, 0]
        local_features = features[:, :, 1:]
        patch_attention = torch.softmax(self.patch_scorer(local_features).squeeze(-1), dim=-1)
        weighted_local = (local_features * patch_attention.unsqueeze(-1)).sum(dim=2)
        frame_embeddings = self.fusion_projection(
            torch.cat((global_features, weighted_local), dim=-1)
        )
        if temporal_mask is None:
            embedding = frame_embeddings.mean(dim=1)
        else:
            weights = temporal_mask.to(frame_embeddings.dtype).unsqueeze(-1)
            embedding = (frame_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {
            "embedding": embedding,
            "logits": self.classifier(embedding),
            "patch_attention": patch_attention,
        }
