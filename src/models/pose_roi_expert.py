from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class PoseROIExpert(nn.Module):
    def __init__(
        self,
        num_classes: int,
        expected_views: int,
        embedding_dim: int = 192,
        frame_feature_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.expected_views = expected_views
        self.local_scorer = nn.Linear(feature_dim, 1)
        self.frame_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, frame_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(frame_feature_dim, embedding_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor, temporal_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if inputs.ndim == 5:
            inputs = inputs.unsqueeze(2)
        if inputs.ndim != 6 or inputs.shape[2] != self.expected_views:
            raise ValueError(f"Expected [B,T,{self.expected_views},C,H,W], got {tuple(inputs.shape)}")
        batch, frames, views, channels, height, width = inputs.shape
        features = self.backbone(inputs.reshape(batch * frames * views, channels, height, width))
        features = features.reshape(batch, frames, views, -1)
        global_features = features[:, :, 0]
        if views > 1:
            locals_ = features[:, :, 1:]
            roi_attention = torch.softmax(self.local_scorer(locals_).squeeze(-1), dim=-1)
            local_summary = (locals_ * roi_attention.unsqueeze(-1)).sum(dim=2)
        else:
            roi_attention = features.new_empty((batch, frames, 0))
            local_summary = torch.zeros_like(global_features)
        frame_features = self.frame_projection(torch.cat((global_features, local_summary), dim=-1))
        if temporal_mask is None:
            _, hidden = self.temporal(frame_features)
        else:
            lengths = temporal_mask.sum(dim=1).clamp_min(1).to(torch.int64).cpu()
            packed = pack_padded_sequence(frame_features, lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.temporal(packed)
        embedding = hidden[-1]
        return {
            "embedding": embedding,
            "logits": self.classifier(self.dropout(embedding)),
            "roi_attention": roi_attention,
        }
