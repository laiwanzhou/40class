from __future__ import annotations

import torch
from torch import nn


class P88Layer3SpatialResidual(nn.Module):
    """Shared low-rank spatial/temporal head over frozen MC3 layer3 descriptors."""

    def __init__(
        self,
        width: int = 128,
        dropout: float = 0.20,
        residual_scale: float = 0.50,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.residual_scale = float(residual_scale)
        self.descriptor_projection = nn.Sequential(
            nn.Linear(4 * 256, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.statistic_embedding = nn.Parameter(torch.zeros(3, width))
        self.view_embedding = nn.Parameter(torch.zeros(3, width))
        self.window_embedding = nn.Parameter(torch.zeros(2, width))
        self.stage_fusion = nn.Sequential(
            nn.Linear(2 * 3 * width + 3 * width, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(width, 40)
        nn.init.normal_(self.statistic_embedding, std=0.01)
        nn.init.normal_(self.view_embedding, std=0.01)
        nn.init.normal_(self.window_embedding, std=0.01)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self, descriptors: torch.Tensor, clip_quality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 6 or descriptors.shape[1:] != (2, 3, 3, 4, 256):
            raise ValueError(
                "descriptors must have shape [B,2,3,3,4,256], got "
                f"{tuple(descriptors.shape)}"
            )
        if clip_quality.shape != descriptors.shape[:3]:
            raise ValueError("clip quality geometry differs")
        batch = descriptors.shape[0]
        encoded = self.descriptor_projection(descriptors.flatten(-2))
        encoded = encoded + self.statistic_embedding.view(1, 1, 1, 3, self.width)
        encoded = encoded + self.view_embedding.view(1, 1, 3, 1, self.width)
        encoded = encoded + self.window_embedding.view(1, 2, 1, 1, self.width)
        quality = clip_quality.clamp_min(0.0)
        weight = quality / quality.sum(dim=2, keepdim=True).clamp_min(1e-6)
        window = (encoded * weight[:, :, :, None, None]).sum(dim=2)
        early = window[:, 0].flatten(1)
        late = window[:, 1].flatten(1)
        delta = (window[:, 1] - window[:, 0]).flatten(1)
        embedding = self.stage_fusion(torch.cat((early, late, delta), dim=1))
        auxiliary_logits = self.classifier(embedding)
        return self.residual_scale * auxiliary_logits, auxiliary_logits
