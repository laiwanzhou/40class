from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


MODALITY_ORDER = ("skeleton", "depth", "thermal", "imu")
EMBEDDING_DIMS = {
    "skeleton": 512,
    "depth": 1024,
    "thermal": 1024,
    "imu": 320,
}
NUM_CLASSES = 40


class ProjectionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ProjectedFeatureMixin:
    projections: nn.ModuleDict

    def project_features(
        self,
        embeddings: Mapping[str, torch.Tensor],
        presence: torch.Tensor,
    ) -> torch.Tensor:
        projected: list[torch.Tensor] = []
        for index, modality in enumerate(MODALITY_ORDER):
            encoded = self.projections[modality](embeddings[modality])
            encoded = encoded * presence[:, index : index + 1]
            projected.append(encoded)
        return torch.cat(projected, dim=1)


class PooledFeatureFusion(nn.Module, ProjectedFeatureMixin):
    """P22-F: pooled feature fusion without a logit bypass."""

    def __init__(
        self,
        projection_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                modality: ProjectionBlock(input_dim, projection_dim)
                for modality, input_dim in EMBEDDING_DIMS.items()
            }
        )
        fusion_dim = len(MODALITY_ORDER) * projection_dim + len(MODALITY_ORDER)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor],
        modality_logits: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        del modality_logits
        projected = self.project_features(embeddings, presence)
        return self.classifier(torch.cat([projected, presence], dim=1))


class MatchedLogitFusion(nn.Module):
    """P22-L: matched four-logit control with fewer parameters than P22-F."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        input_dim = len(MODALITY_ORDER) * num_classes + len(MODALITY_ORDER)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor],
        modality_logits: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        del embeddings
        masked_logits = modality_logits * presence.unsqueeze(-1)
        return self.classifier(
            torch.cat([masked_logits.flatten(1), presence], dim=1)
        )


class FeatureLogitBypassFusion(nn.Module, ProjectedFeatureMixin):
    """P22-U: diagnostic pooled-feature model with raw unimodal-logit bypass."""

    def __init__(
        self,
        projection_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                modality: ProjectionBlock(input_dim, projection_dim)
                for modality, input_dim in EMBEDDING_DIMS.items()
            }
        )
        input_dim = (
            len(MODALITY_ORDER) * projection_dim
            + len(MODALITY_ORDER) * num_classes
            + len(MODALITY_ORDER)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor],
        modality_logits: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.project_features(embeddings, presence)
        masked_logits = modality_logits * presence.unsqueeze(-1)
        inputs = torch.cat(
            [projected, masked_logits.flatten(1), presence],
            dim=1,
        )
        return self.classifier(inputs)


def build_p22_model(
    model_name: str,
    projection_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.20,
) -> nn.Module:
    if model_name == "P22-L":
        return MatchedLogitFusion(hidden_dim=hidden_dim, dropout=dropout)
    if model_name == "P22-F":
        return PooledFeatureFusion(
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if model_name == "P22-U":
        return FeatureLogitBypassFusion(
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown P22 model: {model_name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
