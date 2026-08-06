from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.depth_ir_pose_roi_expert import DepthIRPoseROIExpert


class CrossUserSupConModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        frame_feature_dim: int,
        projection_dim: int,
        dropout: float,
        pretrained: bool,
    ) -> None:
        super().__init__()
        self.visual_model = DepthIRPoseROIExpert(
            num_classes=num_classes,
            expected_views=4,
            embedding_dim=embedding_dim,
            frame_feature_dim=frame_feature_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(
        self, inputs: dict[str, torch.Tensor], temporal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.visual_model(inputs, temporal_mask=temporal_mask)
        output["projection"] = F.normalize(self.projection_head(output["embedding"]), dim=1)
        return output


def cross_user_supcon_loss(
    projection: torch.Tensor,
    labels: torch.Tensor,
    user_indices: torch.Tensor,
    temperature: float,
    same_user_negative_weight: float,
) -> torch.Tensor:
    if projection.ndim != 2 or len(projection) < 2:
        raise ValueError("Projection must be [B,D] with B >= 2")
    if temperature <= 0 or same_user_negative_weight < 1:
        raise ValueError("Invalid contrastive temperature or negative weight")
    logits = projection @ projection.T / float(temperature)
    identity = torch.eye(len(projection), dtype=torch.bool, device=projection.device)
    same_action = labels[:, None] == labels[None, :]
    same_user = user_indices[:, None] == user_indices[None, :]
    positives = same_action & ~same_user & ~identity
    if not positives.any(dim=1).all():
        raise ValueError("Every anchor must have a same-action, cross-user positive")
    pair_weights = torch.ones_like(logits)
    emphasized = same_user & ~same_action & ~identity
    pair_weights[emphasized] = float(same_user_negative_weight)
    denominator_logits = logits + pair_weights.log()
    denominator_logits = denominator_logits.masked_fill(identity, torch.finfo(logits.dtype).min)
    log_denominator = torch.logsumexp(denominator_logits, dim=1)
    positive_log_probability = logits - log_denominator[:, None]
    return -((positive_log_probability * positives).sum(dim=1) / positives.sum(dim=1)).mean()
