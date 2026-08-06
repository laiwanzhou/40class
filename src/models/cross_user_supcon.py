from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.depth_ir_pose_roi_expert import DepthIRPoseROIExpert


class CrossUserPrototypeBank(nn.Module):
    """EMA class-user prototypes used as a detached cross-batch contrast set."""

    def __init__(
        self,
        num_classes: int,
        num_users: int,
        projection_dim: int,
        momentum: float,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or num_users <= 1 or projection_dim <= 0:
            raise ValueError("Invalid prototype-bank dimensions")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Prototype momentum must be in [0, 1)")
        self.num_classes = int(num_classes)
        self.num_users = int(num_users)
        self.projection_dim = int(projection_dim)
        self.momentum = float(momentum)
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, num_users, projection_dim),
        )
        self.register_buffer("valid", torch.zeros(num_classes, num_users, dtype=torch.bool))
        self.register_buffer("update_counts", torch.zeros(num_classes, num_users, dtype=torch.long))

    @property
    def prototype_count(self) -> int:
        return int(self.valid.sum().item())

    def loss(
        self,
        projection: torch.Tensor,
        labels: torch.Tensor,
        user_indices: torch.Tensor,
        temperature: float,
        same_user_negative_weight: float,
    ) -> tuple[torch.Tensor, int]:
        if projection.ndim != 2 or projection.shape[1] != self.projection_dim:
            raise ValueError("Projection shape does not match prototype bank")
        if temperature <= 0 or same_user_negative_weight < 1:
            raise ValueError("Invalid contrastive temperature or negative weight")
        prototype_indices = self.valid.nonzero(as_tuple=False)
        if len(prototype_indices) == 0:
            return projection.sum() * 0.0, 0
        prototype_labels = prototype_indices[:, 0]
        prototype_users = prototype_indices[:, 1]
        prototypes = self.prototypes[self.valid].detach()
        logits = projection @ prototypes.T / float(temperature)
        same_action = labels[:, None] == prototype_labels[None, :]
        same_user = user_indices[:, None] == prototype_users[None, :]
        positives = same_action & ~same_user
        eligible = positives.any(dim=1)
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            return projection.sum() * 0.0, 0
        candidates = positives | ~same_action
        pair_weights = torch.ones_like(logits)
        pair_weights[same_user & ~same_action] = float(same_user_negative_weight)
        denominator_logits = (logits + pair_weights.log()).masked_fill(
            ~candidates, torch.finfo(logits.dtype).min,
        )
        log_denominator = torch.logsumexp(denominator_logits, dim=1)
        positive_log_probability = logits - log_denominator[:, None]
        per_anchor = -(
            (positive_log_probability * positives).sum(dim=1)
            / positives.sum(dim=1).clamp_min(1)
        )
        return per_anchor[eligible].mean(), eligible_count

    @torch.no_grad()
    def update(
        self,
        projection: torch.Tensor,
        labels: torch.Tensor,
        user_indices: torch.Tensor,
    ) -> None:
        projection = F.normalize(projection.detach(), dim=1)
        keys = torch.stack((labels.detach(), user_indices.detach()), dim=1)
        for key in keys.unique(dim=0):
            class_id = int(key[0].item())
            user_id = int(key[1].item())
            if not 0 <= class_id < self.num_classes or not 0 <= user_id < self.num_users:
                raise ValueError("Prototype key is outside configured dimensions")
            selected = (labels == class_id) & (user_indices == user_id)
            value = F.normalize(projection[selected].mean(dim=0), dim=0)
            count = int(selected.sum().item())
            if self.valid[class_id, user_id]:
                value = F.normalize(
                    self.momentum * self.prototypes[class_id, user_id]
                    + (1.0 - self.momentum) * value,
                    dim=0,
                )
            self.prototypes[class_id, user_id].copy_(value)
            self.valid[class_id, user_id] = True
            self.update_counts[class_id, user_id] += count


class CrossUserSupConModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        frame_feature_dim: int,
        projection_dim: int,
        dropout: float,
        pretrained: bool,
        prototype_num_users: int | None = None,
        prototype_momentum: float = 0.9,
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
        self.prototype_bank = (
            CrossUserPrototypeBank(
                num_classes=num_classes,
                num_users=prototype_num_users,
                projection_dim=projection_dim,
                momentum=prototype_momentum,
            )
            if prototype_num_users is not None else None
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
