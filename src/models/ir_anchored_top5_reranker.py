from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class IRAnchoredTop5Reranker(nn.Module):
    """IR-anchored cached-feature fusion with sparse class/view routing."""

    def __init__(
        self,
        *,
        embedding_dim: int = 768,
        fusion_dim: int = 128,
        num_classes: int = 40,
        view_top_k: int = 2,
        class_top_k: int = 5,
        dropout: float = 0.2,
        depth_gate_initial_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if not 1 <= view_top_k <= 4 or not 1 <= class_top_k <= num_classes:
            raise ValueError("invalid sparse reranker top-k contract")
        self.num_classes = int(num_classes)
        self.view_top_k = int(view_top_k)
        self.class_top_k = int(class_top_k)
        self.ir_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, fusion_dim), nn.GELU()
        )
        joint_dim = embedding_dim * 3
        self.depth_delta = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        nn.init.zeros_(self.depth_delta[-1].weight)
        nn.init.zeros_(self.depth_delta[-1].bias)
        self.depth_gate = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, 1))
        nn.init.zeros_(self.depth_gate[-1].weight)
        nn.init.constant_(self.depth_gate[-1].bias, depth_gate_initial_bias)
        self.class_queries = nn.Parameter(torch.empty(num_classes, fusion_dim))
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        self.class_view_bias = nn.Parameter(torch.zeros(num_classes, 4))
        self.residual_weight = nn.Parameter(torch.zeros(num_classes, fusion_dim))
        self.residual_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(
        self,
        *,
        view_embeddings: torch.Tensor,
        base_logits: torch.Tensor,
        availability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if view_embeddings.ndim != 4 or view_embeddings.shape[1:3] != (2, 4):
            raise ValueError("view_embeddings must have shape [B,2,4,D]")
        if base_logits.shape != (view_embeddings.shape[0], self.num_classes):
            raise ValueError("base_logits must have shape [B,C]")
        if availability.shape != view_embeddings.shape[:3] or availability.dtype != torch.bool:
            raise ValueError("availability must be bool [B,2,4]")
        if bool((~availability[:, 0].any(dim=1)).any()):
            raise ValueError("every sample requires at least one IR view")

        ir = view_embeddings[:, 0]
        depth = view_embeddings[:, 1]
        joint = torch.cat((ir, depth, depth - ir), dim=-1)
        depth_gates = torch.sigmoid(self.depth_gate(joint))
        depth_gates = depth_gates * availability[:, 1, :, None].to(depth_gates.dtype)
        fused_views = self.ir_projection(ir) + depth_gates * self.depth_delta(joint)

        scores = torch.einsum("bvd,cd->bcv", fused_views, self.class_queries)
        scores = scores / math.sqrt(fused_views.shape[-1]) + self.class_view_bias[None]
        scores = scores.masked_fill(~availability[:, 0, None], torch.finfo(scores.dtype).min)
        selected_scores, selected_indices = scores.topk(self.view_top_k, dim=2)
        sparse_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
        sparse_scores.scatter_(2, selected_indices, selected_scores)
        view_weights = torch.softmax(sparse_scores, dim=2)
        class_features = torch.einsum("bcv,bvd->bcd", view_weights, fused_views)
        raw_delta = torch.einsum("bcd,cd->bc", class_features, self.residual_weight)
        raw_delta = raw_delta + self.residual_bias[None]
        candidate_mask = torch.zeros_like(base_logits, dtype=torch.bool)
        candidate_mask.scatter_(1, base_logits.topk(self.class_top_k, dim=1).indices, True)
        delta_logits = raw_delta * candidate_mask.to(raw_delta.dtype)
        return {
            "logits": base_logits + delta_logits,
            "base_logits": base_logits,
            "delta_logits": delta_logits,
            "candidate_mask": candidate_mask,
            "view_weights": view_weights,
            "depth_gates": depth_gates,
            "fused_view_embeddings": fused_views,
        }


class Top5LogitReranker(nn.Module):
    """Capacity control that can only transform the frozen IR-anchor logits."""

    def __init__(
        self,
        *,
        num_classes: int = 40,
        hidden_dim: int = 128,
        class_top_k: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.class_top_k = int(class_top_k)
        self.network = nn.Sequential(
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, *, base_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_delta = self.network(base_logits)
        candidate_mask = torch.zeros_like(base_logits, dtype=torch.bool)
        candidate_mask.scatter_(1, base_logits.topk(self.class_top_k, dim=1).indices, True)
        delta_logits = raw_delta * candidate_mask.to(raw_delta.dtype)
        return {
            "logits": base_logits + delta_logits,
            "base_logits": base_logits,
            "delta_logits": delta_logits,
            "candidate_mask": candidate_mask,
            "depth_gates": base_logits.new_zeros((base_logits.shape[0], 4, 1)),
        }


def guarded_reranker_loss(
    *,
    logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    depth_gates: torch.Tensor,
    guard_weight: float,
    depth_l1_weight: float,
) -> dict[str, torch.Tensor]:
    ce_per_sample = F.cross_entropy(logits, labels, reduction="none")
    base_ce = F.cross_entropy(base_logits.detach(), labels, reduction="none")
    ce_loss = ce_per_sample.mean()
    guard_loss = torch.relu(ce_per_sample - base_ce).mean()
    depth_l1 = depth_gates.abs().mean()
    loss = ce_loss + guard_weight * guard_loss + depth_l1_weight * depth_l1
    return {
        "loss": loss,
        "ce_loss": ce_loss,
        "guard_loss": guard_loss,
        "depth_l1_loss": depth_l1,
    }
