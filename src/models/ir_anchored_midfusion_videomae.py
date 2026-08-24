from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils import checkpoint


class IRAnchoredMidFusionVideoMAE(nn.Module):
    """Frozen-prefix IR anchor, zero-init Depth residual, and sparse class/view routing."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        frozen_prefix_blocks: int = 8,
        view_top_k: int = 2,
        hand_prior_bias: float = 0.5,
        depth_hidden_dim: int = 128,
        depth_gate_initial_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if not 0 <= frozen_prefix_blocks < len(backbone.blocks):
            raise ValueError("invalid frozen VideoMAE prefix")
        if not 1 <= view_top_k <= 4:
            raise ValueError("invalid view top-k")
        self.backbone = backbone
        self.frozen_prefix_blocks = int(frozen_prefix_blocks)
        self.view_top_k = int(view_top_k)
        embedding_dim = int(backbone.embed_dim)
        joint_dim = embedding_dim * 3
        self.depth_adapter = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, depth_hidden_dim),
            nn.GELU(),
            nn.Linear(depth_hidden_dim, embedding_dim),
        )
        nn.init.zeros_(self.depth_adapter[-1].weight)
        nn.init.zeros_(self.depth_adapter[-1].bias)
        self.depth_gate = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, 1))
        nn.init.zeros_(self.depth_gate[-1].weight)
        nn.init.constant_(self.depth_gate[-1].bias, depth_gate_initial_bias)
        self.class_queries = nn.Parameter(torch.empty(backbone.head.out_features, embedding_dim))
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        bias = torch.zeros(backbone.head.out_features, 4)
        bias[:, 2:] = float(hand_prior_bias)
        self.class_view_bias = nn.Parameter(bias)
        for parameter in backbone.patch_embed.parameters():
            parameter.requires_grad = False
        for block in backbone.blocks[: self.frozen_prefix_blocks]:
            for parameter in block.parameters():
                parameter.requires_grad = False
        for block in backbone.blocks[self.frozen_prefix_blocks :]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def _frozen_prefix(self, clips: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.patch_embed(clips)
            positional = self.backbone.pos_embed.to(
                device=features.device, dtype=features.dtype
            )
            features = self.backbone.pos_drop(features + positional[:, : features.shape[1]])
            for block in self.backbone.blocks[: self.frozen_prefix_blocks]:
                features = block(features)
        return features.detach()

    def _trainable_tail(self, features: torch.Tensor) -> torch.Tensor:
        for block in self.backbone.blocks[self.frozen_prefix_blocks :]:
            if self.training and torch.is_grad_enabled():
                features = checkpoint.checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        return self.backbone.fc_norm(features.mean(dim=1))

    def forward(
        self,
        *,
        ir: torch.Tensor,
        depth: torch.Tensor,
        availability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if ir.ndim != 6 or ir.shape[1:3] != (4, 3):
            raise ValueError("IR must have shape [B,4,3,T,H,W]")
        if depth.shape != ir.shape:
            raise ValueError("Depth must match IR shape")
        if availability.shape != (ir.shape[0], 2, 4) or availability.dtype != torch.bool:
            raise ValueError("availability must be bool [B,2,4]")
        pooled_views = []
        gate_means = []
        delta_norms = []
        for view in range(4):
            ir_tokens = self._frozen_prefix(ir[:, view])
            depth_tokens = self._frozen_prefix(depth[:, view])
            joint = torch.cat((ir_tokens, depth_tokens, depth_tokens - ir_tokens), dim=-1)
            depth_delta = self.depth_adapter(joint)
            depth_gate = torch.sigmoid(self.depth_gate(joint))
            depth_gate = depth_gate * availability[:, 1, view, None, None].to(depth_gate.dtype)
            fused = ir_tokens + depth_gate * depth_delta
            pooled_views.append(self._trainable_tail(fused))
            gate_means.append(depth_gate.mean(dim=1))
            delta_norms.append(depth_delta.square().mean(dim=(1, 2)).sqrt())
        view_features = torch.stack(pooled_views, dim=1)
        scores = torch.einsum("bvd,cd->bcv", view_features, self.class_queries)
        scores = scores / math.sqrt(view_features.shape[-1]) + self.class_view_bias[None]
        view_available = availability[:, 0] | availability[:, 1]
        scores = scores.masked_fill(~view_available[:, None], torch.finfo(scores.dtype).min)
        selected_scores, selected_indices = scores.topk(self.view_top_k, dim=2)
        sparse_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
        sparse_scores.scatter_(2, selected_indices, selected_scores)
        view_weights = torch.softmax(sparse_scores, dim=2)
        class_features = torch.einsum("bcv,bvd->bcd", view_weights, view_features)
        logits = torch.einsum("bcd,cd->bc", class_features, self.backbone.head.weight)
        if self.backbone.head.bias is not None:
            logits = logits + self.backbone.head.bias[None]
        return {
            "logits": logits,
            "view_weights": view_weights,
            "depth_gates": torch.stack(gate_means, dim=1),
            "depth_delta_norm": torch.stack(delta_norms, dim=1),
            "view_embeddings": view_features,
        }
