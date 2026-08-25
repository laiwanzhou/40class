from __future__ import annotations

import math

import torch
from torch import nn

from src.models.multimodal_token_contract import GroupTokens


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / max(dim, 1))
    )
    result = torch.zeros(length, dim, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies[: result[:, 1::2].shape[1]])
    return result


class ActionQueryLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(
            self.query_norm(queries),
            self.token_norm(tokens),
            self.token_norm(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        queries = self.attention_norm(queries + attended)
        queries = queries + self.mlp(queries)
        return queries, weights


class HierarchicalActionQueryFusion(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 256,
        classes: int = 40,
        heads: int = 8,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if dim < 1 or classes < 1 or layers < 1 or dim % heads:
            raise ValueError("hierarchical fusion dimensions are incompatible")
        self.dim = int(dim)
        self.classes = int(classes)
        self.action_queries = nn.Parameter(torch.empty(classes, dim))
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        self.group_embeddings = nn.Parameter(torch.empty(3, dim))
        nn.init.trunc_normal_(self.group_embeddings, std=0.02)
        self.register_buffer(
            "segment_positions", _sinusoidal_positions(8, dim), persistent=True
        )
        self.layers = nn.ModuleList(ActionQueryLayer(dim, heads) for _ in range(layers))
        self.classifier_weight = nn.Parameter(torch.empty(classes, dim))
        self.classifier_bias = nn.Parameter(torch.zeros(classes))
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def _validate_inputs(self, visual: GroupTokens, body: GroupTokens) -> None:
        batch = visual.tokens.shape[0]
        visual.validate(batch=batch, segments=8, streams=2, dim=self.dim)
        body.validate(batch=batch, segments=8, streams=1, dim=self.dim)
        if body.tokens.shape[0] != batch:
            raise ValueError("visual and body batch sizes differ")

    def forward(
        self, *, visual: GroupTokens, body: GroupTokens
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(visual, body)
        raw = torch.cat((visual.tokens, body.tokens), dim=2)
        mask = torch.cat((visual.mask, body.mask), dim=2)
        batch = raw.shape[0]
        token_values = (
            raw
            + self.segment_positions[None, :, None].to(raw.dtype)
            + self.group_embeddings[None, None].to(raw.dtype)
        )
        tokens = token_values.reshape(batch, 24, self.dim)
        token_mask = mask.reshape(batch, 24)
        core_available = token_mask.any(dim=1)
        safe_mask = token_mask.clone()
        safe_mask[~core_available, 0] = True
        safe_tokens = tokens * safe_mask[:, :, None].to(tokens.dtype)
        key_padding_mask = ~safe_mask
        queries = self.action_queries[None].expand(batch, -1, -1)
        final_weights = torch.zeros(
            batch, self.classes, 24, device=raw.device, dtype=raw.dtype
        )
        for layer in self.layers:
            queries, final_weights = layer(queries, safe_tokens, key_padding_mask)
        queries = queries * core_available[:, None, None].to(queries.dtype)
        final_weights = final_weights * core_available[:, None, None].to(
            final_weights.dtype
        )
        attention_grid = final_weights.reshape(batch, self.classes, 8, 3)
        group_attention = attention_grid.sum(dim=2)
        segment_attention = attention_grid.sum(dim=3)
        logits = torch.einsum(
            "bcd,cd->bc", queries, self.classifier_weight
        ) + self.classifier_bias[None]

        group_mask = mask.any(dim=1)
        group_count = mask.sum(dim=1).clamp_min(1).to(raw.dtype)
        group_features = (raw * mask[:, :, :, None].to(raw.dtype)).sum(dim=1)
        group_features = group_features / group_count[:, :, None]
        group_features = group_features * group_mask[:, :, None].to(raw.dtype)
        return {
            "logits": logits,
            "action_features": queries,
            "group_attention": group_attention,
            "segment_attention": segment_attention,
            "token_attention": final_weights,
            "group_features": group_features,
            "effective_group_mask": group_mask,
            "core_available": core_available,
        }
