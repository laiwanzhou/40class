"""Rich token-level candidate-query model for P103-B2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RichCandidateConfig:
    width: int = 128
    heads: int = 4
    cross_layers: int = 2
    candidate_layers: int = 1
    dropout: float = 0.15
    classes: int = 40

    def validate(self) -> None:
        if self.width % self.heads:
            raise ValueError("width must divide attention heads")
        if self.cross_layers != 2 or self.candidate_layers != 1:
            raise ValueError("P103-B2 architecture depth is frozen at 2+1")
        if self.classes != 40:
            raise ValueError("P103 candidate identity contract requires 40 class IDs")


class CandidateCrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        candidate_padding: torch.Tensor,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, attention = self.attention(
            self.query_norm(query),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query = query + self.attention_dropout(attended)
        query = query + self.feed_forward(self.ff_norm(query))
        query = query.masked_fill(candidate_padding[..., None], 0.0)
        return query, attention


def token_topology() -> dict[str, np.ndarray]:
    encoder: list[int] = []
    window: list[int] = []
    view: list[int] = []
    time: list[int] = []
    token_type: list[int] = []
    for encoder_id in range(2):
        for window_id in range(2):
            for view_id in range(3):
                for time_id in range(8):
                    encoder.append(encoder_id)
                    window.append(window_id)
                    view.append(view_id)
                    time.append(time_id)
                    token_type.append(0)
    for type_id in (1, 2):
        for encoder_id in range(2):
            for window_id in range(2):
                for view_id in range(3):
                    encoder.append(encoder_id)
                    window.append(window_id)
                    view.append(view_id)
                    time.append(-1)
                    token_type.append(type_id)
    return {
        "encoder": np.asarray(encoder, dtype=np.int64),
        "window": np.asarray(window, dtype=np.int64),
        "view": np.asarray(view, dtype=np.int64),
        "time": np.asarray(time, dtype=np.int64),
        "token_type": np.asarray(token_type, dtype=np.int64),
    }


class P103RichCandidateTeacher(nn.Module):
    """Candidate identity changes evidence extraction before a shared score head."""

    ATTENTION_GROUP_NAMES = (
        "encoder_videomaev2",
        "encoder_internvideo2",
        "view_scene",
        "view_person",
        "view_workspace",
        "window_early",
        "window_late",
        "time_0",
        "time_1",
        "time_2",
        "time_3",
        "time_4",
        "time_5",
        "time_6",
        "time_7",
        "type_temporal",
        "type_pooled",
        "type_action",
    )

    def __init__(self, config: RichCandidateConfig = RichCandidateConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.vmae_temporal_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.iv2_temporal_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.vmae_pooled_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.iv2_pooled_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.vmae_action_projection = nn.Sequential(nn.LayerNorm(710), nn.Linear(710, width))
        self.iv2_action_projection = nn.Sequential(nn.LayerNorm(400), nn.Linear(400, width))
        self.encoder_embedding = nn.Embedding(2, width)
        self.window_embedding = nn.Embedding(2, width)
        self.view_embedding = nn.Embedding(3, width)
        self.time_embedding = nn.Embedding(8, width)
        self.type_embedding = nn.Embedding(3, width)
        self.class_query = nn.Embedding(config.classes, width)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(7), nn.Linear(7, width * 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(width * 2, width)
        )
        self.cross_blocks = nn.ModuleList(
            [
                CandidateCrossAttentionBlock(width, config.heads, config.dropout)
                for _ in range(config.cross_layers)
            ]
        )
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.heads,
            dim_feedforward=width * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_comparison = nn.TransformerEncoder(
            candidate_layer, num_layers=config.candidate_layers
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width, 1),
        )
        topology = token_topology()
        for name, values in topology.items():
            self.register_buffer(f"token_{name}", torch.as_tensor(values), persistent=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embedding in (
            self.encoder_embedding,
            self.window_embedding,
            self.view_embedding,
            self.time_embedding,
            self.type_embedding,
            self.class_query,
        ):
            nn.init.trunc_normal_(embedding.weight, std=0.02)

    @property
    def token_count(self) -> int:
        return int(self.token_encoder.numel())

    def _identity(self, content: torch.Tensor) -> torch.Tensor:
        identity = (
            self.encoder_embedding(self.token_encoder)
            + self.window_embedding(self.token_window)
            + self.view_embedding(self.token_view)
            + self.type_embedding(self.token_token_type)
        )
        temporal = self.token_time >= 0
        identity = identity.clone()
        identity[temporal] = identity[temporal] + self.time_embedding(self.token_time[temporal])
        return content + identity[None]

    def encode_tokens(
        self,
        batch: dict[str, torch.Tensor],
        visual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temporal = torch.cat(
            (
                self.vmae_temporal_projection(batch["vmae_temporal"]).flatten(1, 3),
                self.iv2_temporal_projection(batch["iv2_temporal"]).flatten(1, 3),
            ),
            dim=1,
        )
        pooled = torch.cat(
            (
                self.vmae_pooled_projection(batch["vmae_pooled"]).flatten(1, 2),
                self.iv2_pooled_projection(batch["iv2_pooled"]).flatten(1, 2),
            ),
            dim=1,
        )
        action = torch.cat(
            (
                self.vmae_action_projection(batch["vmae_action"]).flatten(1, 2),
                self.iv2_action_projection(batch["iv2_action"]).flatten(1, 2),
            ),
            dim=1,
        )
        content = torch.cat((temporal, pooled, action), dim=1)
        if content.shape[1] != self.token_count:
            raise RuntimeError(f"rich token count changed: {content.shape[1]}")
        if visual_scale is not None:
            content = content * visual_scale[:, None, None]
        return self._identity(content)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        candidates = batch["candidate_ids"].long()
        candidate_padding = candidates < 0
        safe_candidates = candidates.clamp(min=0)
        tokens = self.encode_tokens(batch, batch.get("visual_scale"))
        query = self.class_query(safe_candidates) + self.context_projection(batch["a_context"])
        query = query.masked_fill(candidate_padding[..., None], 0.0)
        last_attention: torch.Tensor | None = None
        for index, block in enumerate(self.cross_blocks):
            query, attention = block(
                query,
                tokens,
                candidate_padding,
                return_attention and index == len(self.cross_blocks) - 1,
            )
            if attention is not None:
                last_attention = attention
        query = self.candidate_comparison(query, src_key_padding_mask=candidate_padding)
        score = self.score_head(query).squeeze(-1)
        score = score.masked_fill(candidate_padding, -1e4)
        output = {"candidate_scores": score, "candidate_evidence": query}
        if last_attention is not None:
            output["token_attention"] = last_attention
            output["attention_groups"] = self.summarize_attention(last_attention)
        return output

    def summarize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        # [B,H,K,T] -> [B,K,18]; groups intentionally overlap by semantic axis.
        value = attention.mean(dim=1)
        masks: list[torch.Tensor] = []
        masks.extend(self.token_encoder == index for index in range(2))
        masks.extend(self.token_view == index for index in range(3))
        masks.extend(self.token_window == index for index in range(2))
        masks.extend(self.token_time == index for index in range(8))
        masks.extend(self.token_token_type == index for index in range(3))
        return torch.stack([value[..., mask].sum(dim=-1) for mask in masks], dim=-1)


def trainable_parameter_audit(model: nn.Module) -> dict[str, int]:
    return {
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameters": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
    }
