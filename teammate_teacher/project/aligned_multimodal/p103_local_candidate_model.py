"""Explicit local-crop candidate-query model for P103-B3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from p103_rich_candidate_model import CandidateCrossAttentionBlock


@dataclass(frozen=True)
class LocalCandidateConfig:
    width: int = 128
    heads: int = 4
    cross_layers: int = 2
    candidate_layers: int = 1
    dropout: float = 0.15
    classes: int = 40

    def validate(self) -> None:
        if self.width % self.heads:
            raise ValueError("width must divide attention heads")
        if (self.cross_layers, self.candidate_layers) != (2, 1):
            raise ValueError("P103-B3 architecture depth is frozen at 2+1")
        if self.classes != 40:
            raise ValueError("P103 candidate identity contract requires 40 classes")


def local_token_topology() -> dict[str, np.ndarray]:
    # ROI: workspace/left/right/interaction. Window: full/early/middle/late/peak.
    vmae_encoder = [0] * 6
    vmae_roi = [1, 2, 3, 1, 2, 3]
    vmae_window = [0, 0, 0, 4, 4, 4]
    vjepa_encoder = [1] * 16
    vjepa_roi = [0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    vjepa_window = [0, 1, 2, 3, 0, 0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 4]
    encoder = np.asarray([*vmae_encoder, *vjepa_encoder] * 2, dtype=np.int64)
    roi = np.asarray([*vmae_roi, *vjepa_roi] * 2, dtype=np.int64)
    window = np.asarray([*vmae_window, *vjepa_window] * 2, dtype=np.int64)
    token_type = np.asarray([0] * 22 + [1] * 22, dtype=np.int64)
    return {"encoder": encoder, "roi": roi, "window": window, "token_type": token_type}


class P103LocalCandidateTeacher(nn.Module):
    """Candidate identity changes local-crop evidence extraction before scoring."""

    ATTENTION_GROUP_NAMES = (
        "encoder_videomaev2",
        "encoder_vjepa2",
        "roi_workspace",
        "roi_left_hand",
        "roi_right_hand",
        "roi_interaction",
        "window_full",
        "window_early",
        "window_middle",
        "window_late",
        "window_motion_peak",
        "type_feature",
        "type_action",
    )

    def __init__(self, config: LocalCandidateConfig = LocalCandidateConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.vmae_feature_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.vmae_action_projection = nn.Sequential(nn.LayerNorm(710), nn.Linear(710, width))
        self.vjepa_feature_projection = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width))
        self.vjepa_action_projection = nn.Sequential(nn.LayerNorm(174), nn.Linear(174, width))
        self.encoder_embedding = nn.Embedding(2, width)
        self.roi_embedding = nn.Embedding(4, width)
        self.window_embedding = nn.Embedding(5, width)
        self.type_embedding = nn.Embedding(2, width)
        self.class_query = nn.Embedding(config.classes, width)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, width * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width * 2, width),
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
        for name, values in local_token_topology().items():
            self.register_buffer(f"token_{name}", torch.as_tensor(values), persistent=True)
        for embedding in (
            self.encoder_embedding,
            self.roi_embedding,
            self.window_embedding,
            self.type_embedding,
            self.class_query,
        ):
            nn.init.trunc_normal_(embedding.weight, std=0.02)

    @property
    def token_count(self) -> int:
        return int(self.token_encoder.numel())

    def encode_tokens(
        self, batch: dict[str, torch.Tensor], local_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        feature = torch.cat(
            (
                self.vmae_feature_projection(batch["vmae_features"]),
                self.vjepa_feature_projection(batch["vjepa_features"]),
            ),
            dim=1,
        )
        action = torch.cat(
            (
                self.vmae_action_projection(batch["vmae_actions"]),
                self.vjepa_action_projection(batch["vjepa_actions"]),
            ),
            dim=1,
        )
        content = torch.cat((feature, action), dim=1)
        if content.shape[1] != self.token_count:
            raise RuntimeError(f"local token count changed: {content.shape[1]}")
        if local_scale is not None:
            content = content * local_scale[:, None, None]
        identity = (
            self.encoder_embedding(self.token_encoder)
            + self.roi_embedding(self.token_roi)
            + self.window_embedding(self.token_window)
            + self.type_embedding(self.token_token_type)
        )
        return content + identity[None]

    def forward(
        self, batch: dict[str, torch.Tensor], return_attention: bool = False
    ) -> dict[str, torch.Tensor]:
        candidates = batch["candidate_ids"].long()
        padding = candidates < 0
        safe_candidates = candidates.clamp(min=0)
        tokens = self.encode_tokens(batch, batch.get("local_scale"))
        query = self.class_query(safe_candidates) + self.context_projection(batch["a_context"])
        query = query.masked_fill(padding[..., None], 0.0)
        last_attention: torch.Tensor | None = None
        for index, block in enumerate(self.cross_blocks):
            query, attention = block(
                query,
                tokens,
                padding,
                return_attention and index == len(self.cross_blocks) - 1,
            )
            if attention is not None:
                last_attention = attention
        query = self.candidate_comparison(query, src_key_padding_mask=padding)
        score = self.score_head(query).squeeze(-1).masked_fill(padding, -1e4)
        output = {"candidate_scores": score, "candidate_evidence": query}
        if last_attention is not None:
            output["token_attention"] = last_attention
            output["attention_groups"] = self.summarize_attention(last_attention)
        return output

    def summarize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        value = attention.mean(dim=1)
        masks: list[torch.Tensor] = []
        masks.extend(self.token_encoder == index for index in range(2))
        masks.extend(self.token_roi == index for index in range(4))
        masks.extend(self.token_window == index for index in range(5))
        masks.extend(self.token_token_type == index for index in range(2))
        return torch.stack([value[..., mask].sum(dim=-1) for mask in masks], dim=-1)


def trainable_parameter_audit(model: nn.Module) -> dict[str, int]:
    return {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
