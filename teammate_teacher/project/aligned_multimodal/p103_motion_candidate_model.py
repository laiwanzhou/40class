"""Staged local-visual/Skeleton/IMU candidate interaction for P103-B4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from p103_local_candidate_model import local_token_topology
from p86_mobind_lite_model import P86IMUPartEncoder, P86SkeletonPartEncoder


@dataclass(frozen=True)
class MotionCandidateConfig:
    width: int = 128
    heads: int = 4
    dropout: float = 0.15
    classes: int = 40

    def validate(self) -> None:
        if self.width % self.heads:
            raise ValueError("width must divide attention heads")
        if self.classes != 40:
            raise ValueError("P103 candidate identity contract requires 40 classes")


class MaskedCandidateAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
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
        token_padding: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_padding is not None:
            token_padding = token_padding.clone()
            empty = token_padding.all(dim=1)
            token_padding[empty, 0] = False
        attended, attention = self.attention(
            self.query_norm(query),
            self.token_norm(tokens),
            self.token_norm(tokens),
            key_padding_mask=token_padding,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query = query + self.dropout(attended)
        query = query + self.feed_forward(self.ff_norm(query))
        query = query.masked_fill(candidate_padding[..., None], 0.0)
        return query, attention


class P103MotionCandidateTeacher(nn.Module):
    """Candidate-local evidence is followed by separate Skeleton/IMU reads."""

    LOCAL_GROUP_NAMES = (
        "local_encoder_videomaev2",
        "local_encoder_vjepa2",
        "local_roi_workspace",
        "local_roi_left_hand",
        "local_roi_right_hand",
        "local_roi_interaction",
        "local_window_full",
        "local_window_early",
        "local_window_middle",
        "local_window_late",
        "local_window_motion_peak",
        "local_type_feature",
        "local_type_action",
    )
    MOTION_GROUP_NAMES = tuple(
        [f"skeleton_window_{index}" for index in range(2)]
        + [f"skeleton_part_{index}" for index in range(5)]
        + [f"imu_window_{index}" for index in range(2)]
        + [f"imu_device_{index}" for index in range(5)]
    )
    ATTENTION_GROUP_NAMES = LOCAL_GROUP_NAMES + MOTION_GROUP_NAMES

    def __init__(self, config: MotionCandidateConfig = MotionCandidateConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.vmae_feature_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.vmae_action_projection = nn.Sequential(nn.LayerNorm(710), nn.Linear(710, width))
        self.vjepa_feature_projection = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width))
        self.vjepa_action_projection = nn.Sequential(nn.LayerNorm(174), nn.Linear(174, width))
        self.local_encoder_embedding = nn.Embedding(2, width)
        self.local_roi_embedding = nn.Embedding(4, width)
        self.local_window_embedding = nn.Embedding(5, width)
        self.local_type_embedding = nn.Embedding(2, width)

        self.skeleton_encoder = P86SkeletonPartEncoder(
            width=width,
            dropout=config.dropout,
            explicit_time_position=True,
            multistream_input=True,
            adaptive_graph=True,
        )
        self.imu_encoder = P86IMUPartEncoder(
            width=width, dropout=config.dropout, instance_normalization=True
        )
        self.motion_modality_embedding = nn.Embedding(2, width)
        self.motion_window_embedding = nn.Embedding(2, width)
        self.motion_time_embedding = nn.Embedding(16, width)
        self.motion_part_embedding = nn.Embedding(5, width)

        self.class_query = nn.Embedding(config.classes, width)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, width * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width * 2, width),
        )
        self.local_attention = MaskedCandidateAttention(width, config.heads, config.dropout)
        self.skeleton_attention = MaskedCandidateAttention(width, config.heads, config.dropout)
        self.imu_attention = MaskedCandidateAttention(width, config.heads, config.dropout)
        self.cross_modal_fusion = nn.Sequential(
            nn.LayerNorm(width * 7),
            nn.Linear(width * 7, width * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width * 2, width),
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
        self.candidate_comparison = nn.TransformerEncoder(candidate_layer, num_layers=1)
        self.score_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width, 1),
        )
        for name, values in local_token_topology().items():
            self.register_buffer(f"local_token_{name}", torch.as_tensor(values), persistent=True)
        for embedding in (
            self.local_encoder_embedding,
            self.local_roi_embedding,
            self.local_window_embedding,
            self.local_type_embedding,
            self.motion_modality_embedding,
            self.motion_window_embedding,
            self.motion_time_embedding,
            self.motion_part_embedding,
            self.class_query,
        ):
            nn.init.trunc_normal_(embedding.weight, std=0.02)

    def encode_local(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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
        identity = (
            self.local_encoder_embedding(self.local_token_encoder)
            + self.local_roi_embedding(self.local_token_roi)
            + self.local_window_embedding(self.local_token_window)
            + self.local_type_embedding(self.local_token_token_type)
        )
        return content + identity[None]

    @staticmethod
    def _motion_fields(
        batch: dict[str, torch.Tensor], modality: str, prefix: str
    ) -> tuple[torch.Tensor, ...]:
        names = (
            (
                "skeleton_features",
                "skeleton_feature_mask",
                "skeleton_joint_mask",
                "skeleton_relations",
                "skeleton_relation_mask",
                "skeleton_frame_quality",
            )
            if modality == "skeleton"
            else (
                "imu_sequences",
                "imu_sequence_mask",
                "imu_bin_statistics",
                "imu_bin_mask",
                "imu_global_statistics",
                "imu_global_mask",
            )
        )
        return tuple(batch[prefix + name] for name in names)

    def _motion_identity(self, modality: int, device: torch.device) -> torch.Tensor:
        window = torch.arange(2, device=device)[:, None, None].expand(2, 16, 5)
        time = torch.arange(16, device=device)[None, :, None].expand(2, 16, 5)
        part = torch.arange(5, device=device)[None, None, :].expand(2, 16, 5)
        return (
            self.motion_modality_embedding(torch.tensor(modality, device=device))
            + self.motion_window_embedding(window)
            + self.motion_time_embedding(time)
            + self.motion_part_embedding(part)
        ).reshape(160, -1)

    def encode_motion(
        self, batch: dict[str, torch.Tensor], prefix: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skeleton, skeleton_mask = self.skeleton_encoder(
            *self._motion_fields(batch, "skeleton", prefix), batch["motion_time"]
        )
        imu, imu_mask = self.imu_encoder(*self._motion_fields(batch, "imu", prefix))
        skeleton_scale = batch.get(prefix + "skeleton_scale", batch["skeleton_scale"])
        imu_scale = batch.get(prefix + "imu_scale", batch["imu_scale"])
        skeleton = skeleton * skeleton_scale[:, None, None, None, None]
        imu = imu * imu_scale[:, None, None, None, None]
        skeleton = skeleton.reshape(len(skeleton), 160, -1)
        imu = imu.reshape(len(imu), 160, -1)
        skeleton = skeleton + self._motion_identity(0, skeleton.device)[None]
        imu = imu + self._motion_identity(1, imu.device)[None]
        return (
            skeleton,
            skeleton_mask.reshape(len(skeleton_mask), 160),
            imu,
            imu_mask.reshape(len(imu_mask), 160),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        motion_prefix: str = "",
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        candidates = batch["candidate_ids"].long()
        candidate_padding = candidates < 0
        query = self.class_query(candidates.clamp(min=0)) + self.context_projection(
            batch["a_context"]
        )
        query = query.masked_fill(candidate_padding[..., None], 0.0)
        local = self.encode_local(batch)
        local_query, local_attention = self.local_attention(
            query, local, candidate_padding, return_attention=return_attention
        )
        skeleton, skeleton_mask, imu, imu_mask = self.encode_motion(batch, motion_prefix)
        skeleton_query, skeleton_attention = self.skeleton_attention(
            local_query,
            skeleton,
            candidate_padding,
            token_padding=~skeleton_mask,
            return_attention=return_attention,
        )
        imu_query, imu_attention = self.imu_attention(
            local_query,
            imu,
            candidate_padding,
            token_padding=~imu_mask,
            return_attention=return_attention,
        )
        fused = self.cross_modal_fusion(
            torch.cat(
                (
                    local_query,
                    skeleton_query,
                    imu_query,
                    local_query * skeleton_query,
                    local_query * imu_query,
                    skeleton_query * imu_query,
                    (skeleton_query - imu_query).abs(),
                ),
                dim=-1,
            )
        )
        fused = fused.masked_fill(candidate_padding[..., None], 0.0)
        fused = self.candidate_comparison(fused, src_key_padding_mask=candidate_padding)
        score = self.score_head(fused).squeeze(-1).masked_fill(candidate_padding, -1e4)
        output = {
            "candidate_scores": score,
            "local_candidate_evidence": local_query,
            "skeleton_candidate_evidence": skeleton_query,
            "imu_candidate_evidence": imu_query,
        }
        if return_attention:
            assert local_attention is not None
            assert skeleton_attention is not None
            assert imu_attention is not None
            output["attention_groups"] = self.summarize_attention(
                local_attention, skeleton_attention, imu_attention
            )
        return output

    def summarize_attention(
        self,
        local_attention: torch.Tensor,
        skeleton_attention: torch.Tensor,
        imu_attention: torch.Tensor,
    ) -> torch.Tensor:
        local = local_attention.mean(dim=1)
        local_masks: list[torch.Tensor] = []
        local_masks.extend(self.local_token_encoder == index for index in range(2))
        local_masks.extend(self.local_token_roi == index for index in range(4))
        local_masks.extend(self.local_token_window == index for index in range(5))
        local_masks.extend(self.local_token_token_type == index for index in range(2))
        groups = [local[..., mask].sum(dim=-1) for mask in local_masks]
        for attention in (skeleton_attention, imu_attention):
            value = attention.mean(dim=1).reshape(*attention.shape[:1], attention.shape[2], 2, 16, 5)
            groups.extend(value.sum(dim=(-2, -1))[..., index] for index in range(2))
            groups.extend(value.sum(dim=(-3, -2))[..., index] for index in range(5))
        return torch.stack(groups, dim=-1)


def trainable_parameter_audit(model: nn.Module) -> dict[str, int]:
    return {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
