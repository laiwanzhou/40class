from __future__ import annotations

import math

import torch
from torch import nn

from p32_part_fusion_temporal_model import P32PartFusionTemporalModel


class TemporalResidual(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, padding=dilation, dilation=dilation, groups=width
        )
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(source.transpose(1, 2))
        residual = self.pointwise(torch.nn.functional.gelu(residual)).transpose(1, 2)
        return self.norm(source + self.dropout(residual)) * mask.unsqueeze(-1)


class SpatialROIEncoder(nn.Module):
    """Preserve within-ROI geometry and explicitly expose adjacent-frame events."""

    def __init__(self, input_dim: int = 128, width: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.width = width
        self.input_project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.difference_project = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        scale = 1.0 / math.sqrt(width)
        self.modality_embedding = nn.Parameter(torch.randn(1, 1, 2, 1, 1, 1, width) * scale)
        self.region_embedding = nn.Parameter(torch.randn(1, 1, 1, 3, 1, 1, width) * scale)
        self.row_embedding = nn.Parameter(torch.randn(1, 1, 1, 1, 3, 1, width) * scale)
        self.column_embedding = nn.Parameter(torch.randn(1, 1, 1, 1, 1, 3, width) * scale)
        frame_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=1)
        self.quality_gate = nn.Sequential(
            nn.Linear(width + 3, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalResidual(width, dilation, dropout) for dilation in (1, 2, 4)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.event_attention = nn.Sequential(
            nn.Linear(width + 1, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        self.embedding = nn.Sequential(
            nn.Linear(width * 3, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        quality: torch.Tensor,
        clipped: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 7 or features.shape[2:6] != (2, 3, 3, 3):
            raise ValueError("spatial_features must be [B,T,2,3,3,3,128]")
        batch, steps = features.shape[:2]
        difference = torch.zeros_like(features)
        if steps > 1:
            difference[:, 1:] = (features[:, 1:] - features[:, :-1]).abs()
        tokens = self.input_project(features) + self.difference_project(difference)
        tokens = (
            tokens
            + self.modality_embedding
            + self.region_embedding
            + self.row_embedding
            + self.column_embedding
        )
        token_mask = valid[:, :, None, :, None, None].expand(batch, steps, 2, 3, 3, 3)
        token_mask = token_mask & frame_mask[:, :, None, None, None, None]
        flat = tokens.reshape(batch * steps, 54, self.width)
        flat_mask = token_mask.reshape(batch * steps, 54)
        # Every real trial has valid fallback-derived hand/workspace tokens.  The
        # guard prevents all-masked synthetic padding frames producing NaNs.
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        encoded = self.frame_encoder(flat, src_key_padding_mask=~safe_mask)
        encoded = encoded * flat_mask.unsqueeze(-1)
        encoded = encoded.reshape(batch, steps, 2, 3, 3, 3, self.width)
        denominator = token_mask.to(encoded.dtype).sum(dim=(2, 4, 5)).clamp_min(1.0)
        regions = encoded.sum(dim=(2, 4, 5)) / denominator.unsqueeze(-1)

        event = difference.square().mean(dim=(2, 4, 5, 6)).sqrt() * valid
        quality_input = torch.cat(
            (regions, quality.unsqueeze(-1), valid.unsqueeze(-1), (1.0 - clipped).unsqueeze(-1)),
            dim=-1,
        )
        region_score = self.quality_gate(quality_input).squeeze(-1)
        region_score = region_score.masked_fill(~valid, -1e4)
        region_weight = torch.softmax(region_score, dim=2) * valid
        region_weight = region_weight / region_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        sequence = (regions * region_weight.unsqueeze(-1)).sum(dim=2)
        frame_event = (event * region_weight).sum(dim=2)
        sequence = sequence * frame_mask.unsqueeze(-1)
        for block in self.temporal_blocks:
            sequence = block(sequence, frame_mask)
        sequence = self.temporal_encoder(sequence, src_key_padding_mask=~frame_mask)
        sequence = sequence * frame_mask.unsqueeze(-1)

        attention_score = self.event_attention(
            torch.cat((sequence, torch.log1p(frame_event).unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        attention_score = attention_score.masked_fill(~frame_mask, -1e4)
        attention = torch.softmax(attention_score, dim=1)
        attended = (sequence * attention.unsqueeze(-1)).sum(dim=1)
        maximum = sequence.masked_fill(~frame_mask.unsqueeze(-1), -1e4).amax(dim=1)
        mean = (sequence * frame_mask.unsqueeze(-1)).sum(dim=1) / frame_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return {
            "embedding": self.embedding(torch.cat((attended, maximum, mean), dim=-1)),
            "sequence": sequence,
            "event_energy": frame_event,
            "event_attention": attention,
            "region_weight": region_weight,
        }


class P44CEncoder(nn.Module):
    def __init__(self, width: int = 192, dropout: float = 0.15) -> None:
        super().__init__()
        self.multimodal = P32PartFusionTemporalModel(
            width=width, dropout=dropout, modality_dropout=0.08
        )
        self.spatial = SpatialROIEncoder(width=128, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(384 + 256, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        multimodal = self.multimodal(batch)
        spatial = self.spatial(
            batch["spatial_features"],
            batch["spatial_valid"],
            batch["spatial_quality"],
            batch["spatial_clipped_ratio"],
            batch["frame_mask"],
        )
        embedding = self.fusion(
            torch.cat((multimodal["trial_embedding"], spatial["embedding"]), dim=-1)
        )
        return {
            "embedding": embedding,
            "multimodal_embedding": multimodal["trial_embedding"],
            "spatial_embedding": spatial["embedding"],
            "spatial_event_energy": spatial["event_energy"],
            "spatial_event_attention": spatial["event_attention"],
            "spatial_region_weight": spatial["region_weight"],
        }


class P44CPretrainModel(nn.Module):
    def __init__(self, num_classes: int = 40) -> None:
        super().__init__()
        self.encoder = P44CEncoder()
        self.classifier = nn.Linear(384, num_classes)
        self.multimodal_aux = nn.Linear(384, num_classes)
        self.spatial_aux = nn.Linear(256, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.encoder(batch)
        return {
            **output,
            "logits": self.classifier(output["embedding"]),
            "multimodal_logits": self.multimodal_aux(output["multimodal_embedding"]),
            "spatial_logits": self.spatial_aux(output["spatial_embedding"]),
        }


GROUP_CLASS_IDS = (6, 7, 8, 9, 10, 11, 14, 37)
INTAKE_GROUP_INDICES = (0, 1, 7)
OPERATION_GROUP_INDICES = (2, 3, 4, 5, 6)


class P44CStructuredExpert(nn.Module):
    """Two structured heads over one 40-class-pretrained shared encoder."""

    def __init__(self, encoder: P44CEncoder, dropout: float = 0.25) -> None:
        super().__init__()
        self.encoder = encoder
        context_dim = 384 + 3 + 3 + 2
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, 192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.intake = nn.Sequential(
            nn.Linear(192, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3)
        )
        self.operation = nn.Sequential(
            nn.Linear(192, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 5)
        )
        self.subgroup = nn.Linear(192, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch)
        valid = batch["spatial_valid"] & batch["frame_mask"].unsqueeze(-1)
        valid_float = valid.to(batch["spatial_quality"].dtype)
        quality = (batch["spatial_quality"] * valid_float).sum(dim=1) / valid_float.sum(
            dim=1
        ).clamp_min(1.0)
        valid_ratio = valid_float.sum(dim=1) / batch["frame_mask"].sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        event = encoded["spatial_event_energy"]
        event_mean = (event * batch["frame_mask"]).sum(dim=1) / batch["frame_mask"].sum(
            dim=1
        ).clamp_min(1)
        event_max = event.masked_fill(~batch["frame_mask"], -1e4).amax(dim=1)
        context = self.context(
            torch.cat(
                (
                    encoded["embedding"],
                    quality,
                    valid_ratio,
                    event_mean.unsqueeze(-1),
                    event_max.unsqueeze(-1),
                ),
                dim=1,
            )
        )
        subgroup_logits = self.subgroup(context)
        subgroup_log_probability = torch.log_softmax(subgroup_logits, dim=1)
        intake_logits = self.intake(context) + subgroup_log_probability[:, :1]
        operation_logits = self.operation(context) + subgroup_log_probability[:, 1:]
        # Concatenate without in-place indexed assignment: autocast keeps the
        # head output and log-softmax in different internal dtypes on CUDA.
        concatenated = torch.cat((intake_logits, operation_logits), dim=1)
        logits = concatenated[:, [0, 1, 3, 4, 5, 6, 7, 2]]
        return {
            **encoded,
            "group_logits": logits,
            "subgroup_logits": subgroup_logits,
            "expert_embedding": context,
            "mean_roi_quality": quality.mean(dim=1),
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
