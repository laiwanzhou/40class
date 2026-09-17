from __future__ import annotations

import math

import torch
from torch import nn

from p30_shared_dir_roi_model import (
    MODALITY_NAMES,
    PYRAMID_FEATURE_DIM,
    REGION_NAMES,
    ContinuousTimeEncoding,
)
from p31_skeleton_imu_model import P31SkeletonIMUPartEncoders
from p31_skeleton_imu_preprocessing import COMMON_PART_NAMES


class P32VisualRegionEncoder(nn.Module):
    """P30 frame-internal D/IR fusion without its visual-only temporal head."""

    def __init__(
        self,
        input_dim: int = PYRAMID_FEATURE_DIM,
        width: int = 256,
        frame_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.width = width
        self.input_project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.modality_embedding = nn.Parameter(
            torch.randn(1, 1, len(MODALITY_NAMES), 1, width) / math.sqrt(width)
        )
        self.region_embedding = nn.Parameter(
            torch.randn(1, 1, 1, len(REGION_NAMES), width) / math.sqrt(width)
        )
        self.source_embedding = nn.Embedding(7, width)
        self.quality_embedding = nn.Sequential(
            nn.Linear(4, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, width),
        )
        self.frame_token = nn.Parameter(torch.zeros(1, 1, width))
        frame_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=frame_layers)
        self.modality_gate = nn.Sequential(
            nn.Linear(width * 2 + 2, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )
        self.region_refine = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
        )

    def forward(
        self,
        features: torch.Tensor,
        roi_quality: torch.Tensor,
        roi_valid: torch.Tensor,
        roi_source: torch.Tensor,
        roi_clipped_ratio: torch.Tensor,
        pose_quality_factor: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 5 or features.shape[2:4] != (
            len(MODALITY_NAMES),
            len(REGION_NAMES),
        ):
            raise ValueError("features must have shape [B,T,2,7,896]")
        batch_size, time_steps, modalities, regions, _ = features.shape
        quality_inputs = torch.stack(
            (
                roi_quality,
                roi_valid.to(roi_quality.dtype),
                1.0 - roi_clipped_ratio,
                pose_quality_factor.unsqueeze(-1).expand_as(roi_quality),
            ),
            dim=-1,
        )
        projected = self.input_project(features)
        projected = projected + self.modality_embedding + self.region_embedding
        projected = projected + self.source_embedding(roi_source.clamp(0, 6)).unsqueeze(2)
        projected = projected + self.quality_embedding(quality_inputs).unsqueeze(2)

        token_valid = roi_valid.unsqueeze(2).expand(
            batch_size, time_steps, modalities, regions
        ) & frame_mask[:, :, None, None]
        flat_tokens = projected.reshape(
            batch_size * time_steps, modalities * regions, self.width
        )
        flat_valid = token_valid.reshape(batch_size * time_steps, modalities * regions)
        cls = self.frame_token.expand(batch_size * time_steps, -1, -1)
        encoded = self.frame_encoder(
            torch.cat((cls, flat_tokens), dim=1),
            src_key_padding_mask=torch.cat(
                (
                    torch.zeros(
                        batch_size * time_steps,
                        1,
                        dtype=torch.bool,
                        device=features.device,
                    ),
                    ~flat_valid,
                ),
                dim=1,
            ),
        )
        frame_context = encoded[:, 0].reshape(batch_size, time_steps, self.width)
        frame_context = frame_context * frame_mask.unsqueeze(-1)
        tokens = encoded[:, 1:].reshape(
            batch_size, time_steps, modalities, regions, self.width
        )
        depth_tokens, ir_tokens = tokens[:, :, 0], tokens[:, :, 1]
        gate_inputs = torch.cat(
            (depth_tokens, ir_tokens, roi_quality.unsqueeze(-1), roi_valid.unsqueeze(-1)),
            dim=-1,
        )
        modality_gate = torch.softmax(self.modality_gate(gate_inputs), dim=-1)
        region_sequence = (
            modality_gate[..., :1] * depth_tokens
            + modality_gate[..., 1:] * ir_tokens
        )
        region_sequence = self.region_refine(region_sequence)
        region_sequence = region_sequence * roi_valid.unsqueeze(-1) * frame_mask[:, :, None, None]
        return {
            "region_sequence": region_sequence,
            "frame_context": frame_context,
            "depth_ir_gate": modality_gate,
        }


def visual_part_region_prior() -> torch.Tensor:
    # Region order: full body, L/R arm, L/R hand, workspace, global fallback.
    allowed = torch.zeros(len(COMMON_PART_NAMES), len(REGION_NAMES), dtype=torch.bool)
    allowed[0, :] = True
    allowed[1, (0, 6)] = True
    allowed[2, (0, 1, 2, 6)] = True
    allowed[3, (0, 1, 3, 5)] = True
    allowed[4, (0, 2, 4, 5)] = True
    allowed[5, (0, 6)] = True
    allowed[6, (0, 6)] = True
    allowed[7, (0, 1, 2, 3, 4, 5)] = True
    return allowed


class VisualCommonPartAdapter(nn.Module):
    """Map seven visual ROI semantics to the shared eight-part vocabulary."""

    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.width = width
        self.register_buffer("allowed", visual_part_region_prior())
        self.register_buffer(
            "specificity",
            torch.tensor((1.0, 0.35, 0.65, 1.0, 1.0, 0.25, 0.25, 1.0)),
        )
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.part_query = nn.Parameter(
            torch.randn(len(COMMON_PART_NAMES), width) / math.sqrt(width)
        )
        self.quality_score = nn.Linear(1, 1)
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        region_sequence: torch.Tensor,
        roi_valid: torch.Tensor,
        roi_quality: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        key = self.key(region_sequence)
        value = self.value(region_sequence)
        score = torch.einsum("btrd,pd->btpr", key, self.part_query)
        score = score / math.sqrt(self.width)
        score = score + self.quality_score(roi_quality.unsqueeze(-1)).squeeze(-1).unsqueeze(2)
        candidate_mask = (
            roi_valid[:, :, None, :]
            & self.allowed[None, None]
            & frame_mask[:, :, None, None]
        )
        score = score.masked_fill(~candidate_mask, -1e4)
        weight = torch.softmax(score, dim=-1) * candidate_mask
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        part_tokens = torch.einsum("btpr,btrd->btpd", weight, value)
        part_mask = candidate_mask.any(dim=-1)
        part_tokens = self.output_norm(part_tokens) * part_mask.unsqueeze(-1)
        part_quality = torch.einsum("btpr,btr->btp", weight, roi_quality)
        part_quality = part_quality * self.specificity * part_mask

        motion_energy = torch.zeros_like(part_quality)
        if part_tokens.shape[1] > 1:
            pair_mask = part_mask[:, 1:] & part_mask[:, :-1]
            delta = torch.linalg.vector_norm(
                part_tokens[:, 1:] - part_tokens[:, :-1], dim=-1
            ) / math.sqrt(self.width)
            motion_energy[:, 1:] = delta * pair_mask
        return {
            "part_tokens": part_tokens,
            "part_mask": part_mask,
            "part_quality": part_quality.clamp(0.0, 1.0),
            "motion_energy": motion_energy,
            "region_to_part_attention": weight,
        }


class PartAwareMultimodalFusion(nn.Module):
    """Step 13: quality-aware soft fusion with a residual path for every modality."""

    MODALITY_COUNT = 3  # visual, Skeleton, IMU

    def __init__(
        self,
        width: int = 256,
        dropout: float = 0.15,
        modality_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.width = width
        self.modality_dropout = modality_dropout
        self.project = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            for _ in range(self.MODALITY_COUNT)
        )
        self.modality_embedding = nn.Parameter(
            torch.randn(self.MODALITY_COUNT, width) / math.sqrt(width)
        )
        self.part_embedding = nn.Parameter(
            torch.randn(len(COMMON_PART_NAMES), width) / math.sqrt(width)
        )
        self.reliability_embedding = nn.Sequential(
            nn.Linear(3, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, width),
        )
        self.gate_score = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        interaction_width = self.MODALITY_COUNT * width + self.MODALITY_COUNT * 3
        self.interaction = nn.Sequential(
            nn.LayerNorm(interaction_width),
            nn.Linear(interaction_width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_dropout = nn.Dropout(dropout)

    def _drop_modalities(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0:
            return mask
        drop = torch.rand(
            mask.shape[0], 1, 1, mask.shape[-1], device=mask.device
        ) < self.modality_dropout
        candidate = mask & ~drop
        all_dropped = ~candidate.any(dim=-1, keepdim=True) & mask.any(
            dim=-1, keepdim=True
        )
        return torch.where(all_dropped, mask, candidate)

    def forward(
        self,
        modality_tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        modality_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        modality_quality: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        modality_motion: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token = torch.stack(
            tuple(project(value) for project, value in zip(self.project, modality_tokens)),
            dim=3,
        )
        mask = torch.stack(modality_masks, dim=3) & frame_mask[:, :, None, None]
        quality = torch.stack(modality_quality, dim=3).clamp(0.0, 1.0)
        motion = torch.log1p(torch.stack(modality_motion, dim=3).clamp_min(0.0))
        motion = motion / (1.0 + motion)
        effective_mask = self._drop_modalities(mask)
        effective_float = effective_mask.to(quality.dtype)
        reliability = torch.stack(
            (
                quality * effective_float,
                effective_float,
                motion * effective_float,
            ),
            dim=-1,
        )
        enriched = token + self.modality_embedding[None, None, None]
        enriched = enriched + self.reliability_embedding(reliability)
        gate_logits = self.gate_score(enriched).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~effective_mask, -1e4)
        gate = torch.softmax(gate_logits, dim=3) * effective_mask
        gate = gate / gate.sum(dim=3, keepdim=True).clamp_min(1e-6)
        weighted = torch.sum(token * gate.unsqueeze(-1), dim=3)

        residual_tokens = (token * effective_mask.unsqueeze(-1)).flatten(start_dim=3)
        interaction_input = torch.cat((residual_tokens, reliability.flatten(start_dim=3)), dim=-1)
        interaction = self.interaction(interaction_input)
        part_mask = mask.any(dim=3)
        fused = weighted + interaction + self.part_embedding[None, None]
        fused = self.output_norm(fused)
        fused = self.output_dropout(fused) * part_mask.unsqueeze(-1)
        fused_quality = torch.sum(quality * gate, dim=3) * part_mask
        return {
            "fused_part_tokens": fused,
            "fused_part_mask": part_mask,
            "fused_part_quality": fused_quality,
            "modality_gate": gate,
            "effective_modality_mask": effective_mask,
        }


class TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=width,
            bias=False,
        )
        self.pointwise = nn.Conv1d(width, width * 2, kernel_size=1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        temporal = self.depthwise(source.transpose(1, 2))
        temporal = self.pointwise(temporal).transpose(1, 2)
        value, gate = temporal.chunk(2, dim=-1)
        temporal = value * torch.sigmoid(gate)
        source = self.norm(source + self.dropout(temporal))
        return source * frame_mask.unsqueeze(-1)


class CompleteJointTemporalEncoder(nn.Module):
    """Step 14: part interaction followed by complete variable-length time modeling."""

    def __init__(
        self,
        width: int = 256,
        temporal_blocks: int = 7,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        part_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=8,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.part_encoder = nn.TransformerEncoder(part_layer, num_layers=1)
        self.part_attention = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.frame_project = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
        )
        self.time_encoding = ContinuousTimeEncoding(width)
        dilations = (1, 2, 4, 8, 16, 32, 64)[:temporal_blocks]
        if len(dilations) != temporal_blocks:
            raise ValueError("temporal_blocks must be between 1 and 7")
        self.temporal_blocks = nn.ModuleList(
            TemporalResidualBlock(width, dilation, dropout) for dilation in dilations
        )
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_attention = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.embedding = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, 384),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        fused_part_tokens: torch.Tensor,
        fused_part_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        time_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, time_steps, parts, width = fused_part_tokens.shape
        flat_source = fused_part_tokens.reshape(batch_size * time_steps, parts, width)
        flat_valid = fused_part_mask.reshape(batch_size * time_steps, parts)
        safe_valid = flat_valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[empty, 0] = True
        encoded_parts = self.part_encoder(
            flat_source, src_key_padding_mask=~safe_valid
        ).reshape(batch_size, time_steps, parts, width)
        encoded_parts = encoded_parts * fused_part_mask.unsqueeze(-1)
        part_score = self.part_attention(encoded_parts).squeeze(-1)
        part_score = part_score.masked_fill(~fused_part_mask, -1e4)
        part_weight = torch.softmax(part_score, dim=2) * fused_part_mask
        part_weight = part_weight / part_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        attended = torch.sum(encoded_parts * part_weight.unsqueeze(-1), dim=2)
        global_part = encoded_parts[:, :, 0]
        frame_sequence = self.frame_project(torch.cat((attended, global_part), dim=-1))
        frame_sequence = (
            frame_sequence + self.time_encoding(time_position)
        ) * frame_mask.unsqueeze(-1)

        temporal = frame_sequence
        for block in self.temporal_blocks:
            temporal = block(temporal, frame_mask)
        temporal = self.temporal_norm(temporal)
        temporal = temporal * frame_mask.unsqueeze(-1)

        temporal_score = self.temporal_attention(temporal).squeeze(-1)
        temporal_score = temporal_score.masked_fill(~frame_mask, -1e4)
        temporal_weight = torch.softmax(temporal_score, dim=1) * frame_mask
        temporal_weight = temporal_weight / temporal_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        attended_time = torch.sum(temporal * temporal_weight.unsqueeze(-1), dim=1)
        maximum = temporal.masked_fill(~frame_mask.unsqueeze(-1), -1e4).amax(dim=1)
        mean = (temporal * frame_mask.unsqueeze(-1)).sum(dim=1) / frame_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        trial_embedding = self.embedding(torch.cat((attended_time, maximum, mean), dim=-1))
        return {
            "trial_embedding": trial_embedding,
            "temporal_sequence": temporal,
            "part_attention": part_weight,
            "temporal_attention": temporal_weight,
        }


class P32PartFusionTemporalModel(nn.Module):
    """Steps 13/14 only. A 40-way head is intentionally external for Step 15."""

    def __init__(
        self,
        width: int = 256,
        dropout: float = 0.15,
        modality_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.visual = P32VisualRegionEncoder(width=width, dropout=dropout)
        self.motion = P31SkeletonIMUPartEncoders(output_width=width, dropout=dropout)
        self.visual_parts = VisualCommonPartAdapter(width=width)
        self.fusion = PartAwareMultimodalFusion(
            width=width, dropout=dropout, modality_dropout=modality_dropout
        )
        self.temporal = CompleteJointTemporalEncoder(width=width, dropout=dropout)

    @staticmethod
    def _token_motion(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        energy = tokens.new_zeros(mask.shape)
        if tokens.shape[1] > 1:
            pair_mask = mask[:, 1:] & mask[:, :-1]
            delta = torch.linalg.vector_norm(tokens[:, 1:] - tokens[:, :-1], dim=-1)
            energy[:, 1:] = delta / math.sqrt(tokens.shape[-1]) * pair_mask
        return energy

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_regions = self.visual(
            batch["features"],
            batch["roi_quality"],
            batch["roi_valid"],
            batch["roi_source"],
            batch["roi_clipped_ratio"],
            batch["pose_quality_factor"],
            batch["frame_mask"],
        )
        visual = self.visual_parts(
            visual_regions["region_sequence"],
            batch["roi_valid"],
            batch["roi_quality"],
            batch["frame_mask"],
        )
        motion = self.motion(batch)
        imu_motion = self._token_motion(
            motion["imu_part_tokens"], motion["imu_part_mask"]
        )
        fusion = self.fusion(
            (
                visual["part_tokens"],
                motion["skeleton_part_tokens"],
                motion["imu_part_tokens"],
            ),
            (
                visual["part_mask"],
                motion["skeleton_part_mask"],
                motion["imu_part_mask"],
            ),
            (
                visual["part_quality"],
                motion["skeleton_part_quality"],
                motion["imu_part_quality"],
            ),
            (
                visual["motion_energy"],
                motion["skeleton_motion_energy"],
                imu_motion,
            ),
            batch["frame_mask"],
        )
        temporal = self.temporal(
            fusion["fused_part_tokens"],
            fusion["fused_part_mask"],
            batch["frame_mask"],
            batch["time_position"],
        )
        return {
            **temporal,
            **fusion,
            "visual_part_tokens": visual["part_tokens"],
            "visual_part_mask": visual["part_mask"],
            "skeleton_part_tokens": motion["skeleton_part_tokens"],
            "skeleton_part_mask": motion["skeleton_part_mask"],
            "imu_part_tokens": motion["imu_part_tokens"],
            "imu_part_mask": motion["imu_part_mask"],
            "depth_ir_gate": visual_regions["depth_ir_gate"],
            "region_to_part_attention": visual["region_to_part_attention"],
            "imu_interval_count": motion["imu_interval_count"],
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def model_size_mib(module: nn.Module, bytes_per_parameter: int = 4) -> float:
    return parameter_count(module) * bytes_per_parameter / (1024**2)
