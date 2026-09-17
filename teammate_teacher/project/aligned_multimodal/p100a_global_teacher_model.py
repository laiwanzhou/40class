"""Hierarchical Visual-conditioned pre-classification fusion for P100-A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from p91_hierarchical_multimodal_teacher import (
    CTRSkeletonEncoder,
    DevicewiseIMUEncoder,
)


@dataclass(frozen=True)
class P100AModelConfig:
    modalities: tuple[str, ...] = ("visual", "skeleton", "imu")
    model_dim: int = 384
    heads: int = 8
    modality_layers: int = 2
    fusion_layers: int = 4
    fusion_latents: int = 8
    dropout: float = 0.20
    evidence_dropout: float = 0.10
    protected_imu_residual: bool = False
    protected_imu_max_scale: float = 0.50

    def __post_init__(self) -> None:
        allowed = {"visual", "skeleton", "imu"}
        if not self.modalities or self.modalities[0] != "visual":
            raise ValueError("P100-A variants must contain Visual first")
        if set(self.modalities) - allowed:
            raise ValueError(f"unknown modalities: {self.modalities}")
        if self.model_dim <= 0 or self.model_dim % self.heads:
            raise ValueError("model_dim must be positive and divisible by heads")
        if self.modality_layers <= 0 or self.fusion_layers <= 0:
            raise ValueError("encoder depths must be positive")
        if not 0 <= self.dropout < 1 or not 0 <= self.evidence_dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.protected_imu_residual:
            if not {"skeleton", "imu"}.issubset(self.modalities):
                raise ValueError("protected IMU residual requires Visual+Skeleton+IMU")
            if self.protected_imu_max_scale <= 0:
                raise ValueError("protected_imu_max_scale must be positive")


def transformer_encoder(
    model_dim: int, heads: int, layers: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=heads,
        dim_feedforward=model_dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class StatisticsTokens(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, tokens: int) -> None:
        super().__init__()
        self.tokens = tokens
        self.model_dim = model_dim
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim * tokens),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.project(values).reshape(-1, self.tokens, self.model_dim)


class VisualSemanticEncoder(nn.Module):
    def __init__(self, config: P100AModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.project = nn.ModuleDict(
            {
                "visual_vmae": nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim)),
                "visual_iv2": nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim)),
                "visual_vmae_action": nn.Sequential(
                    nn.LayerNorm(710), nn.Linear(710, dim)
                ),
                "visual_iv2_action": nn.Sequential(
                    nn.LayerNorm(400), nn.Linear(400, dim)
                ),
            }
        )
        self.type_embedding = nn.Parameter(torch.randn(4, 1, dim) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, 6, dim) * 0.01)
        self.encoder = transformer_encoder(
            dim, config.heads, config.modality_layers, config.dropout
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values = []
        for index, (name, project) in enumerate(self.project.items()):
            token = project(batch[name])
            token = token + self.type_embedding[index] + self.position_embedding
            values.append(token)
        return self.norm(self.encoder(torch.cat(values, dim=1)))


class SkeletonEvidenceEncoder(nn.Module):
    def __init__(self, config: P100AModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.motionbert = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
        self.hdgcn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, dim))
        self.raw = CTRSkeletonEncoder(dim)
        self.statistics = StatisticsTokens(2640, dim, tokens=2)
        self.type_embedding = nn.Parameter(torch.randn(4, 1, dim) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, 46, dim) * 0.01)
        self.encoder = transformer_encoder(
            dim, config.heads, config.modality_layers, config.dropout
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        motionbert = self.motionbert(batch["skeleton_motionbert"])
        # Six HD-GCN streams × 16 temporal tokens -> six × four phase tokens.
        hdgcn = batch["skeleton_hdgcn"].reshape(-1, 6, 4, 4, 256).mean(dim=3)
        hdgcn = self.hdgcn(hdgcn.reshape(-1, 24, 256))
        raw = self.raw(batch["skeleton_sequence"])
        statistics = self.statistics(batch["skeleton_statistics"])
        groups = (motionbert, hdgcn, raw, statistics)
        decorated = [
            values + self.type_embedding[index]
            for index, values in enumerate(groups)
        ]
        values = torch.cat(decorated, dim=1)
        values = values + self.position_embedding[:, : values.shape[1]]
        return self.norm(self.encoder(values))


class IMUEvidenceEncoder(nn.Module):
    def __init__(self, config: P100AModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.raw = DevicewiseIMUEncoder(dim, config.heads, config.dropout)
        self.statistics = StatisticsTokens(3155, dim, tokens=3)
        self.type_embedding = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, 11, dim) * 0.01)
        self.encoder = transformer_encoder(
            dim, config.heads, config.modality_layers, config.dropout
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = self.raw(batch["imu_sequence"], batch["imu_mask"])
        statistics = self.statistics(batch["imu_statistics"])
        values = torch.cat(
            (
                raw + self.type_embedding[0],
                statistics + self.type_embedding[1],
            ),
            dim=1,
        )
        values = values + self.position_embedding[:, : values.shape[1]]
        return self.norm(self.encoder(values))


class CrossAttentionResidual(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        delta, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        return self.dropout(delta)


class FusionBlock(nn.Module):
    def __init__(self, config: P100AModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.modalities = config.modalities
        self.visual = CrossAttentionResidual(dim, config.heads, config.dropout)
        self.evidence = nn.ModuleDict(
            {
                name: CrossAttentionResidual(dim, config.heads, config.dropout)
                for name in ("skeleton", "imu", "cross")
                if (
                    name in config.modalities
                    or name == "cross"
                    and {"skeleton", "imu"}.issubset(config.modalities)
                )
            }
        )
        self.reliability = nn.ModuleDict(
            {name: nn.Linear(dim * 2, 1) for name in self.evidence}
        )
        for gate in self.reliability.values():
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.protected_scale = nn.ParameterDict()
        if config.protected_imu_residual:
            for name in ("imu", "cross"):
                if name in self.evidence:
                    # Exact zero makes the protected model identical to its
                    # source-safe VS anchor before adapter training.
                    self.protected_scale[name] = nn.Parameter(torch.zeros(()))
        self.protected_imu_max_scale = config.protected_imu_max_scale
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        latent: torch.Tensor,
        visual: torch.Tensor,
        evidence: dict[str, torch.Tensor],
        availability: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        latent = latent + self.visual(latent, visual)
        gates: dict[str, torch.Tensor] = {}
        residual_norms: dict[str, torch.Tensor] = {}
        evidence_scales: dict[str, torch.Tensor] = {}
        for name, attention in self.evidence.items():
            context = evidence[name]
            delta = attention(latent, context)
            gate_input = torch.cat((latent.mean(dim=1), context.mean(dim=1)), dim=1)
            gate = torch.sigmoid(self.reliability[name](gate_input))
            gate = gate * availability[name].view(-1, 1)
            scale = torch.ones((), device=latent.device, dtype=latent.dtype)
            if name in self.protected_scale:
                scale = self.protected_imu_max_scale * torch.tanh(
                    self.protected_scale[name]
                )
            latent = latent + delta * gate[:, None] * scale
            gates[name] = gate[:, 0]
            evidence_scales[name] = scale.expand(latent.shape[0])
            residual_norms[name] = (
                delta.square().mean(dim=(1, 2)).sqrt()
                * gate[:, 0]
                * scale.abs()
            )
        value = self.self_norm(latent)
        delta, _ = self.self_attention(value, value, value, need_weights=False)
        latent = latent + delta
        latent = latent + self.ffn(self.ffn_norm(latent))
        return latent, gates, residual_norms, evidence_scales


class P100AGlobalTeacher(nn.Module):
    """One 40-class head after hierarchical cross-modal interaction."""

    def __init__(self, config: P100AModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.visual_encoder = VisualSemanticEncoder(config)
        self.skeleton_encoder = (
            SkeletonEvidenceEncoder(config)
            if "skeleton" in config.modalities
            else None
        )
        self.imu_encoder = (
            IMUEvidenceEncoder(config) if "imu" in config.modalities else None
        )
        self.cross_statistics = (
            StatisticsTokens(400, dim, tokens=1)
            if {"skeleton", "imu"}.issubset(config.modalities)
            else None
        )
        self.latent = nn.Parameter(torch.randn(1, config.fusion_latents, dim) * 0.02)
        self.visual_seed = CrossAttentionResidual(dim, config.heads, config.dropout)
        self.blocks = nn.ModuleList(
            [FusionBlock(config) for _ in range(config.fusion_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 2, 40),
        )

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def _availability(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for name in ("skeleton", "imu"):
            if name not in self.config.modalities:
                continue
            available = batch[f"{name}_available"].to(torch.float32)
            if self.training and self.config.evidence_dropout > 0:
                keep = (
                    torch.rand_like(available) >= self.config.evidence_dropout
                ).to(torch.float32)
                available = available * keep
            output[name] = available
        if {"skeleton", "imu"}.issubset(self.config.modalities):
            output["cross"] = (
                output["skeleton"]
                * output["imu"]
                * batch["cross_available"].to(torch.float32)
            )
        return output

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        visual = self.visual_encoder(batch)
        availability = self._availability(batch)
        evidence: dict[str, torch.Tensor] = {}
        if self.skeleton_encoder is not None:
            evidence["skeleton"] = self.skeleton_encoder(batch)
        if self.imu_encoder is not None:
            evidence["imu"] = self.imu_encoder(batch)
        if self.cross_statistics is not None:
            evidence["cross"] = self.cross_statistics(batch["cross_statistics"])

        latent = self.latent.expand(visual.shape[0], -1, -1)
        latent = latent + self.visual_seed(latent, visual)
        gates: dict[str, list[torch.Tensor]] = {name: [] for name in evidence}
        residual_norms: dict[str, list[torch.Tensor]] = {name: [] for name in evidence}
        evidence_scales: dict[str, list[torch.Tensor]] = {
            name: [] for name in evidence
        }
        for block in self.blocks:
            latent, block_gates, block_norms, block_scales = block(
                latent, visual, evidence, availability
            )
            for name in block_gates:
                gates[name].append(block_gates[name])
                residual_norms[name].append(block_norms[name])
                evidence_scales[name].append(block_scales[name])
        representation = self.norm(latent).mean(dim=1)
        return {
            "logits": self.classifier(representation),
            "representation": representation,
            "reliability": {
                name: torch.stack(values, dim=1).mean(dim=1)
                for name, values in gates.items()
            },
            "evidence_residual_norm": {
                name: torch.stack(values, dim=1).mean(dim=1)
                for name, values in residual_norms.items()
            },
            "evidence_scale": {
                name: torch.stack(values, dim=1).mean(dim=1)
                for name, values in evidence_scales.items()
            },
        }
