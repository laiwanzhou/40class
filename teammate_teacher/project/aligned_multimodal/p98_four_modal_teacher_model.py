"""Configurable pre-classification fusion model for IR/Depth/Skeleton/IMU."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from p91_hierarchical_multimodal_teacher import CTRSkeletonEncoder, DevicewiseIMUEncoder
from p98_four_modal_teacher_data import CANONICAL_MODALITIES, validate_modalities


@dataclass(frozen=True)
class FourModalTeacherConfig:
    modalities: tuple[str, ...] = CANONICAL_MODALITIES
    model_dim: int = 192
    layers: int = 3
    heads: int = 8
    dropout: float = 0.22
    modality_dropout: float = 0.10
    max_stream_tokens: int = 16
    expert_residual: bool = False
    expert_mixture: bool = False
    anchor_margin: float = 0.15
    residual_scale: float = 1.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", validate_modalities(self.modalities))
        if self.model_dim <= 0 or self.model_dim % self.heads:
            raise ValueError("model_dim must be positive and divisible by heads")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        if self.anchor_margin <= 0 or self.residual_scale <= 0:
            raise ValueError("anchor_margin and residual_scale must be positive")
        if self.expert_residual and self.expert_mixture:
            raise ValueError("expert_residual and expert_mixture are mutually exclusive")


class FourModalTeacher(nn.Module):
    """One fusion model with explicitly grouped modality inputs and outputs."""

    def __init__(
        self,
        config: FourModalTeacherConfig,
        stream_dims: dict[str, int],
        stream_groups: dict[str, str],
        statistic_dims: dict[str, int],
        statistic_groups: dict[str, str],
        expert_names: tuple[str, ...] = (),
        expert_groups: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.config = config
        self.modalities = config.modalities
        self.stream_groups = dict(stream_groups)
        self.statistic_groups = dict(statistic_groups)
        expected_streams = set(stream_dims)
        if expected_streams != set(stream_groups):
            raise ValueError("stream dimensions and groups differ")
        if set(statistic_dims) != set(statistic_groups):
            raise ValueError("statistic dimensions and groups differ")
        allowed_groups = set(self.modalities) | {"cross"}
        invalid_groups = sorted(
            (set(stream_groups.values()) | set(statistic_groups.values())) - allowed_groups
        )
        if invalid_groups:
            raise ValueError(f"inactive or invalid input groups: {invalid_groups}")
        if len(expert_names) != len(expert_groups):
            raise ValueError("expert names and groups differ")
        if (config.expert_residual or config.expert_mixture) and not expert_names:
            raise ValueError("expert fusion requires expert distributions")
        self.expert_names = tuple(expert_names)
        self.expert_groups = tuple(expert_groups)

        self.stream_project = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(width), nn.Linear(width, config.model_dim)
                )
                for name, width in stream_dims.items()
            }
        )
        self.statistic_project = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, config.model_dim),
                    nn.GELU(),
                )
                for name, width in statistic_dims.items()
            }
        )
        self.skeleton_encoder = (
            CTRSkeletonEncoder(config.model_dim) if "skeleton" in self.modalities else None
        )
        self.imu_encoder = (
            DevicewiseIMUEncoder(config.model_dim, config.heads, config.dropout)
            if "imu" in self.modalities
            else None
        )

        token_names = list(stream_dims) + list(statistic_dims)
        if self.skeleton_encoder is not None:
            token_names.append("skeleton_raw")
        if self.imu_encoder is not None:
            token_names.append("imu_raw")
        self.type_embedding = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(1, 1, config.model_dim) * 0.02)
                for name in token_names
            }
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.max_stream_tokens, config.model_dim) * 0.01
        )
        if config.expert_residual or config.expert_mixture:
            self.probability_project = nn.Sequential(
                nn.LayerNorm(40), nn.Linear(40, config.model_dim)
            )
            self.expert_type_embedding = nn.Parameter(
                torch.randn(1, len(expert_names), config.model_dim) * 0.02
            )
        else:
            self.probability_project = None
            self.register_parameter("expert_type_embedding", None)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.model_dim)
        self.main_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim * 2, 40),
        )
        if config.expert_residual:
            nn.init.zeros_(self.main_head[-1].weight)
            nn.init.zeros_(self.main_head[-1].bias)
        if config.expert_mixture:
            self.expert_gate = nn.Linear(config.model_dim, len(expert_names))
            gate_prior = torch.zeros(len(expert_names))
            gate_prior[0] = 2.2
            if len(expert_names) > 1:
                gate_prior[1] = 0.8
            self.register_buffer("expert_gate_prior", gate_prior)
            self.residual_strength = nn.Parameter(torch.tensor(-0.5))
        else:
            self.expert_gate = None
            self.register_buffer("expert_gate_prior", None)
            self.register_parameter("residual_strength", None)
        self.family_head = nn.Linear(config.model_dim, 8)
        self.reliability_head = nn.Linear(config.model_dim, 1)
        self.modality_heads = nn.ModuleDict(
            {name: nn.Linear(config.model_dim, 40) for name in self.modalities}
        )

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def _group_keep(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        keep: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            if self.training and self.config.modality_dropout > 0:
                value = (
                    torch.rand(batch_size, 1, 1, device=device)
                    >= self.config.modality_dropout
                ).to(dtype)
            else:
                value = torch.ones(batch_size, 1, 1, device=device, dtype=dtype)
            keep[modality] = value
        stacked = torch.cat([keep[name] for name in self.modalities], dim=1)
        missing_all = stacked.sum(dim=1, keepdim=True) == 0
        if missing_all.any():
            first = self.modalities[0]
            keep[first] = torch.where(
                missing_all,
                torch.ones_like(keep[first]),
                keep[first],
            )
        if "skeleton" in keep and "imu" in keep:
            keep["cross"] = keep["skeleton"] * keep["imu"]
        return keep

    def _decorate(self, name: str, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.config.max_stream_tokens:
            raise ValueError(
                f"{name} has {values.shape[1]} tokens, max is {self.config.max_stream_tokens}"
            )
        return (
            values
            + self.type_embedding[name]
            + self.position_embedding[:, : values.shape[1]]
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        batch_size = int(batch["label"].shape[0])
        reference = next(iter(batch.values()))
        keep = self._group_keep(batch_size, reference.device, torch.float32)
        availability: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            value = batch[f"{modality}_available"].to(torch.float32).view(
                batch_size, 1, 1
            )
            availability[modality] = value
            keep[modality] = keep[modality] * value
        if "skeleton" in keep and "imu" in keep:
            keep["cross"] = keep["skeleton"] * keep["imu"]
        tokens: list[torch.Tensor] = []
        group_pools: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.modalities
        }

        for name, projection in self.stream_project.items():
            group = self.stream_groups[name]
            values = self._decorate(name, projection(batch[name]))
            values = values * keep[group]
            tokens.append(values)
            if group in group_pools:
                group_pools[group].append(values.mean(dim=1))

        if self.skeleton_encoder is not None:
            values = self._decorate(
                "skeleton_raw", self.skeleton_encoder(batch["skeleton_sequence"])
            )
            values = values * keep["skeleton"]
            tokens.append(values)
            group_pools["skeleton"].append(values.mean(dim=1))
        if self.imu_encoder is not None:
            values = self._decorate(
                "imu_raw", self.imu_encoder(batch["imu_sequence"], batch["imu_mask"])
            )
            values = values * keep["imu"]
            tokens.append(values)
            group_pools["imu"].append(values.mean(dim=1))

        for name, projection in self.statistic_project.items():
            group = self.statistic_groups[name]
            values = self._decorate(name, projection(batch[name]).unsqueeze(1))
            values = values * keep[group]
            tokens.append(values)
            if group in group_pools:
                group_pools[group].append(values[:, 0])

        if self.probability_project is not None:
            expert_log_probability = torch.log(
                batch["expert_probability"].clamp_min(1e-8)
            )
            expert_tokens = self.probability_project(expert_log_probability)
            expert_tokens = expert_tokens + self.expert_type_embedding
            expert_values: list[torch.Tensor] = []
            for expert_index, group in enumerate(self.expert_groups):
                value = expert_tokens[:, expert_index : expert_index + 1]
                if group in keep:
                    value = value * keep[group]
                    group_pools[group].append(value[:, 0])
                expert_values.append(value)
            tokens.append(torch.cat(expert_values, dim=1))

        if not tokens:
            raise RuntimeError("no modality tokens reached the fusion encoder")
        cls = self.cls.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat((cls, *tokens), dim=1))
        representation = self.norm(encoded[:, 0])
        modality_embeddings: dict[str, torch.Tensor] = {}
        for name, values in group_pools.items():
            if not values:
                raise RuntimeError(f"active modality {name} produced no tokens")
            modality_embeddings[name] = torch.stack(values, dim=1).mean(dim=1)
        modality_logits = {
            name: self.modality_heads[name](embedding)
            for name, embedding in modality_embeddings.items()
        }
        similarity = torch.stack(
            [
                (representation * modality_embeddings[name]).sum(dim=1)
                / math.sqrt(self.config.model_dim)
                for name in self.modalities
            ],
            dim=1,
        )
        availability_mask = torch.cat(
            [availability[name].view(batch_size, 1) for name in self.modalities], dim=1
        ).bool()
        similarity = similarity.masked_fill(~availability_mask, -1e4)
        raw_logits = self.main_head(representation)
        if self.config.expert_mixture:
            gate_logits = self.expert_gate(representation) + self.expert_gate_prior
            gate_available = torch.ones(
                batch_size,
                len(self.expert_groups),
                device=gate_logits.device,
                dtype=torch.bool,
            )
            for expert_index, group in enumerate(self.expert_groups):
                if group in keep:
                    gate_available[:, expert_index] = keep[group].view(batch_size) > 0
            gate_logits = gate_logits.masked_fill(~gate_available, -1e4)
            expert_gate = torch.softmax(gate_logits, dim=1)
            expert_log_probability = torch.log(
                batch["expert_probability"].clamp_min(1e-8)
            )
            anchor_logits = torch.einsum(
                "be,bec->bc", expert_gate, expert_log_probability
            )
            strength = 2.5 * torch.sigmoid(self.residual_strength)
            residual_logits = strength * torch.tanh(raw_logits)
            logits = anchor_logits + residual_logits
        elif self.config.expert_residual:
            anchor_logits = torch.zeros_like(raw_logits)
            anchor_logits.scatter_(
                1,
                batch["base_prediction"].view(-1, 1),
                self.config.anchor_margin,
            )
            residual_logits = self.config.residual_scale * torch.tanh(raw_logits)
            logits = anchor_logits + residual_logits
            expert_gate = torch.empty(
                batch_size, 0, device=raw_logits.device, dtype=raw_logits.dtype
            )
        else:
            anchor_logits = torch.zeros_like(raw_logits)
            residual_logits = raw_logits
            logits = raw_logits
            expert_gate = torch.empty(
                batch_size, 0, device=raw_logits.device, dtype=raw_logits.dtype
            )
        return {
            "logits": logits,
            "anchor_logits": anchor_logits,
            "residual_logits": residual_logits,
            "expert_gate": expert_gate,
            "representation": representation,
            "family_logits": self.family_head(representation),
            "reliability_logits": self.reliability_head(representation).squeeze(1),
            "modality_logits": modality_logits,
            "modality_embeddings": modality_embeddings,
            "modality_importance": torch.softmax(similarity, dim=1),
        }
