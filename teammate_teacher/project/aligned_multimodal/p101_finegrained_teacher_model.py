"""Fine-grained pre-classification fusion model for the P101 A Teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from p86_mobind_lite_model import P86IMUPartEncoder, P86SkeletonPartEncoder


@dataclass(frozen=True)
class P101ModelConfig:
    modalities: tuple[str, ...] = ("visual", "skeleton", "imu")
    model_dim: int = 384
    motion_dim: int = 192
    heads: int = 8
    global_layers: int = 3
    dropout: float = 0.15
    local_radius: float = 0.18
    workspace_arm_prior: float = 0.35
    maximum_imu_scale: float = 0.25

    def validate(self) -> None:
        if "visual" not in self.modalities:
            raise ValueError("P101 always requires Visual")
        if not set(self.modalities) <= {"visual", "skeleton", "imu"}:
            raise ValueError(f"unexpected P101 modalities: {self.modalities}")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.motion_dim % 4:
            raise ValueError("motion_dim must be divisible by four")
        if not 0.0 < self.local_radius <= 1.0:
            raise ValueError("local_radius must be in (0,1]")
        if not 0.0 < self.maximum_imu_scale <= 1.0:
            raise ValueError("maximum_imu_scale must be in (0,1]")


class InteractionProjection(nn.Module):
    def __init__(self, width: int, dropout: float, zero_output: bool = False) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )
        if zero_output:
            nn.init.zeros_(self.network[4].weight)
            nn.init.zeros_(self.network[4].bias)

    def forward(self, anchor: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.network(
            torch.cat(
                (anchor, evidence, anchor * evidence, (anchor - evidence).abs()),
                dim=-1,
            )
        )


class ContinuousLocalAttention(nn.Module):
    """Attend only to time-nearby body parts/devices inside the same window."""

    def __init__(
        self,
        width: int,
        heads: int,
        radius: float,
        dropout: float,
        workspace_arm_prior: float = 0.0,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("local attention width must divide heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = width // heads
        self.radius = float(radius)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Sequential(nn.Linear(width, width), nn.Dropout(dropout))
        self.distance_rate = nn.Parameter(torch.full((heads,), 2.0))
        part_prior = torch.zeros(3, 5)
        part_prior[2, 1:3] = float(workspace_arm_prior)
        self.register_buffer("part_prior", part_prior, persistent=True)

    def forward(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        evidence_mask: torch.Tensor,
        visual_time: torch.Tensor,
        motion_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # query [B,W,V,T,D], evidence [B,W,S,P,D]
        if query.ndim != 6 - 1 or evidence.ndim != 6 - 1:
            raise ValueError("P101 local attention expects five-dimensional tokens")
        batch, windows, views, visual_steps, width = query.shape
        if evidence.shape[:2] != (batch, windows) or evidence.shape[-1] != width:
            raise ValueError("P101 visual/motion token grids differ")
        motion_steps, parts = evidence.shape[2:4]
        if parts != 5:
            raise ValueError("P101 requires five aligned body/device parts")
        q = self.query(query).reshape(
            batch, windows, views, visual_steps, self.heads, self.head_width
        )
        k = self.key(evidence).reshape(
            batch, windows, motion_steps, parts, self.heads, self.head_width
        )
        v = self.value(evidence).reshape(
            batch, windows, motion_steps, parts, self.heads, self.head_width
        )
        score = torch.einsum("bwvthd,bwsphd->bwvthsp", q, k)
        score = score / math.sqrt(self.head_width)
        distance = (visual_time[:, :, None, :, None] - motion_time[:, :, None, None, :]).abs()
        rate = F.softplus(self.distance_rate).view(1, 1, 1, 1, self.heads, 1, 1)
        score = score - rate * distance[:, :, :, :, None, :, None]
        score = score + self.part_prior[None, None, :, None, None, None, :]
        valid = evidence_mask[:, :, None, None, None, :, :]
        local = distance[:, :, :, :, None, :, None] <= self.radius
        allowed = valid & local
        score = score.masked_fill(~allowed, -1e4)
        attention = torch.softmax(score.flatten(-2), dim=-1).reshape_as(score)
        attention = attention * allowed.to(attention.dtype)
        attention = attention / attention.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        attended = torch.einsum("bwvthsp,bwsphd->bwvthd", attention, v)
        attended = attended.reshape(batch, windows, views, visual_steps, width)
        attended = self.output(attended)
        available = evidence_mask.any(dim=(-2, -1))
        attended = attended * available[:, :, None, None, None].to(attended.dtype)
        mean_attention = attention.mean(dim=4)
        return attended, mean_attention, available


class P101FineGrainedTeacher(nn.Module):
    """One final 40-class head after local V/S/I interaction and global reasoning."""

    def __init__(self, config: P101ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.model_dim
        dropout = config.dropout
        self.vmae_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.iv2_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.visual_interaction = InteractionProjection(width, dropout)
        self.window_embedding = nn.Parameter(torch.zeros(2, width))
        self.view_embedding = nn.Parameter(torch.zeros(3, width))
        self.visual_time_projection = nn.Sequential(
            nn.Linear(1, width), nn.GELU(), nn.Linear(width, width)
        )
        for value in (self.window_embedding, self.view_embedding):
            nn.init.trunc_normal_(value, std=0.02)

        self.pooled_vmae_projection = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, width)
        )
        self.pooled_iv2_projection = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, width)
        )
        self.pooled_interaction = InteractionProjection(width, dropout)
        self.vmae_action_projection = nn.Sequential(
            nn.LayerNorm(710), nn.Linear(710, width), nn.GELU()
        )
        self.iv2_action_projection = nn.Sequential(
            nn.LayerNorm(400), nn.Linear(400, width), nn.GELU()
        )
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(width * 4), nn.Linear(width * 4, width), nn.GELU()
        )

        self.skeleton_encoder: P86SkeletonPartEncoder | None = None
        self.skeleton_projection: nn.Module | None = None
        self.skeleton_attention: ContinuousLocalAttention | None = None
        self.skeleton_delta: InteractionProjection | None = None
        self.skeleton_reliability: nn.Module | None = None
        if "skeleton" in config.modalities:
            self.skeleton_encoder = P86SkeletonPartEncoder(
                width=config.motion_dim,
                dropout=dropout,
                explicit_time_position=True,
                multistream_input=True,
                adaptive_graph=True,
            )
            self.skeleton_projection = nn.Sequential(
                nn.LayerNorm(config.motion_dim), nn.Linear(config.motion_dim, width)
            )
            self.skeleton_attention = ContinuousLocalAttention(
                width,
                config.heads,
                config.local_radius,
                dropout,
                workspace_arm_prior=config.workspace_arm_prior,
            )
            self.skeleton_delta = InteractionProjection(width, dropout)
            self.skeleton_reliability = nn.Sequential(
                nn.LayerNorm(width * 4), nn.Linear(width * 4, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
            )

        self.imu_encoder: P86IMUPartEncoder | None = None
        self.imu_projection: nn.Module | None = None
        self.imu_attention: ContinuousLocalAttention | None = None
        self.imu_delta: InteractionProjection | None = None
        self.imu_reliability: nn.Module | None = None
        self.correspondence: nn.Module | None = None
        self.imu_scale_logit: nn.Parameter | None = None
        if "imu" in config.modalities:
            self.imu_encoder = P86IMUPartEncoder(
                width=config.motion_dim,
                dropout=dropout,
                instance_normalization=True,
            )
            self.imu_projection = nn.Sequential(
                nn.LayerNorm(config.motion_dim), nn.Linear(config.motion_dim, width)
            )
            self.imu_attention = ContinuousLocalAttention(
                width, config.heads, config.local_radius, dropout
            )
            protected = "skeleton" in config.modalities
            self.imu_delta = InteractionProjection(width, dropout, zero_output=protected)
            self.imu_reliability = nn.Sequential(
                nn.LayerNorm(width * 4), nn.Linear(width * 4, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
            )
            self.imu_scale_logit = nn.Parameter(torch.tensor(-2.1972246))
            if protected:
                self.correspondence = nn.Sequential(
                    nn.LayerNorm(width * 4),
                    nn.Linear(width * 4, width // 2),
                    nn.GELU(),
                    nn.Linear(width // 2, 1),
                )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        global_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            global_layer, num_layers=config.global_layers
        )
        self.final_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 40)

    @staticmethod
    def _interaction_values(anchor: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (anchor, evidence, anchor * evidence, (anchor - evidence).abs()), dim=-1
        )

    @staticmethod
    def _motion(batch: dict[str, torch.Tensor], modality: str, prefix: str = "") -> tuple[torch.Tensor, ...]:
        if modality == "skeleton":
            names = (
                "skeleton_features",
                "skeleton_feature_mask",
                "skeleton_joint_mask",
                "skeleton_relations",
                "skeleton_relation_mask",
                "skeleton_frame_quality",
            )
        elif modality == "imu":
            names = (
                "imu_sequences",
                "imu_sequence_mask",
                "imu_bin_statistics",
                "imu_bin_mask",
                "imu_global_statistics",
                "imu_global_mask",
            )
        else:
            raise ValueError(modality)
        return tuple(batch[prefix + name] for name in names)

    def encode_visual(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        vmae = self.vmae_projection(batch["visual_vmae_temporal"])
        iv2 = self.iv2_projection(batch["visual_iv2_temporal"])
        visual = vmae + self.visual_interaction(vmae, iv2)
        visual = visual + self.window_embedding[None, :, None, None]
        visual = visual + self.view_embedding[None, None, :, None]
        visual = visual + self.visual_time_projection(
            batch["visual_time"][..., None]
        )[:, :, None]
        return visual

    def encode_skeleton(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.skeleton_encoder is None or self.skeleton_projection is None:
            raise RuntimeError("Skeleton is not configured")
        tokens, mask = self.skeleton_encoder(
            *self._motion(batch, "skeleton"), batch["motion_time"]
        )
        return self.skeleton_projection(tokens), mask

    def encode_imu(
        self, batch: dict[str, torch.Tensor], prefix: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.imu_encoder is None or self.imu_projection is None:
            raise RuntimeError("IMU is not configured")
        tokens, mask = self.imu_encoder(*self._motion(batch, "imu", prefix))
        return self.imu_projection(tokens), mask

    def _semantic_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        vmae = self.pooled_vmae_projection(batch["visual_vmae_pooled"])
        iv2 = self.pooled_iv2_projection(batch["visual_iv2_pooled"])
        pooled = vmae + self.pooled_interaction(vmae, iv2)
        vmae_action = self.vmae_action_projection(batch["visual_vmae_action"])
        iv2_action = self.iv2_action_projection(batch["visual_iv2_action"])
        return self.semantic_projection(
            torch.cat((pooled, vmae_action, iv2_action, vmae_action * iv2_action), dim=-1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = self.encode_visual(batch)
        output: dict[str, torch.Tensor] = {"visual_tokens": visual}
        fused = visual
        skeleton_attended: torch.Tensor | None = None
        if self.skeleton_encoder is not None:
            assert self.skeleton_attention is not None
            assert self.skeleton_delta is not None
            assert self.skeleton_reliability is not None
            skeleton, skeleton_mask = self.encode_skeleton(batch)
            skeleton_attended, attention, available = self.skeleton_attention(
                visual,
                skeleton,
                skeleton_mask,
                batch["visual_time"],
                batch["motion_time"],
            )
            interaction = self._interaction_values(visual, skeleton_attended)
            reliability = torch.sigmoid(self.skeleton_reliability(interaction))
            reliability = reliability * available[:, :, None, None, None].to(reliability.dtype)
            fused = visual + reliability * self.skeleton_delta(visual, skeleton_attended)
            output.update(
                {
                    "skeleton_attention": attention,
                    "skeleton_reliability": reliability,
                    "skeleton_available": available,
                    "vs_tokens": fused,
                }
            )

        if self.imu_encoder is not None:
            assert self.imu_attention is not None
            assert self.imu_delta is not None
            assert self.imu_reliability is not None
            imu, imu_mask = self.encode_imu(batch)
            imu_attended, attention, available = self.imu_attention(
                visual,
                imu,
                imu_mask,
                batch["visual_time"],
                batch["motion_time"],
            )
            interaction = self._interaction_values(visual, imu_attended)
            reliability = torch.sigmoid(self.imu_reliability(interaction))
            reliability = reliability * available[:, :, None, None, None].to(reliability.dtype)
            imu_delta = reliability * self.imu_delta(visual, imu_attended)
            scale = torch.sigmoid(self.imu_scale_logit) * self.config.maximum_imu_scale
            correspondence_logits: torch.Tensor | None = None
            if skeleton_attended is not None:
                assert self.correspondence is not None
                correspondence_values = self._interaction_values(skeleton_attended, imu_attended)
                correspondence_logits = self.correspondence(correspondence_values)
                imu_delta = imu_delta * torch.sigmoid(correspondence_logits)
                output["positive_correspondence_logits"] = correspondence_logits
                if "negative_imu_sequences" in batch:
                    negative, negative_mask = self.encode_imu(batch, prefix="negative_")
                    negative_attended, _, _ = self.imu_attention(
                        visual,
                        negative,
                        negative_mask,
                        batch["visual_time"],
                        batch["motion_time"],
                    )
                    output["negative_correspondence_logits"] = self.correspondence(
                        self._interaction_values(skeleton_attended, negative_attended)
                    )
            fused = fused + scale * imu_delta
            output.update(
                {
                    "imu_attention": attention,
                    "imu_reliability": reliability,
                    "imu_available": available,
                    "imu_scale": scale,
                    "imu_residual_rms": imu_delta.square().mean().sqrt(),
                }
            )

        event_tokens = fused.flatten(1, 3)
        semantic_tokens = self._semantic_tokens(batch).flatten(1, 2)
        cls = self.cls_token.expand(event_tokens.shape[0], -1, -1)
        global_tokens = self.global_encoder(
            torch.cat((cls, event_tokens, semantic_tokens), dim=1)
        )
        embedding = self.final_norm(global_tokens[:, 0])
        output["fused_embedding"] = embedding
        output["fused_tokens"] = fused
        output["logits"] = self.classifier(embedding)
        return output


def trainable_imu_adapter_parameters(
    model: P101FineGrainedTeacher,
) -> tuple[list[nn.Parameter], list[str], list[str]]:
    """Freeze the exact VS anchor and expose only P101's new IMU path."""
    trainable_prefixes = (
        "imu_encoder.",
        "imu_projection.",
        "imu_attention.",
        "imu_delta.",
        "imu_reliability.",
        "correspondence.",
        "imu_scale_logit",
    )
    parameters: list[nn.Parameter] = []
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        selected = any(name == prefix or name.startswith(prefix) for prefix in trainable_prefixes)
        parameter.requires_grad_(selected)
        if selected:
            parameters.append(parameter)
            trainable.append(name)
        else:
            frozen.append(name)
    if not parameters or not any(name.startswith("imu_encoder.") for name in trainable):
        raise RuntimeError("P101 IMU adapter parameter selection failed")
    return parameters, trainable, frozen
