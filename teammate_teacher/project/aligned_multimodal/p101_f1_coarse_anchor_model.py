"""P101-F1: P100 coarse VS anchor plus fine local VSI pre-classifier residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from p100a_global_teacher_model import P100AGlobalTeacher
from p101_finegrained_teacher_model import (
    ContinuousLocalAttention,
    InteractionProjection,
    P101FineGrainedTeacher,
    P101ModelConfig,
)


@dataclass(frozen=True)
class P101F1Config:
    model_dim: int = 384
    motion_dim: int = 192
    heads: int = 8
    dropout: float = 0.15
    local_radius: float = 0.18
    workspace_arm_prior: float = 0.35
    evidence_layers: int = 2

    def validate(self) -> None:
        if self.model_dim % self.heads:
            raise ValueError("P101-F1 model_dim must divide heads")
        if self.motion_dim % 4:
            raise ValueError("P101-F1 motion_dim must divide four")
        if self.evidence_layers < 1:
            raise ValueError("P101-F1 needs at least one evidence layer")


class P101LocalVSIEncoder(nn.Module):
    """F0-compatible local modules without a second 40-class classifier."""

    F0_KEYS = (
        "vmae_projection",
        "iv2_projection",
        "visual_interaction",
        "window_embedding",
        "view_embedding",
        "visual_time_projection",
        "skeleton_encoder",
        "skeleton_projection",
        "skeleton_attention",
        "skeleton_delta",
        "skeleton_reliability",
        "imu_encoder",
        "imu_projection",
        "imu_attention",
        "imu_reliability",
        "correspondence",
    )

    def __init__(self, config: P101F1Config) -> None:
        super().__init__()
        template = P101FineGrainedTeacher(
            P101ModelConfig(
                modalities=("visual", "skeleton", "imu"),
                model_dim=config.model_dim,
                motion_dim=config.motion_dim,
                heads=config.heads,
                global_layers=1,
                dropout=config.dropout,
                local_radius=config.local_radius,
                workspace_arm_prior=config.workspace_arm_prior,
            )
        )
        for name in self.F0_KEYS:
            module = getattr(template, name)
            if module is None:
                raise RuntimeError(f"P101-F1 local module is absent: {name}")
            setattr(self, name, module)

    @staticmethod
    def _interaction(anchor: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (anchor, evidence, anchor * evidence, (anchor - evidence).abs()), dim=-1
        )

    @staticmethod
    def _motion(
        batch: dict[str, torch.Tensor], modality: str, prefix: str = ""
    ) -> tuple[torch.Tensor, ...]:
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

    def load_f0_state(self, state: dict[str, torch.Tensor]) -> dict[str, Any]:
        local = self.state_dict()
        missing = sorted(set(local) - set(state))
        if missing:
            raise RuntimeError(f"P101-F1 cannot initialize local F0 fields: {missing[:5]}")
        self.load_state_dict({name: state[name] for name in local}, strict=True)
        return {"loaded_tensors": len(local), "source_tensors": len(state)}

    def encode_visual(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        vmae = self.vmae_projection(batch["visual_vmae_temporal"])
        iv2 = self.iv2_projection(batch["visual_iv2_temporal"])
        visual = vmae + self.visual_interaction(vmae, iv2)
        visual = visual + self.window_embedding[None, :, None, None]
        visual = visual + self.view_embedding[None, None, :, None]
        return visual + self.visual_time_projection(
            batch["visual_time"][..., None]
        )[:, :, None]

    def encode_skeleton(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.skeleton_encoder(
            *self._motion(batch, "skeleton"), batch["motion_time"]
        )
        return self.skeleton_projection(tokens), mask

    def encode_imu(
        self, batch: dict[str, torch.Tensor], prefix: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.imu_encoder(*self._motion(batch, "imu", prefix))
        return self.imu_projection(tokens), mask

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = self.encode_visual(batch)
        skeleton, skeleton_mask = self.encode_skeleton(batch)
        skeleton_attended, skeleton_attention, skeleton_available = self.skeleton_attention(
            visual,
            skeleton,
            skeleton_mask,
            batch["visual_time"],
            batch["motion_time"],
        )
        skeleton_values = self._interaction(visual, skeleton_attended)
        skeleton_gate = torch.sigmoid(self.skeleton_reliability(skeleton_values))
        skeleton_gate = skeleton_gate * skeleton_available[:, :, None, None, None].to(
            skeleton_gate.dtype
        )
        vs_event = visual + skeleton_gate * self.skeleton_delta(
            visual, skeleton_attended
        )

        imu, imu_mask = self.encode_imu(batch)
        imu_attended, imu_attention, imu_available = self.imu_attention(
            visual,
            imu,
            imu_mask,
            batch["visual_time"],
            batch["motion_time"],
        )
        correspondence = self.correspondence(
            self._interaction(skeleton_attended, imu_attended)
        )
        negative_correspondence: torch.Tensor | None = None
        negative_attended: torch.Tensor | None = None
        negative_available: torch.Tensor | None = None
        if "negative_imu_sequences" in batch:
            negative_imu, negative_mask = self.encode_imu(batch, prefix="negative_")
            negative_attended, _, negative_available = self.imu_attention(
                visual,
                negative_imu,
                negative_mask,
                batch["visual_time"],
                batch["motion_time"],
            )
            negative_correspondence = self.correspondence(
                self._interaction(skeleton_attended, negative_attended)
            )

        def pool(values: torch.Tensor) -> torch.Tensor:
            return values.mean(dim=(1, 2, 3))

        def evidence_tokens(imu_values: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                (
                    pool(visual),
                    pool(vs_event),
                    pool(skeleton_attended),
                    pool(imu_values),
                    pool(skeleton_attended * imu_values),
                    pool((skeleton_attended - imu_values).abs()),
                    pool(vs_event * imu_values),
                    pool((vs_event - imu_values).abs()),
                ),
                dim=1,
            )

        positive_evidence = evidence_tokens(imu_attended)
        output = {
            "evidence_tokens": positive_evidence,
            "zero_imu_evidence_tokens": evidence_tokens(torch.zeros_like(imu_attended)),
            "skeleton_attention": skeleton_attention,
            "imu_attention": imu_attention,
            "skeleton_available": skeleton_available,
            "imu_available": imu_available,
            "skeleton_reliability": skeleton_gate,
            "positive_correspondence_logits": correspondence,
        }
        if negative_correspondence is not None:
            output["negative_correspondence_logits"] = negative_correspondence
            output["negative_evidence_tokens"] = evidence_tokens(negative_attended)
            output["negative_imu_available"] = negative_available
        return output


class P101F1CoarseAnchoredTeacher(nn.Module):
    """Fine local VSI evidence writes into frozen coarse VS before its classifier."""

    def __init__(
        self, anchor: P100AGlobalTeacher, config: P101F1Config
    ) -> None:
        super().__init__()
        config.validate()
        if anchor.config.modalities != ("visual", "skeleton"):
            raise ValueError("P101-F1 anchor must be the P100 coarse VS variant")
        if anchor.config.model_dim != config.model_dim:
            raise ValueError("P101-F1 anchor and local width differ")
        self.config = config
        self.anchor = anchor
        self.local = P101LocalVSIEncoder(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.evidence_encoder = nn.TransformerEncoder(
            layer, num_layers=config.evidence_layers
        )
        self.evidence_norm = nn.LayerNorm(config.model_dim)
        self.residual_projection = InteractionProjection(
            config.model_dim, config.dropout, zero_output=True
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        anchor = self.anchor(batch)
        local = self.local(batch)
        evidence = self.evidence_norm(
            self.evidence_encoder(local["evidence_tokens"]).mean(dim=1)
        )
        residual = self.residual_projection(anchor["representation"], evidence)
        probability = torch.softmax(anchor["logits"].detach(), dim=-1)
        top = probability.topk(2, dim=-1).values
        inference_uncertainty = (1.0 - (top[:, 0] - top[:, 1])).clamp(0.0, 1.0)
        uncertainty = batch.get("nested_anchor_uncertainty", inference_uncertainty)
        available = local["imu_available"].any(dim=1).to(residual.dtype)
        effective = residual * uncertainty[:, None] * available[:, None]
        representation = anchor["representation"] + effective
        output = {
            "logits": self.anchor.classifier(representation),
            "representation": representation,
            "anchor_logits": anchor["logits"],
            "anchor_representation": anchor["representation"],
            "uncertainty": uncertainty,
            "inference_uncertainty": inference_uncertainty,
            "fine_residual_rms": effective.square().mean(dim=-1).sqrt(),
            "raw_residual_rms": residual.square().mean(dim=-1).sqrt(),
            "evidence_embedding": evidence,
        }
        output.update(local)
        return output


def select_f1_trainable_parameters(
    model: P101F1CoarseAnchoredTeacher,
) -> tuple[list[nn.Parameter], list[str], list[str]]:
    trainable_prefixes = (
        "local.imu_encoder.",
        "local.imu_projection.",
        "local.imu_attention.",
        "local.imu_reliability.",
        "local.correspondence.",
        "evidence_encoder.",
        "evidence_norm.",
        "residual_projection.",
    )
    parameters: list[nn.Parameter] = []
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        selected = any(name.startswith(prefix) for prefix in trainable_prefixes)
        parameter.requires_grad_(selected)
        if selected:
            parameters.append(parameter)
            trainable.append(name)
        else:
            frozen.append(name)
    if not parameters or any(name.startswith("anchor.") for name in trainable):
        raise RuntimeError("P101-F1 parameter freeze contract failed")
    return parameters, trainable, frozen
