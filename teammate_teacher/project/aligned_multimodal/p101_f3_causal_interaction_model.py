"""P101-F3 causal-centered, correspondence-gated pre-classifier interaction."""

from __future__ import annotations

import torch
from torch import nn

from p100a_global_teacher_model import P100AGlobalTeacher
from p101_f1_coarse_anchor_model import P101F1Config, P101LocalVSIEncoder
from p101_finegrained_teacher_model import InteractionProjection


class P101F3CausalInteractionTeacher(nn.Module):
    """Only IMU-caused evidence differences may write into the coarse VS anchor."""

    def __init__(self, anchor: P100AGlobalTeacher, config: P101F1Config) -> None:
        super().__init__()
        config.validate()
        if anchor.config.modalities != ("visual", "skeleton"):
            raise ValueError("P101-F3 anchor must be P100 coarse VS")
        if anchor.config.model_dim != config.model_dim:
            raise ValueError("P101-F3 anchor and interaction width differ")
        self.config = config
        self.anchor = anchor
        self.local = P101LocalVSIEncoder(config)
        # Centering must be algebraic rather than two independently dropped views.
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.evidence_encoder = nn.TransformerEncoder(
            layer, num_layers=config.evidence_layers
        )
        self.evidence_norm = nn.LayerNorm(config.model_dim)
        self.residual_projection = InteractionProjection(
            config.model_dim, 0.0, zero_output=True
        )

    def _evidence(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.evidence_norm(self.evidence_encoder(tokens).mean(dim=1))

    def _centered_residual(
        self,
        anchor_representation: torch.Tensor,
        evidence_tokens: torch.Tensor,
        zero_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        evidence_difference = self._evidence(evidence_tokens) - self._evidence(zero_tokens)
        zero_difference = torch.zeros_like(evidence_difference)
        residual = self.residual_projection(
            anchor_representation, evidence_difference
        ) - self.residual_projection(anchor_representation, zero_difference)
        return residual, evidence_difference

    @staticmethod
    def _correspondence_gate(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits.mean(dim=tuple(range(1, logits.ndim))))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        anchor = self.anchor(batch)
        local = self.local(batch)
        raw_residual, evidence = self._centered_residual(
            anchor["representation"],
            local["evidence_tokens"],
            local["zero_imu_evidence_tokens"],
        )
        probability = torch.softmax(anchor["logits"].detach(), dim=-1)
        top = probability.topk(2, dim=-1).values
        inference_uncertainty = (1.0 - (top[:, 0] - top[:, 1])).clamp(0.0, 1.0)
        uncertainty = batch.get("nested_anchor_uncertainty", inference_uncertainty)
        available = local["imu_available"].any(dim=1).to(raw_residual.dtype)
        pair_gate = self._correspondence_gate(
            local["positive_correspondence_logits"]
        ).to(raw_residual.dtype)
        effective = (
            raw_residual
            * uncertainty[:, None]
            * available[:, None]
            * pair_gate[:, None]
        )
        representation = anchor["representation"] + effective
        output = {
            "logits": self.anchor.classifier(representation),
            "representation": representation,
            "anchor_logits": anchor["logits"],
            "anchor_representation": anchor["representation"],
            "uncertainty": uncertainty,
            "inference_uncertainty": inference_uncertainty,
            "fine_residual_rms": effective.square().mean(dim=-1).sqrt(),
            "raw_residual_rms": raw_residual.square().mean(dim=-1).sqrt(),
            "evidence_embedding": evidence,
            "pair_gate": pair_gate,
        }
        if "negative_evidence_tokens" in local:
            negative_raw, negative_evidence = self._centered_residual(
                anchor["representation"],
                local["negative_evidence_tokens"],
                local["zero_imu_evidence_tokens"],
            )
            negative_available = local["negative_imu_available"].any(dim=1).to(
                negative_raw.dtype
            )
            negative_gate = self._correspondence_gate(
                local["negative_correspondence_logits"]
            ).to(negative_raw.dtype)
            negative_effective = (
                negative_raw
                * uncertainty[:, None]
                * negative_available[:, None]
                * negative_gate[:, None]
            )
            output.update(
                {
                    "negative_logits": self.anchor.classifier(
                        anchor["representation"] + negative_effective
                    ),
                    "negative_fine_residual_rms": negative_effective.square()
                    .mean(dim=-1)
                    .sqrt(),
                    "negative_pair_gate": negative_gate,
                    "negative_evidence_embedding": negative_evidence,
                }
            )
        output.update(local)
        return output
