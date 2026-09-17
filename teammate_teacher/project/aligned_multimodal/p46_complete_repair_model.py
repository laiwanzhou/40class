from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from p46_step10_model import P46Step10Model
from p46r_event_bottleneck_model import P46REventModel


class P46CompleteRepairModel(nn.Module):
    """Feature-level repair of P46 using the validated P46-R relationship evidence.

    This is deliberately not an output gate.  The original P46 trial embedding and
    the P46-R same-part/same-time evidence are combined before the original P46
    Detail21 head.  The final adapter layer is zero-initialised, so loading the two
    pretrained branches preserves the original P46 logits exactly at step zero.
    """

    def __init__(
        self,
        *,
        base_width: int = 192,
        relation_width: int = 128,
        subjects: int = 14,
        dropout: float = 0.12,
        adapter_rank: int = 32,
        maximum_delta_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_width = int(base_width)
        self.relation_width = int(relation_width)
        self.adapter_rank = int(adapter_rank)
        self.maximum_delta_norm = float(maximum_delta_norm)
        if self.adapter_rank < 1:
            raise ValueError("adapter_rank must be positive")
        if self.maximum_delta_norm <= 0.0:
            raise ValueError("maximum_delta_norm must be positive")
        self.base = P46Step10Model(
            width=self.base_width,
            dropout=dropout,
            subjects=subjects,
        )
        self.relation = P46REventModel(
            width=self.relation_width,
            dropout=dropout,
        )
        # P46-R relationship evidence: ordered event embedding (256), five-part
        # lag correlations (5 offsets x 5 parts), and five learned event centres.
        self.relationship_width = 256 + 25 + 5
        self.relationship_adapter = nn.Sequential(
            nn.LayerNorm(self.relationship_width),
            nn.Linear(self.relationship_width, self.adapter_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.adapter_rank, 384),
        )
        nn.init.zeros_(self.relationship_adapter[-1].weight)
        nn.init.zeros_(self.relationship_adapter[-1].bias)
        self._pretrained_frozen = False

    def load_pretrained(
        self,
        base_checkpoint: str | Path,
        relation_checkpoint: str | Path,
    ) -> dict[str, Any]:
        base_path = Path(base_checkpoint).resolve()
        relation_path = Path(relation_checkpoint).resolve()
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        relation = torch.load(relation_path, map_location="cpu", weights_only=False)
        if base.get("stage") != "P46_step10_stageB_detail21":
            raise RuntimeError(f"not a P46 Stage-B checkpoint: {base_path}")
        if relation.get("stage") != "P46-R_full_mechanism_training":
            raise RuntimeError(f"not a formal P46-R checkpoint: {relation_path}")
        self.base.load_state_dict(base["model_state_dict"], strict=True)
        self.relation.load_state_dict(relation["model_state_dict"], strict=True)
        return {
            "base_checkpoint": str(base_path),
            "base_epoch": int(base["epoch"]),
            "base_metrics": base.get("metrics", {}),
            "relation_checkpoint": str(relation_path),
            "relation_epoch": int(relation["epoch"]),
            "relation_metrics": relation.get("validation_metrics", {}),
            "relation_offset_metrics": relation.get("validation_offset_metrics", {}),
        }

    def freeze_pretrained(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.relation.parameters():
            parameter.requires_grad_(False)
        for parameter in self.relationship_adapter.parameters():
            parameter.requires_grad_(True)
        self._pretrained_frozen = True
        self.base.eval()
        self.relation.eval()

    def train(self, mode: bool = True) -> P46CompleteRepairModel:
        super().train(mode)
        if self._pretrained_frozen:
            # Frozen pretrained branches must not change through dropout state.
            self.base.eval()
            self.relation.eval()
            self.relationship_adapter.train(mode)
        return self

    def relationship_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.relationship_adapter.parameters()
            if parameter.requires_grad
        ]

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        relationship_scale: float = 1.0,
        relationship_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self.base(batch, subject_adversarial_scale=0.0)
        relation = self.relation(batch if relationship_batch is None else relationship_batch)
        relationship_input = torch.cat(
            (
                relation["embedding"],
                relation["offset_correlation"].to(relation["embedding"].dtype),
                relation["event_centre"].to(relation["embedding"].dtype),
            ),
            dim=-1,
        )
        if relationship_input.shape[-1] != self.relationship_width:
            raise RuntimeError(
                f"P46/P46-R relationship contract changed: {relationship_input.shape}"
            )
        relationship_raw_delta = self.relationship_adapter(relationship_input)
        # This is a deterministic feature-space norm bound, not a learned gate.
        # P46-R still directly changes the embedding consumed by P46's head, but
        # it cannot erase the pretrained P46 representation with a large update.
        raw_norm = relationship_raw_delta.float().norm(dim=-1, keepdim=True)
        norm_scale = torch.clamp(
            self.maximum_delta_norm / raw_norm.clamp_min(1e-6),
            max=1.0,
        ).to(relationship_raw_delta.dtype)
        relationship_delta = relationship_raw_delta * norm_scale
        fused_embedding = base["trial_embedding"] + float(relationship_scale) * relationship_delta
        detail_logits = self.base.detail_head(fused_embedding)
        return {
            **base,
            "detail_logits": detail_logits,
            "base_detail_logits": base["detail_logits"],
            "base_trial_embedding": base["trial_embedding"],
            "fused_trial_embedding": fused_embedding,
            "relationship_input": relationship_input,
            "relationship_raw_delta": relationship_raw_delta,
            "relationship_delta": relationship_delta,
            "relation_embedding": relation["embedding"],
            "relation_part_summary": relation["part_summary"],
            "relation_event_centre": relation["event_centre"],
            "relation_offset_logits": relation["offset_logits"],
            "relation_offset_correlation": relation["offset_correlation"],
        }


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
