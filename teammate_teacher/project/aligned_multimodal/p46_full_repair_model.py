from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from p46_step10_model import DETAIL_CLASSES, P46Step10Model, gradient_reverse
from p46r_event_bottleneck_model import P46REventModel


class P46FullRepairModel(nn.Module):
    """End-to-end P46 with the P46-R relationship evidence inside its classifier input.

    Every module is randomly initialised.  There is no checkpoint loading, frozen
    branch, prediction gate, or post-hoc residual patch.  The final Detail21 loss
    flows through the shared fusion representation into both P46 and P46-R.
    """

    def __init__(
        self,
        *,
        base_width: int = 192,
        relation_width: int = 128,
        fusion_width: int = 384,
        dropout: float = 0.12,
        subjects: int = 14,
    ) -> None:
        super().__init__()
        self.base_width = int(base_width)
        self.relation_width = int(relation_width)
        self.fusion_width = int(fusion_width)
        self.p46 = P46Step10Model(
            width=self.base_width,
            dropout=dropout,
            subjects=subjects,
        )
        self.relationship = P46REventModel(
            width=self.relation_width,
            dropout=dropout,
            classes=DETAIL_CLASSES,
        )
        # 384 P46 trial features + 256 ordered event features + 25 part/lag
        # correlations + 5 event centres.  These are one classifier input, not
        # two independently trained predictions combined after classification.
        self.relationship_input_width = 256 + 25 + 5
        self.classifier_input_width = 384 + self.relationship_input_width
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.classifier_input_width),
            nn.Linear(self.classifier_input_width, self.fusion_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_width, self.fusion_width),
            nn.GELU(),
        )
        self.detail_head = nn.Sequential(
            nn.LayerNorm(self.fusion_width),
            nn.Dropout(0.15),
            nn.Linear(self.fusion_width, DETAIL_CLASSES),
        )
        self.contrast_projection = nn.Sequential(
            nn.LayerNorm(self.fusion_width),
            nn.Linear(self.fusion_width, 128, bias=False),
        )
        self.subject_head = nn.Sequential(
            nn.LayerNorm(self.fusion_width),
            nn.Linear(self.fusion_width, 128),
            nn.GELU(),
            nn.Linear(128, subjects),
        )
        self._p46_pretraining = False

    def set_p46_pretraining(self, enabled: bool) -> None:
        """Skip the relationship branch during the original P46 Stage-A protocol."""

        self._p46_pretraining = bool(enabled)

    def forward_relationship(
        self,
        batch: dict[str, torch.Tensor],
        *,
        visual_scale: float = 1.0,
        visual_difference_scale: float = 1.0,
        skeleton_scale: float = 1.0,
        imu_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        return self.relationship(
            batch,
            visual_scale=visual_scale,
            visual_difference_scale=visual_difference_scale,
            skeleton_scale=skeleton_scale,
            imu_scale=imu_scale,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        subject_adversarial_scale: float = 0.0,
        *,
        relationship_batch: dict[str, torch.Tensor] | None = None,
        relationship_scale: float = 1.0,
        visual_scale: float = 1.0,
        visual_difference_scale: float = 1.0,
        skeleton_scale: float = 1.0,
        imu_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        base = self.p46(batch, subject_adversarial_scale=subject_adversarial_scale)
        if self._p46_pretraining:
            return base

        relation = self.forward_relationship(
            batch if relationship_batch is None else relationship_batch,
            visual_scale=visual_scale,
            visual_difference_scale=visual_difference_scale,
            skeleton_scale=skeleton_scale,
            imu_scale=imu_scale,
        )
        relationship_input = torch.cat(
            (
                relation["embedding"],
                relation["offset_correlation"].to(relation["embedding"].dtype),
                relation["event_centre"].to(relation["embedding"].dtype),
            ),
            dim=-1,
        )
        if relationship_input.shape[-1] != self.relationship_input_width:
            raise RuntimeError(
                "P46-R relationship contract changed: "
                f"expected {self.relationship_input_width}, got {relationship_input.shape}"
            )
        classifier_input = torch.cat(
            (
                base["trial_embedding"],
                float(relationship_scale) * relationship_input,
            ),
            dim=-1,
        )
        fused = self.fusion(classifier_input)
        return {
            **base,
            "detail_logits": self.detail_head(fused),
            "contrast_embedding": F.normalize(
                self.contrast_projection(fused).float(), dim=-1, eps=1e-6
            ),
            "subject_logits": self.subject_head(
                gradient_reverse(fused, subject_adversarial_scale)
            ),
            "p46_detail_logits": base["detail_logits"],
            "p46_contrast_embedding": base["contrast_embedding"],
            "p46_subject_logits": base["subject_logits"],
            "p46_trial_embedding": base["trial_embedding"],
            "relationship_detail_logits": relation["detail_logits"],
            "relationship_contrast_embedding": relation["contrast_embedding"],
            "relationship_embedding": relation["embedding"],
            "relationship_input": relationship_input,
            "relationship_part_summary": relation["part_summary"],
            "relationship_event_centre": relation["event_centre"],
            "relationship_event_localizer_logits": relation[
                "event_localizer_logits"
            ],
            "relationship_event_mask": relation["event_mask"],
            "relationship_evidence_target": relation["evidence_target"],
            "relationship_evidence_mask": relation["evidence_mask"],
            "relationship_offset_logits": relation["offset_logits"],
            "relationship_offset_correlation": relation["offset_correlation"],
            "classifier_input": classifier_input,
            "fused_trial_embedding": fused,
        }


def relationship_localization_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    target = output["relationship_evidence_target"]
    mask = output["relationship_evidence_mask"]
    log_probability = F.log_softmax(
        output["relationship_event_localizer_logits"].float().masked_fill(~mask, -1e4),
        dim=1,
    )
    valid_part = mask.any(dim=1)
    loss = -(target * log_probability).sum(dim=1)
    if not valid_part.any():
        return output["relationship_embedding"].sum() * 0.0
    return loss[valid_part].mean()


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
