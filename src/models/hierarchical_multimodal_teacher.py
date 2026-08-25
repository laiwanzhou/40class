from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.multimodal_token_contract import GroupTokens


@dataclass(frozen=True)
class GroupDropout:
    context: float = 0.10
    wrist: float = 0.10
    body: float = 0.15
    visual: float = 0.10

    def __post_init__(self) -> None:
        if any(not 0.0 <= value <= 1.0 for value in (
            self.context, self.wrist, self.body, self.visual
        )):
            raise ValueError("group dropout probabilities must be in [0,1]")

    @classmethod
    def disabled(cls) -> "GroupDropout":
        return cls(context=0.0, wrist=0.0, body=0.0, visual=0.0)


def _replace_mask(group: GroupTokens, mask: torch.Tensor) -> GroupTokens:
    tokens = group.tokens * mask[:, :, :, None].to(group.tokens.dtype)
    quality = group.quality * mask[:, :, :, None].to(group.quality.dtype)
    return GroupTokens(
        tokens=tokens,
        mask=mask,
        quality=quality,
        quality_mask=group.quality_mask & mask[:, :, :, None],
    )


class HierarchicalMultimodalTeacher(nn.Module):
    def __init__(
        self,
        *,
        visual_encoder: nn.Module,
        body_encoder: nn.Module,
        fusion: nn.Module,
        dim: int = 256,
        classes: int = 40,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.body_encoder = body_encoder
        self.fusion = fusion
        self.context_head = nn.Linear(dim, classes)
        self.wrist_head = nn.Linear(dim, classes)
        self.body_head = nn.Linear(dim, classes)

    def _drop_groups(
        self,
        visual: GroupTokens,
        body: GroupTokens,
        policy: GroupDropout,
    ) -> tuple[GroupTokens, GroupTokens]:
        if not self.training or policy == GroupDropout.disabled():
            return visual, body
        batch = visual.tokens.shape[0]
        original = torch.cat((visual.mask, body.mask), dim=2)
        context_drop = torch.rand(batch, device=visual.tokens.device) < policy.context
        wrist_drop = torch.rand(batch, device=visual.tokens.device) < policy.wrist
        both_subgroups = context_drop & wrist_drop
        wrist_drop = wrist_drop & ~both_subgroups
        visual_drop = torch.rand(batch, device=visual.tokens.device) < policy.visual
        body_drop = torch.rand(batch, device=visual.tokens.device) < policy.body
        visual_mask = visual.mask.clone()
        body_mask = body.mask.clone()
        visual_mask[context_drop | visual_drop, :, 0] = False
        visual_mask[wrist_drop | visual_drop, :, 1] = False
        body_mask[body_drop, :, 0] = False

        effective = torch.cat((visual_mask, body_mask), dim=2)
        empty = ~effective.any(dim=(1, 2)) & original.any(dim=(1, 2))
        for row in torch.nonzero(empty, as_tuple=False).flatten().tolist():
            # Preserve one real group, preferring wrist, then context, then body.
            for group_index in (1, 0, 2):
                available = original[row, :, group_index]
                if bool(available.any()):
                    if group_index < 2:
                        visual_mask[row, :, group_index] = available
                    else:
                        body_mask[row, :, 0] = available
                    break
        return _replace_mask(visual, visual_mask), _replace_mask(body, body_mask)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        dropout_policy: GroupDropout | None = None,
        enabled_modalities: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        enabled = set(
            ("ir", "depth_color", "skeleton", "imu")
            if enabled_modalities is None
            else enabled_modalities
        )
        unknown = enabled - {"ir", "depth_color", "skeleton", "imu"}
        if unknown:
            raise ValueError(f"unknown Stage-1 modalities: {sorted(unknown)}")
        visual_availability = batch["visual_view_availability"].clone()
        if "ir" not in enabled:
            visual_availability[:, 0] = False
        if "depth_color" not in enabled:
            visual_availability[:, 1] = False
        skeleton_mask = batch["skeleton_mask"].clone()
        imu_role_mask = batch["imu_role_mask"].clone()
        if "skeleton" not in enabled:
            skeleton_mask.zero_()
        if "imu" not in enabled:
            imu_role_mask.zero_()
        visual = self.visual_encoder(
            ir=batch["visual"][:, 0],
            depth=batch["visual"][:, 1],
            availability=visual_availability,
        )
        body = self.body_encoder(
            batch["skeleton"],
            batch["imu"],
            skeleton_mask,
            imu_role_mask,
        )
        visual, body = self._drop_groups(
            visual, body, dropout_policy or GroupDropout.disabled()
        )
        output = self.fusion(visual=visual, body=body)
        group_features = output["group_features"]
        output.update(
            {
                "context_logits": self.context_head(group_features[:, 0]),
                "wrist_logits": self.wrist_head(group_features[:, 1]),
                "body_logits": self.body_head(group_features[:, 2]),
                "visual_tokens": visual.tokens,
                "visual_mask": visual.mask,
                "body_tokens": body.tokens,
                "body_mask": body.mask,
            }
        )
        return output
