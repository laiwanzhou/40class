from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _masked_ce(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if bool(mask.any()):
        return F.cross_entropy(logits[mask], labels[mask])
    return logits.sum() * 0.0


def hierarchical_teacher_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    epoch: int,
    natural_pattern: bool,
) -> dict[str, torch.Tensor]:
    core = output["core_available"].bool()
    group_mask = output["effective_group_mask"].bool()
    fused_ce = _masked_ce(output["logits"], labels, core)
    context_ce = _masked_ce(output["context_logits"], labels, group_mask[:, 0])
    wrist_ce = _masked_ce(output["wrist_logits"], labels, group_mask[:, 1])
    body_ce = _masked_ce(output["body_logits"], labels, group_mask[:, 2])
    entropy_floor = output["logits"].sum() * 0.0
    if epoch <= 2 and bool(core.any()):
        attention = output["group_attention"][core].clamp_min(1e-12)
        entropy = -(attention * attention.log()).sum(dim=2)
        available_count = group_mask[core].sum(dim=1).clamp_min(1).to(entropy.dtype)
        target = 0.5 * available_count.log()[:, None]
        entropy_floor = torch.relu(target - entropy).mean()
    total = (
        fused_ce
        + 0.15 * context_ce
        + 0.15 * wrist_ce
        + 0.20 * body_ce
        + 0.02 * entropy_floor
    )
    if not natural_pattern:
        total = total * 0.5
    return {
        "loss": total,
        "fused_ce": fused_ce,
        "context_ce": context_ce,
        "wrist_ce": wrist_ce,
        "body_ce": body_ce,
        "entropy_floor": entropy_floor,
        "supervised_rows": core.sum(),
    }
