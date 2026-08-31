from __future__ import annotations

import torch
from torch.nn import functional as F


def motion_attribute_loss(
    output: dict[str, torch.Tensor],
    *,
    family_targets: torch.Tensor,
    attribute_targets: torch.Tensor,
    labels: torch.Tensor,
    available: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    selected = available.bool()
    if selected.shape != labels.shape:
        raise ValueError("motion loss availability shape changed")
    if set(weights) != {"families", "attributes", "action"}:
        raise ValueError("motion loss weights changed")
    if bool(selected.any()):
        family = F.binary_cross_entropy_with_logits(
            output["family_logits"][selected], family_targets[selected].float()
        )
        attribute = F.smooth_l1_loss(
            output["attribute_predictions"][selected],
            attribute_targets[selected].float(),
        )
        action = F.cross_entropy(
            output["action_logits"][selected], labels[selected].long()
        )
    else:
        zero = output["action_logits"].sum() * 0.0
        family = attribute = action = zero
    total = (
        float(weights["families"]) * family
        + float(weights["attributes"]) * attribute
        + float(weights["action"]) * action
    )
    return {
        "loss": total,
        "family_loss": family,
        "attribute_loss": attribute,
        "action_loss": action,
        "supervised_rows": selected.sum(),
    }
