from __future__ import annotations

import torch

from src.models.motion_attribute_expert import MotionAttributeExpert
from src.training.motion_attribute_loss import motion_attribute_loss


def _inputs(batch: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    mask = torch.ones(batch, 96, dtype=torch.bool)
    mask[1, 40:55] = False
    mask[-1].zero_()
    features = torch.randn(batch, 96, 17, 6)
    features[~mask] = 0
    return {
        "features": features,
        "mask": mask,
        "families": torch.randint(0, 2, (batch, 6)).float(),
        "attributes": torch.randn(batch, 16),
        "labels": torch.tensor([0, 1, 2]),
        "available": mask.any(dim=1),
    }


def test_motion_attribute_expert_shapes_masks_and_parameter_budget() -> None:
    model = MotionAttributeExpert()
    inputs = _inputs()

    output = model(inputs["features"], inputs["mask"])

    assert output["embedding"].shape == (3, 256)
    assert output["family_logits"].shape == (3, 6)
    assert output["attribute_predictions"].shape == (3, 16)
    assert output["action_logits"].shape == (3, 40)
    assert torch.count_nonzero(output["embedding"][-1]) == 0
    assert torch.count_nonzero(output["family_logits"][-1]) == 0
    assert torch.isfinite(output["action_logits"]).all()
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_000_000


def test_weighted_multitask_loss_reaches_encoder_and_all_heads() -> None:
    model = MotionAttributeExpert()
    inputs = _inputs()
    output = model(inputs["features"], inputs["mask"])

    losses = motion_attribute_loss(
        output,
        family_targets=inputs["families"],
        attribute_targets=inputs["attributes"],
        labels=inputs["labels"],
        available=inputs["available"],
        weights={"families": 1.0, "attributes": 0.5, "action": 0.25},
    )
    expected = (
        losses["family_loss"]
        + 0.5 * losses["attribute_loss"]
        + 0.25 * losses["action_loss"]
    )
    torch.testing.assert_close(losses["loss"], expected)
    losses["loss"].backward()

    assert model.input_projection.weight.grad.abs().sum() > 0
    assert model.family_head.weight.grad.abs().sum() > 0
    assert model.attribute_head.weight.grad.abs().sum() > 0
    assert model.action_head.weight.grad.abs().sum() > 0
    assert losses["supervised_rows"].item() == 2
