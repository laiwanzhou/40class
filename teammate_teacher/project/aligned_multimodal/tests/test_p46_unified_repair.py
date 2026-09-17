from __future__ import annotations

import torch

from p31_skeleton_imu_model import IMUIntervalPartEncoder, SkeletonPartEncoder
from p46_event_model import LocalVisualObjectEncoder
from p46_unified_repair_model import (
    P46UnifiedRepairModel,
    SharedSameTimeRelationshipRefiner,
    relationship_localization_loss,
)
from test_p46_event_token import synthetic_batch


def _changed_embedding(
    model: P46UnifiedRepairModel,
    batch: dict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
) -> float:
    changed = dict(batch)
    changed[key] = value
    with torch.inference_mode():
        baseline = model(batch)["trial_embedding"]
        modified = model(changed)["trial_embedding"]
    return float((baseline - modified).abs().amax())


def test_unified_repair_has_one_encoder_per_main_modality_and_one_head() -> None:
    model = P46UnifiedRepairModel(width=48, dropout=0.0, subjects=3)
    assert sum(isinstance(module, SkeletonPartEncoder) for module in model.modules()) == 1
    assert sum(isinstance(module, IMUIntervalPartEncoder) for module in model.modules()) == 1
    assert sum(isinstance(module, LocalVisualObjectEncoder) for module in model.modules()) == 1
    assert sum(
        isinstance(module, SharedSameTimeRelationshipRefiner)
        for module in model.modules()
    ) == 1
    assert not hasattr(model, "p46")
    assert not hasattr(model, "relationship")
    assert not hasattr(model, "relationship_detail_head")


def test_unified_repair_forward_uses_single_time_aware_trial_embedding() -> None:
    model = P46UnifiedRepairModel(width=48, dropout=0.0, subjects=3).eval()
    with torch.inference_mode():
        output = model(synthetic_batch())
    assert output["detail_logits"].shape == (2, 21)
    assert output["trial_embedding"].shape == (2, 384)
    assert output["relationship_offset_logits"].shape == (2, 5)
    assert output["relationship_event_localizer_logits"].shape == (2, 5, 5)
    assert output["raw_imu_event_tokens"].shape == (2, 5, 10, 48)
    assert output["roi_metadata_embedding"].shape == (2, 5, 5, 48)
    assert torch.isfinite(output["detail_logits"]).all()


def test_unified_repair_logits_are_sensitive_to_every_new_input_contract() -> None:
    torch.manual_seed(7)
    batch = synthetic_batch()
    model = P46UnifiedRepairModel(width=48, dropout=0.0, subjects=3).eval()
    changes = {
        "time_position": _changed_embedding(
            model, batch, "time_position", torch.zeros_like(batch["time_position"])
        ),
        "frame_time_seconds": _changed_embedding(
            model,
            batch,
            "frame_time_seconds",
            batch["frame_time_seconds"] * 2.0,
        ),
        "imu_raw_vectors": _changed_embedding(
            model,
            batch,
            "imu_raw_vectors",
            batch["imu_raw_vectors"] + 0.5,
        ),
        "oriented_roi_geometry": _changed_embedding(
            model,
            batch,
            "oriented_roi_geometry",
            batch["oriented_roi_geometry"] + 0.25,
        ),
        "oriented_angle_valid": _changed_embedding(
            model,
            batch,
            "oriented_angle_valid",
            torch.zeros_like(batch["oriented_angle_valid"]),
        ),
        "local_roi_source": _changed_embedding(
            model,
            batch,
            "local_roi_source",
            torch.full_like(batch["local_roi_source"], 6),
        ),
    }
    assert all(value > 1e-7 for value in changes.values()), changes


def test_unified_repair_main_loss_reaches_new_continuous_inputs() -> None:
    batch = synthetic_batch()
    for key in (
        "time_position",
        "frame_time_seconds",
        "imu_raw_vectors",
        "oriented_roi_geometry",
    ):
        batch[key] = batch[key].clone().requires_grad_(True)
    model = P46UnifiedRepairModel(width=48, dropout=0.0, subjects=3).train()
    output = model(batch)
    loss = torch.nn.functional.cross_entropy(
        output["detail_logits"], torch.tensor((0, 1))
    )
    loss.backward()
    for key in (
        "time_position",
        "frame_time_seconds",
        "imu_raw_vectors",
        "oriented_roi_geometry",
    ):
        gradient = batch[key].grad
        assert gradient is not None, key
        assert torch.isfinite(gradient).all(), key
        assert gradient.abs().sum() > 0, key
    source_gradient = model.encoder.visual.roi_source_embedding.weight.grad
    assert source_gradient is not None
    assert source_gradient.abs().sum() > 0


def test_unified_offset_and_localization_losses_train_the_shared_tokens() -> None:
    batch = synthetic_batch()
    model = P46UnifiedRepairModel(width=48, dropout=0.0, subjects=3).train()
    output = model(batch)
    offsets = torch.tensor((0, 4))
    offset_logits = model.synthetic_offset_logits(
        output, offsets, batch["frame_mask"]
    )
    loss = (
        torch.nn.functional.cross_entropy(output["detail_logits"], torch.tensor((0, 1)))
        + 0.5
        * torch.nn.functional.cross_entropy(
            offset_logits, offsets
        )
        + 0.15 * relationship_localization_loss(output)
    )
    loss.backward()
    for module in (
        model.encoder.motion,
        model.encoder.visual,
        model.encoder.fusion,
        model.encoder.relationship,
        model.encoder.temporal,
        model.detail_head,
    ):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(value).all() for value in gradients)
        assert any(value.abs().sum() > 0 for value in gradients)
