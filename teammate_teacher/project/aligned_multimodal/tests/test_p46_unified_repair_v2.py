from __future__ import annotations

from argparse import Namespace
import math

import torch

from p46_unified_repair_model import (
    GeometryAwareLocalVisualEncoder,
    P46UnifiedRepairModel,
    P46UnifiedRepairV2Model,
    P46UnifiedRepairV3Model,
    RawIMUResidualEncoder,
    parameter_count,
)
from test_p46_event_token import synthetic_batch
from train_p46_unified_repair import (
    dynamic_auxiliary_weights,
    group_dro_classification_loss,
    protected_boundary_loss,
)


def test_relative_roi_geometry_removes_global_translation_and_scale() -> None:
    geometry = torch.tensor(
        [[[[0.30, 0.40, 0.08, 0.12, 0.0, 1.0],
           [0.70, 0.40, 0.08, 0.12, 0.0, 1.0],
           [0.22, 0.60, 0.05, 0.05, 0.5, 0.866],
           [0.78, 0.60, 0.05, 0.05, -0.5, 0.866],
           [0.50, 0.75, 0.25, 0.12, 0.0, 1.0]]]]
    )
    valid = torch.ones(1, 1, 5, dtype=torch.bool)
    transformed = geometry.clone()
    transformed[..., :4] *= 1.7
    transformed[..., 0] += 0.13
    transformed[..., 1] -= 0.09
    baseline = GeometryAwareLocalVisualEncoder.torso_relative_geometry(
        geometry, valid
    )
    changed = GeometryAwareLocalVisualEncoder.torso_relative_geometry(
        transformed, valid
    )
    torch.testing.assert_close(baseline, changed, atol=1e-5, rtol=1e-5)


def test_v2_is_smaller_and_bounds_relationship_residual() -> None:
    v1 = P46UnifiedRepairModel(subjects=3)
    v2 = P46UnifiedRepairV2Model(subjects=3).eval()
    assert parameter_count(v2) < parameter_count(v1)
    with torch.inference_mode():
        output = v2(synthetic_batch())
    scale = output["relationship_residual_scale"]
    torch.testing.assert_close(scale, torch.full_like(scale, 0.05))
    assert float(scale.max()) <= 0.25


def test_v3_clean_trunk_starts_with_a_stronger_bounded_relationship_residual() -> None:
    v3 = P46UnifiedRepairV3Model(subjects=3).eval()
    with torch.inference_mode():
        output = v3(synthetic_batch())
    scale = output["relationship_residual_scale"]
    torch.testing.assert_close(scale, torch.full_like(scale, 0.10))
    assert float(scale.max()) <= 0.30
    assert v3.encoder.motion.raw.axis_rotation_radians == math.radians(10.0)
    assert v3.encoder.motion.raw.coordinate_dropout == 0.05


def test_raw_imu_axis_augmentation_is_training_only() -> None:
    encoder = RawIMUResidualEncoder(
        width=24,
        dropout=0.0,
        axis_rotation_degrees=60.0,
        coordinate_dropout=0.0,
    )
    raw = torch.zeros(2, 5, 4, 6)
    raw[..., 0] = 1.0
    raw[..., 4] = 2.0
    encoder.eval()
    torch.testing.assert_close(encoder._augment_raw_coordinates(raw), raw)
    encoder.train()
    torch.manual_seed(3)
    augmented = encoder._augment_raw_coordinates(raw)
    assert not torch.allclose(augmented, raw)
    torch.testing.assert_close(
        torch.linalg.vector_norm(augmented[..., :3], dim=-1),
        torch.linalg.vector_norm(raw[..., :3], dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_dynamic_auxiliary_controller_turns_off_saturated_offset() -> None:
    args = Namespace(
        protocol="v2",
        offset_weight=0.5,
        localization_weight=0.03,
        offset_minimum_epochs=2,
        offset_saturation_accuracy=0.95,
        localization_decay_epochs=8,
    )
    first = dynamic_auxiliary_weights(args, 1, {"previous_offset_accuracy": 0.0})
    saturated = dynamic_auxiliary_weights(
        args, 3, {"previous_offset_accuracy": 0.98}
    )
    assert first == {"offset": 0.5, "localization": 0.03}
    assert saturated["offset"] == 0.0
    assert 0.0 < saturated["localization"] < first["localization"]


def test_group_dro_upweights_harder_subject_and_boundary_loss_backpropagates() -> None:
    logits = torch.tensor(
        [[5.0, -2.0, -2.0], [4.0, -1.0, -1.0], [-2.0, 4.0, -1.0], [-2.0, 3.0, -1.0]],
        requires_grad=True,
    )
    labels = torch.tensor((0, 0, 2, 2))
    subjects = torch.tensor((0, 0, 1, 1))
    group_weights = torch.tensor((0.5, 0.5))
    classification = group_dro_classification_loss(
        logits,
        labels,
        subjects,
        torch.ones(3),
        0.0,
        group_weights,
        0.1,
    )
    assert group_weights[1] > group_weights[0]

    detail_logits = torch.zeros(3, 21, requires_grad=True)
    detail_labels = torch.tensor((0, 2, 3))  # class IDs 7, 9, 10
    boundary = protected_boundary_loss(detail_logits, detail_labels)
    (classification + boundary).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert detail_logits.grad is not None and detail_logits.grad.abs().sum() > 0
