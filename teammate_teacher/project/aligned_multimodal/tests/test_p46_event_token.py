from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from p46_event_model import EVENT_PART_NAMES, EVENT_PHASE_NAMES, P46EventTokenEncoder
from p46_complete_repair_model import P46CompleteRepairModel
from p46_full_repair_model import P46FullRepairModel, relationship_localization_loss
from p46_event_preprocessing import (
    body_coordinate_axes,
    quaternion_rotate,
    upper_body_human_prior,
)
from p46_protocol import EXPECTED, freeze_rows, read_rows
from p46_step10_model import (
    P46Step10Model,
    contact_and_phase_losses,
    cross_subject_supervised_contrastive,
    hardest_rival_loss,
    left_right_swap_batch,
    mask_modalities,
    modality_summary_targets,
    same_part_alignment_loss,
    selected_modality_reconstruction_loss,
    temporal_order_loss,
)
from p46r_event_bottleneck_model import (
    P46R_OFFSET_FRACTIONS,
    P46REventModel,
    circular_shift_visual_batch,
    event_localization_loss,
)
from train_p46_step10 import (
    FrameBudgetBatchSampler,
    FullCoverageCrossSubjectBatchSampler,
    batch_coverage_report,
    make_class_weights,
    source_coverage_report,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_p46_protocol_is_one_subject_disjoint_split() -> None:
    rows, summary = freeze_rows(
        read_rows(PROJECT_DIR / "data" / "subject_folds" / "fold_0.csv")
    )
    assert len(rows) == EXPECTED["train_trials"] + EXPECTED["val_trials"]
    for key, value in EXPECTED.items():
        assert summary[key] == value
    assert set(summary["train_subject_ids"]).isdisjoint(summary["val_subject_ids"])


def test_body_coordinate_normalisation_removes_camera_rotation() -> None:
    xyz = np.zeros((3, 17, 3), dtype=np.float32)
    xyz[:, 11] = (-1.0, 1.0, 0.0)
    xyz[:, 14] = (1.0, 1.0, 0.0)
    xyz[:, 8] = (0.0, 1.0, 0.0)
    xyz[:, 9] = (0.0, 1.5, 0.0)
    xyz[:, 10] = (0.0, 2.0, 0.0)
    xyz[:, 13] = (-1.5, 0.5, 0.2)
    xyz[:, 16] = (1.5, 0.5, -0.2)
    valid = np.ones((3, 17), dtype=bool)
    angle = 0.73
    rotation = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    rotated = np.einsum("tjc,cd->tjd", xyz, rotation.T)
    axes_a, raw_a, _ = body_coordinate_axes(xyz, valid)
    axes_b, raw_b, _ = body_coordinate_axes(rotated, valid)
    body_a = np.einsum("tjc,tck->tjk", xyz, axes_a)
    body_b = np.einsum("tjc,tck->tjk", rotated, axes_b)
    assert raw_a.all() and raw_b.all()
    np.testing.assert_allclose(body_a, body_b, atol=1e-5)


def test_quaternion_rotation_uses_wxyz_contract() -> None:
    half = math.pi / 4.0
    quaternion = np.asarray([[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=np.float32)
    vector = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    rotated = quaternion_rotate(quaternion, vector)
    np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-5)


def test_upper_body_prior_survives_missing_person_box() -> None:
    keypoints = np.full((1, 17, 3), np.nan, dtype=np.float32)
    keypoints[0, 5] = (30.0, 20.0, 0.9)
    keypoints[0, 7] = (35.0, 40.0, 0.9)
    keypoints[0, 9] = (40.0, 60.0, 0.9)
    boxes = np.full((1, 4), np.nan, dtype=np.float32)
    prior = upper_body_human_prior(keypoints, boxes, width=80, height=80)
    assert prior.shape == (1, 80, 80)
    assert prior.dtype == np.uint8
    assert prior.any()


def synthetic_batch() -> dict[str, torch.Tensor]:
    batch, steps, points = 2, 5, 8
    frame_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    skeleton_mask = frame_mask[:, :, None].expand(batch, steps, 17).clone()
    feature_mask = skeleton_mask[:, :, :, None].expand(batch, steps, 17, 13).clone()
    relation_mask = frame_mask[:, :, None].expand(batch, steps, 18).clone()
    imu_point_mask = torch.ones(batch, 5, points, dtype=torch.bool)
    imu_frame_index = torch.arange(points).remainder(steps)[None, None].expand(batch, 5, -1).clone()
    imu_values = torch.randn(batch, 5, points, 10) * 0.05
    imu_values[..., 6] = 1.0
    imu_values[..., 7:10] = 0.0
    body_axes = torch.eye(3)[None, None].expand(batch, steps, -1, -1).clone()
    return {
        "frame_mask": frame_mask,
        "frame_time_seconds": torch.arange(steps).float()[None].expand(batch, -1) * 0.1,
        "time_position": torch.linspace(0.0, 1.0, steps)[None].expand(batch, -1),
        "skeleton_features": torch.randn(batch, steps, 17, 13) * 0.05,
        "skeleton_feature_mask": feature_mask,
        "skeleton_joint_mask": skeleton_mask,
        "skeleton_relations": torch.randn(batch, steps, 18) * 0.05,
        "skeleton_relation_mask": relation_mask,
        "skeleton_frame_quality": frame_mask.float(),
        "body_axes_camera": body_axes,
        "body_axes_raw_valid": frame_mask,
        "imu_values": imu_values,
        "imu_raw_vectors": torch.randn(batch, 5, points, 6) * 0.05,
        "imu_time_seconds": torch.linspace(0.0, 0.5, points)[None, None].expand(batch, 5, -1),
        "imu_frame_index": imu_frame_index,
        "imu_point_mask": imu_point_mask,
        "imu_device_mask": torch.ones(batch, 5, dtype=torch.bool),
        "imu_interval_counts": torch.ones(batch, steps, 5, dtype=torch.long),
        "arm_spatial_features": torch.randn(batch, steps, 2, 2, 3, 3, 128) * 0.05,
        "detail_spatial_features": torch.randn(batch, steps, 2, 3, 5, 5, 128) * 0.05,
        "local_geometry_features": torch.rand(batch, steps, 5, 5, 5, 6),
        "oriented_roi_geometry": torch.rand(batch, steps, 5, 6),
        "oriented_angle_valid": frame_mask[:, :, None].expand(batch, steps, 5),
        "local_roi_valid": frame_mask[:, :, None].expand(batch, steps, 5),
        "local_roi_quality": frame_mask[:, :, None].expand(batch, steps, 5).float(),
        "local_roi_source": torch.ones(batch, steps, 5, dtype=torch.long),
        "local_roi_clipped_ratio": torch.zeros(batch, steps, 5),
        "pose_quality_factor": frame_mask.float(),
        "context_features": torch.randn(batch, steps, 2, 2, 896) * 0.05,
        "context_valid": frame_mask[:, :, None].expand(batch, steps, 2),
        "context_quality": frame_mask[:, :, None].expand(batch, steps, 2).float(),
    }


def test_p46_steps_7_to_9_forward_without_classification_head() -> None:
    model = P46EventTokenEncoder(width=48, dropout=0.0).eval()
    with torch.inference_mode():
        output = model(synthetic_batch())
    assert output["event_tokens"].shape == (2, 5, len(EVENT_PART_NAMES), 48)
    assert output["phase_logits"].shape == (2, 5, len(EVENT_PHASE_NAMES))
    assert output["trial_embedding"].shape == (2, 384)
    assert output["explicit_event_statistics"].shape == (2, 12)
    assert "logits" not in output
    for key in (
        "event_tokens",
        "soft_event_gate",
        "trial_embedding",
        "phase_logits",
        "explicit_event_statistics",
    ):
        assert torch.isfinite(output[key]).all()


def test_p46_event_encoder_has_end_to_end_gradients_before_step10() -> None:
    model = P46EventTokenEncoder(width=48, dropout=0.0).train()
    output = model(synthetic_batch())
    loss = output["trial_embedding"].square().mean() + output["phase_logits"].square().mean()
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


def test_p46_step10_mask_swap_and_auxiliary_losses_are_finite() -> None:
    batch = synthetic_batch()
    model = P46Step10Model(width=48, dropout=0.0, subjects=3).train()
    output = model(batch, subject_adversarial_scale=1.0)
    assert output["detail_logits"].shape == (2, 21)
    assert output["subject_logits"].shape == (2, 3)
    assert output["modality_reconstruction"].shape == (2, 12)

    assignment = torch.tensor((0, 2))
    targets = modality_summary_targets(batch)
    masked = mask_modalities(batch, assignment)
    assert not masked["local_roi_valid"][0].any()
    assert not masked["imu_point_mask"][1].any()
    masked_output = model(masked)
    reconstruction = selected_modality_reconstruction_loss(
        masked_output["modality_reconstruction"], targets, assignment
    )
    contact, phase, coverage = contact_and_phase_losses(output, batch)
    losses = (
        reconstruction,
        contact,
        phase,
        coverage,
        same_part_alignment_loss(output, maximum_per_trial=3),
        temporal_order_loss(output),
    )
    assert all(torch.isfinite(value) for value in losses)

    swapped_twice = left_right_swap_batch(left_right_swap_batch(batch))
    for key in (
        "arm_spatial_features",
        "detail_spatial_features",
        "local_geometry_features",
        "oriented_roi_geometry",
        "skeleton_features",
        "skeleton_relations",
        "imu_values",
        "imu_interval_counts",
    ):
        torch.testing.assert_close(swapped_twice[key], batch[key])


def test_p46_step10_combined_supervised_gradients_are_finite() -> None:
    batch = synthetic_batch()
    model = P46Step10Model(width=48, dropout=0.0, subjects=2).train()
    output = model(batch, subject_adversarial_scale=1.0)
    labels = torch.tensor((0, 0))
    subjects = torch.tensor((0, 1))
    contact, phase, _ = contact_and_phase_losses(output, batch)
    total = (
        torch.nn.functional.cross_entropy(output["detail_logits"], labels)
        + 0.15 * hardest_rival_loss(output["detail_logits"], labels)
        + 0.08
        * cross_subject_supervised_contrastive(
            output["contrast_embedding"], labels, subjects
        )
        + 0.03 * torch.nn.functional.cross_entropy(output["subject_logits"], subjects)
        + 0.05 * same_part_alignment_loss(output, maximum_per_trial=3)
        + 0.05 * contact
        + 0.05 * phase
        + 0.02 * temporal_order_loss(output)
    )
    total.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_p46r_event_bottleneck_forward_and_gradients_are_finite() -> None:
    batch = synthetic_batch()
    model = P46REventModel(width=48, dropout=0.0).train()
    output = model(batch)
    assert output["detail_logits"].shape == (2, 21)
    assert output["offset_logits"].shape == (2, len(P46R_OFFSET_FRACTIONS))
    assert output["part_summary"].shape == (2, 5, 48)
    assert output["event_during_weight"].shape == (2, 5, 5)
    assert torch.isfinite(output["detail_logits"]).all()
    assert torch.isfinite(output["offset_logits"]).all()
    loss = (
        torch.nn.functional.cross_entropy(output["detail_logits"], torch.tensor((0, 1)))
        + event_localization_loss(output)
    )
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_p46r_circular_visual_shift_has_no_zero_padding_shortcut() -> None:
    batch = synthetic_batch()
    positive = torch.full((2,), 4, dtype=torch.long)
    negative = torch.full((2,), 0, dtype=torch.long)
    shifted, forward_steps = circular_shift_visual_batch(batch, positive)
    restored, backward_steps = circular_shift_visual_batch(shifted, negative)
    assert forward_steps.tolist() == [1, 1]
    assert backward_steps.tolist() == [-1, -1]
    for key in (
        "arm_spatial_features",
        "detail_spatial_features",
        "local_geometry_features",
        "local_roi_valid",
    ):
        torch.testing.assert_close(restored[key], batch[key])


def test_p46r_classifier_has_no_context_or_global_scene_bypass() -> None:
    batch = synthetic_batch()
    changed = dict(batch)
    changed["context_features"] = batch["context_features"] + 1000.0
    changed["context_quality"] = torch.zeros_like(batch["context_quality"])
    model = P46REventModel(width=48, dropout=0.0).eval()
    with torch.inference_mode():
        original = model(batch)["detail_logits"]
        modified = model(changed)["detail_logits"]
    torch.testing.assert_close(original, modified)


def test_p46_complete_repair_is_exactly_p46_at_zero_initialisation() -> None:
    batch = synthetic_batch()
    model = P46CompleteRepairModel(
        base_width=48, relation_width=48, subjects=3, dropout=0.0
    ).eval()
    with torch.inference_mode():
        output = model(batch)
        without_relationship = model(batch, relationship_scale=0.0)
    torch.testing.assert_close(output["relationship_delta"], torch.zeros_like(output["relationship_delta"]))
    torch.testing.assert_close(output["detail_logits"], output["base_detail_logits"])
    torch.testing.assert_close(output["detail_logits"], without_relationship["detail_logits"])
    assert output["relationship_input"].shape == (2, 286)


def test_p46_complete_repair_freezes_pretrained_branches_and_trains_input_adapter() -> None:
    batch = synthetic_batch()
    model = P46CompleteRepairModel(
        base_width=48, relation_width=48, subjects=3, dropout=0.0
    )
    model.freeze_pretrained()
    model.train()
    assert not model.base.training
    assert not model.relation.training
    assert model.relationship_adapter.training
    output = model(batch)
    loss = torch.nn.functional.cross_entropy(output["detail_logits"], torch.tensor((0, 1)))
    loss.backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(parameter.grad is None for parameter in model.relation.parameters())
    adapter_gradients = [
        parameter.grad
        for parameter in model.relationship_adapter.parameters()
        if parameter.grad is not None
    ]
    assert adapter_gradients
    assert all(torch.isfinite(value).all() for value in adapter_gradients)
    assert any(value.abs().sum() > 0 for value in adapter_gradients)


def test_p46_complete_repair_bounds_feature_delta_without_output_gate() -> None:
    batch = synthetic_batch()
    model = P46CompleteRepairModel(
        base_width=48,
        relation_width=48,
        subjects=3,
        dropout=0.0,
        adapter_rank=8,
        maximum_delta_norm=0.25,
    ).eval()
    with torch.no_grad():
        model.relationship_adapter[-1].bias.fill_(10.0)
        output = model(batch)
    assert torch.all(output["relationship_delta"].norm(dim=1) <= 0.25001)
    torch.testing.assert_close(
        output["fused_trial_embedding"],
        output["base_trial_embedding"] + output["relationship_delta"],
    )


def test_p46_full_repair_uses_one_joint_classifier_input_from_random_branches() -> None:
    batch = synthetic_batch()
    model = P46FullRepairModel(
        base_width=48,
        relation_width=48,
        fusion_width=96,
        subjects=3,
        dropout=0.0,
    ).eval()
    assert all(parameter.requires_grad for parameter in model.parameters())
    with torch.inference_mode():
        output = model(batch)
    assert output["classifier_input"].shape == (2, 670)
    assert output["relationship_input"].shape == (2, 286)
    assert output["fused_trial_embedding"].shape == (2, 96)
    assert output["detail_logits"].shape == (2, 21)


def test_p46_full_repair_main_and_mechanism_losses_reach_both_branches() -> None:
    batch = synthetic_batch()
    model = P46FullRepairModel(
        base_width=48,
        relation_width=48,
        fusion_width=96,
        subjects=3,
        dropout=0.0,
    )
    labels = torch.tensor((0, 1))
    offset_labels = torch.tensor((0, 4))
    shifted, _ = circular_shift_visual_batch(batch, offset_labels)
    output = model(batch)
    shifted_relationship = model.forward_relationship(shifted)
    loss = (
        torch.nn.functional.cross_entropy(output["detail_logits"], labels)
        + 0.5
        * torch.nn.functional.cross_entropy(
            shifted_relationship["offset_logits"], offset_labels
        )
        + 0.15 * relationship_localization_loss(output)
    )
    loss.backward()
    for module in (model.p46, model.relationship, model.fusion, model.detail_head):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(value).all() for value in gradients)
        assert any(value.abs().sum() > 0 for value in gradients)


def test_p46_step10_dynamic_batch_samplers_respect_frame_budgets() -> None:
    lengths = [31, 33, 63, 65, 95, 97, 127, 129] * 2
    labels = [0, 0, 1, 1, 0, 0, 1, 1] * 2
    users = ["a", "b", "a", "b", "c", "d", "c", "d"] * 2
    source_ids = [f"trial_{index}" for index in range(len(lengths))]
    sampler = FullCoverageCrossSubjectBatchSampler(
        lengths,
        labels,
        users,
        source_ids,
        maximum_batch_size=8,
        seed=2026,
        frame_budget=520,
        bucket_multiplier=2,
    )
    batches = list(sampler)
    assert len(sampler) == len(batches)
    flattened = [index for batch in batches for index in batch]
    assert len(flattened) == len(lengths)
    assert sorted(flattened) == list(range(len(lengths)))
    assert len(set(flattened)) == len(lengths)
    coverage = sampler.coverage_report()
    assert coverage["exact_once"]
    assert coverage["unique_samples"] == len(lengths)
    assert coverage["missing_count"] == 0
    assert coverage["duplicate_positions"] == 0
    assert coverage["max_repeat_count"] == 1
    assert coverage["cross_subject_positive_samples"] > 0
    for batch in batches:
        assert len(batch) * max(lengths[index] for index in batch) <= 520

    sampler.set_epoch(1)
    second_epoch = [index for batch in sampler for index in batch]
    assert sorted(second_epoch) == list(range(len(lengths)))
    assert second_epoch != flattened

    validation = FrameBudgetBatchSampler(
        lengths, maximum_batch_size=8, frame_budget=520, bucket_multiplier=2
    )
    validation_batches = list(validation)
    assert len(validation) == len(validation_batches)
    assert sorted(index for batch in validation_batches for index in batch) == list(
        range(len(lengths))
    )
    assert all(
        len(batch) * max(lengths[index] for index in batch) <= 520
        for batch in validation_batches
    )


def test_p46_step10_coverage_audits_reject_duplicates_and_missing_trials() -> None:
    correct = batch_coverage_report([[2, 0], [1, 3]], expected_samples=4)
    assert correct["exact_once"]

    broken = batch_coverage_report([[0, 0], [1, 3]], expected_samples=4)
    assert not broken["exact_once"]
    assert broken["missing_count"] == 1
    assert broken["duplicate_positions"] == 1

    expected = ["a", "b", "c", "d"]
    delivered = source_coverage_report(["a", "b", "c", "d"], expected)
    assert delivered["exact_once"]
    repeated = source_coverage_report(["a", "b", "b", "d"], expected)
    assert not repeated["exact_once"]
    assert repeated["missing_count"] == 1
    assert repeated["duplicate_positions"] == 1


def test_p46_step10_class_balance_uses_loss_weights_not_replacement() -> None:
    weights, counts = make_class_weights([0, 0, 0, 0, 1, 1, 2], classes=3, power=0.5)
    assert counts == [4, 2, 1]
    assert weights[0] < weights[1] < weights[2]
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
