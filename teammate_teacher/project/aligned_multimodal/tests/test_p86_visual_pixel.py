from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p86_visual_pixel_model import P86TrainableVisualStudent, model_size_mib
from p86_mc3_visual_model import P86MC3VisualStudent
from p86_videomae_small_visual_model import P86VideoMAESmallVisualStudent
from p86_visual_pixel_data import P86VisualPixelDataset


def test_trainable_visual_student_shapes_and_cost() -> None:
    model = P86TrainableVisualStudent(enable_distillation_projection=True)
    model.freeze_low_level()
    model.eval()
    images = torch.zeros(1, 2, 8, 3, 64, 64, dtype=torch.uint8)
    valid = torch.ones(1, 2, 8, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 8, 3)
    with torch.inference_mode():
        output = model(images, valid, quality)
    assert output["logits"].shape == (1, 40)
    assert output["clip_embeddings"].shape == (1, 2, 3, 512)
    assert output["projected_clip_embeddings"].shape == (1, 2, 3, 1024)
    assert output["clip_mask"].shape == (1, 2, 3)
    assert model_size_mib(model, 4) < 70.0
    assert not any(parameter.requires_grad for parameter in model.conv1.parameters())
    assert any(parameter.requires_grad for parameter in model.layer4.parameters())


def test_pixel_model_rejects_wrong_temporal_shape() -> None:
    model = P86TrainableVisualStudent().eval()
    images = torch.zeros(1, 2, 7, 3, 64, 64, dtype=torch.uint8)
    valid = torch.ones(1, 2, 7, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 7, 3)
    try:
        model(images, valid, quality)
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong temporal shape must be rejected")


def test_structured_fusion_preserves_six_tokens_within_budget() -> None:
    model = P86TrainableVisualStudent(fusion_mode="structured").eval()
    images = torch.zeros(1, 2, 8, 3, 64, 64, dtype=torch.uint8)
    valid = torch.ones(1, 2, 8, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 8, 3)
    with torch.inference_mode():
        output = model(images, valid, quality)
    assert output["logits"].shape == (1, 40)
    assert model.structured_fusion is not None
    assert model.quality_gate is None
    assert model_size_mib(model, 4) < 70.0


def test_gated_structured_residual_stays_one_classifier_and_under_budget() -> None:
    model = P86TrainableVisualStudent(fusion_mode="gated_residual").eval()
    images = torch.zeros(1, 2, 8, 3, 64, 64, dtype=torch.uint8)
    valid = torch.ones(1, 2, 8, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 8, 3)
    with torch.inference_mode():
        output = model(images, valid, quality)
    assert output["logits"].shape == (1, 40)
    assert model.quality_gate is not None and model.structured_fusion is not None
    assert model.structured_residual_scale is not None
    assert sum(isinstance(module, torch.nn.Linear) and module.out_features == 40 for module in model.modules()) == 1
    assert model_size_mib(model, 4) < 75.0


def test_layer2_can_be_unfrozen_without_unfreezing_stem() -> None:
    model = P86TrainableVisualStudent()
    model.freeze_low_level("layer1")
    assert not any(parameter.requires_grad for parameter in model.layer1.parameters())
    assert any(parameter.requires_grad for parameter in model.layer2.parameters())
    assert not any(parameter.requires_grad for parameter in model.conv1.parameters())


def test_twelve_frame_model_keeps_same_parameter_budget_class() -> None:
    model = P86TrainableVisualStudent(frames=12).eval()
    images = torch.zeros(1, 2, 12, 3, 64, 64, dtype=torch.uint8)
    valid = torch.ones(1, 2, 12, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 12, 3)
    with torch.inference_mode():
        output = model(images, valid, quality)
    assert output["logits"].shape == (1, 40)
    assert model_size_mib(model, 4) < 66.0


def test_mc3_visual_student_is_a_single_small_pure_visual_model() -> None:
    model = P86MC3VisualStudent(frames=12, kinetics_pretrained=False).eval()
    model.freeze_low_level("layer2")
    images = torch.zeros(1, 2, 12, 3, 64, 64, dtype=torch.uint8)
    valid = torch.ones(1, 2, 12, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 12, 3)
    with torch.inference_mode():
        output = model(images, valid, quality)
    assert output["logits"].shape == (1, 40)
    assert output["clip_embeddings"].shape == (1, 2, 3, 512)
    assert sum(
        isinstance(module, torch.nn.Linear) and module.out_features == 40
        for module in model.modules()
    ) == 1
    assert model_size_mib(model, 4) < 65.0
    assert not any(parameter.requires_grad for parameter in model.layer2.parameters())
    assert any(parameter.requires_grad for parameter in model.layer4.parameters())


def test_mc3_all_missing_visual_mask_remains_finite() -> None:
    model = P86MC3VisualStudent(
        frames=4,
        kinetics_pretrained=False,
        temporal_modeling=True,
    ).eval()
    sequence = torch.zeros(2, 2, 3, 4, 512)
    valid = torch.zeros(2, 2, 4, 3, dtype=torch.bool)
    quality = torch.zeros(2, 2, 4, 3)
    global_time = torch.zeros(2, 2, 4)
    with torch.inference_mode():
        output = model.forward_from_backbone_sequence(
            sequence, valid, quality, global_time
        )
    assert torch.isfinite(output["logits"]).all()
    assert not output["clip_mask"].any()
    assert torch.count_nonzero(output["clip_embeddings"]) == 0


def test_temporal_mc3_preserves_order_within_budget() -> None:
    model = P86MC3VisualStudent(
        frames=12,
        kinetics_pretrained=False,
        temporal_modeling=True,
    ).eval()
    model.freeze_low_level("layer3")
    images = torch.zeros(1, 2, 12, 3, 64, 64, dtype=torch.uint8)
    images[:, :, 6:] = 255
    valid = torch.ones(1, 2, 12, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 12, 3)
    with torch.inference_mode():
        output = model(images, valid, quality)
    assert output["logits"].shape == (1, 40)
    assert output["clip_embeddings"].shape == (1, 2, 3, 512)
    assert model.time_position is not None
    assert model.temporal_encoder is not None
    assert model.temporal_fusion is not None
    assert model_size_mib(model, 4) < 80.0
    assert not any(parameter.requires_grad for parameter in model.layer3.parameters())
    assert any(parameter.requires_grad for parameter in model.layer4.parameters())
    model.train()
    with torch.inference_mode():
        training_output = model(images, valid, quality)
    assert training_output["stage_logits"].shape == (1, 3, 40)
    assert sum(
        isinstance(module, torch.nn.Linear) and module.out_features == 40
        for module in model.modules()
    ) == 1


def test_subject_robust_augmentation_keeps_uint8_contract() -> None:
    np.random.seed(20260811)
    images = np.arange(2 * 12 * 3 * 32 * 32, dtype=np.uint32)
    images = (images % 256).astype(np.uint8).reshape(2, 12, 3, 32, 32)
    augmented, temporal_index = P86VisualPixelDataset._subject_robust_augment(images)
    assert augmented.shape == images.shape
    assert augmented.dtype == np.uint8
    assert int(augmented.min()) >= 0
    assert int(augmented.max()) <= 255
    assert temporal_index.shape == (12,)
    assert int(temporal_index.min()) >= 0
    assert int(temporal_index.max()) < 12


def test_temporal_mc3_can_use_exact_source_time() -> None:
    model = P86MC3VisualStudent(
        frames=4,
        kinetics_pretrained=False,
        temporal_modeling=True,
        exact_time_modeling=True,
    ).eval()
    sequence = torch.randn(1, 2, 3, 4, 512)
    valid = torch.ones(1, 2, 4, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 4, 3)
    global_time = torch.tensor([[[0.0, 0.2, 0.4, 0.7], [0.3, 0.5, 0.8, 1.0]]])
    with torch.inference_mode():
        output = model.forward_from_backbone_sequence(
            sequence, valid, quality, global_time
        )
    assert output["logits"].shape == (1, 40)
    assert model.exact_time_projection is not None
    assert torch.count_nonzero(model.exact_time_projection[3].weight) == 0


def test_same_time_cross_view_starts_as_exact_anchor_and_stays_small() -> None:
    torch.manual_seed(20260811)
    anchor = P86MC3VisualStudent(
        frames=4,
        kinetics_pretrained=False,
        temporal_modeling=True,
    ).eval()
    candidate = P86MC3VisualStudent(
        frames=4,
        kinetics_pretrained=False,
        temporal_modeling=True,
        cross_view_time_modeling=True,
    ).eval()
    missing, unexpected = candidate.load_state_dict(anchor.state_dict(), strict=False)
    assert missing
    assert all(
        key.startswith(("same_time_view_encoder.", "same_time_view_projection."))
        for key in missing
    )
    assert not unexpected
    sequence = torch.randn(2, 2, 3, 4, 512)
    valid = torch.ones(2, 2, 4, 3, dtype=torch.bool)
    valid[0, 0, 1, 2] = False
    quality = torch.ones(2, 2, 4, 3)
    with torch.inference_mode():
        anchor_output = anchor.forward_from_backbone_sequence(sequence, valid, quality)
        candidate_output = candidate.forward_from_backbone_sequence(sequence, valid, quality)
    torch.testing.assert_close(candidate_output["logits"], anchor_output["logits"])
    assert candidate.same_time_view_encoder is not None
    assert candidate.same_time_view_projection is not None
    assert torch.count_nonzero(candidate.same_time_view_projection[1].weight) == 0
    assert model_size_mib(candidate, 4) < 100.0


def test_spatial_regions_start_as_global_pool_anchor_and_stay_small() -> None:
    torch.manual_seed(20260811)
    anchor = P86MC3VisualStudent(
        frames=4,
        kinetics_pretrained=False,
        temporal_modeling=True,
    ).eval()
    candidate = P86MC3VisualStudent(
        frames=4,
        kinetics_pretrained=False,
        temporal_modeling=True,
        spatial_region_modeling=True,
    ).eval()
    missing, unexpected = candidate.load_state_dict(anchor.state_dict(), strict=False)
    assert missing
    assert all(key.startswith("spatial_region_") for key in missing)
    assert not unexpected
    regions = torch.randn(2, 2, 3, 4, 4, 512)
    valid = torch.ones(2, 2, 4, 3, dtype=torch.bool)
    quality = torch.ones(2, 2, 4, 3)
    with torch.inference_mode():
        anchor_output = anchor.forward_from_backbone_sequence(
            regions.mean(dim=4), valid, quality
        )
        candidate_output = candidate.forward_from_backbone_region_sequence(
            regions, valid, quality
        )
    torch.testing.assert_close(candidate_output["logits"], anchor_output["logits"])
    assert candidate.spatial_region_encoder is not None
    assert candidate.spatial_region_projection is not None
    assert torch.count_nonzero(candidate.spatial_region_projection[1].weight) == 0
    assert model_size_mib(candidate, 4) < 100.0


def test_region_local_temporal_path_is_zero_residual_and_mask_aware() -> None:
    torch.manual_seed(20260811)
    anchor = P86MC3VisualStudent(
        frames=4, kinetics_pretrained=False, temporal_modeling=True
    ).eval()
    candidate = P86MC3VisualStudent(
        frames=4,
        kinetics_pretrained=False,
        temporal_modeling=True,
        spatial_region_modeling=True,
        region_temporal_modeling=True,
    ).eval()
    missing, unexpected = candidate.load_state_dict(anchor.state_dict(), strict=False)
    assert any(key.startswith("region_temporal_encoder.") for key in missing)
    assert not unexpected
    regions = torch.randn(1, 2, 3, 4, 4, 512)
    valid = torch.ones(1, 2, 4, 3, dtype=torch.bool)
    valid[:, 0, 1, 2] = False
    quality = torch.ones(1, 2, 4, 3)
    with torch.inference_mode():
        anchor_logits = anchor.forward_from_backbone_sequence(
            regions.mean(dim=4), valid, quality
        )["logits"]
        candidate_logits = candidate.forward_from_backbone_region_sequence(
            regions, valid, quality
        )["logits"]
    torch.testing.assert_close(candidate_logits, anchor_logits)
    assert candidate.region_temporal_encoder is not None
    assert model_size_mib(candidate, 4) < 100.0


def test_structured_spatial_region_path_is_zero_residual_and_compact() -> None:
    torch.manual_seed(20260811)
    anchor = P86MC3VisualStudent(
        frames=4, kinetics_pretrained=False, temporal_modeling=True
    ).eval()
    candidate = P86MC3VisualStudent(
        frames=4,
        kinetics_pretrained=False,
        temporal_modeling=True,
        spatial_region_modeling=True,
        structured_region_modeling=True,
    ).eval()
    missing, unexpected = candidate.load_state_dict(anchor.state_dict(), strict=False)
    assert missing
    assert all(key.startswith("structured_region_projection.") for key in missing)
    assert not unexpected
    regions = torch.randn(1, 2, 3, 4, 4, 512)
    valid = torch.ones(1, 2, 4, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 4, 3)
    with torch.inference_mode():
        anchor_logits = anchor.forward_from_backbone_sequence(
            regions.mean(dim=4), valid, quality
        )["logits"]
        candidate_logits = candidate.forward_from_backbone_region_sequence(
            regions, valid, quality
        )["logits"]
    torch.testing.assert_close(candidate_logits, anchor_logits)
    assert candidate.structured_region_projection is not None
    assert candidate.spatial_region_encoder is None
    assert model_size_mib(candidate, 4) < 80.0


def test_videomae_small_is_one_sub_100mib_visual_model() -> None:
    model = P86VideoMAESmallVisualStudent(
        frames=4,
        resolution=64,
        kinetics_pretrained=False,
    ).eval()
    model.freeze_low_level("layer8")
    images = torch.zeros(1, 2, 4, 3, 64, 64, dtype=torch.uint8)
    valid = torch.ones(1, 2, 4, 3, dtype=torch.bool)
    quality = torch.ones(1, 2, 4, 3)
    with torch.inference_mode():
        output = model(images, valid, quality)
    assert output["logits"].shape == (1, 40)
    assert output["clip_embeddings"].shape == (1, 2, 3, 384)
    assert sum(
        isinstance(module, torch.nn.Linear) and module.out_features == 40
        for module in model.modules()
    ) == 1
    assert model_size_mib(model, 4) < 100.0
    assert not any(
        parameter.requires_grad for parameter in model.backbone.encoder.layer[7].parameters()
    )
    assert any(
        parameter.requires_grad for parameter in model.backbone.encoder.layer[8].parameters()
    )
