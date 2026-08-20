from __future__ import annotations

from pathlib import Path

import torch

from src.models.thermal_mobilenet_tsm import BNLinearClassifier, ThermalMobileNetExpert
from src.models.thermal_tsm import MobileNetV3SmallTSM, TemporalShift
from src.train_thermal_native_expert import T1BRecipe, build_optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model() -> ThermalMobileNetExpert:
    return ThermalMobileNetExpert(
        MobileNetV3SmallTSM(
            weights=None,
            num_classes=40,
            num_segments=16,
            fold_div=8,
        )
    )


def test_mobilenet_expert_matches_trial_contract() -> None:
    model = _model().eval()
    clips = torch.zeros(2, 16, 3, 224, 224)
    quality = torch.zeros(2, 6)
    quality_mask = torch.ones(2, 6, dtype=torch.bool)
    availability = torch.tensor([True, False])

    with torch.inference_mode():
        output = model(clips, quality, quality_mask, availability)

    assert output.main_logits.shape == (2, 40)
    assert output.embedding.shape == (2, 576)
    assert torch.isfinite(output.main_logits).all()
    assert torch.equal(output.availability, availability)


def test_mobilenet_optimizer_uses_same_disjoint_recipe_groups() -> None:
    model = _model()
    optimizer = build_optimizer(model, T1BRecipe())

    assert [group["name"] for group in optimizer.param_groups] == ["backbone", "head"]
    assert optimizer.param_groups[0]["lr"] == 3e-5
    assert optimizer.param_groups[1]["lr"] == 3e-4
    backbone_ids = {id(parameter) for parameter in model.backbone_parameters()}
    head_ids = {id(parameter) for parameter in model.head_parameters()}
    assert backbone_ids
    assert head_ids
    assert backbone_ids.isdisjoint(head_ids)
    assert isinstance(model.spatial.classifier, BNLinearClassifier)
    assert sum(parameter.numel() for parameter in model.head_parameters()) == 24_232


def test_mobilenet_tsm_is_parameter_free() -> None:
    model = _model()

    assert isinstance(model.spatial.temporal_shift, TemporalShift)
    assert sum(parameter.numel() for parameter in model.spatial.temporal_shift.parameters()) == 0
    assert model.spatial.num_segments == 16


def test_mobilenet_matched_config_freezes_full_recipe_and_route() -> None:
    text = (
        PROJECT_ROOT
        / "configs/experiments/thermal_mobilenetv3_tsm_train12_val2.yaml"
    ).read_text(encoding="utf-8")

    assert "status: preregistered_authorized_not_started" in text
    assert "route: full_frame" in text
    assert "epochs: 30" in text
    assert "hard_stop_epoch: 30" in text
    assert "automatic_extension: false" in text
    assert "batch_size: 4" in text
    assert "gradient_accumulation: 4" in text
    assert "backbone_lr: 0.00003" in text
    assert "head_lr: 0.0003" in text
    assert "seed: 20260715" in text
    assert "yolo_crop: false" in text
    assert "classifier_structure_matches_iformer_t: true" in text
    assert "promotion_authorized: false" in text
