from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.thermal_teachers import (
    OFFICIAL_R2PLUS1D18,
    ThermalR2Plus1D18Teacher,
)
from src.train_thermal_teacher import (
    TeacherTrainingAuthorizationError,
    build_teacher_optimizer_and_scheduler,
    load_teacher_config,
    require_teacher_training_authorization,
    sequential_trial_backward,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/thermal_c1_r2plus1d18_train12_val2.yaml"


class TinyVideoBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        value = clip.mean(dim=(1, 2, 3, 4)) * self.scale
        return torch.stack((value, -value), dim=1)


def teacher_batch() -> dict[str, torch.Tensor]:
    full = torch.arange(1, 7, dtype=torch.float32).reshape(1, 3, 1, 1, 1, 2)
    crop = full + 1.0
    return {
        "full_rgb": full,
        "crop_rgb": crop,
        "window_mask": torch.ones(1, 3, dtype=torch.bool),
        "availability": torch.tensor([[True, True]]),
        "label": torch.tensor([0]),
        "loss_eligible": torch.tensor([True]),
    }


def test_config_freezes_c1_authorization_data_and_optimization() -> None:
    config = load_teacher_config(CONFIG)

    assert config["route"] == "c1_r2plus1d18"
    assert config["training_authorized"] is True
    assert config["authorization"]["approval_text"] == "请在独立worktree中开启对教师的训练"
    assert config["data"]["modality"] == "thermal_only"
    assert config["data"]["development_split"].endswith(
        "train12_val2_user6_user7_development.json"
    )
    assert config["optimization"]["maximum_epochs"] == 30
    assert config["optimization"]["physical_batch_trials"] == 1
    assert config["optimization"]["effective_batch_trials"] == 8
    assert config["runtime"]["sequential_clip_execution"] is True
    assert config["runtime"]["maximum_peak_allocated_mib_exclusive"] == 7300
    assert all(config["data"][field] is False for field in (
        "read_heldout4_labels",
        "read_competition_test",
        "read_quarantined_evidence",
        "import_ir_depth_inputs",
    ))


def test_authorization_requires_exact_c1_token() -> None:
    config = load_teacher_config(CONFIG)

    with pytest.raises(TeacherTrainingAuthorizationError):
        require_teacher_training_authorization(config, token="wrong")
    require_teacher_training_authorization(config, token="thermal-c1-r2plus1d18")


def test_official_provenance_is_pinned() -> None:
    assert OFFICIAL_R2PLUS1D18["repository_revision"] == (
        "9eb57cd5c96be7fe31923eb65399c3819d064587"
    )
    assert OFFICIAL_R2PLUS1D18["license"] == "BSD-3-Clause"
    assert OFFICIAL_R2PLUS1D18["license_scope"].startswith("torchvision source code")
    assert OFFICIAL_R2PLUS1D18["checkpoint_sha256"] == (
        "91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8"
    )


def test_teacher_averages_only_available_full_and_crop_clips_sequentially() -> None:
    backbone = TinyVideoBackbone()
    model = ThermalR2Plus1D18Teacher(backbone=backbone, num_classes=2)
    batch = teacher_batch()

    output = model(
        batch["full_rgb"],
        batch["crop_rgb"],
        window_mask=batch["window_mask"],
        availability=batch["availability"],
    )

    expected = torch.stack(
        [backbone(batch[view][:, window]) for window in range(3) for view in ("full_rgb", "crop_rgb")]
    ).mean(dim=0)
    assert torch.equal(output["logits"], expected)
    assert output["clip_count"].tolist() == [6]
    assert model.last_execution_trace == [
        "full:0", "crop:0", "full:1", "crop:1", "full:2", "crop:2"
    ]


def test_sequential_backward_matches_joint_mean_logit_gradient() -> None:
    batch = teacher_batch()
    joint_backbone = TinyVideoBackbone()
    sequential_backbone = TinyVideoBackbone()
    sequential_backbone.load_state_dict(joint_backbone.state_dict())
    joint = ThermalR2Plus1D18Teacher(backbone=joint_backbone, num_classes=2)
    sequential = ThermalR2Plus1D18Teacher(backbone=sequential_backbone, num_classes=2)

    logits = joint(
        batch["full_rgb"], batch["crop_rgb"],
        window_mask=batch["window_mask"], availability=batch["availability"],
    )["logits"]
    loss = torch.nn.functional.cross_entropy(logits, batch["label"], label_smoothing=0.1)
    loss.backward()

    result = sequential_trial_backward(
        model=sequential,
        batch=batch,
        device=torch.device("cpu"),
        label_smoothing=0.1,
        loss_scale=1.0,
        amp_enabled=False,
    )

    assert result["clip_count"] == 6
    assert result["loss"] == pytest.approx(loss.item())
    assert sequential_backbone.scale.grad.item() == pytest.approx(
        joint_backbone.scale.grad.item(), abs=1e-6
    )


def test_optimizer_keeps_backbone_and_classifier_learning_rates_separate() -> None:
    config = load_teacher_config(CONFIG)
    model = ThermalR2Plus1D18Teacher(backbone=TinyVideoBackbone(), num_classes=2)

    optimizer, scheduler = build_teacher_optimizer_and_scheduler(
        model, config=config, steps_per_epoch=2
    )

    assert scheduler.base_lrs == pytest.approx([1e-5, 1e-4])
    assert optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"] == pytest.approx(10.0)


def test_deployment_model_namespace_does_not_import_teacher() -> None:
    source = (ROOT / "src/models/__init__.py").read_text(encoding="utf-8")
    assert "thermal_teachers" not in source
    assert "ThermalR2Plus1D18Teacher" not in source
