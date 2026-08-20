from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import torch

from src.train_thermal_generation2 import (
    TrainingAuthorizationError,
    build_generation2_student,
    build_optimizer_and_scheduler,
    checkpoint_rank,
    collect_generation2_predictions,
    compute_generation2_loss,
    fixed_label_metrics,
    load_generation2_config,
    require_training_authorization,
    run_generation2_epoch,
    set_deterministic_seed,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "configs/experiments/thermal_b_x3d_xs_train12_val2.yaml"
DIRECT_CONFIG = ROOT / "configs/experiments/thermal_a_multistream_direct_train12_val2.yaml"


@pytest.mark.parametrize("path", (BASELINE_CONFIG, DIRECT_CONFIG))
def test_frozen_config_is_random_fixed_and_unauthorized(path: Path) -> None:
    config = load_generation2_config(path)

    assert config["training_authorized"] is False
    assert config["student"]["initialization"] == "random"
    assert config["student"]["pretrained"] is False
    assert config["data"]["development_split"] == (
        "metadata/splits/train12_val2_user6_user7_development.json"
    )
    assert config["data"]["class_ids"] == list(range(40))
    assert config["data"]["tensor_shapes"] == {
        "full_rgb": [3, 16, 3, 160, 160],
        "crop_rgb": [3, 16, 3, 160, 160],
        "motion": [3, 16, 1, 160, 160],
        "pose": [3, 16, 56],
        "availability": [4],
        "quality": [8],
    }
    assert config["optimization"]["maximum_epochs"] == 50
    assert config["optimization"]["automatic_resume"] is False
    assert config["optimization"]["automatic_extension"] is False
    assert config["optimization"]["checkpoint_rank"] == [
        "macro_f1_fixed_0_39",
        "accuracy",
        "worst_user_accuracy",
        "lower_epoch",
    ]


def test_authorization_requires_config_flag_and_exact_token() -> None:
    config = load_generation2_config(DIRECT_CONFIG)

    with pytest.raises(TrainingAuthorizationError):
        require_training_authorization(config, token="thermal-a-direct")
    approved = dict(config, training_authorized=True)
    with pytest.raises(TrainingAuthorizationError):
        require_training_authorization(approved, token="wrong-token")
    require_training_authorization(approved, token="thermal-a-direct")


def test_masked_direct_and_kd_losses_share_hard_label_term() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 5.0]], requires_grad=True)
    labels = torch.tensor([0, 0])
    eligible = torch.tensor([True, False])
    direct = compute_generation2_loss(
        logits=logits,
        labels=labels,
        eligible=eligible,
        objective="direct",
        label_smoothing=0.0,
    )
    changed = compute_generation2_loss(
        logits=torch.tensor([[2.0, 0.0], [100.0, -100.0]]),
        labels=labels,
        eligible=eligible,
        objective="direct",
        label_smoothing=0.0,
    )
    kd = compute_generation2_loss(
        logits=logits,
        labels=labels,
        eligible=eligible,
        objective="kd",
        label_smoothing=0.0,
        teacher_logits=logits.detach(),
        temperature=4.0,
        hard_label_weight=0.5,
        teacher_kl_weight=0.5,
    )

    assert direct.item() == pytest.approx(changed.item())
    assert kd.item() == pytest.approx(0.5 * direct.item(), abs=1e-6)
    direct.backward()
    assert torch.isfinite(logits.grad).all()


def test_fixed_metrics_include_all_classes_and_worst_user() -> None:
    labels = np.asarray([0, 0, 1, 1])
    logits = np.zeros((4, 40), dtype=np.float32)
    logits[0, 0] = logits[1, 0] = logits[2, 1] = logits[3, 0] = 4.0
    users = np.asarray(["user6", "user6", "user7", "user7"])

    metrics = fixed_label_metrics(labels=labels, logits=logits, users=users)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["worst_user_accuracy"] == pytest.approx(0.5)
    assert metrics["zero_recall_classes"] == 38
    assert len(metrics["per_class_recall"]) == 40
    assert checkpoint_rank(metrics, epoch=7) == (
        metrics["macro_f1"],
        0.75,
        0.5,
        -7,
    )


@pytest.mark.parametrize("path", (BASELINE_CONFIG, DIRECT_CONFIG))
def test_model_builder_preserves_random_initialization(path: Path) -> None:
    config = load_generation2_config(path)
    model = build_generation2_student(config)

    assert model.initialization_provenance["pretrained_student_weights"] is False


class TinyGeneration2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 40))

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        size = batch["full_rgb"].shape[0]
        logits = self.logits.expand(size, -1)
        return {
            "logits": logits,
            "embedding": logits,
            "availability": batch["availability"],
            "quality": batch["quality"],
        }


def tiny_batch() -> dict[str, object]:
    return {
        "full_rgb": torch.zeros(2, 1),
        "crop_rgb": torch.zeros(2, 1),
        "motion": torch.zeros(2, 1),
        "pose": torch.zeros(2, 1),
        "window_mask": torch.ones(2, 1, dtype=torch.bool),
        "pose_mask": torch.ones(2, 1, dtype=torch.bool),
        "availability": torch.ones(2, 4, dtype=torch.bool),
        "quality": torch.ones(2, 8),
        "label": torch.tensor([0, 1]),
        "loss_eligible": torch.tensor([True, False]),
        "sample_id": ["eligible", "retained-unavailable"],
        "user_id": ["user6", "user7"],
    }


def test_epoch_core_is_deterministic_masked_and_archive_ready() -> None:
    config = load_generation2_config(DIRECT_CONFIG)
    config["optimization"]["physical_batch_trials"] = 1
    config["optimization"]["effective_batch_trials"] = 1
    set_deterministic_seed(config["optimization"]["seed"])
    first = (random.random(), np.random.rand(), torch.rand(()).item())
    set_deterministic_seed(config["optimization"]["seed"])
    second = (random.random(), np.random.rand(), torch.rand(()).item())
    assert first == second

    model = TinyGeneration2()
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config=config, steps_per_epoch=1
    )
    train_metrics = run_generation2_epoch(
        model=model,
        loader=[tiny_batch()],
        config=config,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
    )
    archive = collect_generation2_predictions(
        model=model,
        loader=[tiny_batch()],
        config=config,
        device=torch.device("cpu"),
    )

    assert train_metrics["eligible_samples"] == 1
    assert train_metrics["optimizer_steps"] == 1
    assert archive["sample_ids"].tolist() == ["eligible"]
    assert archive["logits"].shape == (1, 40)
    assert archive["availability"].shape == (1, 4)
    assert archive["quality"].shape == (1, 8)
