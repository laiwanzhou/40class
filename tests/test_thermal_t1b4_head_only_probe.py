from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.train_thermal_head_only_probe import (
    T1B4HeadOnlyRecipe,
    build_head_only_optimizer,
    configure_head_only_mode,
    freeze_backbone_for_head_only,
)
from scripts.run_thermal_t1b4_head_only_probe import evaluate_tail_gate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _ProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone_bn = nn.BatchNorm1d(4)
        self.classifier = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 40))

    def head_parameters(self):
        yield from self.classifier.parameters()

    def backbone_parameters(self):
        head_ids = {id(parameter) for parameter in self.head_parameters()}
        yield from (parameter for parameter in self.parameters() if id(parameter) not in head_ids)


def test_t1b4_recipe_is_a_hard_eight_epoch_stop() -> None:
    recipe = T1B4HeadOnlyRecipe()

    assert recipe.epochs == 8
    assert recipe.hard_stop_epoch == 8
    assert recipe.automatic_extension is False
    assert recipe.resume is False
    assert recipe.batch_size == 4
    assert recipe.gradient_accumulation == 4
    assert recipe.head_lr == 3e-4
    assert recipe.seed == 20260715


def test_head_only_freeze_and_mode_leave_backbone_bn_eval() -> None:
    model = _ProbeModel()

    trainable_names = freeze_backbone_for_head_only(model)
    configure_head_only_mode(model, training=True)

    assert trainable_names == (
        "classifier.0.bias",
        "classifier.0.weight",
        "classifier.1.bias",
        "classifier.1.weight",
    )
    assert model.training is False
    assert model.backbone_bn.training is False
    assert model.classifier.training is True
    assert model.classifier[0].training is True
    assert all(not parameter.requires_grad for parameter in model.backbone_bn.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_head_only_step_updates_head_but_not_backbone_bn_or_parameters() -> None:
    torch.manual_seed(7)
    model = _ProbeModel()
    recipe = T1B4HeadOnlyRecipe()
    freeze_backbone_for_head_only(model)
    optimizer = build_head_only_optimizer(model, recipe)
    configure_head_only_mode(model, training=True)
    backbone_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("classifier.")
    }
    head_weight_before = model.classifier[1].weight.detach().clone()
    head_running_mean_before = model.classifier[0].running_mean.detach().clone()

    inputs = torch.randn(4, 4)
    logits = model.classifier(model.backbone_bn(inputs))
    nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    optimizer.step()

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "head"
    assert all(
        torch.equal(model.state_dict()[name], value) for name, value in backbone_state.items()
    )
    assert not torch.equal(model.classifier[1].weight, head_weight_before)
    assert not torch.equal(model.classifier[0].running_mean, head_running_mean_before)


def test_preregistered_config_prohibits_resume_and_extension() -> None:
    text = (
        PROJECT_ROOT / "configs/experiments/thermal_iformer_t_tsm_head_only_probe.yaml"
    ).read_text(encoding="utf-8")

    assert "hard_stop_epoch: 8" in text
    assert "automatic_extension: false" in text
    assert "resume: false" in text
    assert "epoch16_backbone_loaded: false" in text
    assert "promotion_authorized: false" in text


def test_tail_gate_requires_all_pairs_and_both_block_and_embedding_limits() -> None:
    passing = [
        {"block5_output_rms_ratio": 1.1, "embedding_rms_ratio": 1.2}
        for _ in range(23)
    ]
    assert evaluate_tail_gate(passing)["passed"] is True

    one_extreme = list(passing)
    one_extreme[0] = {"block5_output_rms_ratio": 2.01, "embedding_rms_ratio": 1.0}
    assert evaluate_tail_gate(one_extreme)["passed"] is False
