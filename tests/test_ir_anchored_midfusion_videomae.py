from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.run_ir_depth_videomaev2_aggressive_dev import (
    ClassUserBalancedSampler,
    load_aggressive_config,
)
from src.models.ir_anchored_midfusion_videomae import IRAnchoredMidFusionVideoMAE


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/ir_depth_videomaev2_aggressive_user6_user7.yaml"


class TinyPatchEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(3, dim, kernel_size=1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.proj(clips).flatten(2).transpose(1, 2)


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.projection(inputs)


class TinyBackbone(nn.Module):
    def __init__(self, dim: int = 12, blocks: int = 6, classes: int = 8) -> None:
        super().__init__()
        self.embed_dim = dim
        self.patch_embed = TinyPatchEmbed(dim)
        self.pos_embed = torch.zeros(1, 8, dim)
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList(TinyBlock(dim) for _ in range(blocks))
        self.fc_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)


def test_aggressive_config_keeps_validation_users_out_of_training() -> None:
    config = load_aggressive_config(CONFIG)

    assert set(config["split"]["train_user_ids"]).isdisjoint({"user6", "user7"})
    assert config["split"]["validation_user_ids"] == ["user6", "user7"]
    assert config["split"]["train_class_count"] == 40
    assert config["split"]["validation_class_count"] == 40
    assert config["model"]["frozen_prefix_blocks"] == 8
    assert config["policy"]["load_previous_teacher_checkpoint"] is False


def test_zero_depth_adapter_makes_logits_independent_of_depth_input() -> None:
    torch.manual_seed(7)
    model = IRAnchoredMidFusionVideoMAE(
        backbone=TinyBackbone(), frozen_prefix_blocks=2, view_top_k=2, hand_prior_bias=0.5
    )
    ir = torch.randn(2, 4, 3, 2, 2, 2)
    first_depth = torch.randn_like(ir)
    second_depth = torch.randn_like(ir) * 10.0
    availability = torch.ones(2, 2, 4, dtype=torch.bool)

    first = model(ir=ir, depth=first_depth, availability=availability)
    second = model(ir=ir, depth=second_depth, availability=availability)

    assert torch.equal(first["logits"], second["logits"])
    assert torch.count_nonzero(first["depth_delta_norm"]) == 0


def test_midfusion_freezes_prefix_and_trains_tail() -> None:
    backbone = TinyBackbone(blocks=6)
    IRAnchoredMidFusionVideoMAE(
        backbone=backbone, frozen_prefix_blocks=2, view_top_k=2, hand_prior_bias=0.5
    )

    assert all(not parameter.requires_grad for block in backbone.blocks[:2] for parameter in block.parameters())
    assert all(parameter.requires_grad for block in backbone.blocks[2:] for parameter in block.parameters())
    assert all(not parameter.requires_grad for parameter in backbone.patch_embed.parameters())


def test_each_class_uses_at_most_two_views_and_masks_unavailable_views() -> None:
    model = IRAnchoredMidFusionVideoMAE(
        backbone=TinyBackbone(), frozen_prefix_blocks=2, view_top_k=2, hand_prior_bias=0.5
    )
    availability = torch.ones(1, 2, 4, dtype=torch.bool)
    availability[:, :, 3] = False
    output = model(
        ir=torch.randn(1, 4, 3, 2, 2, 2),
        depth=torch.randn(1, 4, 3, 2, 2, 2),
        availability=availability,
    )

    assert int((output["view_weights"] > 0).sum(dim=2).max()) <= 2
    assert torch.count_nonzero(output["view_weights"][:, :, 3]) == 0
    assert torch.allclose(output["view_weights"].sum(dim=2), torch.ones(1, 8))


def test_class_user_sampler_balances_observed_class_user_groups() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    users = np.asarray(["a", "a", "b", "b", "a", "a", "c", "c"])
    sampler = ClassUserBalancedSampler(labels=labels, users=users, samples=8000, seed=11)
    counts: dict[tuple[int, str], int] = {}
    for index in sampler:
        key = (int(labels[index]), str(users[index]))
        counts[key] = counts.get(key, 0) + 1

    class_counts = {label: sum(value for (candidate, _), value in counts.items() if candidate == label) for label in (0, 1)}
    assert abs(class_counts[0] - class_counts[1]) < 300
    assert set(counts) == {(0, "a"), (0, "b"), (1, "a"), (1, "c")}
