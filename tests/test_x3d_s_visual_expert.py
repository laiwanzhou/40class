from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.expert_contract import ExpertOutput
from src.models.x3d_s_visual_expert import X3DSVisualExpert, build_x3d_s_feature_backbone


class TinyBackbone(nn.Module):
    output_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, self.output_dim, kernel_size=1, stride=(1, 8, 8), bias=False)
        self.bn = nn.BatchNorm3d(self.output_dim)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.bn(self.conv(clips)))).flatten(1)


class BlockBackbone(nn.Module):
    output_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(3, 3, kernel_size=1, bias=False),
                    nn.BatchNorm3d(3),
                    nn.ReLU(),
                )
                for _ in range(4)
            ]
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            clips = block(clips)
        return self.pool(clips).flatten(1)


def build_model() -> X3DSVisualExpert:
    torch.manual_seed(7)
    return X3DSVisualExpert(
        backbone=TinyBackbone(),
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
        update_backbone_bn_running_stats=False,
    )


def fixture_inputs() -> dict[str, torch.Tensor]:
    return {
        "clips": torch.randn(2, 3, 3, 13, 32, 32),
        "clip_mask": torch.tensor([[True, True, True], [True, False, False]]),
        "quality": torch.ones(2, 6),
        "quality_mask": torch.ones(2, 6, dtype=torch.bool),
        "availability": torch.ones(2, 1, dtype=torch.bool),
    }


def test_x3d_visual_expert_emits_standard_expert_output() -> None:
    model = build_model().eval()
    inputs = fixture_inputs()

    output = model(**inputs)

    assert isinstance(output, ExpertOutput)
    assert output.main_logits.shape == (2, 40)
    assert output.embedding.shape == (2, 16)
    assert torch.isfinite(output.main_logits).all()
    assert output.quality is inputs["quality"]
    assert output.quality_mask is inputs["quality_mask"]
    assert output.availability is inputs["availability"]


def test_padded_clip_values_cannot_change_trial_output() -> None:
    model = build_model().eval()
    inputs = fixture_inputs()
    first = model(**inputs)
    changed = inputs["clips"].clone()
    changed[1, 1:] = 10_000.0

    second = model(**{**inputs, "clips": changed})

    torch.testing.assert_close(first.main_logits, second.main_logits)
    torch.testing.assert_close(first.embedding, second.embedding)


def test_trial_with_zero_valid_clips_is_rejected() -> None:
    model = build_model()
    inputs = fixture_inputs()
    inputs["clip_mask"][1] = False

    with pytest.raises(ValueError, match="zero valid clips"):
        model(**inputs)


def test_parameter_groups_split_backbone_head_and_zero_decay_parameters() -> None:
    model = build_model()

    groups = model.parameter_groups(backbone_lr=3e-5, head_lr=3e-4, weight_decay=0.05)

    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}
    assert {(group["lr"], group["weight_decay"]) for group in groups} == {
        (3e-5, 0.0),
        (3e-5, 0.05),
        (3e-4, 0.0),
        (3e-4, 0.05),
    }


def test_backbone_warmup_freezes_weights_and_bn_running_statistics() -> None:
    model = build_model()
    backbone = model.backbone
    assert isinstance(backbone, TinyBackbone)

    model.set_backbone_trainable(False)
    model.train()
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    assert not backbone.training
    assert not backbone.bn.training

    model.set_backbone_trainable(True)
    model.train()
    assert backbone.conv.weight.requires_grad
    assert backbone.bn.weight.requires_grad
    assert backbone.bn.bias.requires_grad
    assert backbone.training
    assert not backbone.bn.training

    running_mean = backbone.bn.running_mean.clone()
    model(**fixture_inputs())
    torch.testing.assert_close(backbone.bn.running_mean, running_mean, atol=0.0, rtol=0.0)


def test_selective_unfreeze_enables_only_last_backbone_blocks() -> None:
    backbone = BlockBackbone()
    model = X3DSVisualExpert(
        backbone=backbone,
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
        update_backbone_bn_running_stats=False,
    )

    model.set_backbone_trainable(False)
    assert not any(parameter.requires_grad for parameter in backbone.parameters())

    model.set_backbone_trainable(True, last_blocks=2)
    assert not any(parameter.requires_grad for parameter in backbone.blocks[0].parameters())
    assert not any(parameter.requires_grad for parameter in backbone.blocks[1].parameters())
    assert all(parameter.requires_grad for parameter in backbone.blocks[2].parameters())
    assert all(parameter.requires_grad for parameter in backbone.blocks[3].parameters())
    model.train()
    assert all(not block[1].training for block in backbone.blocks)


@pytest.mark.parametrize("last_blocks", [0, 5])
def test_selective_unfreeze_rejects_invalid_block_count(last_blocks: int) -> None:
    model = X3DSVisualExpert(
        backbone=BlockBackbone(),
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="last_blocks"):
        model.set_backbone_trainable(True, last_blocks=last_blocks)


def test_official_x3d_builder_removes_only_kinetics_projection() -> None:
    backbone = build_x3d_s_feature_backbone(pretrained=False)

    assert backbone.output_dim == 2048
    assert backbone.blocks[-1].proj is None
    assert backbone.blocks[-1].activation is None
