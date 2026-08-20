from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.expert_contract import ExpertOutput
from src.models.x3d_s_visual_expert import (
    IRAnchoredDepthAdapter,
    X3DSVisualExpert,
    build_x3d_s_feature_backbone,
    expand_first_conv3d_input_channels,
)


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


def build_model(*, head_type: str = "projected") -> X3DSVisualExpert:
    torch.manual_seed(7)
    embedding_dim = 8 if head_type == "direct" else 16
    return X3DSVisualExpert(
        backbone=TinyBackbone(),
        num_classes=40,
        embedding_dim=embedding_dim,
        dropout=0.0,
        head_type=head_type,
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


def test_ir_anchored_depth_adapter_is_exact_ir_baseline_at_initialization() -> None:
    adapter = IRAnchoredDepthAdapter()
    inputs = torch.randn(2, 4, 13, 8, 8)
    expected = inputs[:, 3:4].expand(-1, 3, -1, -1, -1)

    output = adapter(inputs)

    torch.testing.assert_close(output, expected, atol=0.0, rtol=0.0)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 9
    assert torch.count_nonzero(adapter.depth_projection.weight).item() == 0


def test_ir_anchored_depth_adapter_has_zero_initial_depth_sensitivity_but_gradient() -> None:
    adapter = IRAnchoredDepthAdapter()
    inputs = torch.randn(2, 4, 13, 8, 8)
    changed_depth = inputs.clone()
    changed_depth[:, :3].mul_(7.0).add_(2.0)

    torch.testing.assert_close(
        adapter(inputs), adapter(changed_depth), atol=0.0, rtol=0.0
    )
    adapter(inputs).square().mean().backward()
    gradient = adapter.depth_projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def test_ir_anchored_four_channel_expert_feeds_three_channels_to_backbone() -> None:
    model = X3DSVisualExpert(
        backbone=TinyBackbone(),
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
        input_channels=4,
        input_adapter_mode="ir_anchored_depth_residual",
    ).eval()
    inputs = fixture_inputs()
    inputs["clips"] = torch.randn(2, 3, 4, 13, 32, 32)

    output = model(**inputs)

    assert output.main_logits.shape == (2, 40)
    assert isinstance(model.input_adapter, IRAnchoredDepthAdapter)


def test_expand_first_conv_to_four_channels_preserves_rgb_and_mean_initializes_ir() -> None:
    backbone = nn.Sequential(nn.Conv3d(3, 2, kernel_size=(1, 2, 2), bias=True))
    original = backbone[0]
    with torch.no_grad():
        original.weight.copy_(torch.arange(original.weight.numel()).reshape_as(original.weight))
        original.bias.copy_(torch.tensor([1.0, 2.0]))
    expected_rgb = original.weight.detach().clone()
    expected_bias = original.bias.detach().clone()

    expanded = expand_first_conv3d_input_channels(backbone, input_channels=4)

    replacement = expanded[0]
    assert replacement.in_channels == 4
    torch.testing.assert_close(replacement.weight[:, :3], expected_rgb)
    torch.testing.assert_close(replacement.weight[:, 3], expected_rgb.mean(dim=1))
    torch.testing.assert_close(replacement.bias, expected_bias)


def test_four_channel_expert_accepts_depth_rgb_plus_ir_clip() -> None:
    backbone = TinyBackbone()
    backbone.conv = nn.Conv3d(4, backbone.output_dim, kernel_size=1, stride=(1, 8, 8))
    model = X3DSVisualExpert(
        backbone=backbone,
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
        input_channels=4,
    ).eval()
    inputs = fixture_inputs()
    inputs["clips"] = torch.randn(2, 3, 4, 13, 32, 32)

    output = model(**inputs)

    assert output.main_logits.shape == (2, 40)


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


@pytest.mark.parametrize("head_type", ["projected", "direct"])
def test_padded_clip_values_cannot_change_trial_output(head_type: str) -> None:
    model = build_model(head_type=head_type).eval()
    inputs = fixture_inputs()
    first = model(**inputs)
    changed = inputs["clips"].clone()
    changed[1, 1:] = 10_000.0

    second = model(**{**inputs, "clips": changed})

    torch.testing.assert_close(first.main_logits, second.main_logits)
    torch.testing.assert_close(first.embedding, second.embedding)


def test_direct_head_emits_backbone_dimensional_embedding_and_logits() -> None:
    model = X3DSVisualExpert(
        backbone=TinyBackbone(),
        num_classes=40,
        embedding_dim=8,
        dropout=0.25,
        head_type="direct",
    ).eval()

    output = model(**fixture_inputs())

    assert model.head_type == "direct"
    assert model.output_embedding_dim == 8
    assert isinstance(model.embedding_head, nn.Identity)
    assert isinstance(model.direct_classifier_dropout, nn.Dropout)
    assert model.direct_classifier_dropout.p == pytest.approx(0.25)
    assert model.classifier.in_features == 8
    assert output.main_logits.shape == (2, 40)
    assert output.embedding.shape == (2, 8)


def test_direct_head_dropout_changes_logits_but_not_runtime_embedding() -> None:
    class AddOne(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + 1.0

    model = build_model(head_type="direct").eval()
    inputs = fixture_inputs()
    baseline = model(**inputs)

    model.direct_classifier_dropout = AddOne()
    changed = model(**inputs)

    torch.testing.assert_close(changed.embedding, baseline.embedding, atol=0.0, rtol=0.0)
    assert not torch.equal(changed.main_logits, baseline.main_logits)


@pytest.mark.parametrize(
    ("head_type", "embedding_dim", "message"),
    [
        ("direct", 16, "backbone output_dim"),
        ("unknown", 8, "head_type"),
    ],
)
def test_invalid_head_contract_is_rejected(
    head_type: str, embedding_dim: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        X3DSVisualExpert(
            backbone=TinyBackbone(),
            num_classes=40,
            embedding_dim=embedding_dim,
            dropout=0.25,
            head_type=head_type,
        )


def test_explicit_projected_head_strict_loads_legacy_state_and_matches_exactly() -> None:
    torch.manual_seed(91)
    legacy = X3DSVisualExpert(
        backbone=TinyBackbone(), num_classes=40, embedding_dim=16, dropout=0.25
    )
    torch.manual_seed(91)
    explicit = X3DSVisualExpert(
        backbone=TinyBackbone(),
        num_classes=40,
        embedding_dim=16,
        dropout=0.25,
        head_type="projected",
    )

    load_result = explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert tuple(explicit.state_dict()) == tuple(legacy.state_dict())
    assert explicit.head_type == "projected"
    assert explicit.output_embedding_dim == 16

    torch.manual_seed(13)
    inputs = fixture_inputs()
    legacy.eval()
    explicit.eval()
    legacy_output = legacy(**inputs)
    explicit_output = explicit(**inputs)
    torch.testing.assert_close(
        legacy_output.main_logits, explicit_output.main_logits, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        legacy_output.embedding, explicit_output.embedding, atol=0.0, rtol=0.0
    )


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


def test_ir_anchored_adapter_has_separate_optimizer_scope() -> None:
    model = X3DSVisualExpert(
        backbone=TinyBackbone(),
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
        input_channels=4,
        input_adapter_mode="ir_anchored_depth_residual",
    )

    groups = model.parameter_groups(
        backbone_lr=3e-5,
        head_lr=3e-4,
        input_adapter_lr=3e-4,
        weight_decay=0.05,
    )

    adapter_parameter_ids = {
        id(parameter) for parameter in model.input_adapter.parameters()
    }
    adapter_groups = [group for group in groups if group["group_name"] == "input_adapter"]
    assert len(adapter_groups) == 1
    assert float(adapter_groups[0]["lr"]) == pytest.approx(3e-4)
    assert float(adapter_groups[0]["weight_decay"]) == pytest.approx(0.05)
    assert {id(parameter) for parameter in adapter_groups[0]["params"]} == (
        adapter_parameter_ids
    )


def test_parameter_groups_apply_block_specific_learning_rates() -> None:
    backbone = BlockBackbone()
    model = X3DSVisualExpert(
        backbone=backbone,
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
    )

    groups = model.parameter_groups(
        backbone_lr=3e-5,
        head_lr=3e-4,
        weight_decay=0.05,
        backbone_block_lrs={2: 3e-6, 3: 1e-5},
    )
    learning_rate_by_parameter = {
        id(parameter): float(group["lr"])
        for group in groups
        for parameter in group["params"]
    }

    assert all(
        learning_rate_by_parameter[id(parameter)] == pytest.approx(3e-6)
        for parameter in backbone.blocks[2].parameters()
    )
    assert all(
        learning_rate_by_parameter[id(parameter)] == pytest.approx(1e-5)
        for parameter in backbone.blocks[3].parameters()
    )
    assert {group["group_name"] for group in groups} >= {
        "backbone_default",
        "backbone_block_2",
        "backbone_block_3",
        "custom_head",
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
