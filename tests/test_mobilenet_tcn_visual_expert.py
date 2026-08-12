import torch
from torch import nn

from src.models.mobilenet_tcn_visual_expert import MobileNetTCNVisualExpert
from src.train_x3d_s_visual_expert import _build_model, validate_config


class TinyFrameBackbone(nn.Module):
    output_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.GELU())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images).mean(dim=(2, 3))


def test_mobilenet_tcn_matches_trial_expert_contract_and_ignores_padding() -> None:
    torch.manual_seed(7)
    model = MobileNetTCNVisualExpert(
        backbone=TinyFrameBackbone(), tcn_channels=16, embedding_dim=12, dilations=(1, 2)
    ).eval()
    clips = torch.randn(2, 3, 3, 13, 16, 16)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    kwargs = {
        "clip_mask": mask,
        "quality": torch.ones(2, 2),
        "quality_mask": torch.ones(2, 2, dtype=torch.bool),
        "availability": torch.ones(2, 1, dtype=torch.bool),
    }
    first = model(clips, **kwargs)
    clips[~mask] = 999.0
    second = model(clips, **kwargs)
    assert first.main_logits.shape == (2, 40)
    assert first.embedding.shape == (2, 12)
    assert torch.allclose(first.main_logits, second.main_logits)
    assert torch.allclose(first.embedding, second.embedding)


def test_mobilenet_tcn_bn_running_stats_remain_frozen() -> None:
    model = MobileNetTCNVisualExpert(
        backbone=TinyFrameBackbone(), tcn_channels=16, embedding_dim=12, dilations=(1,)
    )
    model.train()
    assert not model.backbone.layers[1].training
    model.set_backbone_trainable(False)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_trainer_builds_matched_family_without_pretraining() -> None:
    config = {
        "model_family": "mobilenet_v3_small_tcn",
        "input_view": "ir_context_path",
        "num_classes": 40,
        "embedding_dim": 16,
        "dropout": 0.2,
        "pretrained": False,
        "matched_baseline": {"tcn_channels": 16},
        "loader": {"max_trials_per_batch": 2, "max_valid_clips_per_batch": 8, "num_workers": 0},
        "temporal": {
            "local_frames": 13,
            "target_window_frames": 32,
            "max_clips": 8,
            "train_views_per_window": 1,
            "val_views_per_window": 1,
            "aggregation": "mean_probability",
        },
        "backbone_bn": {
            "update_running_stats": False,
            "train_affine_after_unfreeze": True,
        },
        "optimizer": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "weight_decay": 0.05,
            "gradient_accumulation": 4,
            "gradient_clip": 1.0,
        },
        "training": {
            "epochs": 30,
            "scheduler_horizon_epochs": 30,
            "warmup_epochs": 2,
            "patience": 8,
        },
        "size_gate": {"internal_limit_bytes": 95_000_000},
    }
    validate_config(config)
    assert isinstance(_build_model(config), MobileNetTCNVisualExpert)
