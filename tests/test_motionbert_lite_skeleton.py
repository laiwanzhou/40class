from __future__ import annotations

from pathlib import Path

import torch

from src.experiments.motionbert_p6b_config import (
    load_motionbert_p6b_config,
    project_path,
)
from src.models.motionbert_lite_skeleton import (
    MotionBERTLiteSkeletonExpert,
    build_motionbert_lite_expert,
    set_motionbert_train_stage,
)
from third_party.motionbert import DSTformer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml"


def _tiny_expert() -> MotionBERTLiteSkeletonExpert:
    backbone = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=32,
        dim_rep=64,
        depth=5,
        num_heads=4,
        mlp_ratio=2,
        num_joints=17,
        maxlen=96,
        att_fuse=True,
    )
    return MotionBERTLiteSkeletonExpert(
        backbone=backbone, dim_rep=64, classes=40, dropout=0.5
    )


def test_motionbert_expert_returns_embedding_and_prior_safe_logits() -> None:
    model = _tiny_expert()
    sequence = torch.randn(2, 96, 17, 3)
    available = torch.tensor([True, False])

    output = model(sequence, available)

    assert output["sequence_features"].shape == (2, 96, 17, 64)
    assert output["backbone_embedding"].shape == (2, 64)
    assert output["embedding"].shape == (2, 64)
    assert output["logits"].shape == (2, 40)
    assert torch.isfinite(output["logits"]).all()
    assert torch.count_nonzero(output["embedding"][1]) == 0
    assert torch.count_nonzero(output["logits"][1]) == 0


def test_b1_changes_only_head_and_b2_unfreezes_last_two_block_pairs() -> None:
    model = _tiny_expert()

    set_motionbert_train_stage(model, "B1")
    b1 = {name for name, value in model.named_parameters() if value.requires_grad}
    assert b1 == {
        "head_norm.weight",
        "head_norm.bias",
        "classifier.weight",
        "classifier.bias",
    }

    set_motionbert_train_stage(model, "B2")
    b2 = {name for name, value in model.named_parameters() if value.requires_grad}
    assert b1.issubset(b2)
    assert any(name.startswith("backbone.blocks_st.3.") for name in b2)
    assert any(name.startswith("backbone.blocks_ts.4.") for name in b2)
    assert any(name.startswith("backbone.ts_attn.3.") for name in b2)
    assert not any(name.startswith("backbone.blocks_st.2.") for name in b2)
    assert not model.backbone.joints_embed.weight.requires_grad


def test_real_checkpoint_covers_complete_motionbert_lite_backbone() -> None:
    config = load_motionbert_p6b_config(CONFIG)

    model, coverage = build_motionbert_lite_expert(config)

    assert coverage.element_fraction == 1.0
    assert coverage.missing_keys == ()
    assert coverage.unexpected_keys == ()
    assert coverage.shape_mismatches == ()
    assert sum(parameter.numel() for parameter in model.backbone.parameters()) == 16_001_549
    assert project_path(str(config["checkpoint"]["path"])).is_file()
