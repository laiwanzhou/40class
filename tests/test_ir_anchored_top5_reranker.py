from __future__ import annotations

from pathlib import Path

import torch
import numpy as np

from scripts.cache_ir_depth_videomaev2_train12_p2r0 import build_ir_anchor_logits
from scripts.run_ir_depth_videomaev2_p2r0 import _train_cv_fold, load_p2r0_config
from src.models.ir_anchored_top5_reranker import (
    IRAnchoredTop5Reranker,
    Top5LogitReranker,
    guarded_reranker_loss,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/ir_depth_videomaev2_p2r0.yaml"


def test_p2r0_config_reserves_user6_user7_for_one_final_evaluation() -> None:
    config = load_p2r0_config(CONFIG)

    assert config["stage"] == "P2-R0"
    assert config["validation_policy"]["final_users"] == ["user6", "user7"]
    assert config["validation_policy"]["final_evaluation_count"] == 1
    assert config["validation_policy"]["use_final_users_for_training"] is False
    assert config["policy"]["update_videomae"] is False
    assert config["policy"]["p2b_authorized"] is False


def test_zero_initialized_feature_reranker_is_exact_ir_anchor() -> None:
    torch.manual_seed(3)
    model = IRAnchoredTop5Reranker(
        embedding_dim=12, fusion_dim=8, num_classes=7, view_top_k=2, class_top_k=5
    )
    embeddings = torch.randn(4, 2, 4, 12)
    base_logits = torch.randn(4, 7)
    availability = torch.ones(4, 2, 4, dtype=torch.bool)

    output = model(
        view_embeddings=embeddings,
        base_logits=base_logits,
        availability=availability,
    )

    assert torch.equal(output["logits"], base_logits)
    assert torch.count_nonzero(output["delta_logits"]) == 0
    assert torch.allclose(output["view_weights"].sum(dim=2), torch.ones(4, 7))
    assert int((output["view_weights"] > 0).sum(dim=2).max()) <= 2


def test_residual_can_change_only_base_top5_classes() -> None:
    torch.manual_seed(5)
    model = IRAnchoredTop5Reranker(
        embedding_dim=10, fusion_dim=6, num_classes=8, view_top_k=2, class_top_k=5
    )
    with torch.no_grad():
        model.residual_weight.fill_(0.1)
        model.residual_bias.fill_(0.1)
    embeddings = torch.randn(2, 2, 4, 10)
    base_logits = torch.tensor(
        [[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    )
    output = model(
        view_embeddings=embeddings,
        base_logits=base_logits,
        availability=torch.ones(2, 2, 4, dtype=torch.bool),
    )

    changed = output["logits"] != base_logits
    expected = torch.zeros_like(changed)
    expected.scatter_(1, base_logits.topk(5, dim=1).indices, True)
    assert torch.equal(changed, expected)


def test_depth_unavailable_forces_zero_depth_gate() -> None:
    model = IRAnchoredTop5Reranker(
        embedding_dim=10, fusion_dim=6, num_classes=8, view_top_k=2, class_top_k=5
    )
    output = model(
        view_embeddings=torch.randn(2, 2, 4, 10),
        base_logits=torch.randn(2, 8),
        availability=torch.tensor(
            [[[True] * 4, [False] * 4], [[True] * 4, [True, False, True, False]]]
        ),
    )

    assert torch.count_nonzero(output["depth_gates"][0]) == 0
    assert output["depth_gates"][1, 1, 0] == 0
    assert output["depth_gates"][1, 3, 0] == 0


def test_guard_loss_penalizes_reranker_worse_than_anchor() -> None:
    labels = torch.tensor([0, 1])
    anchor = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    worse = torch.tensor([[0.0, 3.0], [3.0, 0.0]])

    loss = guarded_reranker_loss(
        logits=worse,
        base_logits=anchor,
        labels=labels,
        depth_gates=torch.zeros(2, 4, 1),
        guard_weight=0.2,
        depth_l1_weight=0.02,
    )

    assert loss["guard_loss"] > 0
    assert loss["loss"] > loss["ce_loss"]


def test_train_cache_ir_anchor_uses_only_the_four_ir_views() -> None:
    view_logits = np.zeros((1, 2, 4, 2), dtype=np.float32)
    view_logits[0, 0, :, 0] = [1.0, 2.0, 3.0, 4.0]
    view_logits[0, 1, :, 1] = 100.0
    availability = np.ones((1, 2, 4), dtype=bool)
    gate = np.zeros((2, 2, 4), dtype=np.float32)

    anchor = build_ir_anchor_logits(view_logits, gate, availability)

    assert np.allclose(anchor, [[2.5, 0.0]])


def test_logit_only_capacity_control_is_zero_initialized_and_top5_limited() -> None:
    model = Top5LogitReranker(num_classes=8, hidden_dim=6, class_top_k=5)
    base = torch.randn(3, 8)

    initial = model(base_logits=base)

    assert torch.equal(initial["logits"], base)
    with torch.no_grad():
        model.network[-1].bias.fill_(0.2)
    changed = model(base_logits=base)["logits"] != base
    expected = torch.zeros_like(changed)
    expected.scatter_(1, base.topk(5, dim=1).indices, True)
    assert torch.equal(changed, expected)


def test_cached_feature_cv_training_runs_without_final_validation_users() -> None:
    config = load_p2r0_config(CONFIG)
    config["model"] = {
        **config["model"],
        "embedding_dim": 10,
        "fusion_dim": 6,
        "num_classes": 8,
        "class_top_k": 5,
    }
    config["training"] = {
        **config["training"],
        "batch_size": 8,
        "max_epochs": 3,
        "early_stopping_patience": 2,
    }
    rng = np.random.default_rng(19)
    labels = np.tile(np.arange(8), 3)
    cache = {
        "view_embeddings": rng.normal(size=(24, 2, 4, 10)).astype(np.float32),
        "ir_anchor_logits": rng.normal(size=(24, 8)).astype(np.float32),
        "availability": np.ones((24, 2, 4), dtype=bool),
        "labels": labels.astype(np.int64),
        "user_ids": np.asarray(["train_a"] * 12 + ["train_b"] * 12),
        "sample_ids": np.asarray([f"sample_{index}" for index in range(24)]),
    }

    result = _train_cv_fold(
        candidate="feature_router",
        config=config,
        cache=cache,
        train_indices=np.arange(16),
        val_indices=np.arange(16, 24),
        seed=23,
        device=torch.device("cpu"),
    )

    assert 1 <= result["best_epoch"] <= 3
    assert result["epochs_completed"] <= 3
    assert "accuracy" in result["best_metrics"]
