from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.run_ir_depth_videomaev2_wrist_person_residual import (
    _candidate_seed_offset,
    _final_evaluation_candidates,
    load_wrist_person_config,
    train_cv_fold,
)
from src.models.wrist_person_residual_fusion import (
    WristPersonResidualFusion,
    wrist_person_loss,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/ir_depth_videomaev2_wrist_person_residual.yaml"


def _inputs(batch: int = 2, classes: int = 6, embedding_dim: int = 8) -> dict[str, torch.Tensor]:
    view_embeddings = torch.randn(batch, 4, embedding_dim)
    anchor_logits = torch.randn(batch, classes)
    anchor_view_weights = torch.zeros(batch, classes, 4)
    anchor_view_weights[:, :, 2:] = 0.5
    return {
        "view_embeddings": view_embeddings,
        "anchor_logits": anchor_logits,
        "anchor_view_weights": anchor_view_weights,
        "availability": torch.ones(batch, 4, dtype=torch.bool),
        "num_frames": torch.full((batch,), 24.0),
    }


def test_zero_initialized_person_residual_is_exact_wrist_anchor() -> None:
    model = WristPersonResidualFusion(
        embedding_dim=8,
        num_classes=6,
        class_embedding_dim=4,
        hidden_dim=12,
        gating_mode="margin",
    )
    inputs = _inputs()

    output = model(**inputs)

    assert torch.equal(output["logits"], inputs["anchor_logits"])
    assert torch.count_nonzero(output["delta_logits"]) == 0


def test_margin_gate_decreases_for_confident_wrist_anchor() -> None:
    model = WristPersonResidualFusion(
        embedding_dim=8,
        num_classes=6,
        class_embedding_dim=4,
        hidden_dim=12,
        gating_mode="margin",
    )
    inputs = _inputs()
    inputs["anchor_logits"] = torch.tensor(
        [[3.0, 2.9, 2.8, 0.0, -1.0, -2.0], [6.0, 2.0, 1.0, 0.0, -1.0, -2.0]]
    )

    output = model(**inputs)

    assert output["person_gate"][1].mean() < output["person_gate"][0].mean()


def test_missing_person_forces_exact_anchor_and_zero_gate() -> None:
    model = WristPersonResidualFusion(
        embedding_dim=8,
        num_classes=6,
        class_embedding_dim=4,
        hidden_dim=12,
        gating_mode="no_margin",
    )
    inputs = _inputs()
    inputs["availability"][:, 1] = False
    output = model(**inputs)

    assert torch.equal(output["logits"], inputs["anchor_logits"])
    assert torch.count_nonzero(output["person_gate"]) == 0


def test_person_auxiliary_loss_gives_person_head_gradient_at_initialization() -> None:
    model = WristPersonResidualFusion(
        embedding_dim=8,
        num_classes=6,
        class_embedding_dim=4,
        hidden_dim=12,
        gating_mode="fixed10",
    )
    inputs = _inputs()
    output = model(**inputs)
    losses = wrist_person_loss(
        output=output,
        labels=torch.tensor([0, 1]),
        person_available=inputs["availability"][:, 1],
        person_aux_weight=0.2,
        guard_weight=0.2,
    )

    losses["loss"].backward()

    assert model.person_head.weight.grad is not None
    assert torch.count_nonzero(model.person_head.weight.grad) > 0
    assert model.person_adapter.weight.grad is not None
    assert torch.count_nonzero(model.person_adapter.weight.grad) > 0


def test_margin_auxiliary_ablation_uses_paired_seed() -> None:
    assert _candidate_seed_offset("person_margin") == _candidate_seed_offset(
        "person_margin_no_aux"
    )


def test_final_evaluation_contains_only_grouped_cv_selected_candidate() -> None:
    assert _final_evaluation_candidates("person_fixed10") == ("person_fixed10",)


def test_wrist_person_config_preserves_validation_isolation() -> None:
    config = load_wrist_person_config(CONFIG)

    assert set(config["split"]["train_user_ids"]).isdisjoint({"user6", "user7"})
    assert config["split"]["validation_user_ids"] == ["user6", "user7"]
    assert config["policy"]["validation_users_enter_gradient"] is False
    assert config["policy"]["validation_users_enter_cv_selection"] is False
    assert config["policy"]["fusion_grouped_cv_authorized"] is True
    assert config["policy"]["full_videomae_oof_authorized"] is False


def test_fixed_epoch_grouped_fold_runs_on_cached_embeddings() -> None:
    config = load_wrist_person_config(CONFIG)
    config["model"] = {
        **config["model"],
        "embedding_dim": 8,
        "class_embedding_dim": 4,
        "hidden_dim": 12,
    }
    config["training"] = {
        **config["training"],
        "fixed_epochs": 2,
        "batch_size": 8,
    }
    rng = np.random.default_rng(5)
    samples, classes = 30, 6
    cache = {
        "sample_ids": np.asarray([f"sample_{index}" for index in range(samples)]),
        "user_ids": np.asarray(["a"] * 10 + ["b"] * 10 + ["c"] * 10),
        "labels": np.tile(np.arange(classes), 5).astype(np.int64),
        "num_frames": np.full(samples, 16, dtype=np.int64),
        "availability": np.ones((samples, 2, 4), dtype=bool),
        "fused_view_embeddings": rng.normal(size=(samples, 4, 8)).astype(np.float32),
        "route_logits": rng.normal(size=(samples, 1, classes)).astype(np.float32),
        "route_view_weights": np.zeros((samples, 1, classes, 4), dtype=np.float32),
    }
    cache["route_view_weights"][:, 0, :, 2:] = 0.5

    result = train_cv_fold(
        candidate="person_margin",
        config=config,
        cache=cache,
        train_indices=np.arange(20),
        validation_indices=np.arange(20, 30),
        seed=7,
        device=torch.device("cpu"),
    )

    assert result["epochs"] == 2
    assert result["validation_sample_ids"] == [f"sample_{index}" for index in range(20, 30)]
