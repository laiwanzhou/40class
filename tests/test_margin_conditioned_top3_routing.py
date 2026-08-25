from __future__ import annotations

import numpy as np
import torch

from scripts.cache_ir_depth_videomaev2_p3r1 import (
    load_p3r1_config,
    validate_cache_membership,
    validate_reference_predictions,
)
from scripts.run_ir_depth_videomaev2_p3r1 import (
    compare_uniform_reference,
    train_cv_fold,
)
from src.models.margin_conditioned_top3_routing import (
    MarginConditionedTop3Reranker,
    build_route_bank,
    reroute_cached_embeddings,
)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/ir_depth_videomaev2_p3r1.yaml"


def _routing_fixture() -> dict[str, torch.Tensor]:
    return {
        "view_embeddings": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]]
        ),
        "class_queries": torch.zeros(2, 2),
        "class_view_bias": torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]
        ),
        "head_weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "head_bias": torch.zeros(2),
        "availability": torch.ones(1, 4, dtype=torch.bool),
    }


def test_hard_top2_reproduces_wrist_only_routing() -> None:
    output = reroute_cached_embeddings(**_routing_fixture(), mode="hard", top_k=2)

    assert torch.equal(output["selected_views"], torch.tensor([[[2, 3], [2, 3]]]))
    assert torch.count_nonzero(output["view_weights"][:, :, :2]) == 0
    assert torch.allclose(output["logits"], torch.tensor([[1.0, 1.0]]))


def test_grouped_routing_reserves_context_mass() -> None:
    output = reroute_cached_embeddings(
        **_routing_fixture(), mode="grouped", context_share=0.25
    )

    context = output["view_weights"][:, :, :2].sum(dim=2)
    wrists = output["view_weights"][:, :, 2:].sum(dim=2)
    assert torch.allclose(context, torch.full_like(context, 0.25))
    assert torch.allclose(wrists, torch.full_like(wrists, 0.75))
    assert torch.allclose(output["view_weights"].sum(dim=2), torch.ones(1, 2))


def test_grouped_routing_falls_back_to_context_when_both_wrists_are_missing() -> None:
    fixture = _routing_fixture()
    fixture["availability"][:, 2:] = False

    output = reroute_cached_embeddings(
        **fixture, mode="grouped", context_share=0.25
    )

    assert torch.count_nonzero(output["view_weights"][:, :, 2:]) == 0
    assert torch.allclose(
        output["view_weights"][:, :, :2].sum(dim=2), torch.ones(1, 2)
    )


def test_hard_routing_with_large_topk_keeps_missing_views_at_zero_weight() -> None:
    fixture = _routing_fixture()
    fixture["availability"][:, 3] = False

    output = reroute_cached_embeddings(**fixture, mode="hard", top_k=4)

    assert torch.count_nonzero(output["view_weights"][:, :, 3]) == 0
    assert torch.allclose(output["view_weights"].sum(dim=2), torch.ones(1, 2))


def test_hard_routing_rejects_sample_with_no_available_view() -> None:
    fixture = _routing_fixture()
    fixture["availability"][:] = False

    with np.testing.assert_raises_regex(ValueError, "available view"):
        reroute_cached_embeddings(**fixture, mode="hard", top_k=2)


def test_zero_initialized_top3_reranker_is_exact_anchor() -> None:
    torch.manual_seed(3)
    model = MarginConditionedTop3Reranker(
        num_classes=8, route_count=4, class_embedding_dim=5, hidden_dim=12
    )
    anchor = torch.randn(3, 8)
    routes = torch.randn(3, 4, 8)

    output = model(
        anchor_logits=anchor,
        route_logits=routes,
        num_frames=torch.tensor([8.0, 32.0, 64.0]),
    )

    assert torch.equal(output["logits"], anchor)
    assert torch.count_nonzero(output["delta_logits"]) == 0


def test_top3_reranker_can_change_only_anchor_candidates() -> None:
    model = MarginConditionedTop3Reranker(
        num_classes=6, route_count=3, class_embedding_dim=4, hidden_dim=10
    )
    with torch.no_grad():
        model.residual[-1].bias.fill_(1.0)
    anchor = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    output = model(
        anchor_logits=anchor,
        route_logits=torch.randn(1, 3, 6),
        num_frames=torch.tensor([24.0]),
    )

    expected = torch.tensor([[True, True, True, False, False, False]])
    assert torch.equal(output["candidate_mask"], expected)
    assert torch.equal(output["logits"] != anchor, expected)


def test_margin_gate_is_monotonically_smaller_for_confident_samples() -> None:
    model = MarginConditionedTop3Reranker(
        num_classes=5, route_count=2, class_embedding_dim=3, hidden_dim=8
    )
    low_margin = torch.tensor([[3.0, 2.9, 2.8, 0.0, -1.0]])
    high_margin = torch.tensor([[6.0, 2.0, 1.0, 0.0, -1.0]])
    anchor = torch.cat((low_margin, high_margin), dim=0)
    output = model(
        anchor_logits=anchor,
        route_logits=torch.randn(2, 2, 5),
        num_frames=torch.tensor([16.0, 16.0]),
    )

    assert output["margin_gate"][1] < output["margin_gate"][0]


def test_non_margin_control_uses_constant_gate() -> None:
    model = MarginConditionedTop3Reranker(
        num_classes=5,
        route_count=2,
        class_embedding_dim=3,
        hidden_dim=8,
        use_margin_gate=False,
    )
    output = model(
        anchor_logits=torch.tensor(
            [[3.0, 2.9, 2.8, 0.0, -1.0], [6.0, 2.0, 1.0, 0.0, -1.0]]
        ),
        route_logits=torch.randn(2, 2, 5),
        num_frames=torch.tensor([16.0, 64.0]),
    )

    assert torch.equal(output["margin_gate"], torch.ones(2))


def test_routing_rejects_missing_view_for_a_group() -> None:
    fixture = _routing_fixture()
    fixture["availability"][:, :2] = False

    with np.testing.assert_raises_regex(ValueError, "context view"):
        reroute_cached_embeddings(**fixture, mode="grouped", context_share=0.25)


def test_route_bank_contains_anchor_depth_and_context_controls() -> None:
    fixture = _routing_fixture()
    bank = build_route_bank(
        fused_view_embeddings=fixture.pop("view_embeddings"),
        ir_view_embeddings=torch.tensor(
            [[[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]]]
        ),
        **fixture,
    )

    assert bank["route_names"] == (
        "full_hard2",
        "ir_hard2",
        "full_hard3",
        "full_hard4",
        "full_soft",
        "full_group_context10",
        "full_group_context25",
        "full_group_context50",
        "ir_group_context25",
        "full_context_only",
        "full_wrists_only",
    )
    assert bank["route_logits"].shape == (1, 11, 2)
    assert torch.allclose(bank["route_logits"][:, 0], torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(bank["route_logits"][:, 1], torch.tensor([[0.5, 0.5]]))


def test_route_bank_wrist_control_falls_back_to_context_when_wrists_missing() -> None:
    fixture = _routing_fixture()
    fixture["availability"][:, 2:] = False
    bank = build_route_bank(
        fused_view_embeddings=fixture.pop("view_embeddings"),
        ir_view_embeddings=torch.tensor(
            [[[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]]]
        ),
        **fixture,
    )

    context_index = bank["route_names"].index("full_context_only")
    wrists_index = bank["route_names"].index("full_wrists_only")
    assert torch.equal(
        bank["route_logits"][:, wrists_index], bank["route_logits"][:, context_index]
    )


def test_p3r1_config_keeps_validation_users_out_of_all_selection_paths() -> None:
    config = load_p3r1_config(CONFIG)

    assert set(config["split"]["train_user_ids"]).isdisjoint({"user6", "user7"})
    assert config["split"]["validation_user_ids"] == ["user6", "user7"]
    assert config["policy"]["update_videomae"] is False
    assert config["policy"]["validation_users_enter_gradient"] is False
    assert config["policy"]["validation_users_enter_cv_selection"] is False


def test_cache_membership_rejects_validation_user_in_train_partition() -> None:
    config = load_p3r1_config(CONFIG)

    with np.testing.assert_raises_regex(ValueError, "train user membership"):
        validate_cache_membership(
            partition="train",
            sample_ids=np.asarray(["a", "b"]),
            user_ids=np.asarray(["user1", "user6"]),
            labels=np.asarray([0, 1]),
            expected_samples=2,
            config=config,
        )


def test_validation_cache_must_reproduce_selected_checkpoint_predictions() -> None:
    current = {
        "sample_ids": np.asarray(["a", "b"]),
        "labels": np.asarray([0, 1]),
        "route_logits": np.asarray([[[3.0, 0.0]], [[0.0, 3.0]]], dtype=np.float32),
    }
    reference = {
        "sample_ids": np.asarray(["b", "a"]),
        "labels": np.asarray([1, 0]),
        "logits": np.asarray([[0.0, 3.0], [3.0, 0.0]], dtype=np.float32),
    }

    result = validate_reference_predictions(
        current=current, reference=reference, maximum_logit_delta=1e-5
    )

    assert result == {"maximum_logit_delta": 0.0, "prediction_disagreement": 0}


def test_validation_cache_rejects_changed_anchor_logits() -> None:
    current = {
        "sample_ids": np.asarray(["a"]),
        "labels": np.asarray([0]),
        "route_logits": np.asarray([[[2.0, 0.0]]], dtype=np.float32),
    }
    reference = {
        "sample_ids": np.asarray(["a"]),
        "labels": np.asarray([0]),
        "logits": np.asarray([[1.0, 0.0]], dtype=np.float32),
    }

    with np.testing.assert_raises_regex(RuntimeError, "anchor logits"):
        validate_reference_predictions(
            current=current, reference=reference, maximum_logit_delta=1e-5
        )


def test_grouped_cv_fold_trains_only_on_supplied_indices() -> None:
    config = load_p3r1_config(CONFIG)
    config["reranker"] = {
        **config["reranker"],
        "class_embedding_dim": 4,
        "hidden_dim": 8,
        "batch_size": 8,
        "max_epochs": 3,
        "early_stopping_patience": 2,
    }
    rng = np.random.default_rng(17)
    labels = np.tile(np.arange(5), 6).astype(np.int64)
    anchor = rng.normal(size=(30, 5)).astype(np.float32)
    routes = rng.normal(size=(30, 3, 5)).astype(np.float32)
    cache = {
        "route_logits": routes,
        "labels": labels,
        "num_frames": np.full(30, 16, dtype=np.int64),
        "user_ids": np.asarray(["a"] * 10 + ["b"] * 10 + ["c"] * 10),
        "sample_ids": np.asarray([f"sample_{index}" for index in range(30)]),
    }
    cache["route_logits"][:, 0] = anchor

    result = train_cv_fold(
        candidate="margin_routes",
        config=config,
        cache=cache,
        train_indices=np.arange(20),
        validation_indices=np.arange(20, 30),
        seed=19,
        device=torch.device("cpu"),
    )

    assert 1 <= result["best_epoch"] <= 3
    assert result["epochs_completed"] <= 3
    assert result["validation_sample_ids"] == [f"sample_{index}" for index in range(20, 30)]


def test_uniform_reference_comparison_aligns_sample_ids_before_deltas() -> None:
    current = {
        "sample_ids": np.asarray(["a", "b", "c", "d"]),
        "user_ids": np.asarray(["u1", "u1", "u2", "u2"]),
        "labels": np.asarray([0, 0, 1, 1]),
        "route_logits": np.asarray(
            [
                [[3.0, 0.0]],
                [[0.0, 3.0]],
                [[0.0, 3.0]],
                [[3.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    }
    reference = {
        "sample_ids": np.asarray(["d", "c", "b", "a"]),
        "labels": np.asarray([1, 1, 0, 0]),
        "full_logits": np.asarray(
            [[0.0, 3.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0]],
            dtype=np.float32,
        ),
        "class_view_weights": np.full((4, 2, 2, 4), 0.125, dtype=np.float32),
    }

    result = compare_uniform_reference(
        current=current, reference=reference, action_names=("zero", "one")
    )

    assert result["current_correct"] == 2
    assert result["uniform_correct"] == 3
    assert result["current_vs_uniform"] == {
        "rescued": 1,
        "harmed": 2,
        "net": -1,
        "disagreement": 3,
    }
    assert np.isclose(result["uniform_mean_view_entropy"], np.log(8.0))
