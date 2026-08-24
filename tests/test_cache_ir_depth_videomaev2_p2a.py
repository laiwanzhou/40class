from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.cache_ir_depth_videomaev2_p2a import (
    evaluate_cached_ablation,
    fuse_cached_view_logits,
    load_p2a_config,
    validate_reference_membership,
)
from src.models.ir_depth_videomaev2_teacher import IRDepthVideoMAEV2Teacher


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/ir_depth_videomaev2_p2a.yaml"


def test_p2a_config_forbids_training_and_records_only_the_deferred_p2b_candidate() -> None:
    config = load_p2a_config(CONFIG)

    assert config["stage"] == "P2-A"
    assert config["policy"]["training_allowed"] is False
    assert config["policy"]["p2b_authorized"] is False
    assert config["policy"]["exploratory_validation_diagnostics"] is True
    assert config["policy"]["may_select_p2b_without_new_evaluation_boundary"] is False
    assert config["execution"]["deterministic_algorithms"] is True
    assert config["execution"]["cublas_workspace_config"] == ":4096:8"
    assert config["deferred_p2b"]["long_trial_frames"] == 32
    assert config["deferred_p2b"]["motion_peak_sampling"] is False


def test_cached_fusion_uses_class_specific_weights_and_availability() -> None:
    view_logits = np.zeros((1, 2, 2, 2), dtype=np.float32)
    view_logits[0, 0, 0] = [4.0, 0.0]
    view_logits[0, 1, 1] = [0.0, 4.0]
    gate = np.zeros((2, 2, 2), dtype=np.float32)
    gate[0, 0, 0] = 2.0
    gate[1, 1, 1] = 1.0
    availability = np.ones((1, 2, 2), dtype=bool)
    availability[0, 0, 1] = False

    logits, weights = fuse_cached_view_logits(view_logits, gate, availability)

    assert logits.shape == (1, 2)
    assert weights.shape == (1, 2, 2, 2)
    assert logits[0, 0] > logits[0, 1]
    assert weights[0, 0, 0, 0] > weights[0, 0, 1, 1]
    assert weights[0, 1, 1, 1] > weights[0, 1, 0, 0]
    assert weights[0, :, 0, 1].sum() == 0.0


def test_numpy_cached_fusion_matches_training_torch_fusion() -> None:
    rng = np.random.default_rng(11)
    view_logits = rng.normal(size=(3, 2, 4, 5)).astype(np.float32)
    gate = rng.normal(size=(5, 2, 4)).astype(np.float32)
    availability = np.ones((3, 2, 4), dtype=bool)
    availability[1, 0, 3] = False
    model = IRDepthVideoMAEV2Teacher(backbone=nn.Identity(), num_classes=5)
    with torch.no_grad():
        model.class_view_gate.copy_(torch.from_numpy(gate))

    with torch.no_grad():
        expected_logits, expected_weights = model.fuse_view_logits(
            torch.from_numpy(view_logits), torch.from_numpy(availability)
        )
    actual_logits, actual_weights = fuse_cached_view_logits(
        view_logits, gate, availability
    )

    assert np.allclose(actual_logits, expected_logits.numpy(), atol=1e-6)
    assert np.allclose(actual_weights, expected_weights.numpy(), atol=1e-6)


def test_reference_membership_requires_two_exact_unique_id_sets() -> None:
    current = np.asarray(["a", "b", "c"])
    reference = np.asarray(["c", "a", "b"])

    take = validate_reference_membership(current, reference)

    assert take.tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="unique"):
        validate_reference_membership(current, np.asarray(["a", "a", "c"]))
    with pytest.raises(ValueError, match="exactly match"):
        validate_reference_membership(current[:2], reference)


def test_only_and_drop_ablation_are_computed_from_cached_view_logits() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    users = np.asarray(["u1", "u1", "u2", "u2"])
    view_logits = np.zeros((4, 2, 2, 2), dtype=np.float32)
    view_logits[np.arange(4), 0, 0, labels] = 8.0
    gate = np.zeros((2, 2, 2), dtype=np.float32)
    availability = np.ones((4, 2, 2), dtype=bool)

    report = evaluate_cached_ablation(
        view_logits=view_logits,
        class_view_gate=gate,
        availability=availability,
        labels=labels,
        users=users,
        modality_names=("ir", "depth"),
        view_names=("global", "person"),
    )

    assert report["full"]["accuracy"] == 1.0
    assert report["stream_only"]["ir:global"]["accuracy"] == 1.0
    assert report["stream_drop"]["ir:global"]["accuracy"] == 0.5
    assert report["modality_only"]["ir"]["accuracy"] == 1.0
    assert report["modality_only"]["depth"]["accuracy"] == 0.5
