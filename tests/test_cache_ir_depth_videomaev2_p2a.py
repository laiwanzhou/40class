from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.cache_ir_depth_videomaev2_p2a import (
    evaluate_cached_ablation,
    fuse_cached_view_logits,
    load_p2a_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/ir_depth_videomaev2_p2a.yaml"


def test_p2a_config_forbids_training_and_records_only_the_deferred_p2b_candidate() -> None:
    config = load_p2a_config(CONFIG)

    assert config["stage"] == "P2-A"
    assert config["policy"]["training_allowed"] is False
    assert config["policy"]["p2b_authorized"] is False
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

    logits, weights = fuse_cached_view_logits(view_logits, gate, availability)

    assert logits.shape == (1, 2)
    assert weights.shape == (1, 2, 2, 2)
    assert logits[0, 0] > logits[0, 1]
    assert weights[0, 0, 0, 0] > weights[0, 0, 1, 1]
    assert weights[0, 1, 1, 1] > weights[0, 1, 0, 0]


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
