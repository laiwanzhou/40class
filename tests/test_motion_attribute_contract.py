from __future__ import annotations

from pathlib import Path

import torch

from src.experiments.motion_attribute_config import (
    FAMILY_NAMES,
    load_motion_attribute_config,
    motion_family_targets,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motion_attribute_expert.yaml"


def test_motion_attribute_contract_freezes_population_model_and_training() -> None:
    config = load_motion_attribute_config(CONFIG)

    assert config["stage"] == "MOTION-ATTRIBUTE-1"
    assert config["population"]["train_samples"] == 2039
    assert config["population"]["validation_samples"] == 388
    assert config["input"] == {
        "frames": 96,
        "joints": 17,
        "channels": ["x", "y", "z", "vx", "vy", "vz"],
    }
    assert config["attributes"]["count"] == 16
    assert config["families"] == list(FAMILY_NAMES)
    assert config["loss_weights"] == {
        "families": 1.0,
        "attributes": 0.5,
        "action": 0.25,
    }
    assert config["training"]["epochs"] == 15
    assert config["training"]["batch_size"] == 32
    assert config["policy"]["grouped_cv_allowed"] is False
    assert config["policy"]["multiple_seeds_allowed"] is False


def test_motion_family_targets_preserve_overlapping_semantics() -> None:
    targets = motion_family_targets(torch.tensor([36, 32, 17, 3, 7]))

    assert targets.shape == (5, 6)
    assert targets[0].tolist() == [1, 0, 0, 1, 0, 0]  # Walk
    assert targets[1].tolist() == [0, 1, 0, 1, 0, 0]  # Stand_up
    assert targets[2].tolist() == [0, 0, 0, 0, 1, 1]  # keyboard
    assert targets[3].tolist() == [0, 0, 0, 1, 1, 0]  # take off clothes
    assert targets[4].tolist() == [0, 0, 0, 0, 1, 0]  # Eat_food
