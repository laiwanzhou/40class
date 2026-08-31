from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAMILY_NAMES = (
    "locomotion",
    "posture_transition",
    "exercise",
    "whole_body_motion",
    "upper_body_dominant",
    "mostly_static_fine",
)
TRAIN_USERS = {
    "user1", "user2", "user3", "user5", "user8", "user9", "user16",
    "user18", "user19", "user20", "user21", "user22",
}
VALIDATION_USERS = {"user6", "user7"}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _family_table() -> torch.Tensor:
    table = torch.zeros(40, len(FAMILY_NAMES), dtype=torch.float32)
    groups = {
        0: {28, 36},
        1: {32, 33, 34},
        2: {29, 30, 31, 35},
        3: {3, 5, 12, 13, 15, 16, 28, 29, 30, 31, 32, 33, 34, 35, 36},
        4: {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 37, 38, 39,
        },
        5: {17, 18, 20, 21, 22, 23, 25, 26},
    }
    for family, classes in groups.items():
        table[list(sorted(classes)), family] = 1.0
    return table


FAMILY_TABLE = _family_table()


def motion_family_targets(class_ids: torch.Tensor) -> torch.Tensor:
    values = class_ids.long()
    if bool((values < 0).any()) or bool((values >= 40).any()):
        raise ValueError("motion family class ID is outside 0..39")
    return FAMILY_TABLE.to(values.device)[values]


def load_motion_attribute_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("motion attribute config must be a mapping")
    if config.get("schema_version") != 1 or config.get("stage") != "MOTION-ATTRIBUTE-1":
        raise ValueError("motion attribute header changed")
    if config.get("seed") != 20260715 or config.get("class_ids") != list(range(40)):
        raise ValueError("motion attribute seed or class order changed")
    population = config.get("population", {})
    if population.get("train_samples") != 2039 or population.get("validation_samples") != 388:
        raise ValueError("motion attribute population changed")
    if set(population.get("train_user_ids", [])) != TRAIN_USERS:
        raise ValueError("motion attribute train users changed")
    if set(population.get("validation_user_ids", [])) != VALIDATION_USERS:
        raise ValueError("motion attribute validation users changed")
    if config.get("input") != {
        "frames": 96,
        "joints": 17,
        "channels": ["x", "y", "z", "vx", "vy", "vz"],
    }:
        raise ValueError("motion attribute input changed")
    if config.get("attributes") != {"count": 16}:
        raise ValueError("motion attribute targets changed")
    if config.get("families") != list(FAMILY_NAMES):
        raise ValueError("motion family order changed")
    if config.get("loss_weights") != {
        "families": 1.0,
        "attributes": 0.5,
        "action": 0.25,
    }:
        raise ValueError("motion attribute loss changed")
    training = config.get("training", {})
    if training != {
        "epochs": 15,
        "batch_size": 32,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "num_workers": 0,
    }:
        raise ValueError("motion attribute training changed")
    policy = config.get("policy", {})
    if any(value is not False for value in policy.values()):
        raise ValueError("motion attribute authorization changed")
    return config
