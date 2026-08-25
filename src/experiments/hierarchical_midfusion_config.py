from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODALITIES = ["ir", "depth_color", "skeleton", "imu"]
EXPECTED_TRAIN_USERS = {
    "user1", "user2", "user3", "user5", "user8", "user9",
    "user16", "user18", "user19", "user20", "user21", "user22",
}
EXPECTED_VALIDATION_USERS = {"user6", "user7"}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_midfusion_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("midfusion config must be a mapping")
    if config.get("stage") != "P5-HMF0":
        raise ValueError("midfusion stage changed")
    if config.get("modalities") != EXPECTED_MODALITIES:
        raise ValueError("midfusion modalities changed")
    if int(config.get("segment_count", -1)) != 8:
        raise ValueError("midfusion segment count changed")
    if config.get("class_ids") != list(range(40)):
        raise ValueError("midfusion class order changed")

    population = config.get("population", {})
    if int(population.get("train_samples", -1)) != 2039:
        raise ValueError("midfusion train population changed")
    if int(population.get("validation_samples", -1)) != 388:
        raise ValueError("midfusion validation population changed")
    if int(population.get("class_count", -1)) != 40:
        raise ValueError("midfusion class count changed")
    if set(population.get("train_user_ids", [])) != EXPECTED_TRAIN_USERS:
        raise ValueError("midfusion train users changed")
    if set(population.get("validation_user_ids", [])) != EXPECTED_VALIDATION_USERS:
        raise ValueError("midfusion validation users changed")

    policy = config.get("policy", {})
    required_false = (
        "validation_users_enter_training",
        "validation_users_enter_normalization",
        "validation_users_enter_sampler",
        "validation_users_enter_selection",
        "nonselected_candidates_enter_final_evaluation",
        "thermal_allowed",
        "radar_allowed",
    )
    if any(policy.get(key) is not False for key in required_false):
        raise ValueError("midfusion isolation policy changed")

    grouped_path = project_path(str(population["grouped_split"]))
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    if grouped.get("schema_version") != 1 or grouped.get("seed") != 20260715:
        raise ValueError("midfusion grouped split header changed")
    validation_owners: list[str] = []
    folds: list[dict[str, Any]] = []
    for expected_fold, row in enumerate(grouped.get("folds", [])):
        if int(row.get("fold", -1)) != expected_fold:
            raise ValueError("midfusion grouped fold order changed")
        validation_users = [str(value) for value in row.get("validation_user_ids", [])]
        fit_users = sorted(EXPECTED_TRAIN_USERS - set(validation_users))
        if not validation_users or set(validation_users) & set(fit_users):
            raise ValueError("midfusion grouped fold overlap")
        validation_owners.extend(validation_users)
        folds.append(
            {
                "fold": expected_fold,
                "fit_user_ids": fit_users,
                "validation_user_ids": validation_users,
            }
        )
    if len(folds) != 3 or len(validation_owners) != len(set(validation_owners)):
        raise ValueError("midfusion grouped ownership repeated")
    if set(validation_owners) != EXPECTED_TRAIN_USERS:
        raise ValueError("midfusion grouped ownership incomplete")
    config["grouped_folds"] = folds
    return config

