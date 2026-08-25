from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.hierarchical_midfusion_config import load_midfusion_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml"
SPLIT = ROOT / "metadata/splits/train12_grouped_3fold_midfusion.json"


def test_midfusion_contract_freezes_population_modalities_and_folds() -> None:
    config = load_midfusion_config(CONFIG)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    assert config["stage"] == "P5-HMF0"
    assert config["modalities"] == ["ir", "depth_color", "skeleton", "imu"]
    assert config["segment_count"] == 8
    assert config["population"]["train_samples"] == 2039
    assert config["population"]["validation_samples"] == 388
    assert config["population"]["class_count"] == 40
    assert config["policy"]["validation_users_enter_training"] is False
    assert config["policy"]["nonselected_candidates_enter_final_evaluation"] is False
    fold_users = [user for fold in split["folds"] for user in fold["validation_user_ids"]]
    assert len(fold_users) == len(set(fold_users)) == 12
    assert set(fold_users).isdisjoint({"user6", "user7"})
    for fold in config["grouped_folds"]:
        assert set(fold["fit_user_ids"]).isdisjoint(fold["validation_user_ids"])
        assert len(fold["fit_user_ids"]) + len(fold["validation_user_ids"]) == 12


def test_midfusion_contract_rejects_forbidden_thermal_modality(tmp_path: Path) -> None:
    config = CONFIG.read_text(encoding="utf-8").replace(
        "[ir, depth_color, skeleton, imu]",
        "[ir, depth_color, skeleton, imu, thermal]",
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="modalities"):
        load_midfusion_config(path)
