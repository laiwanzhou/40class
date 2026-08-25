from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.hierarchical_midfusion_config import load_midfusion_config
from src.experiments.hierarchical_midfusion_config import (
    assert_grouped_cv_authorized,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml"


def test_midfusion_contract_freezes_fixed_user6_user7_protocol() -> None:
    config = load_midfusion_config(CONFIG)

    assert config["stage"] == "P5-HMF0"
    assert config["evaluation_protocol"] == "fixed_user6_user7"
    assert config["modalities"] == ["ir", "depth_color", "skeleton", "imu"]
    assert config["segment_count"] == 8
    assert config["population"]["train_samples"] == 2039
    assert config["population"]["validation_samples"] == 388
    assert config["population"]["class_count"] == 40
    assert config["policy"]["validation_users_enter_training"] is False
    assert config["policy"]["validation_users_enter_selection"] is True
    assert config["policy"]["grouped_cv_authorized"] is False
    assert "grouped_folds" not in config


def test_grouped_cv_requires_explicit_authorization() -> None:
    config = load_midfusion_config(CONFIG)

    with pytest.raises(PermissionError, match="explicit authorization"):
        assert_grouped_cv_authorized(config)


def test_authorization_alone_cannot_bypass_incomplete_fold_classes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authorized.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "grouped_cv_authorized: false", "grouped_cv_authorized: true"
        ),
        encoding="utf-8",
    )
    config = load_midfusion_config(path)

    with pytest.raises(ValueError, match="class coverage"):
        assert_grouped_cv_authorized(config)


def test_midfusion_contract_rejects_forbidden_thermal_modality(tmp_path: Path) -> None:
    config = CONFIG.read_text(encoding="utf-8").replace(
        "[ir, depth_color, skeleton, imu]",
        "[ir, depth_color, skeleton, imu, thermal]",
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="modalities"):
        load_midfusion_config(path)
