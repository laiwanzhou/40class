from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.imu_rf_finalization import FROZEN_ESTIMATOR_PARAMS, load_final_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_final_config_freezes_every_rf_parameter_and_schema() -> None:
    config = load_final_config(REPOSITORY_ROOT / "configs" / "imu_rf_final_v1.json")
    assert config["config_version"] == "imu-rf-finalization-v1"
    assert config["candidate_id"] == "trees_150_leaf4"
    assert config["feature_schema_version"] == "imu-rf-summary-v1"
    assert config["feature_count"] == 2310
    assert config["fold"] == 0
    assert config["num_classes"] == 40
    assert config["random_state"] == 20260725
    assert config["model_compression"] == {"method": "lzma", "level": 3}
    assert config["estimator_params"] == FROZEN_ESTIMATOR_PARAMS


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("random_state",), 20260724),
        (("feature_schema_version",), "imu-rf-physical-v1"),
        (("feature_count",), 4710),
        (("estimator_params", "n_estimators"), 149),
        (("estimator_params", "criterion"), "entropy"),
        (("estimator_params", "max_depth"), 20),
        (("estimator_params", "min_samples_split"), 3),
        (("estimator_params", "min_samples_leaf"), 2),
        (("estimator_params", "max_features"), None),
        (("estimator_params", "max_leaf_nodes"), 100),
        (("estimator_params", "bootstrap"), False),
        (("estimator_params", "max_samples"), 0.8),
        (("estimator_params", "class_weight"), None),
        (("estimator_params", "n_jobs"), 1),
        (("estimator_params", "random_state"), 20260726),
    ],
)
def test_final_config_rejects_any_frozen_contract_change(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    original = json.loads(
        (REPOSITORY_ROOT / "configs" / "imu_rf_final_v1.json").read_text(encoding="utf-8")
    )
    target = original
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path = tmp_path / "changed.json"
    config_path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        load_final_config(config_path)


def test_final_config_rejects_unknown_fields(tmp_path: Path) -> None:
    original = json.loads(
        (REPOSITORY_ROOT / "configs" / "imu_rf_final_v1.json").read_text(encoding="utf-8")
    )
    original["experimental_features"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="field set"):
        load_final_config(path)
