from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.data.imu_stage2_contracts import sha256_file
from src.features.imu_rf_features import build_feature_schema
from src.training.imu_rf_trainer import (
    RF_RUN_FILES,
    load_model_bundle,
    load_rf_config,
    train_random_forest,
    validate_rf_run,
)


def _class_order(count: int = 3) -> list[dict[str, object]]:
    return [
        {"class_id": index, "class_name": f"class-{index}", "label_index": index}
        for index in range(count)
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _feature_root(path: Path) -> Path:
    path.mkdir()
    rng = np.random.default_rng(7)
    train_ids = np.asarray([f"train-{index:03d}" for index in range(18)])
    validation_ids = np.asarray([f"val-{index:03d}" for index in range(9)])
    train_labels = np.repeat(np.arange(3, dtype=np.int64), 6)
    validation_labels = np.repeat(np.arange(3, dtype=np.int64), 3)
    train_features = rng.normal(size=(18, 8)).astype(np.float64)
    validation_features = rng.normal(size=(9, 8)).astype(np.float64)
    train_features[:, 0] += train_labels * 4
    validation_features[:, 0] += validation_labels * 4
    np.savez(
        path / "train_features.npz",
        sample_ids=train_ids,
        user_ids=np.asarray(["user-train"] * 18),
        labels=train_labels,
        features=train_features,
    )
    np.savez(
        path / "validation_features.npz",
        sample_ids=validation_ids,
        user_ids=np.asarray(["user-a", "user-b", "user-c"] * 3),
        labels=validation_labels,
        features=validation_features,
    )
    schema = {
        "schema_version": "imu-rf-summary-v1",
        "feature_names": [f"feature-{index}" for index in range(8)],
        "feature_count": 8,
    }
    _write_json(path / "feature_schema.json", schema)
    _write_json(
        path / "imputer.json",
        {
            "imputer_version": "imu-rf-median-imputer-v1",
            "fit_split": "train",
            "fit_sample_count": 18,
            "fit_sample_id_sha256": "a" * 64,
            "feature_schema_sha256": sha256_file(path / "feature_schema.json"),
            "medians": [0.0] * 8,
        },
    )
    _write_json(path / "provenance.json", {"provenance_version": "imu-rf-features-v1"})
    (path / "manifest.csv").write_text("sample_id,split\n", encoding="utf-8")
    return path


def _config(path: Path) -> Path:
    payload = {
        "config_version": "imu-rf-screen-v1",
        "fold": 0,
        "num_classes": 3,
        "feature_schema_version": "imu-rf-summary-v1",
        "n_estimators": 24,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "n_jobs": 1,
        "variants": {"plain": None, "balanced": "balanced_subsample"},
        "random_states": [20260724, 20260725, 20260726],
    }
    _write_json(path, payload)
    return path


def test_rf_config_is_strict_and_rejects_unknown_or_wrong_typed_fields(tmp_path: Path) -> None:
    path = _config(tmp_path / "config.json")
    loaded = load_rf_config(path)
    assert loaded["random_states"] == [20260724, 20260725, 20260726]
    bad = dict(loaded)
    bad["extra"] = True
    _write_json(path, bad)
    with pytest.raises(ValueError, match="field set"):
        load_rf_config(path)
    bad.pop("extra")
    bad["n_estimators"] = 24.0
    _write_json(path, bad)
    with pytest.raises(ValueError, match="n_estimators"):
        load_rf_config(path)


def test_same_seed_is_deterministic_and_different_seed_runs(tmp_path: Path) -> None:
    features = _feature_root(tmp_path / "features")
    config = _config(tmp_path / "config.json")
    first = train_random_forest(
        feature_root=features,
        config_path=config,
        variant="plain",
        random_state=20260724,
        output_dir=None,
        preflight_only=True,
    )
    second = train_random_forest(
        feature_root=features,
        config_path=config,
        variant="plain",
        random_state=20260724,
        output_dir=None,
        preflight_only=True,
    )
    third = train_random_forest(
        feature_root=features,
        config_path=config,
        variant="plain",
        random_state=20260725,
        output_dir=None,
        preflight_only=True,
    )
    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    assert third["probabilities"].shape == (9, 3)
    assert np.isfinite(third["probabilities"]).all()


def test_rf_run_publishes_exact_ten_files_and_valid_probabilities(tmp_path: Path) -> None:
    features = _feature_root(tmp_path / "features")
    config = _config(tmp_path / "config.json")
    output = tmp_path / "run"
    summary = train_random_forest(
        feature_root=features,
        config_path=config,
        variant="balanced",
        random_state=20260724,
        output_dir=output,
        preflight_only=False,
    )
    assert {path.name for path in output.iterdir()} == RF_RUN_FILES
    arrays = validate_rf_run(output)
    assert arrays["class_probabilities"].shape == (9, 3)
    np.testing.assert_allclose(arrays["class_probabilities"].sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        arrays["predictions"], np.argmax(arrays["class_probabilities"], axis=1)
    )
    assert summary["validation_samples"] == 9
    assert summary["feature_count"] == 8


def test_preflight_does_not_create_run_directory(tmp_path: Path) -> None:
    features = _feature_root(tmp_path / "features")
    config = _config(tmp_path / "config.json")
    output = tmp_path / "must-not-exist"
    result = train_random_forest(
        feature_root=features,
        config_path=config,
        variant="plain",
        random_state=20260724,
        output_dir=output,
        preflight_only=True,
    )
    assert not output.exists()
    assert result["probabilities"].shape == (9, 3)


def test_model_reload_rejects_schema_imputer_class_order_or_config_mismatch(tmp_path: Path) -> None:
    features = _feature_root(tmp_path / "features")
    config = _config(tmp_path / "config.json")
    output = tmp_path / "run"
    train_random_forest(
        feature_root=features,
        config_path=config,
        variant="plain",
        random_state=20260724,
        output_dir=output,
        preflight_only=False,
    )
    bundle = joblib.load(output / "model.joblib")
    bindings = dict(bundle["metadata"])
    load_model_bundle(output / "model.joblib", expected_metadata=bindings)
    for field in ("feature_schema_sha256", "imputer_sha256", "class_order_sha256", "rf_config_sha256"):
        bad = dict(bindings)
        bad[field] = "f" * 64
        with pytest.raises(ValueError, match=field):
            load_model_bundle(output / "model.joblib", expected_metadata=bad)


def test_failed_training_leaves_no_output_or_staging(tmp_path: Path) -> None:
    features = _feature_root(tmp_path / "features")
    config = _config(tmp_path / "config.json")
    with np.load(features / "validation_features.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["features"] = arrays["features"].copy()
    arrays["features"][0, 0] = np.nan
    np.savez(features / "validation_features.npz", **arrays)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="finite"):
        train_random_forest(
            feature_root=features,
            config_path=config,
            variant="plain",
            random_state=20260724,
            output_dir=output,
            preflight_only=False,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".run.staging-*"))

