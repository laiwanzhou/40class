from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.imu_stage2_contracts import sha256_file


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_feature_root(path: Path) -> Path:
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
        "class_order": [
            {"class_id": index, "class_name": f"class-{index}", "label_index": index}
            for index in range(3)
        ],
    }
    write_json(path / "feature_schema.json", schema)
    write_json(
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
    write_json(path / "provenance.json", {"provenance_version": "imu-rf-features-v1"})
    (path / "manifest.csv").write_text("sample_id,split\n", encoding="utf-8")
    return path


def write_rf_config(path: Path) -> Path:
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
    write_json(path, payload)
    return path
