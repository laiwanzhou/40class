from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_imu_training_index import generate_training_index_artifacts
from src.data.imu_stage2_io import build_stage2_schema
from src.features.imu_rf_features import build_rf_feature_artifacts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_npz(path: Path, base: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.empty((3, 5, 16), dtype=np.float32)
    for time in range(3):
        for sensor in range(5):
            values[time, sensor] = base + time + sensor + np.arange(16) / 100
    np.savez(
        path,
        values=values,
        sensor_mask=np.ones(5, dtype=bool),
        valid_mask=np.ones((3, 5), dtype=bool),
        timestamps_ms=np.arange(3, dtype=np.int64) * 100,
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    rows = [
        {
            "sample_id": "train-sample",
            "class_id": "10",
            "class_name": "Ten",
            "user_id": "train-user",
            "action_id": "a",
            "stage2_npz_relpath": "train/imu_stage2.npz",
            "status": "success",
            "imu_usable": "True",
            "sensor_mask": "LL;RL;LA;RA;C",
            "usable_sensor_mask": "LL;RL;LA;RA;C",
        },
        {
            "sample_id": "validation-sample",
            "class_id": "30",
            "class_name": "Thirty",
            "user_id": "validation-user",
            "action_id": "b",
            "stage2_npz_relpath": "validation/imu_stage2.npz",
            "status": "success",
            "imu_usable": "True",
            "sensor_mask": "LL;RL;LA;RA;C",
            "usable_sensor_mask": "LL;RL;LA;RA;C",
        },
    ]
    pd.DataFrame(rows).to_csv(stage2 / "manifest.csv", index=False, encoding="utf-8-sig")
    schema = build_stage2_schema(
        {
            "implementation_version": "fixture",
            "generator_script": "scripts/preprocess_imu_stage2.py",
            "git_commit": "0" * 40,
            "created_at": "2026-07-29T00:00:00Z",
            "source_stage1_manifest": "manifest.csv",
            "source_stage1_manifest_sha256": "a" * 64,
        }
    )
    (stage2 / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    _write_npz(stage2 / "train" / "imu_stage2.npz", 1.0)
    _write_npz(stage2 / "validation" / "imu_stage2.npz", 1000.0)
    split = tmp_path / "fold.json"
    split.write_text(
        json.dumps({"fold": 0, "train_users": ["train-user"], "val_users": ["validation-user"]}),
        encoding="utf-8",
    )
    index = tmp_path / "index"
    generate_training_index_artifacts(stage2 / "manifest.csv", index, split, repository_root=tmp_path)
    normalization = tmp_path / "normalization"
    normalization.mkdir()
    (normalization / "unused.txt").write_text("RF does not standardize\n", encoding="utf-8")
    return stage2, index, normalization


def test_feature_builder_publishes_six_bound_artifacts_with_train_only_imputer(
    tmp_path: Path,
) -> None:
    stage2, index, normalization = _inputs(tmp_path)
    output = tmp_path / "features"
    summary = build_rf_feature_artifacts(
        stage2_root=stage2,
        training_index_dir=index,
        normalization_dir=normalization,
        output_dir=output,
        preflight_only=False,
        repository_root=REPOSITORY_ROOT,
    )
    assert {path.name for path in output.iterdir()} == {
        "train_features.npz",
        "validation_features.npz",
        "feature_schema.json",
        "imputer.json",
        "manifest.csv",
        "provenance.json",
    }
    imputer = json.loads((output / "imputer.json").read_text(encoding="utf-8"))
    assert imputer["fit_split"] == "train"
    assert imputer["fit_sample_count"] == 1
    with np.load(output / "train_features.npz", allow_pickle=False) as train:
        assert train["sample_ids"].tolist() == ["train-sample"]
        assert np.isfinite(train["features"]).all()
    with np.load(output / "validation_features.npz", allow_pickle=False) as validation:
        assert validation["sample_ids"].tolist() == ["validation-sample"]
        assert np.isfinite(validation["features"]).all()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_stage2_tree_sha256"]
    assert provenance["training_index_sha256"]
    assert provenance["normalization_tree_sha256"]
    assert summary["train_samples"] == 1
    assert summary["validation_samples"] == 1


def test_feature_builder_preflight_is_zero_write_and_fresh_publish_refuses_overwrite(
    tmp_path: Path,
) -> None:
    stage2, index, normalization = _inputs(tmp_path)
    output = tmp_path / "features"
    summary = build_rf_feature_artifacts(
        stage2_root=stage2,
        training_index_dir=index,
        normalization_dir=normalization,
        output_dir=output,
        preflight_only=True,
        repository_root=REPOSITORY_ROOT,
    )
    assert summary["feature_matrix_finite"] is True
    assert not output.exists()
    build_rf_feature_artifacts(
        stage2_root=stage2,
        training_index_dir=index,
        normalization_dir=normalization,
        output_dir=output,
        preflight_only=False,
        repository_root=REPOSITORY_ROOT,
    )
    with pytest.raises(FileExistsError):
        build_rf_feature_artifacts(
            stage2_root=stage2,
            training_index_dir=index,
            normalization_dir=normalization,
            output_dir=output,
            preflight_only=False,
            repository_root=REPOSITORY_ROOT,
        )
