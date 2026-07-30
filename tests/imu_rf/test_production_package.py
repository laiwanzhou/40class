from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.features.imu_rf_features import build_feature_schema
from src.training.imu_rf_production import (
    ProductionDataset,
    fit_production_imputer,
    train_production_package,
    validate_production_package,
    validate_production_package_size,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _dataset() -> ProductionDataset:
    rng = np.random.default_rng(99)
    schema = build_feature_schema()
    count = 80
    raw = rng.normal(size=(count, 2310)).astype(np.float64)
    labels = np.tile(np.arange(40, dtype=np.int64), 2)
    raw[:, 0] += labels * 0.2
    sample_ids = np.asarray([f"sample-{index:03d}" for index in range(count)])
    names = schema["feature_names"]
    imputer = fit_production_imputer(raw, sample_ids.tolist(), names)
    class_order = [
        {"class_id": index, "class_name": f"class-{index}", "label_index": index}
        for index in range(40)
    ]
    manifest = {
        "manifest_version": "imu-rf-production-training-data-v1",
        "sample_count": count,
        "sample_id_sha256": "a" * 64,
        "label_vector_sha256": "b" * 64,
        "class_counts": {str(index): 2 for index in range(40)},
        "class_order": class_order,
        "class_order_sha256": "c" * 64,
        "train_sample_id_sha256": "d" * 64,
        "validation_sample_id_sha256": "e" * 64,
        "selected_sample_id_sha256": "a" * 64,
        "union_verification": "exact_match",
        "train_count": 40,
        "validation_count": 40,
        "overlap_count": 0,
        "excluded_samples": 0,
        "stage2_contract_sha256": "f" * 64,
        "stage2_manifest_sha256": "1" * 64,
        "training_index_sha256": "2" * 64,
        "feature_schema_sha256": "3" * 64,
        "feature_reconstruction": "formal_fold0_exact_match",
    }
    return ProductionDataset(
        sample_ids=sample_ids,
        user_ids=np.asarray(["user"] * count),
        labels=labels,
        raw_features=raw,
        features=raw.copy(),
        feature_schema={**schema, "class_order": class_order, "class_order_sha256": "c" * 64},
        class_order=class_order,
        imputer=imputer,
        training_data_manifest=manifest,
    )


def test_production_package_publishes_exact_nine_files_and_reloads(tmp_path: Path) -> None:
    output = tmp_path / "production"
    result = train_production_package(
        dataset=_dataset(),
        config_path=REPOSITORY_ROOT / "configs" / "imu_rf_final_v1.json",
        output_dir=output,
        creation_commit="deadbeef",
    )
    assert result["status"] == "success"
    assert result["package_total_mib"] <= 8
    assert {path.name for path in output.iterdir()} == {
        "model.joblib",
        "feature_schema.json",
        "imputer.json",
        "class_order.json",
        "resolved_config.json",
        "inference_metadata.json",
        "training_data_manifest.json",
        "training_summary.json",
        "package_manifest.json",
    }
    loaded = validate_production_package(output)
    assert loaded["model"].n_estimators == 150
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    diagnostics = summary["resubstitution_diagnostics"]
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["not_generalization_metric"] is True
    assert diagnostics["not_for_model_selection"] is True
    with pytest.raises(FileExistsError):
        train_production_package(
            dataset=_dataset(),
            config_path=REPOSITORY_ROOT / "configs" / "imu_rf_final_v1.json",
            output_dir=output,
            creation_commit="deadbeef",
        )


def test_production_package_size_gate_is_hard_eight_mib() -> None:
    validate_production_package_size(8 * 1048576)
    with pytest.raises(ValueError, match="8 MiB"):
        validate_production_package_size(8 * 1048576 + 1)


def test_production_cli_help_runs_outside_repository(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "train_imu_rf_production.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for option in (
        "--stage2-root",
        "--training-index-dir",
        "--feature-root",
        "--output-dir",
        "--preflight-only",
    ):
        assert option in completed.stdout
