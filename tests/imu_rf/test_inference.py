from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.features.imu_rf_features import build_feature_schema
from src.inference.imu_rf_inference import (
    load_imu_rf_package,
    predict_stage2_records,
    run_imu_rf_inference,
)
from src.training.imu_rf_production import (
    ProductionDataset,
    fit_production_imputer,
    train_production_package,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _production_package(tmp_path: Path) -> Path:
    rng = np.random.default_rng(101)
    schema = build_feature_schema()
    raw = rng.normal(size=(80, 2310)).astype(np.float64)
    labels = np.tile(np.arange(40, dtype=np.int64), 2)
    sample_ids = np.asarray([f"fit-{index:03d}" for index in range(80)])
    class_order = [
        {"class_id": index, "class_name": f"class-{index}", "label_index": index}
        for index in range(40)
    ]
    imputer = fit_production_imputer(raw, sample_ids.tolist(), schema["feature_names"])
    manifest = {
        "manifest_version": "imu-rf-production-training-data-v1",
        "sample_count": 80,
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
    dataset = ProductionDataset(
        sample_ids=sample_ids,
        user_ids=np.asarray(["user"] * 80),
        labels=labels,
        raw_features=raw,
        features=raw,
        feature_schema={**schema, "class_order": class_order, "class_order_sha256": "c" * 64},
        class_order=class_order,
        imputer=imputer,
        training_data_manifest=manifest,
    )
    output = tmp_path / "package"
    train_production_package(
        dataset=dataset,
        config_path=REPOSITORY_ROOT / "configs" / "imu_rf_final_v1.json",
        output_dir=output,
        creation_commit="deadbeef",
    )
    return output


def _write_stage2(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True)
    values = np.full((3, 5, 16), value, dtype=np.float32)
    valid = np.ones((3, 5), dtype=np.bool_)
    np.savez(
        path,
        values=values,
        sensor_mask=np.ones(5, dtype=np.bool_),
        valid_mask=valid,
        timestamps_ms=np.asarray([0, 100, 200], dtype=np.int64),
    )


def test_inference_reloads_package_and_publishes_consistent_csv_npz(tmp_path: Path) -> None:
    package = _production_package(tmp_path)
    stage2 = tmp_path / "stage2"
    _write_stage2(stage2 / "b" / "imu_stage2.npz", 2.0)
    _write_stage2(stage2 / "a" / "imu_stage2.npz", 1.0)
    index = tmp_path / "index.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "stage2_npz_relpath", "status"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sample_id": "sample-b",
                    "stage2_npz_relpath": "b/imu_stage2.npz",
                    "status": "success",
                },
                {
                    "sample_id": "sample-a",
                    "stage2_npz_relpath": "a/imu_stage2.npz",
                    "status": "success",
                },
            ]
        )
    output = tmp_path / "inference"
    result = run_imu_rf_inference(
        model_dir=package,
        input_index=index,
        stage2_root=stage2,
        output_dir=output,
    )
    assert result["sample_count"] == 2
    assert {path.name for path in output.iterdir()} == {
        "predictions.csv",
        "outputs.npz",
        "inference_metadata.json",
        "output_manifest.json",
    }
    rows = list(csv.DictReader((output / "predictions.csv").open(encoding="utf-8")))
    assert [row["sample_id"] for row in rows] == ["sample-b", "sample-a"]
    with np.load(output / "outputs.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "sample_ids",
            "input_positions",
            "predictions",
            "class_probabilities",
            "class_order",
        }
        probabilities = archive["class_probabilities"]
        assert probabilities.shape == (2, 40)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(
            archive["predictions"], np.argmax(probabilities, axis=1)
        )
        for row_index, row in enumerate(rows):
            csv_probabilities = np.asarray(
                [float(row[f"probability_{label}"]) for label in range(40)]
            )
            np.testing.assert_array_equal(csv_probabilities, probabilities[row_index])
    with pytest.raises(FileExistsError):
        run_imu_rf_inference(
            model_dir=package,
            input_index=index,
            stage2_root=stage2,
            output_dir=output,
        )


def test_package_loader_rejects_member_tampering(tmp_path: Path) -> None:
    package = _production_package(tmp_path)
    (package / "imputer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_imu_rf_package(package)


def test_prediction_interface_accepts_validated_fold0_train_only_imputer(
    tmp_path: Path,
) -> None:
    package_dir = _production_package(tmp_path)
    package = load_imu_rf_package(package_dir)
    production_imputer = package["imputer"]
    package["imputer"] = {
        "imputer_version": "imu-rf-median-imputer-v1",
        "fit_split": "train",
        "fit_sample_count": 80,
        "fit_sample_id_sha256": "a" * 64,
        "feature_schema_version": "imu-rf-summary-v1",
        "feature_count": 2310,
        "all_missing_features": production_imputer["all_missing_features"],
        "medians": production_imputer["medians"],
    }
    stage2 = tmp_path / "stage2-reference"
    _write_stage2(stage2 / "a" / "imu_stage2.npz", 1.0)
    records = __import__("pandas").DataFrame(
        [
            {
                "sample_id": "sample-a",
                "stage2_npz_relpath": "a/imu_stage2.npz",
                "status": "success",
            }
        ]
    )

    arrays = predict_stage2_records(
        package=package, records=records, stage2_root=stage2
    )

    assert arrays["class_probabilities"].shape == (1, 40)
    assert np.isfinite(arrays["class_probabilities"]).all()


def test_inference_cli_help_runs_outside_repository(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_imu_rf_inference.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for option in ("--model-dir", "--input-index", "--stage2-root", "--output-dir"):
        assert option in completed.stdout
