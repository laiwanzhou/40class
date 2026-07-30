from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.imu_stage2_contracts import DataStatus, sha256_file
from src.data.imu_stage2_io import load_and_validate_npz
from src.features.imu_rf_features import extract_summary_features
from src.training.imu_rf_finalization import _staged_directory, directory_snapshot
from src.training.imu_rf_production import (
    apply_production_imputer,
    validate_production_package,
)
from src.training.imu_rf_trainer import _class_order_array

def load_imu_rf_package(model_dir: Path) -> dict[str, object]:
    return validate_production_package(model_dir)


def predict_stage2_records(
    *, package: dict[str, object], records: pd.DataFrame, stage2_root: Path
) -> dict[str, np.ndarray]:
    required = {"sample_id", "stage2_npz_relpath", "status"}
    if not required.issubset(records.columns):
        raise ValueError("IMU RF inference index columns are missing")
    stage2_root = Path(stage2_root).resolve(strict=True)
    sample_ids = records["sample_id"].astype(str).tolist()
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("IMU RF inference sample IDs must be non-empty and unique")
    rows: list[np.ndarray] = []
    for record in records.to_dict(orient="records"):
        relative = Path(str(record["stage2_npz_relpath"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("IMU RF inference Stage 2 path is unsafe")
        artifact = (stage2_root / relative).resolve(strict=True)
        try:
            artifact.relative_to(stage2_root)
        except ValueError as error:
            raise ValueError("IMU RF inference Stage 2 path escapes root") from error
        result = load_and_validate_npz(
            artifact,
            sample_id=str(record["sample_id"]),
            status=DataStatus(str(record["status"])),
            qc={},
        )
        rows.append(extract_summary_features(result.values, result.valid_mask, result.timestamps_ms))
    raw = np.stack(rows).astype(np.float64, copy=False) if rows else np.empty((0, 2310))
    names = package["schema"]["feature_names"]
    features = apply_production_imputer(raw, package["imputer"], names)
    model = package["model"]
    partial = model.predict_proba(features)
    probabilities = np.zeros((len(records), 40), dtype=np.float64)
    probabilities[:, model.classes_.astype(np.int64)] = partial
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    if (
        probabilities.shape != (len(records), 40)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not np.array_equal(predictions, np.argmax(probabilities, axis=1))
    ):
        raise ValueError("IMU RF inference probability contract mismatch")
    return {
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        "input_positions": np.arange(len(sample_ids), dtype=np.int64),
        "predictions": predictions,
        "class_probabilities": probabilities,
        "class_order": _class_order_array(package["class_order"]),
    }


def run_imu_rf_inference(
    *,
    model_dir: Path,
    input_index: Path,
    stage2_root: Path,
    output_dir: Path,
    split: str | None = None,
) -> dict[str, object]:
    package = load_imu_rf_package(model_dir)
    input_index = Path(input_index).resolve(strict=True)
    records = pd.read_csv(input_index, encoding="utf-8-sig", keep_default_na=False)
    if split is not None:
        if "split" not in records.columns:
            raise ValueError("Inference split requested but index has no split column")
        records = records.loc[records["split"].astype(str) == split].copy()
    if "selected_for_run" in records.columns:
        selected = records["selected_for_run"].map(
            lambda value: value is True
            or value == 1
            or str(value).strip().lower() == "true"
        )
        records = records.loc[selected].copy()
    records = records.reset_index(drop=True)
    arrays = predict_stage2_records(package=package, records=records, stage2_root=stage2_root)
    fieldnames = ["input_position", "sample_id", "prediction"] + [
        f"probability_{label}" for label in range(40)
    ]
    with _staged_directory(output_dir) as staging:
        with (staging / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for index, sample_id in enumerate(arrays["sample_ids"]):
                writer.writerow(
                    {
                        "input_position": index,
                        "sample_id": str(sample_id),
                        "prediction": int(arrays["predictions"][index]),
                        **{
                            f"probability_{label}": repr(
                                float(arrays["class_probabilities"][index, label])
                            )
                            for label in range(40)
                        },
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        np.savez(staging / "outputs.npz", **arrays)
        metadata = {
            "inference_output_version": "imu-rf-inference-output-v1",
            "model_package_canonical_sha256": package["package_manifest"][
                "package_canonical_sha256"
            ],
            "input_index_sha256": sha256_file(input_index),
            "input_index_path": str(input_index),
            "input_order": "input_positions restore source CSV row order after optional split filter",
            "split_filter": split,
            "sample_count": len(records),
            "class_count": 40,
        }
        with (staging / "inference_metadata.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(metadata, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        members = [
            {
                "relative_path": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.iterdir(), key=lambda value: value.name)
            if path.is_file()
        ]
        with (staging / "output_manifest.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                {"manifest_version": "imu-rf-inference-output-manifest-v1", "files": members},
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(staging / "outputs.npz", allow_pickle=False) as archive:
            reopened = {name: archive[name].copy() for name in archive.files}
        for name, expected in arrays.items():
            if not np.array_equal(reopened[name], expected):
                raise ValueError(f"IMU RF inference reopened {name} mismatch")
        csv_rows = list(
            csv.DictReader((staging / "predictions.csv").open(encoding="utf-8"))
        )
        if len(csv_rows) != len(records):
            raise ValueError("IMU RF inference CSV row count mismatch")
        for index, row in enumerate(csv_rows):
            restored = np.asarray(
                [float(row[f"probability_{label}"]) for label in range(40)],
                dtype=np.float64,
            )
            if (
                row["sample_id"] != str(arrays["sample_ids"][index])
                or int(row["prediction"]) != int(arrays["predictions"][index])
                or not np.array_equal(restored, arrays["class_probabilities"][index])
            ):
                raise ValueError("IMU RF inference CSV/NPZ mismatch")
    snapshot = directory_snapshot(output_dir)
    return {"status": "success", "sample_count": len(records), **snapshot}
