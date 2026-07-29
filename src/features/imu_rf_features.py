from __future__ import annotations

import hashlib
import math
import csv
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from scripts.build_imu_training_index import hash_training_index, load_class_order
from src.data.imu_stage2_contracts import (
    FEATURE_ORDER,
    SENSOR_ORDER,
    DataStatus,
    canonical_json_bytes,
    sha256_file,
)
from src.data.imu_stage2_io import load_and_validate_npz, load_stage2_schema


FEATURE_SCHEMA_VERSION = "imu-rf-summary-v1"
IMPUTER_VERSION = "imu-rf-median-imputer-v1"

CHANNEL_STATISTICS = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "median",
    "q25",
    "q75",
    "peak_to_peak",
    "rms",
    "first_last_delta",
    "mean_absolute_first_difference",
    "first_difference_std",
    "valid_count",
    "valid_ratio",
)
SENSOR_STATISTICS = (
    "valid_span_seconds",
    "valid_timepoint_count",
    "invalid_ratio",
    "valid_segment_count",
    "longest_invalid_run",
    "sensor_missing",
)
SAMPLE_STATISTICS = (
    "usable_sensor_count",
    "all_sensor_valid_ratio",
    "sequence_length",
    "duration_seconds",
    "overall_missing_ratio",
)


def _with_missing_indicators(base_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for name in base_names:
        names.extend((name, f"{name}__missing"))
    return names


def build_feature_schema() -> dict[str, object]:
    base_names = [
        f"{sensor}__{channel}__{statistic}"
        for sensor in SENSOR_ORDER
        for channel in FEATURE_ORDER
        for statistic in CHANNEL_STATISTICS
    ]
    base_names.extend(
        f"{sensor}__{statistic}"
        for sensor in SENSOR_ORDER
        for statistic in SENSOR_STATISTICS
    )
    base_names.extend(f"sample__{statistic}" for statistic in SAMPLE_STATISTICS)
    names = _with_missing_indicators(base_names)
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "sensor_order": list(SENSOR_ORDER),
        "channel_order": list(FEATURE_ORDER),
        "feature_names": names,
        "feature_count": len(names),
        "imputation": "train-only-median-with-zero-for-all-missing-and-explicit-indicator",
    }


def _append_scalar(output: list[float], value: float) -> None:
    missing = not np.isfinite(value)
    output.extend((float(value) if not missing else math.nan, float(missing)))


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask.tolist():
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _true_segments(mask: np.ndarray) -> int:
    if not mask.size:
        return 0
    starts = mask & np.concatenate((np.asarray([True]), ~mask[:-1]))
    return int(starts.sum())


def _channel_statistics(values: np.ndarray, total_count: int) -> tuple[float, ...]:
    count = int(values.size)
    if count:
        differences = np.diff(values)
        return (
            float(np.mean(values, dtype=np.float64)),
            float(np.std(values, dtype=np.float64)),
            float(np.min(values)),
            float(np.max(values)),
            float(np.median(values)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
            float(np.ptp(values)),
            float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))),
            float(values[-1] - values[0]),
            float(np.mean(np.abs(differences), dtype=np.float64)) if count > 1 else math.nan,
            float(np.std(differences, dtype=np.float64)) if count > 1 else math.nan,
            float(count),
            float(count / total_count),
        )
    return (math.nan,) * 12 + (0.0, 0.0)


def extract_summary_features(
    values: np.ndarray,
    valid_mask: np.ndarray,
    timestamps_ms: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values)
    valid_mask = np.asarray(valid_mask)
    timestamps_ms = np.asarray(timestamps_ms)
    if (
        values.dtype != np.float32
        or values.ndim != 3
        or values.shape[1:] != (len(SENSOR_ORDER), len(FEATURE_ORDER))
        or valid_mask.dtype != np.bool_
        or valid_mask.shape != values.shape[:2]
        or timestamps_ms.dtype != np.int64
        or timestamps_ms.shape != (values.shape[0],)
        or values.shape[0] < 1
    ):
        raise ValueError("Stage 2 summary input shape or dtype mismatch")
    if timestamps_ms[0] != 0 or (
        len(timestamps_ms) > 1 and np.any(np.diff(timestamps_ms) <= 0)
    ):
        raise ValueError("Stage 2 summary timestamps are invalid")
    if not np.isfinite(values[valid_mask]).all():
        raise ValueError("Valid Stage 2 cells must be finite")

    output: list[float] = []
    total_count = values.shape[0]
    for sensor_index in range(len(SENSOR_ORDER)):
        sensor_valid = valid_mask[:, sensor_index]
        for channel_index in range(len(FEATURE_ORDER)):
            observed = values[sensor_valid, sensor_index, channel_index].astype(
                np.float64, copy=False
            )
            for statistic in _channel_statistics(observed, total_count):
                _append_scalar(output, statistic)
    for sensor_index in range(len(SENSOR_ORDER)):
        sensor_valid = valid_mask[:, sensor_index]
        valid_indices = np.flatnonzero(sensor_valid)
        valid_span = (
            float(timestamps_ms[valid_indices[-1]] - timestamps_ms[valid_indices[0]]) / 1000.0
            if valid_indices.size
            else math.nan
        )
        sensor_values = (
            valid_span,
            float(sensor_valid.sum()),
            float(1.0 - sensor_valid.mean()),
            float(_true_segments(sensor_valid)),
            float(_longest_false_run(sensor_valid)),
            float(not sensor_valid.any()),
        )
        for statistic in sensor_values:
            _append_scalar(output, statistic)
    sample_values = (
        float(valid_mask.any(axis=0).sum()),
        float(valid_mask.all(axis=1).mean()),
        float(total_count),
        float(timestamps_ms[-1] - timestamps_ms[0]) / 1000.0,
        float(1.0 - valid_mask.mean()),
    )
    for statistic in sample_values:
        _append_scalar(output, statistic)
    result = np.asarray(output, dtype=np.float64)
    expected = int(build_feature_schema()["feature_count"])
    if result.shape != (expected,):
        raise AssertionError("RF feature schema and extractor disagree")
    return result


def fit_median_imputer(
    train_features: np.ndarray,
    train_sample_ids: Sequence[str],
) -> dict[str, object]:
    train_features = np.asarray(train_features, dtype=np.float64)
    if train_features.ndim != 2 or train_features.shape[0] != len(train_sample_ids):
        raise ValueError("Train feature matrix and sample IDs are misaligned")
    schema = build_feature_schema()
    if train_features.shape[1] != schema["feature_count"]:
        raise ValueError("Train feature count disagrees with schema")
    sample_ids = list(map(str, train_sample_ids))
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Train sample IDs must be non-empty and unique")
    medians: list[float] = []
    all_missing: list[str] = []
    feature_names = list(schema["feature_names"])
    for index, name in enumerate(feature_names):
        finite = train_features[np.isfinite(train_features[:, index]), index]
        if finite.size:
            medians.append(float(np.median(finite)))
        else:
            medians.append(0.0)
            all_missing.append(name)
    payload = "".join(f"{sample_id}\n" for sample_id in sorted(sample_ids)).encode("utf-8")
    return {
        "imputer_version": IMPUTER_VERSION,
        "fit_split": "train",
        "fit_sample_count": len(sample_ids),
        "fit_sample_id_sha256": hashlib.sha256(payload).hexdigest(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(feature_names),
        "all_missing_features": all_missing,
        "medians": medians,
    }


def apply_median_imputer(
    features: np.ndarray,
    imputer: Mapping[str, object],
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional")
    expected_keys = {
        "imputer_version",
        "fit_split",
        "fit_sample_count",
        "fit_sample_id_sha256",
        "feature_schema_version",
        "feature_count",
        "all_missing_features",
        "medians",
    }
    if set(imputer) != expected_keys or imputer.get("imputer_version") != IMPUTER_VERSION:
        raise ValueError("RF imputer contract mismatch")
    if imputer.get("fit_split") != "train" or imputer.get(
        "feature_schema_version"
    ) != FEATURE_SCHEMA_VERSION:
        raise ValueError("RF imputer provenance mismatch")
    medians = np.asarray(imputer["medians"], dtype=np.float64)
    if medians.shape != (features.shape[1],) or imputer.get("feature_count") != features.shape[1]:
        raise ValueError("RF imputer feature count mismatch")
    if not np.isfinite(medians).all():
        raise ValueError("RF imputer medians must be finite")
    result = features.copy()
    missing_rows, missing_columns = np.where(~np.isfinite(result))
    result[missing_rows, missing_columns] = medians[missing_columns]
    if not np.isfinite(result).all():
        raise ValueError("Imputed RF feature matrix must be finite")
    return result


def build_rf_feature_artifacts(
    *,
    stage2_root: Path,
    training_index_dir: Path,
    normalization_dir: Path,
    output_dir: Path,
    preflight_only: bool,
    repository_root: Path,
) -> dict[str, object]:
    stage2_root = Path(stage2_root).resolve(strict=True)
    training_index_dir = Path(training_index_dir).resolve(strict=True)
    normalization_dir = Path(normalization_dir).resolve(strict=True)
    repository_root = Path(repository_root).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists() and not preflight_only:
        raise FileExistsError(output_dir)
    frame = pd.read_csv(
        training_index_dir / "training_index.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    metadata = _load_json(training_index_dir / "training_index.json")
    if metadata.get("training_index_sha256") != hash_training_index(frame):
        raise ValueError("RF source training index hash mismatch")
    if metadata.get("fold") != 0 or metadata.get("source_stage2_manifest_path") != "manifest.csv":
        raise ValueError("RF source training-index fold or manifest path mismatch")
    if metadata.get("source_stage2_manifest_sha256") != sha256_file(
        stage2_root / "manifest.csv"
    ):
        raise ValueError("RF source Stage 2 manifest hash mismatch")
    stage2_schema = load_stage2_schema(stage2_root / "schema.json")
    if metadata.get("stage2_contract_sha256") != stage2_schema.get("contract_sha256"):
        raise ValueError("RF source Stage 2 contract mismatch")
    class_order_contract = load_class_order(training_index_dir / "class_order.json")
    if metadata.get("class_order_sha256") != class_order_contract.class_order_sha256:
        raise ValueError("RF source class order mismatch")
    selected = frame[frame["selected_for_run"].map(_as_bool)].copy()
    if set(selected["split"].astype(str)) != {"train", "validation"}:
        raise ValueError("RF selected rows must contain train and validation")
    if selected["sample_id"].astype(str).duplicated().any():
        raise ValueError("RF selected sample IDs must be unique")
    split_payloads: dict[str, dict[str, np.ndarray]] = {}
    raw_by_split: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, object]] = []
    for split in ("train", "validation"):
        rows = selected[selected["split"].astype(str) == split].copy()
        rows = rows.sort_values("sample_id", kind="stable")
        raw_rows: list[np.ndarray] = []
        sample_ids: list[str] = []
        user_ids: list[str] = []
        labels: list[int] = []
        for record in rows.to_dict(orient="records"):
            relative = Path(str(record["stage2_npz_relpath"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("RF Stage 2 feature source path is unsafe")
            artifact = (stage2_root / relative).resolve(strict=True)
            try:
                artifact.relative_to(stage2_root)
            except ValueError as error:
                raise ValueError("RF Stage 2 feature source escapes root") from error
            result = load_and_validate_npz(
                artifact,
                sample_id=str(record["sample_id"]),
                status=DataStatus(str(record["status"])),
                qc={},
            )
            raw_rows.append(
                extract_summary_features(result.values, result.valid_mask, result.timestamps_ms)
            )
            sample_ids.append(str(record["sample_id"]))
            user_ids.append(str(record["user_id"]))
            labels.append(int(record["label_index"]))
            manifest_rows.append(
                {
                    "sample_id": str(record["sample_id"]),
                    "split": split,
                    "user_id": str(record["user_id"]),
                    "label_index": int(record["label_index"]),
                    "stage2_npz_relpath": relative.as_posix(),
                }
            )
        raw_matrix = np.stack(raw_rows).astype(np.float64, copy=False)
        raw_by_split[split] = raw_matrix
        split_payloads[split] = {
            "sample_ids": np.asarray(sample_ids, dtype=np.str_),
            "user_ids": np.asarray(user_ids, dtype=np.str_),
            "labels": np.asarray(labels, dtype=np.int64),
        }
    imputer = fit_median_imputer(
        raw_by_split["train"], split_payloads["train"]["sample_ids"].tolist()
    )
    for split in ("train", "validation"):
        split_payloads[split]["features"] = apply_median_imputer(raw_by_split[split], imputer)
    schema = build_feature_schema()
    schema["class_order"] = [dict(record) for record in class_order_contract.classes]
    schema["class_order_sha256"] = class_order_contract.class_order_sha256
    summary = {
        "status": "preflight_ok" if preflight_only else "written",
        "train_samples": len(split_payloads["train"]["labels"]),
        "validation_samples": len(split_payloads["validation"]["labels"]),
        "selected_samples": len(selected),
        "classes": class_order_contract.num_classes,
        "omitted_train": 0,
        "omitted_validation": 0,
        "feature_count": int(schema["feature_count"]),
        "feature_matrix_finite": bool(
            np.isfinite(split_payloads["train"]["features"]).all()
            and np.isfinite(split_payloads["validation"]["features"]).all()
        ),
        "leakage_fields_present": False,
    }
    if preflight_only:
        return summary
    with _staged_output(output_dir) as staging:
        np.savez(staging / "train_features.npz", **split_payloads["train"])
        np.savez(staging / "validation_features.npz", **split_payloads["validation"])
        _write_json(staging / "feature_schema.json", schema)
        _write_json(staging / "imputer.json", imputer)
        _write_manifest(staging / "manifest.csv", manifest_rows)
        member_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
        ).strip()
        provenance = {
            "provenance_version": "imu-rf-features-v1",
            "git_commit": git_commit,
            "source_stage2_tree_sha256": canonical_tree_hash(stage2_root),
            "source_stage2_manifest_sha256": metadata["source_stage2_manifest_sha256"],
            "training_index_sha256": metadata["training_index_sha256"],
            "training_index_tree_sha256": canonical_tree_hash(training_index_dir),
            "normalization_tree_sha256": canonical_tree_hash(normalization_dir),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_count": schema["feature_count"],
            "train_sample_id_sha256": imputer["fit_sample_id_sha256"],
            "imputer_fit_split": "train",
            "artifact_sha256": member_hashes,
            "canonical_member_tree_sha256": _member_hash(member_hashes),
        }
        _write_json(staging / "provenance.json", provenance)
        _validate_feature_output(staging)
    _validate_feature_output(output_dir)
    return summary


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError("RF training-index boolean is invalid")


def _load_json(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("RF JSON payload must be an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = ["sample_id", "split", "user_id", "label_index", "stage2_npz_relpath"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _member_hash(member_hashes: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes({"files": dict(sorted(member_hashes.items()))})).hexdigest()


def canonical_tree_hash(root: Path) -> str:
    root = Path(root).resolve(strict=True)
    records = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return hashlib.sha256(canonical_json_bytes({"files": records})).hexdigest()


@contextmanager
def _staged_output(output_dir: Path):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    residues = list(output_dir.parent.glob(f".{output_dir.name}.staging-*"))
    if residues:
        raise FileExistsError("Unknown RF feature staging residue exists")
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_feature_output(root: Path) -> None:
    expected = {
        "train_features.npz",
        "validation_features.npz",
        "feature_schema.json",
        "imputer.json",
        "manifest.csv",
        "provenance.json",
    }
    if {path.name for path in root.iterdir() if path.is_file()} != expected:
        raise ValueError("RF feature artifact file set mismatch")
    schema = _load_json(root / "feature_schema.json")
    feature_count = int(schema["feature_count"])
    for split in ("train", "validation"):
        with np.load(root / f"{split}_features.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"sample_ids", "user_ids", "labels", "features"}:
                raise ValueError("RF feature NPZ keys mismatch")
            if archive["features"].dtype != np.float64 or archive["features"].shape[1] != feature_count:
                raise ValueError("RF feature NPZ shape mismatch")
            if not np.isfinite(archive["features"]).all():
                raise ValueError("RF feature NPZ must be finite")
    provenance = _load_json(root / "provenance.json")
    member_hashes = provenance.get("artifact_sha256")
    if not isinstance(member_hashes, dict):
        raise ValueError("RF feature provenance artifact hashes are missing")
    for name, expected_hash in member_hashes.items():
        if sha256_file(root / name) != expected_hash:
            raise ValueError("RF feature artifact SHA-256 mismatch")
    if provenance.get("canonical_member_tree_sha256") != _member_hash(member_hashes):
        raise ValueError("RF feature canonical member tree mismatch")
