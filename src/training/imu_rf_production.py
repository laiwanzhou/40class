from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from scripts.build_imu_training_index import (
    hash_training_index,
    load_class_order,
    validate_training_index_metadata,
)
from src.data.imu_stage2_contracts import DataStatus, canonical_json_bytes, sha256_file
from src.data.imu_stage2_io import load_and_validate_npz, load_stage2_schema
from src.features.imu_rf_features import (
    apply_median_imputer,
    build_feature_schema,
    extract_summary_features,
)
from src.training.imu_rf_trainer import _load_features
from src.training.imu_rf_compact import forest_structure_metrics
from src.training.imu_rf_finalization import (
    FINAL_CONFIG,
    directory_snapshot,
    load_final_config,
    validate_frozen_estimator,
)
from src.training.imu_stage2_trainer import classification_metrics


PRODUCTION_IMPUTER_VERSION = "imu-rf-production-median-imputer-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProductionDataset:
    sample_ids: np.ndarray
    user_ids: np.ndarray
    labels: np.ndarray
    raw_features: np.ndarray
    features: np.ndarray
    feature_schema: dict[str, object]
    class_order: list[dict[str, object]]
    imputer: dict[str, object]
    training_data_manifest: dict[str, object]


def _as_bool(value: object) -> bool:
    if value is True or value == 1 or str(value).strip().lower() == "true":
        return True
    if value is False or value == 0 or str(value).strip().lower() == "false":
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _sample_id_sha256(sample_ids: Sequence[str]) -> str:
    normalized = sorted(map(str, sample_ids))
    return hashlib.sha256("".join(f"{value}\n" for value in normalized).encode("utf-8")).hexdigest()


def _feature_names_sha256(feature_names: Sequence[str]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"feature_names": list(map(str, feature_names))})
    ).hexdigest()


def validate_production_split(
    frame: pd.DataFrame, *, selected_ids: set[str]
) -> dict[str, object]:
    required = {
        "sample_id",
        "user_id",
        "label_index",
        "selected_for_run",
        "split",
        "stage2_npz_relpath",
        "status",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Production training index columns are missing")
    selected_mask = frame["selected_for_run"].map(_as_bool)
    selected = frame.loc[selected_mask].copy()
    ids = selected["sample_id"].astype(str).tolist()
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Production sample IDs must be non-empty and unique")
    split_names = set(selected["split"].astype(str))
    if split_names != {"train", "validation"}:
        raise ValueError("Production set must contain only train and validation")
    train_ids = set(selected.loc[selected["split"].astype(str) == "train", "sample_id"].astype(str))
    validation_ids = set(
        selected.loc[selected["split"].astype(str) == "validation", "sample_id"].astype(str)
    )
    if len(train_ids) != 2184 or len(validation_ids) != 573 or len(ids) != 2757:
        raise ValueError("Production split count mismatch")
    if train_ids.intersection(validation_ids):
        raise ValueError("Production train/validation overlap is non-zero")
    if set(ids) != set(map(str, selected_ids)):
        raise ValueError("Production union differs from selected2757")
    labels = selected["label_index"]
    if labels.isna().any():
        raise ValueError("Production labels must be present")
    normalized_labels = labels.astype(np.int64)
    if not np.array_equal(normalized_labels.to_numpy(), labels.to_numpy()):
        raise ValueError("Production labels must be integers")
    if set(normalized_labels.tolist()) != set(range(40)):
        raise ValueError("Production class order must cover 40 labels")
    class_counts = {
        str(label): int((normalized_labels == label).sum()) for label in range(40)
    }
    return {
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "union_count": len(ids),
        "overlap_count": 0,
        "excluded_samples": int((~selected_mask).sum()),
        "sample_id_sha256": _sample_id_sha256(ids),
        "class_counts": class_counts,
    }


def fit_production_imputer(
    raw_features: np.ndarray,
    sample_ids: Sequence[str],
    feature_names: Sequence[str],
) -> dict[str, object]:
    raw = np.asarray(raw_features)
    names = list(map(str, feature_names))
    ids = list(map(str, sample_ids))
    if raw.dtype != np.float64 or raw.ndim != 2 or raw.shape != (len(ids), len(names)):
        raise ValueError("Production raw feature shape or dtype mismatch")
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Production imputer sample IDs must be unique")
    if any(not value for value in names) or len(names) != len(set(names)):
        raise ValueError("Production imputer feature names are invalid")
    medians: list[float] = []
    all_missing: list[str] = []
    for index, name in enumerate(names):
        finite = raw[np.isfinite(raw[:, index]), index]
        if finite.size:
            medians.append(float(np.median(finite)))
        else:
            medians.append(0.0)
            all_missing.append(name)
    return {
        "imputer_version": PRODUCTION_IMPUTER_VERSION,
        "fit_scope": "all_labeled",
        "fit_sample_count": len(ids),
        "fit_sample_id_sha256": _sample_id_sha256(ids),
        "feature_schema_version": "imu-rf-summary-v1",
        "feature_count": len(names),
        "feature_names_sha256": _feature_names_sha256(names),
        "all_missing_features": all_missing,
        "medians": medians,
    }


def apply_production_imputer(
    raw_features: np.ndarray,
    imputer: dict[str, object],
    feature_names: Sequence[str],
) -> np.ndarray:
    raw = np.asarray(raw_features)
    names = list(map(str, feature_names))
    expected_fields = {
        "imputer_version",
        "fit_scope",
        "fit_sample_count",
        "fit_sample_id_sha256",
        "feature_schema_version",
        "feature_count",
        "feature_names_sha256",
        "all_missing_features",
        "medians",
    }
    if set(imputer) != expected_fields or imputer.get("imputer_version") != PRODUCTION_IMPUTER_VERSION:
        raise ValueError("Production imputer contract mismatch")
    if imputer.get("fit_scope") != "all_labeled":
        raise ValueError("Production imputer scope mismatch")
    if (
        raw.dtype != np.float64
        or raw.ndim != 2
        or raw.shape[1] != len(names)
        or imputer.get("feature_count") != len(names)
        or imputer.get("feature_names_sha256") != _feature_names_sha256(names)
    ):
        raise ValueError("Production imputer feature names or matrix mismatch")
    medians = np.asarray(imputer["medians"], dtype=np.float64)
    if medians.shape != (len(names),) or not np.isfinite(medians).all():
        raise ValueError("Production imputer medians are invalid")
    result = raw.copy()
    rows, columns = np.where(~np.isfinite(result))
    result[rows, columns] = medians[columns]
    if not np.isfinite(result).all():
        raise ValueError("Production imputed features must be finite")
    return result


def _strict_json(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _label_sha256(labels: np.ndarray) -> str:
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    return hashlib.sha256(labels.tobytes()).hexdigest()


def build_production_dataset(
    *, stage2_root: Path, training_index_dir: Path, feature_root: Path
) -> ProductionDataset:
    stage2_root = Path(stage2_root).resolve(strict=True)
    training_index_dir = Path(training_index_dir).resolve(strict=True)
    feature_root = Path(feature_root).resolve(strict=True)
    frame = pd.read_csv(
        training_index_dir / "training_index.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    metadata = _strict_json(training_index_dir / "training_index.json")
    class_contract = load_class_order(training_index_dir / "class_order.json")
    schema = load_stage2_schema(stage2_root / "schema.json")
    split_path = REPOSITORY_ROOT / str(metadata["split_definition_path"])
    split_definition = _strict_json(split_path)
    validate_training_index_metadata(
        metadata,
        frame,
        class_contract,
        expected_source_manifest_sha256=sha256_file(stage2_root / "manifest.csv"),
        expected_stage2_contract_sha256=str(schema["contract_sha256"]),
        expected_split_definition_sha256=sha256_file(split_path),
        expected_fold=0,
        expected_split_definition_path=str(metadata["split_definition_path"]),
        expected_source_manifest_path="manifest.csv",
        expected_split_definition=split_definition,
    )
    if metadata.get("training_index_sha256") != hash_training_index(frame):
        raise ValueError("Production training index SHA-256 mismatch")
    selected_mask = frame["selected_for_run"].map(_as_bool)
    selected_ids = set(frame.loc[selected_mask, "sample_id"].astype(str))
    split_summary = validate_production_split(frame, selected_ids=selected_ids)
    if split_summary["sample_id_sha256"] != metadata.get("selected_sample_id_sha256"):
        raise ValueError("Production selected2757 hash mismatch")
    selected = frame.loc[selected_mask].copy()
    selected = selected.sort_values("sample_id", kind="stable")
    formal_artifacts = _load_features(
        feature_root,
        {
            "feature_schema_version": "imu-rf-summary-v1",
            "num_classes": 40,
        },
    )
    feature_schema = _strict_json(feature_root / "feature_schema.json")
    expected_schema = build_feature_schema()
    if (
        feature_schema.get("schema_version") != expected_schema["schema_version"]
        or feature_schema.get("feature_count") != 2310
        or feature_schema.get("feature_names") != expected_schema["feature_names"]
        or feature_schema.get("sensor_order") != expected_schema["sensor_order"]
        or feature_schema.get("channel_order") != expected_schema["channel_order"]
    ):
        raise ValueError("Production feature schema differs from imu-rf-summary-v1")
    raw_rows: list[np.ndarray] = []
    sample_ids: list[str] = []
    user_ids: list[str] = []
    labels: list[int] = []
    for record in selected.to_dict(orient="records"):
        relative = Path(str(record["stage2_npz_relpath"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Production Stage 2 path is unsafe")
        artifact_path = (stage2_root / relative).resolve(strict=True)
        try:
            artifact_path.relative_to(stage2_root)
        except ValueError as error:
            raise ValueError("Production Stage 2 path escapes root") from error
        result = load_and_validate_npz(
            artifact_path,
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
    raw = np.stack(raw_rows).astype(np.float64, copy=False)
    old_imputer = _strict_json(feature_root / "imputer.json")
    old_transformed = apply_median_imputer(raw, old_imputer)
    formal_by_id: dict[str, np.ndarray] = {}
    for split in ("train", "validation"):
        payload = formal_artifacts[split]
        formal_by_id.update(
            {
                str(sample_id): row
                for sample_id, row in zip(
                    payload["sample_ids"].tolist(), payload["features"], strict=True
                )
            }
        )
    expected_old = np.stack([formal_by_id[sample_id] for sample_id in sample_ids])
    if not np.array_equal(old_transformed, expected_old):
        mismatch = int(np.count_nonzero(old_transformed != expected_old))
        raise ValueError(f"Rebuilt imu-rf-summary-v1 features mismatch: {mismatch}")
    names = list(map(str, feature_schema["feature_names"]))
    production_imputer = fit_production_imputer(raw, sample_ids, names)
    transformed = apply_production_imputer(raw, production_imputer, names)
    label_array = np.asarray(labels, dtype=np.int64)
    manifest = {
        "manifest_version": "imu-rf-production-training-data-v1",
        "sample_count": len(sample_ids),
        "sample_id_sha256": _sample_id_sha256(sample_ids),
        "label_vector_sha256": _label_sha256(label_array),
        "class_counts": split_summary["class_counts"],
        "class_order": [dict(record) for record in class_contract.classes],
        "class_order_sha256": class_contract.class_order_sha256,
        "train_sample_id_sha256": metadata["train_sample_id_sha256"],
        "validation_sample_id_sha256": metadata["validation_sample_id_sha256"],
        "selected_sample_id_sha256": metadata["selected_sample_id_sha256"],
        "union_verification": "exact_match",
        "train_count": 2184,
        "validation_count": 573,
        "overlap_count": 0,
        "excluded_samples": split_summary["excluded_samples"],
        "stage2_contract_sha256": schema["contract_sha256"],
        "stage2_manifest_sha256": sha256_file(stage2_root / "manifest.csv"),
        "training_index_sha256": metadata["training_index_sha256"],
        "feature_schema_sha256": sha256_file(feature_root / "feature_schema.json"),
        "feature_reconstruction": "formal_fold0_exact_match",
    }
    return ProductionDataset(
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        user_ids=np.asarray(user_ids, dtype=np.str_),
        labels=label_array,
        raw_features=raw,
        features=transformed,
        feature_schema=feature_schema,
        class_order=[dict(record) for record in class_contract.classes],
        imputer=production_imputer,
        training_data_manifest=manifest,
    )


def validate_production_package_size(total_bytes: int) -> None:
    if type(total_bytes) is not int or total_bytes < 0:
        raise ValueError("Production package size is invalid")
    if total_bytes > 8 * 1048576:
        raise ValueError("Production package exceeds the hard 8 MiB limit")


def _write_json(path: Path, payload: object) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _weighted_f1(per_class: Sequence[dict[str, object]]) -> float:
    total = sum(int(record["support"]) for record in per_class)
    return (
        sum(float(record["f1"]) * int(record["support"]) for record in per_class) / total
        if total
        else 0.0
    )


def _package_member_records(staging: Path) -> list[dict[str, object]]:
    return [
        {"relative_path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(staging.iterdir(), key=lambda value: value.name)
        if path.is_file() and path.name != "package_manifest.json"
    ]


def _write_package_manifest(
    staging: Path, *, creation_commit: str, member_records: list[dict[str, object]]
) -> dict[str, object]:
    canonical = _json_sha256(
        {"package_version": "imu-rf-final-v1", "files": member_records}
    )
    manifest: dict[str, object] = {
        "manifest_version": "imu-rf-final-package-manifest-v1",
        "package_version": "imu-rf-final-v1",
        "package_file_count": 9,
        "files": member_records,
        "package_total_bytes": 0,
        "package_total_mib": "0.000000000000",
        "package_canonical_sha256": canonical,
        "creation_code_commit": creation_commit,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
    }
    previous = -1
    for _ in range(10):
        _write_json(staging / "package_manifest.json", manifest)
        total = sum(path.stat().st_size for path in staging.iterdir() if path.is_file())
        manifest["package_total_bytes"] = total
        # A fixed-width decimal avoids a self-referential JSON-size oscillation:
        # the manifest contributes to package_total_bytes, while this field
        # contributes to the manifest's own byte count.
        manifest["package_total_mib"] = f"{total / 1048576:.12f}"
        if total == previous:
            _write_json(staging / "package_manifest.json", manifest)
            final_total = sum(path.stat().st_size for path in staging.iterdir() if path.is_file())
            if final_total == int(manifest["package_total_bytes"]):
                return manifest
        previous = total
    raise RuntimeError("Production package manifest size did not stabilize")


def train_production_package(
    *, dataset: ProductionDataset, config_path: Path, output_dir: Path, creation_commit: str
) -> dict[str, object]:
    config = load_final_config(config_path)
    if dataset.features.dtype != np.float64 or dataset.features.shape != (
        len(dataset.sample_ids),
        2310,
    ):
        raise ValueError("Production feature matrix contract mismatch")
    if len(dataset.sample_ids) != len(dataset.labels) or not np.isfinite(dataset.features).all():
        raise ValueError("Production training data is invalid")
    if dataset.training_data_manifest.get("union_verification") != "exact_match":
        raise ValueError("Production union verification failed")
    model = RandomForestClassifier(**config["estimator_params"])
    started = time.perf_counter()
    model.fit(dataset.features, dataset.labels)
    duration = time.perf_counter() - started
    validate_frozen_estimator(model)
    partial = model.predict_proba(dataset.features)
    probabilities = np.zeros((len(dataset.labels), 40), dtype=np.float64)
    probabilities[:, model.classes_.astype(np.int64)] = partial
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    metrics = classification_metrics(
        labels=dataset.labels, predictions=predictions, num_classes=40
    )
    per_class = metrics["per_class"]
    feature_schema_sha = _json_sha256(dataset.feature_schema)
    imputer_sha = _json_sha256(dataset.imputer)
    class_order_payload = {
        "class_order_version": "imu-class-order-v1",
        "num_classes": 40,
        "classes": dataset.class_order,
        "class_order_sha256": dataset.training_data_manifest["class_order_sha256"],
    }
    class_order_sha = _json_sha256(class_order_payload)
    config_sha = _json_sha256(config)
    training_data_sha = _json_sha256(dataset.training_data_manifest)
    model_metadata = {
        "model_version": "imu-rf-final-v1",
        "model_role": "imu_rf_production",
        "estimator": "RandomForestClassifier",
        "feature_schema_sha256": feature_schema_sha,
        "imputer_sha256": imputer_sha,
        "class_order_file_sha256": class_order_sha,
        "training_data_manifest_sha256": training_data_sha,
        "resolved_config_sha256": config_sha,
        "random_state": 20260725,
        "feature_count": 2310,
        "class_count": 40,
    }
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    residues = list(output_dir.parent.glob(f".{output_dir.name}.staging-*"))
    if residues:
        raise FileExistsError("Unknown production package staging residue exists")
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid4().hex}"
    staging.mkdir()
    preserve_oversize = False
    try:
        joblib.dump(
            {"model": model, "metadata": model_metadata},
            staging / "model.joblib",
            compress=("lzma", 3),
        )
        _write_json(staging / "feature_schema.json", dataset.feature_schema)
        _write_json(staging / "imputer.json", dataset.imputer)
        _write_json(staging / "class_order.json", class_order_payload)
        _write_json(staging / "resolved_config.json", config)
        _write_json(
            staging / "inference_metadata.json",
            {
                "inference_metadata_version": "imu-rf-final-inference-v1",
                **model_metadata,
                "probability_order": "class_order.json label_index order",
                "input_feature_schema": "imu-rf-summary-v1",
            },
        )
        _write_json(
            staging / "training_data_manifest.json", dataset.training_data_manifest
        )
        structure = forest_structure_metrics(model)
        _write_json(
            staging / "training_summary.json",
            {
                "summary_version": "imu-rf-production-training-summary-v1",
                "status": "success",
                "sample_count": len(dataset.sample_ids),
                "feature_count": 2310,
                "class_count": 40,
                "training_duration_seconds": duration,
                "model_structure": structure,
                "resubstitution_diagnostics": {
                    "diagnostic_only": True,
                    "not_generalization_metric": True,
                    "not_for_model_selection": True,
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": _weighted_f1(per_class),
                    "zero_f1_class_count": sum(
                        float(record["f1"]) == 0.0 for record in per_class
                    ),
                },
            },
        )
        records = _package_member_records(staging)
        manifest = _write_package_manifest(
            staging, creation_commit=creation_commit, member_records=records
        )
        try:
            validate_production_package_size(int(manifest["package_total_bytes"]))
        except ValueError:
            preserve_oversize = True
            raise
        validate_production_package(staging)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and not preserve_oversize:
            shutil.rmtree(staging)
        raise
    validated = validate_production_package(output_dir)
    manifest = validated["package_manifest"]
    return {
        "status": "success",
        "model_sha256": sha256_file(output_dir / "model.joblib"),
        "imputer_sha256": sha256_file(output_dir / "imputer.json"),
        "package_canonical_sha256": manifest["package_canonical_sha256"],
        "package_total_bytes": manifest["package_total_bytes"],
        "package_total_mib": float(manifest["package_total_mib"]),
        "training_duration_seconds": duration,
    }


def validate_production_package(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir).resolve(strict=True)
    expected = {
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
    if {path.name for path in output_dir.iterdir() if path.is_file()} != expected:
        raise ValueError("Production package file set mismatch")
    manifest = _strict_json(output_dir / "package_manifest.json")
    if (
        manifest.get("manifest_version") != "imu-rf-final-package-manifest-v1"
        or manifest.get("package_version") != "imu-rf-final-v1"
        or manifest.get("package_file_count") != 9
    ):
        raise ValueError("Production package manifest contract mismatch")
    records = manifest.get("files")
    if not isinstance(records, list) or {record.get("relative_path") for record in records} != (
        expected - {"package_manifest.json"}
    ):
        raise ValueError("Production package member records mismatch")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"relative_path", "size", "sha256"}:
            raise ValueError("Production package member record mismatch")
        path = output_dir / str(record["relative_path"])
        if record["size"] != path.stat().st_size or record["sha256"] != sha256_file(path):
            raise ValueError("Production package member hash mismatch")
    canonical = _json_sha256(
        {"package_version": "imu-rf-final-v1", "files": records}
    )
    if manifest.get("package_canonical_sha256") != canonical:
        raise ValueError("Production package canonical SHA-256 mismatch")
    total = sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file())
    if (
        manifest.get("package_total_bytes") != total
        or not isinstance(manifest.get("package_total_mib"), str)
        or abs(float(manifest["package_total_mib"]) - total / 1048576) > 5e-13
    ):
        raise ValueError("Production package total size mismatch")
    validate_production_package_size(total)
    config = load_final_config(output_dir / "resolved_config.json")
    schema = _strict_json(output_dir / "feature_schema.json")
    imputer = _strict_json(output_dir / "imputer.json")
    class_order = _strict_json(output_dir / "class_order.json")
    training_data = _strict_json(output_dir / "training_data_manifest.json")
    inference = _strict_json(output_dir / "inference_metadata.json")
    bundle = joblib.load(output_dir / "model.joblib")
    if not isinstance(bundle, dict) or set(bundle) != {"model", "metadata"}:
        raise ValueError("Production model bundle contract mismatch")
    model = bundle["model"]
    metadata = bundle["metadata"]
    validate_frozen_estimator(model)
    expected_metadata = {
        "model_version": "imu-rf-final-v1",
        "model_role": "imu_rf_production",
        "estimator": "RandomForestClassifier",
        "feature_schema_sha256": _json_sha256(schema),
        "imputer_sha256": _json_sha256(imputer),
        "class_order_file_sha256": _json_sha256(class_order),
        "training_data_manifest_sha256": _json_sha256(training_data),
        "resolved_config_sha256": _json_sha256(config),
        "random_state": 20260725,
        "feature_count": 2310,
        "class_count": 40,
    }
    if metadata != expected_metadata:
        raise ValueError("Production model metadata mismatch")
    if any(inference.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError("Production inference metadata mismatch")
    names = schema.get("feature_names")
    if not isinstance(names, list) or len(names) != 2310:
        raise ValueError("Production feature schema mismatch")
    apply_production_imputer(np.empty((0, 2310), dtype=np.float64), imputer, names)
    classes = class_order.get("classes")
    if (
        not isinstance(classes, list)
        or len(classes) != 40
        or [record.get("label_index") for record in classes] != list(range(40))
        or not np.array_equal(model.classes_, np.arange(40))
    ):
        raise ValueError("Production class order mismatch")
    return {
        "model": model,
        "model_metadata": metadata,
        "config": config,
        "schema": schema,
        "imputer": imputer,
        "class_order": classes,
        "training_data_manifest": training_data,
        "package_manifest": manifest,
        "snapshot": directory_snapshot(output_dir),
    }
