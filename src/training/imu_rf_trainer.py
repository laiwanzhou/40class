from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.imu_stage2_contracts import canonical_json_bytes, sha256_file
from src.training.imu_stage2_trainer import classification_metrics


RF_CONFIG_VERSION = "imu-rf-screen-v1"
RF_MODEL_VERSION = "imu-rf-model-v1"
RF_RUN_MANIFEST_VERSION = "imu-rf-run-manifest-v1"
RF_RUN_FILES = {
    "model.joblib",
    "feature_schema.json",
    "feature_importances.csv",
    "validation_predictions.csv",
    "validation_outputs.npz",
    "confusion_matrix.csv",
    "per_class_metrics.csv",
    "resolved_config.json",
    "training_summary.json",
    "run_manifest.json",
}
CONFIG_FIELDS = {
    "config_version",
    "fold",
    "num_classes",
    "feature_schema_version",
    "n_estimators",
    "max_features",
    "max_depth",
    "min_samples_leaf",
    "bootstrap",
    "n_jobs",
    "variants",
    "random_states",
}


def _strict_json(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant: {value}")
            ),
        )
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_rf_config(path: Path) -> dict[str, object]:
    payload = _strict_json(path)
    if set(payload) != CONFIG_FIELDS:
        raise ValueError("RF config field set mismatch")
    exact_types: dict[str, type] = {
        "config_version": str,
        "fold": int,
        "num_classes": int,
        "feature_schema_version": str,
        "n_estimators": int,
        "max_features": str,
        "min_samples_leaf": int,
        "bootstrap": bool,
        "n_jobs": int,
        "variants": dict,
        "random_states": list,
    }
    for field, expected_type in exact_types.items():
        if type(payload[field]) is not expected_type:
            raise ValueError(f"RF config {field} has invalid type")
    if payload["max_depth"] is not None and type(payload["max_depth"]) is not int:
        raise ValueError("RF config max_depth has invalid type")
    if payload["config_version"] != RF_CONFIG_VERSION:
        raise ValueError("RF config version mismatch")
    if int(payload["fold"]) != 0 or int(payload["num_classes"]) < 2:
        raise ValueError("RF config fold or class count is invalid")
    if int(payload["n_estimators"]) < 1 or int(payload["min_samples_leaf"]) < 1:
        raise ValueError("RF config tree counts are invalid")
    if payload["max_features"] != "sqrt" or payload["bootstrap"] is not True:
        raise ValueError("RF config forest policy mismatch")
    if payload["variants"] != {"plain": None, "balanced": "balanced_subsample"}:
        raise ValueError("RF config variants mismatch")
    states = payload["random_states"]
    if any(type(value) is not int for value in states) or len(states) != len(set(states)):
        raise ValueError("RF config random_states are invalid")
    return payload


def _load_class_order(schema: Mapping[str, object], num_classes: int) -> list[dict[str, object]]:
    records = schema.get("class_order")
    if records is None:
        records = [
            {"class_id": index, "class_name": f"class-{index}", "label_index": index}
            for index in range(num_classes)
        ]
    if not isinstance(records, list) or len(records) != num_classes:
        raise ValueError("RF feature class order mismatch")
    normalized: list[dict[str, object]] = []
    class_ids: set[int] = set()
    class_names: set[str] = set()
    for label_index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "class_id",
            "class_name",
            "label_index",
        }:
            raise ValueError("RF feature class order record mismatch")
        normalized.append(
            {
                "class_id": int(record["class_id"]),
                "class_name": str(record["class_name"]),
                "label_index": int(record["label_index"]),
            }
        )
        if normalized[-1]["label_index"] != label_index:
            raise ValueError("RF class order labels must be consecutive")
        if (
            normalized[-1]["class_id"] in class_ids
            or normalized[-1]["class_name"] in class_names
            or not normalized[-1]["class_name"]
        ):
            raise ValueError("RF class order identities must be unique and named")
        class_ids.add(int(normalized[-1]["class_id"]))
        class_names.add(str(normalized[-1]["class_name"]))
    return normalized


def _load_matrix(path: Path, *, expected_feature_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"sample_ids", "user_ids", "labels", "features"}
        if set(archive.files) != expected:
            raise ValueError("RF feature matrix NPZ keys mismatch")
        arrays = {name: archive[name].copy() for name in expected}
    count = len(arrays["sample_ids"])
    if (
        arrays["sample_ids"].ndim != 1
        or arrays["sample_ids"].dtype.kind != "U"
        or arrays["user_ids"].shape != (count,)
        or arrays["user_ids"].dtype.kind != "U"
        or arrays["labels"].dtype != np.int64
        or arrays["labels"].shape != (count,)
        or arrays["features"].dtype != np.float64
        or arrays["features"].shape != (count, expected_feature_count)
    ):
        raise ValueError("RF feature matrix dtype or shape mismatch")
    sample_ids = arrays["sample_ids"].tolist()
    if any(not value for value in sample_ids) or len(set(sample_ids)) != count:
        raise ValueError("RF feature sample IDs must be non-empty and unique")
    if not np.isfinite(arrays["features"]).all():
        raise ValueError("RF feature matrix must be finite")
    order = np.argsort(arrays["sample_ids"], kind="stable")
    return {name: value[order] for name, value in arrays.items()}


def _load_features(feature_root: Path, config: Mapping[str, object]) -> dict[str, object]:
    feature_root = Path(feature_root).resolve(strict=True)
    required = {
        "train_features.npz",
        "validation_features.npz",
        "feature_schema.json",
        "imputer.json",
        "manifest.csv",
        "provenance.json",
    }
    if {path.name for path in feature_root.iterdir() if path.is_file()} != required:
        raise ValueError("RF feature artifact file set mismatch")
    schema = _strict_json(feature_root / "feature_schema.json")
    if schema.get("schema_version") != config["feature_schema_version"]:
        raise ValueError("RF feature schema version mismatch")
    names = schema.get("feature_names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
        or schema.get("feature_count") != len(names)
    ):
        raise ValueError("RF feature schema columns are invalid")
    train = _load_matrix(feature_root / "train_features.npz", expected_feature_count=len(names))
    validation = _load_matrix(
        feature_root / "validation_features.npz", expected_feature_count=len(names)
    )
    num_classes = int(config["num_classes"])
    for split in (train, validation):
        labels = split["labels"]
        if labels.size == 0 or labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError("RF feature labels are outside class range")
    imputer = _strict_json(feature_root / "imputer.json")
    class_order = _load_class_order(schema, num_classes)
    return {
        "root": feature_root,
        "schema": schema,
        "imputer": imputer,
        "train": train,
        "validation": validation,
        "class_order": class_order,
        "feature_schema_sha256": sha256_file(feature_root / "feature_schema.json"),
        "imputer_sha256": sha256_file(feature_root / "imputer.json"),
        "class_order_sha256": _sha256_json({"classes": class_order}),
    }


def _resolved_config(
    config: Mapping[str, object], variant: str, random_state: int
) -> dict[str, object]:
    if variant not in config["variants"]:
        raise ValueError("Unknown RF variant")
    if random_state not in config["random_states"]:
        raise ValueError("Unapproved RF random_state")
    return {
        **dict(config),
        "variant": variant,
        "random_state": random_state,
        "class_weight": config["variants"][variant],
    }


def _metadata(
    artifacts: Mapping[str, object], resolved: Mapping[str, object]
) -> dict[str, object]:
    return {
        "model_version": RF_MODEL_VERSION,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "feature_schema_sha256": artifacts["feature_schema_sha256"],
        "imputer_sha256": artifacts["imputer_sha256"],
        "class_order_sha256": artifacts["class_order_sha256"],
        "random_state": resolved["random_state"],
        "rf_config_sha256": _sha256_json(resolved),
        "rf_config": dict(resolved),
    }


def load_model_bundle(
    path: Path,
    *,
    expected_metadata: Mapping[str, object],
) -> dict[str, object]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or set(payload) != {"model", "metadata"}:
        raise ValueError("RF model bundle contract mismatch")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != set(expected_metadata):
        raise ValueError("RF model metadata field set mismatch")
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"RF model {field} mismatch")
    if not isinstance(payload["model"], RandomForestClassifier):
        raise ValueError("RF model bundle estimator mismatch")
    return payload


def _probabilities(model: RandomForestClassifier, features: np.ndarray, num_classes: int) -> np.ndarray:
    partial = model.predict_proba(features)
    probabilities = np.zeros((len(features), num_classes), dtype=np.float64)
    probabilities[:, model.classes_.astype(np.int64)] = partial
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("RF class probabilities are invalid")
    return probabilities


def _class_order_array(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    width = max(len(str(record["class_name"])) for record in records)
    result = np.empty(
        len(records),
        dtype=np.dtype(
            [("label_index", np.int64), ("class_id", np.int64), ("class_name", f"U{width}")]
        ),
    )
    for index, record in enumerate(records):
        result[index] = (index, int(record["class_id"]), str(record["class_name"]))
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _staged_directory(output_dir: Path):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    residues = list(output_dir.parent.glob(f".{output_dir.name}.staging-*"))
    if residues:
        raise FileExistsError("Unknown RF staging residue exists")
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _weighted_f1(per_class: Sequence[Mapping[str, object]]) -> float:
    total = sum(int(record["support"]) for record in per_class)
    return (
        sum(float(record["f1"]) * int(record["support"]) for record in per_class) / total
        if total
        else 0.0
    )


def _write_run(
    staging: Path,
    *,
    model: RandomForestClassifier,
    metadata: Mapping[str, object],
    artifacts: Mapping[str, object],
    resolved: Mapping[str, object],
    probabilities: np.ndarray,
    duration_seconds: float,
) -> dict[str, object]:
    validation = artifacts["validation"]
    sample_ids = validation["sample_ids"]
    user_ids = validation["user_ids"]
    labels = validation["labels"]
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    num_classes = int(resolved["num_classes"])
    metrics = classification_metrics(labels=labels, predictions=predictions, num_classes=num_classes)
    per_class = metrics["per_class"]
    weighted_f1 = _weighted_f1(per_class)
    zero_f1_count = sum(float(record["f1"]) == 0.0 for record in per_class)
    joblib.dump({"model": model, "metadata": dict(metadata)}, staging / "model.joblib")
    shutil.copyfile(artifacts["root"] / "feature_schema.json", staging / "feature_schema.json")
    feature_names = artifacts["schema"]["feature_names"]
    importances = [
        {"feature_index": index, "feature_name": name, "importance": float(value)}
        for index, (name, value) in enumerate(zip(feature_names, model.feature_importances_, strict=True))
    ]
    importances.sort(key=lambda row: (-float(row["importance"]), int(row["feature_index"])))
    _write_csv(
        staging / "feature_importances.csv",
        ["feature_index", "feature_name", "importance"],
        importances,
    )
    prediction_rows = [
        {
            "sample_id": str(sample_id),
            "user_id": str(user_id),
            "label": int(label),
            "prediction": int(prediction),
            "correct": int(label == prediction),
        }
        for sample_id, user_id, label, prediction in zip(
            sample_ids, user_ids, labels, predictions, strict=True
        )
    ]
    _write_csv(
        staging / "validation_predictions.csv",
        ["sample_id", "user_id", "label", "prediction", "correct"],
        prediction_rows,
    )
    np.savez(
        staging / "validation_outputs.npz",
        sample_ids=sample_ids,
        labels=labels,
        predictions=predictions,
        class_probabilities=probabilities,
        class_order=_class_order_array(artifacts["class_order"]),
    )
    confusion = metrics["confusion_matrix"]
    _write_csv(
        staging / "confusion_matrix.csv",
        ["true_label_index"] + [f"predicted_{index}" for index in range(num_classes)],
        [
            {
                "true_label_index": index,
                **{f"predicted_{column}": int(confusion[index, column]) for column in range(num_classes)},
            }
            for index in range(num_classes)
        ],
    )
    class_order = artifacts["class_order"]
    class_rows = [
        {
            **class_order[index],
            "precision": record["precision"],
            "recall": record["recall"],
            "f1": record["f1"],
            "support": record["support"],
        }
        for index, record in enumerate(per_class)
    ]
    _write_csv(
        staging / "per_class_metrics.csv",
        ["label_index", "class_id", "class_name", "precision", "recall", "f1", "support"],
        class_rows,
    )
    _write_json(staging / "resolved_config.json", dict(resolved))
    depths = [estimator.tree_.max_depth for estimator in model.estimators_]
    summary = {
        "status": "success",
        "variant": resolved["variant"],
        "random_state": resolved["random_state"],
        "train_samples": len(artifacts["train"]["labels"]),
        "validation_samples": len(labels),
        "num_classes": num_classes,
        "feature_count": len(feature_names),
        "validation_accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": weighted_f1,
        "zero_f1_class_count": zero_f1_count,
        "maximum_prediction_share": float(np.bincount(predictions, minlength=num_classes).max() / len(labels)),
        "training_duration_seconds": duration_seconds,
        "n_estimators": len(model.estimators_),
        "mean_tree_depth": float(np.mean(depths)),
    }
    _write_json(staging / "training_summary.json", summary)
    members = [
        {"relative_path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(staging.iterdir(), key=lambda item: item.name)
        if path.is_file()
    ]
    _write_json(
        staging / "run_manifest.json",
        {"manifest_version": RF_RUN_MANIFEST_VERSION, "files": members},
    )
    return summary


def validate_rf_run(output_dir: Path) -> dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    if {path.name for path in output_dir.iterdir() if path.is_file()} != RF_RUN_FILES:
        raise ValueError("RF run file set mismatch")
    manifest = _strict_json(output_dir / "run_manifest.json")
    if manifest.get("manifest_version") != RF_RUN_MANIFEST_VERSION:
        raise ValueError("RF run manifest version mismatch")
    records = manifest.get("files")
    expected_members = RF_RUN_FILES - {"run_manifest.json"}
    if not isinstance(records, list) or {record.get("relative_path") for record in records} != expected_members:
        raise ValueError("RF run manifest members mismatch")
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "size",
            "sha256",
        }:
            raise ValueError("RF run manifest record mismatch")
        path = output_dir / str(record["relative_path"])
        if record.get("size") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ValueError("RF run manifest member mismatch")
    schema = _strict_json(output_dir / "feature_schema.json")
    resolved = _strict_json(output_dir / "resolved_config.json")
    class_order = _load_class_order(schema, int(resolved["num_classes"]))
    with np.load(output_dir / "validation_outputs.npz", allow_pickle=False) as archive:
        expected = {"sample_ids", "labels", "predictions", "class_probabilities", "class_order"}
        if set(archive.files) != expected:
            raise ValueError("RF validation output keys mismatch")
        arrays = {name: archive[name].copy() for name in expected}
    count = len(arrays["sample_ids"])
    probabilities = arrays["class_probabilities"]
    if (
        arrays["sample_ids"].ndim != 1
        or arrays["sample_ids"].dtype.kind != "U"
        or arrays["sample_ids"].tolist() != sorted(arrays["sample_ids"].tolist())
        or len(set(arrays["sample_ids"].tolist())) != count
        or arrays["labels"].dtype != np.int64
        or arrays["labels"].shape != (count,)
        or arrays["predictions"].dtype != np.int64
        or arrays["predictions"].shape != (count,)
        or (count and (arrays["labels"].min() < 0 or arrays["labels"].max() >= len(class_order)))
        or (count and (arrays["predictions"].min() < 0 or arrays["predictions"].max() >= len(class_order)))
        or probabilities.dtype != np.float64
        or probabilities.shape != (count, len(class_order))
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not np.array_equal(arrays["predictions"], np.argmax(probabilities, axis=1))
    ):
        raise ValueError("RF validation output contract mismatch")
    if arrays["class_order"].dtype.names != ("label_index", "class_id", "class_name"):
        raise ValueError("RF validation class order dtype mismatch")
    reopened_order = [
        {
            "label_index": int(record["label_index"]),
            "class_id": int(record["class_id"]),
            "class_name": str(record["class_name"]),
        }
        for record in arrays["class_order"]
    ]
    if reopened_order != class_order:
        raise ValueError("RF validation class order mismatch")
    return arrays


def train_random_forest(
    *,
    feature_root: Path,
    config_path: Path,
    variant: str,
    random_state: int,
    output_dir: Path | None,
    preflight_only: bool,
) -> dict[str, Any]:
    config = load_rf_config(config_path)
    resolved = _resolved_config(config, variant, random_state)
    artifacts = _load_features(feature_root, config)
    train = artifacts["train"]
    validation = artifacts["validation"]
    model = RandomForestClassifier(
        n_estimators=int(resolved["n_estimators"]),
        max_features=str(resolved["max_features"]),
        max_depth=resolved["max_depth"],
        min_samples_leaf=int(resolved["min_samples_leaf"]),
        bootstrap=bool(resolved["bootstrap"]),
        class_weight=resolved["class_weight"],
        random_state=random_state,
        n_jobs=int(resolved["n_jobs"]),
    )
    started = time.perf_counter()
    model.fit(train["features"], train["labels"])
    duration = time.perf_counter() - started
    probabilities = _probabilities(model, validation["features"], int(config["num_classes"]))
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    preflight = {
        "status": "preflight_ok",
        "variant": variant,
        "random_state": random_state,
        "train_samples": len(train["labels"]),
        "validation_samples": len(validation["labels"]),
        "num_classes": int(config["num_classes"]),
        "feature_count": train["features"].shape[1],
        "feature_matrix_finite": True,
        "probability_shape": list(probabilities.shape),
        "probabilities_finite": True,
        "predictions": predictions,
        "probabilities": probabilities,
    }
    if preflight_only:
        return preflight
    if output_dir is None:
        raise ValueError("output_dir is required for RF training")
    metadata = _metadata(artifacts, resolved)
    with _staged_directory(output_dir) as staging:
        summary = _write_run(
            staging,
            model=model,
            metadata=metadata,
            artifacts=artifacts,
            resolved=resolved,
            probabilities=probabilities,
            duration_seconds=duration,
        )
        load_model_bundle(staging / "model.joblib", expected_metadata=metadata)
        validate_rf_run(staging)
    validate_rf_run(output_dir)
    return summary


def summarize_rf_experiment(
    *,
    experiment_root: Path,
    baseline_validation_outputs: Path,
    config_path: Path,
) -> dict[str, object]:
    experiment_root = Path(experiment_root).resolve(strict=True)
    config = load_rf_config(config_path)
    report_names = {
        "random_forest_comparison.csv",
        "random_forest_comparison.json",
        "per_class_multiseed_summary.csv",
        "per_user_multiseed_summary.csv",
    }
    if any((experiment_root / name).exists() for name in report_names):
        raise FileExistsError("RF comparison report already exists")
    with np.load(baseline_validation_outputs, allow_pickle=False) as archive:
        required = {"sample_ids", "labels", "predictions"}
        if not required.issubset(archive.files):
            raise ValueError("TCN baseline validation output is missing comparison arrays")
        baseline_ids = archive["sample_ids"].copy()
        baseline_labels = archive["labels"].copy()
        baseline_predictions = archive["predictions"].copy()
    baseline_order = np.argsort(baseline_ids, kind="stable")
    baseline_ids = baseline_ids[baseline_order]
    baseline_labels = baseline_labels[baseline_order]
    baseline_predictions = baseline_predictions[baseline_order]
    run_rows: list[dict[str, object]] = []
    class_records: list[dict[str, object]] = []
    user_records: list[dict[str, object]] = []
    zero_sets: dict[str, list[set[int]]] = {"plain": [], "balanced": []}
    for variant in ("plain", "balanced"):
        for seed in config["random_states"]:
            run_name = f"rf_{variant}_seed{seed}"
            run = experiment_root / run_name
            arrays = validate_rf_run(run)
            if not np.array_equal(arrays["sample_ids"], baseline_ids) or not np.array_equal(
                arrays["labels"], baseline_labels
            ):
                raise ValueError("RF and TCN validation samples are misaligned")
            labels = arrays["labels"]
            predictions = arrays["predictions"]
            rf_correct = predictions == labels
            tcn_correct = baseline_predictions == labels
            summary = _strict_json(run / "training_summary.json")
            row = {
                "run_name": run_name,
                "variant": variant,
                "random_state": seed,
                "validation_accuracy": summary["validation_accuracy"],
                "macro_precision": summary["macro_precision"],
                "macro_recall": summary["macro_recall"],
                "macro_f1": summary["macro_f1"],
                "weighted_f1": summary["weighted_f1"],
                "zero_f1_class_count": summary["zero_f1_class_count"],
                "maximum_prediction_share": summary["maximum_prediction_share"],
                "training_duration_seconds": summary["training_duration_seconds"],
                "model_size_bytes": (run / "model.joblib").stat().st_size,
                "feature_count": summary["feature_count"],
                "n_estimators": summary["n_estimators"],
                "mean_tree_depth": summary["mean_tree_depth"],
                "rf_only_correct": int((rf_correct & ~tcn_correct).sum()),
                "tcn_only_correct": int((~rf_correct & tcn_correct).sum()),
                "both_correct": int((rf_correct & tcn_correct).sum()),
                "both_wrong": int((~rf_correct & ~tcn_correct).sum()),
                "ideal_selector_accuracy": float((rf_correct | tcn_correct).mean()),
            }
            run_rows.append(row)
            per_class = pd.read_csv(run / "per_class_metrics.csv")
            zero_sets[variant].append(
                set(per_class.loc[per_class["f1"] == 0.0, "label_index"].astype(int))
            )
            for record in per_class.to_dict(orient="records"):
                class_records.append({"variant": variant, "random_state": seed, **record})
            prediction_frame = pd.read_csv(run / "validation_predictions.csv")
            for user_id, group in prediction_frame.groupby("user_id", sort=True):
                user_records.append(
                    {
                        "variant": variant,
                        "random_state": seed,
                        "user_id": str(user_id),
                        "support": len(group),
                        "accuracy": float(group["correct"].mean()),
                    }
                )
    comparison_fields = list(run_rows[0])
    _write_csv(
        experiment_root / "random_forest_comparison.csv", comparison_fields, run_rows
    )
    class_frame = pd.DataFrame(class_records)
    class_rows: list[dict[str, object]] = []
    for (variant, label_index), group in class_frame.groupby(
        ["variant", "label_index"], sort=True
    ):
        class_rows.append(
            {
                "variant": variant,
                "label_index": int(label_index),
                "class_id": int(group.iloc[0]["class_id"]),
                "class_name": str(group.iloc[0]["class_name"]),
                "support": int(group.iloc[0]["support"]),
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=0)),
                "f1_min": float(group["f1"].min()),
                "f1_max": float(group["f1"].max()),
            }
        )
    _write_csv(
        experiment_root / "per_class_multiseed_summary.csv",
        list(class_rows[0]),
        class_rows,
    )
    user_frame = pd.DataFrame(user_records)
    user_rows: list[dict[str, object]] = []
    for (variant, user_id), group in user_frame.groupby(["variant", "user_id"], sort=True):
        user_rows.append(
            {
                "variant": variant,
                "user_id": str(user_id),
                "support": int(group.iloc[0]["support"]),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": float(group["accuracy"].std(ddof=0)),
                "accuracy_min": float(group["accuracy"].min()),
                "accuracy_max": float(group["accuracy"].max()),
            }
        )
    _write_csv(
        experiment_root / "per_user_multiseed_summary.csv",
        list(user_rows[0]),
        user_rows,
    )
    variant_summaries: dict[str, object] = {}
    for variant in ("plain", "balanced"):
        records = [row for row in run_rows if row["variant"] == variant]
        metric_summary: dict[str, object] = {}
        for metric in (
            "validation_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "zero_f1_class_count",
        ):
            values = np.asarray([float(row[metric]) for row in records], dtype=np.float64)
            metric_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        zeros = zero_sets[variant]
        metric_summary["zero_f1_intersection"] = sorted(set.intersection(*zeros))
        metric_summary["zero_f1_union"] = sorted(set.union(*zeros))
        variant_summaries[variant] = metric_summary
    result = {
        "comparison_version": "imu-rf-comparison-v1",
        "formal_tcn_baseline": {
            "source": "formal IMU run",
            "validation_accuracy": 0.287958,
            "macro_f1": 0.223470,
            "weighted_f1": 0.290659,
            "zero_f1_class_count": 13,
        },
        "ce_supcon_candidate": {
            "source": "IMU_test experiment; not the formal IMU model",
            "validation_accuracy_mean": 0.30657,
            "macro_f1_mean": 0.24758,
        },
        "runs": run_rows,
        "variants": variant_summaries,
    }
    _write_json(experiment_root / "random_forest_comparison.json", result)
    return result
