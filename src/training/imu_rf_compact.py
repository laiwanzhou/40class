from __future__ import annotations

import hashlib
import json
import csv
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data.imu_stage2_contracts import canonical_json_bytes, sha256_file
from src.training.imu_rf_trainer import (
    _load_features,
    _metadata,
    _probabilities,
    _staged_directory,
    _write_run,
    _write_json,
    load_model_bundle,
    validate_rf_run,
)


COMPACT_CONFIG_VERSION = "imu-rf-compact-screen-v1"
CONFIG_FIELDS = {
    "config_version",
    "fold",
    "num_classes",
    "feature_schema_version",
    "model_compression",
    "compression_screen",
    "structural_seed",
    "random_states",
    "structural_candidates",
    "selection_thresholds",
    "budgets_mib",
}
CANDIDATE_FIELDS = {
    "candidate_id",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "max_samples",
    "max_features",
    "bootstrap",
    "class_weight",
}
THRESHOLD_FIELDS = {
    "primary_max_mib",
    "primary_macro_f1_drop",
    "primary_accuracy_drop",
    "primary_weighted_f1_drop",
    "primary_zero_f1_increase",
    "fallback_max_mib",
    "fallback_macro_f1_drop",
    "fallback_accuracy_drop",
    "fallback_weighted_f1_drop",
    "fallback_zero_f1_increase",
    "maximum_prediction_share",
}
SUPPORTED_COMPRESSION = {"zlib", "gzip", "bz2", "lzma", "xz"}


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
        raise ValueError("Compact RF config must be an object")
    return payload


def _compression(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"method", "level"}:
        raise ValueError(f"Compact RF {field} field set mismatch")
    method = value["method"]
    level = value["level"]
    if type(method) is not str or method not in SUPPORTED_COMPRESSION:
        raise ValueError(f"Compact RF {field} method is invalid")
    if type(level) is not int or not 0 <= level <= 9:
        raise ValueError(f"Compact RF {field} level is invalid")
    return {"method": method, "level": level}


def _model_and_metadata(payload: object) -> tuple[RandomForestClassifier, dict[str, object]]:
    if not isinstance(payload, dict) or set(payload) != {"model", "metadata"}:
        raise ValueError("RF model bundle contract mismatch")
    model = payload["model"]
    metadata = payload["metadata"]
    if not isinstance(model, RandomForestClassifier) or not isinstance(metadata, dict):
        raise ValueError("RF model bundle types mismatch")
    return model, metadata


def _forest_digest(model: RandomForestClassifier) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"params": model.get_params(deep=False)}))
    digest.update(np.asarray(model.classes_).tobytes())
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        nodes = np.asarray(state["nodes"])
        for name in nodes.dtype.names or ():
            digest.update(name.encode("ascii"))
            digest.update(np.ascontiguousarray(nodes[name]).tobytes())
        digest.update(np.asarray(state["values"]).tobytes())
    return digest.hexdigest()


def probabilities_equivalent(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(
        first.dtype == np.float64
        and second.dtype == np.float64
        and first.shape == second.shape
        and np.isfinite(first).all()
        and np.isfinite(second).all()
        and np.allclose(first, second, rtol=0.0, atol=1e-15)
    )


def measure_model_roundtrip(
    *,
    source_path: Path,
    destination_path: Path,
    validation_features: np.ndarray,
    compression_method: str,
    compression_level: int,
) -> dict[str, object]:
    compression = _compression(
        {"method": compression_method, "level": compression_level}, field="compression"
    )
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    source_model, source_metadata = _model_and_metadata(joblib.load(source_path))
    source_predictions = source_model.predict(validation_features)
    source_probabilities = source_model.predict_proba(validation_features)
    source_digest = _forest_digest(source_model)
    started = time.perf_counter()
    joblib.dump(
        {"model": source_model, "metadata": source_metadata},
        destination_path,
        compress=(str(compression["method"]), int(compression["level"])),
    )
    compression_seconds = time.perf_counter() - started
    started = time.perf_counter()
    loaded_model, loaded_metadata = _model_and_metadata(joblib.load(destination_path))
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    loaded_predictions = loaded_model.predict(validation_features)
    loaded_probabilities = loaded_model.predict_proba(validation_features)
    batch_seconds = time.perf_counter() - started
    started = time.perf_counter()
    loaded_model.predict(validation_features[:1])
    single_seconds = time.perf_counter() - started
    return {
        "compression_method": compression["method"],
        "compression_level": compression["level"],
        "uncompressed_bytes": source_path.stat().st_size,
        "compressed_bytes": destination_path.stat().st_size,
        "compression_ratio": destination_path.stat().st_size / source_path.stat().st_size,
        "compression_seconds": compression_seconds,
        "load_seconds": load_seconds,
        "batch_inference_seconds": batch_seconds,
        "single_inference_seconds": single_seconds,
        "prediction_equal": bool(np.array_equal(source_predictions, loaded_predictions)),
        "probability_equal": probabilities_equivalent(
            source_probabilities, loaded_probabilities
        ),
        "metadata_equal": source_metadata == loaded_metadata,
        "structure_equal": source_digest == _forest_digest(loaded_model),
    }


def forest_structure_metrics(model: object) -> dict[str, object]:
    if not isinstance(model, RandomForestClassifier) or not hasattr(model, "estimators_"):
        raise ValueError("Expected a fitted RandomForestClassifier")
    node_counts = [estimator.tree_.node_count for estimator in model.estimators_]
    depths = [estimator.tree_.max_depth for estimator in model.estimators_]
    array_bytes = 0
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        array_bytes += np.asarray(state["nodes"]).nbytes
        array_bytes += np.asarray(state["values"]).nbytes
    return {
        "n_estimators": len(model.estimators_),
        "total_nodes": int(sum(node_counts)),
        "mean_nodes_per_tree": float(np.mean(node_counts)),
        "minimum_nodes_per_tree": int(min(node_counts)),
        "maximum_nodes_per_tree": int(max(node_counts)),
        "mean_tree_depth": float(np.mean(depths)),
        "maximum_tree_depth": int(max(depths)),
        "estimated_model_array_bytes": int(array_bytes),
    }


def load_compact_config(path: Path) -> dict[str, object]:
    payload = _strict_json(path)
    if set(payload) != CONFIG_FIELDS:
        raise ValueError("Compact RF config field set mismatch")
    if payload["config_version"] != COMPACT_CONFIG_VERSION:
        raise ValueError("Compact RF config version mismatch")
    for name in ("fold", "num_classes", "structural_seed"):
        if type(payload[name]) is not int:
            raise ValueError(f"Compact RF {name} has invalid type")
    if payload["fold"] != 0 or payload["num_classes"] < 2:
        raise ValueError("Compact RF fold or class count is invalid")
    if type(payload["feature_schema_version"]) is not str:
        raise ValueError("Compact RF feature schema version is invalid")
    payload["model_compression"] = _compression(
        payload["model_compression"], field="model_compression"
    )
    screens = payload["compression_screen"]
    if not isinstance(screens, list) or not screens:
        raise ValueError("Compact RF compression_screen is invalid")
    compression_ids: set[str] = set()
    for record in screens:
        if not isinstance(record, dict) or set(record) != {"compression_id", "method", "level"}:
            raise ValueError("Compact RF compression_screen field set mismatch")
        compression_id = record["compression_id"]
        if type(compression_id) is not str or not compression_id or compression_id in compression_ids:
            raise ValueError("Compact RF compression_id is invalid")
        compression_ids.add(compression_id)
        _compression({"method": record["method"], "level": record["level"]}, field="compression_screen")
    states = payload["random_states"]
    if (
        not isinstance(states, list)
        or not states
        or any(type(value) is not int for value in states)
        or len(states) != len(set(states))
        or payload["structural_seed"] not in states
    ):
        raise ValueError("Compact RF random_states are invalid")
    candidates = payload["structural_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Compact RF structural_candidates are invalid")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
            raise ValueError("Compact RF candidate field set mismatch")
        candidate_id = candidate["candidate_id"]
        if type(candidate_id) is not str or not candidate_id or candidate_id in candidate_ids:
            raise ValueError("Compact RF candidate_id is invalid")
        candidate_ids.add(candidate_id)
        for name in ("n_estimators", "min_samples_leaf", "min_samples_split"):
            if type(candidate[name]) is not int or int(candidate[name]) < 1:
                raise ValueError(f"Compact RF candidate {name} is invalid")
        if candidate["max_depth"] is not None and (
            type(candidate["max_depth"]) is not int or int(candidate["max_depth"]) < 1
        ):
            raise ValueError("Compact RF candidate max_depth is invalid")
        if candidate["max_samples"] is not None and (
            type(candidate["max_samples"]) is not float
            or not 0.0 < float(candidate["max_samples"]) <= 1.0
        ):
            raise ValueError("Compact RF candidate max_samples is invalid")
        if candidate["max_features"] != "sqrt" or candidate["bootstrap"] is not True:
            raise ValueError("Compact RF candidate forest policy mismatch")
        if candidate["class_weight"] != "balanced_subsample":
            raise ValueError("Compact RF candidate class_weight mismatch")
    thresholds = payload["selection_thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != THRESHOLD_FIELDS:
        raise ValueError("Compact RF selection_thresholds field set mismatch")
    for name, value in thresholds.items():
        expected = int if name.endswith("increase") or name.endswith("mib") else float
        if type(value) is not expected or value < 0:
            raise ValueError(f"Compact RF threshold {name} is invalid")
    budgets = payload["budgets_mib"]
    if (
        not isinstance(budgets, list)
        or any(type(value) is not int or value <= 0 for value in budgets)
        or budgets != sorted(set(budgets))
    ):
        raise ValueError("Compact RF budgets_mib are invalid")
    return payload


def train_compact_candidate(
    *,
    feature_root: Path,
    config_path: Path,
    candidate_id: str,
    random_state: int,
    output_dir: Path | None,
    preflight_only: bool,
) -> dict[str, Any]:
    config = load_compact_config(config_path)
    if random_state not in config["random_states"]:
        raise ValueError("Unapproved compact RF random_state")
    matches = [
        candidate
        for candidate in config["structural_candidates"]
        if candidate["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("Unknown compact RF candidate")
    candidate = dict(matches[0])
    artifacts = _load_features(feature_root, config)
    train = artifacts["train"]
    validation = artifacts["validation"]
    model = RandomForestClassifier(
        n_estimators=int(candidate["n_estimators"]),
        max_depth=candidate["max_depth"],
        min_samples_leaf=int(candidate["min_samples_leaf"]),
        min_samples_split=int(candidate["min_samples_split"]),
        max_samples=candidate["max_samples"],
        max_features=str(candidate["max_features"]),
        bootstrap=bool(candidate["bootstrap"]),
        class_weight=str(candidate["class_weight"]),
        random_state=random_state,
        n_jobs=-1,
    )
    started = time.perf_counter()
    model.fit(train["features"], train["labels"])
    duration = time.perf_counter() - started
    probabilities = _probabilities(model, validation["features"], int(config["num_classes"]))
    structure = forest_structure_metrics(model)
    compression = dict(config["model_compression"])
    resolved = {
        "config_version": config["config_version"],
        "fold": config["fold"],
        "num_classes": config["num_classes"],
        "feature_schema_version": config["feature_schema_version"],
        "variant": "balanced_compact",
        "candidate_id": candidate_id,
        "random_state": random_state,
        "candidate": candidate,
        "model_compression": compression,
    }
    preflight = {
        "status": "preflight_ok",
        "candidate_id": candidate_id,
        "random_state": random_state,
        "train_samples": len(train["labels"]),
        "validation_samples": len(validation["labels"]),
        "num_classes": int(config["num_classes"]),
        "feature_count": train["features"].shape[1],
        "probability_shape": list(probabilities.shape),
        "probabilities_finite": bool(np.isfinite(probabilities).all()),
        **structure,
    }
    if preflight_only:
        return preflight
    if output_dir is None:
        raise ValueError("output_dir is required for compact RF training")
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
            model_compression=(str(compression["method"]), int(compression["level"])),
            summary_fields={
                "candidate_id": candidate_id,
                "model_compression": compression,
                **{
                    key: value
                    for key, value in structure.items()
                    if key not in {"n_estimators", "mean_tree_depth"}
                },
            },
        )
        temporary_uncompressed = staging / ".model-uncompressed.joblib"
        joblib.dump({"model": model, "metadata": metadata}, temporary_uncompressed, compress=0)
        uncompressed_bytes = temporary_uncompressed.stat().st_size
        temporary_uncompressed.unlink()
        started = time.perf_counter()
        loaded_payload = joblib.load(staging / "model.joblib")
        load_seconds = time.perf_counter() - started
        loaded_model, loaded_metadata = _model_and_metadata(loaded_payload)
        started = time.perf_counter()
        reopened_probabilities = _probabilities(
            loaded_model, validation["features"], int(config["num_classes"])
        )
        batch_inference_seconds = time.perf_counter() - started
        started = time.perf_counter()
        _probabilities(loaded_model, validation["features"][:1], int(config["num_classes"]))
        single_inference_seconds = time.perf_counter() - started
        if loaded_metadata != metadata or not probabilities_equivalent(
            reopened_probabilities, probabilities
        ):
            raise ValueError("Compact RF serialized model changed metadata or probabilities")
        summary["uncompressed_model_bytes"] = uncompressed_bytes
        summary["compressed_model_bytes"] = (staging / "model.joblib").stat().st_size
        summary["compressed_model_mib"] = summary["compressed_model_bytes"] / 1048576
        summary["compression_ratio"] = summary["compressed_model_bytes"] / uncompressed_bytes
        summary["load_seconds"] = load_seconds
        summary["batch_inference_seconds"] = batch_inference_seconds
        summary["single_inference_seconds"] = single_inference_seconds
        # Refresh summary and manifest after the exact serialized size becomes known.
        _write_json(staging / "training_summary.json", summary)
        manifest_path = staging / "run_manifest.json"
        manifest = _strict_json(manifest_path)
        manifest["files"] = [
            {"relative_path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "run_manifest.json"
        ]
        _write_json(manifest_path, manifest)
        load_model_bundle(staging / "model.joblib", expected_metadata=metadata)
        validate_rf_run(staging)
    validate_rf_run(output_dir)
    return summary


def select_compact_finalists(
    records: list[dict[str, object]],
    baseline: dict[str, object],
    thresholds: dict[str, object],
) -> dict[str, str | None]:
    def qualified(record: Mapping[str, object], prefix: str) -> bool:
        return (
            float(record["compressed_mib"]) <= float(thresholds[f"{prefix}_max_mib"])
            and float(record["macro_f1"])
            >= float(baseline["macro_f1"]) - float(thresholds[f"{prefix}_macro_f1_drop"])
            and float(record["validation_accuracy"])
            >= float(baseline["validation_accuracy"])
            - float(thresholds[f"{prefix}_accuracy_drop"])
            and float(record["weighted_f1"])
            >= float(baseline["weighted_f1"])
            - float(thresholds[f"{prefix}_weighted_f1_drop"])
            and int(record["zero_f1_class_count"])
            <= int(baseline["zero_f1_class_count"])
            + int(thresholds[f"{prefix}_zero_f1_increase"])
            and float(record["maximum_prediction_share"])
            <= float(thresholds["maximum_prediction_share"])
        )

    def choose(prefix: str, excluded: set[str]) -> str | None:
        eligible = [
            record
            for record in records
            if str(record["candidate_id"]) not in excluded and qualified(record, prefix)
        ]
        if not eligible:
            return None
        eligible.sort(
            key=lambda record: (
                -float(record["macro_f1"]),
                -float(record["validation_accuracy"]),
                float(record["compressed_mib"]),
                str(record["candidate_id"]),
            )
        )
        return str(eligible[0]["candidate_id"])

    primary = choose("primary", set())
    fallback = choose("fallback", {primary} if primary else set())
    return {"primary": primary, "fallback": fallback}


def compact_pareto_frontier(
    records: list[dict[str, object]], budgets_mib: list[int]
) -> dict[str, object]:
    ordered = sorted(
        records,
        key=lambda record: (
            float(record["compressed_mib"]),
            -float(record["macro_f1"]),
            str(record["candidate_id"]),
        ),
    )
    frontier: list[dict[str, object]] = []
    best_f1 = float("-inf")
    for record in ordered:
        score = float(record["macro_f1"])
        if score > best_f1:
            frontier.append(record)
            best_f1 = score
    winners: dict[str, str | None] = {}
    for budget in budgets_mib:
        eligible = [record for record in records if float(record["compressed_mib"]) <= budget]
        eligible.sort(
            key=lambda record: (
                -float(record["macro_f1"]),
                float(record["compressed_mib"]),
                str(record["candidate_id"]),
            )
        )
        winners[str(budget)] = str(eligible[0]["candidate_id"]) if eligible else None
    return {"frontier": frontier, "budget_winners": winners}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty compact RF table: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Compact RF table columns mismatch: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _directory_snapshot(root: Path) -> dict[str, object]:
    root = Path(root).resolve(strict=True)
    records = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    ]
    digest = hashlib.sha256(canonical_json_bytes({"files": records})).hexdigest()
    return {"root": str(root), "file_count": len(records), "canonical_sha256": digest}


def _baseline_run(baseline_root: Path, seed: int) -> Path:
    path = Path(baseline_root) / f"rf_balanced_seed{seed}"
    validate_rf_run(path)
    return path


def _summary(path: Path) -> dict[str, object]:
    return _strict_json(path / "training_summary.json")


def _candidate_record(candidate_id: str, random_state: int, run: Path) -> dict[str, object]:
    summary = _summary(run)
    return {
        "candidate_id": candidate_id,
        "random_state": random_state,
        "compressed_bytes": int(summary["compressed_model_bytes"]),
        "compressed_mib": float(summary["compressed_model_mib"]),
        "uncompressed_bytes": int(summary["uncompressed_model_bytes"]),
        "validation_accuracy": float(summary["validation_accuracy"]),
        "macro_precision": float(summary["macro_precision"]),
        "macro_recall": float(summary["macro_recall"]),
        "macro_f1": float(summary["macro_f1"]),
        "weighted_f1": float(summary["weighted_f1"]),
        "zero_f1_class_count": int(summary["zero_f1_class_count"]),
        "maximum_prediction_share": float(summary["maximum_prediction_share"]),
        "n_estimators": int(summary["n_estimators"]),
        "total_nodes": int(summary["total_nodes"]),
        "mean_tree_depth": float(summary["mean_tree_depth"]),
        "maximum_tree_depth": int(summary["maximum_tree_depth"]),
        "estimated_model_array_bytes": int(summary["estimated_model_array_bytes"]),
        "load_seconds": float(summary["load_seconds"]),
        "single_inference_seconds": float(summary["single_inference_seconds"]),
        "batch_inference_seconds": float(summary["batch_inference_seconds"]),
        "training_duration_seconds": float(summary["training_duration_seconds"]),
        "model_sha256": sha256_file(run / "model.joblib"),
        "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
    }


def _baseline_metrics(run: Path) -> dict[str, object]:
    summary = _summary(run)
    return {
        "validation_accuracy": float(summary["validation_accuracy"]),
        "macro_f1": float(summary["macro_f1"]),
        "weighted_f1": float(summary["weighted_f1"]),
        "zero_f1_class_count": int(summary["zero_f1_class_count"]),
    }


def _multiseed_rows(
    *, phase_c_root: Path, selected: Sequence[str], random_states: Sequence[int]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    user_rows: list[dict[str, object]] = []
    for candidate_id in selected:
        for seed in random_states:
            run = phase_c_root / candidate_id / f"seed{seed}"
            records.append(_candidate_record(candidate_id, seed, run))
            with (run / "per_class_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    class_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "random_state": seed,
                            "label_index": int(row["label_index"]),
                            "class_id": int(row["class_id"]),
                            "class_name": row["class_name"],
                            "support": int(row["support"]),
                            "f1": float(row["f1"]),
                        }
                    )
            with (run / "validation_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
                by_user: dict[str, list[int]] = {}
                for row in csv.DictReader(handle):
                    by_user.setdefault(row["user_id"], []).append(int(row["correct"]))
            for user_id, correct in sorted(by_user.items()):
                user_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "random_state": seed,
                        "user_id": user_id,
                        "support": len(correct),
                        "accuracy": float(np.mean(correct)),
                    }
                )
    return records, class_rows, user_rows


def _aggregate_rows(
    rows: Sequence[Mapping[str, object]], group_fields: Sequence[str], metric: str
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    prototypes: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(float(row[metric]))
        prototypes[key] = row
    output: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda value: tuple(str(item) for item in value)):
        values = np.asarray(groups[key], dtype=np.float64)
        prototype = prototypes[key]
        record = {field: prototype[field] for field in group_fields}
        for optional in ("class_id", "class_name", "support"):
            if optional in prototype and optional not in record:
                record[optional] = prototype[optional]
        record.update(
            {
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std": float(values.std()),
                f"{metric}_minimum": float(values.min()),
                f"{metric}_maximum": float(values.max()),
            }
        )
        output.append(record)
    return output


def run_compact_screen(
    *,
    feature_root: Path,
    baseline_root: Path,
    config_path: Path,
    output_root: Path,
    preflight_only: bool,
) -> dict[str, object]:
    config = load_compact_config(config_path)
    feature_root = Path(feature_root).resolve(strict=True)
    baseline_root = Path(baseline_root).resolve(strict=True)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    for seed in config["random_states"]:
        _baseline_run(baseline_root, int(seed))
    artifacts = _load_features(feature_root, config)
    before = {
        "feature_root": _directory_snapshot(feature_root),
        "baseline_root": _directory_snapshot(baseline_root),
    }
    if preflight_only:
        return {
            "status": "preflight_ok",
            "feature_count": int(artifacts["train"]["features"].shape[1]),
            "train_samples": len(artifacts["train"]["labels"]),
            "validation_samples": len(artifacts["validation"]["labels"]),
            "baseline_runs": len(config["random_states"]),
            "structural_candidates": len(config["structural_candidates"]),
            "input_snapshot": before,
        }
    with _staged_directory(output_root) as staging:
        phase_a = staging / "artifact_compression_screen"
        phase_b = staging / "structural_screen_seed20260725"
        phase_c = staging / "multiseed_confirmation"
        phase_a.mkdir()
        phase_b.mkdir()
        phase_c.mkdir()
        compression_rows: list[dict[str, object]] = []
        for seed in config["random_states"]:
            source = _baseline_run(baseline_root, int(seed)) / "model.joblib"
            seed_root = phase_a / f"seed{seed}"
            seed_root.mkdir()
            for compression in config["compression_screen"]:
                destination = seed_root / f"{compression['compression_id']}.joblib"
                result = measure_model_roundtrip(
                    source_path=source,
                    destination_path=destination,
                    validation_features=artifacts["validation"]["features"],
                    compression_method=str(compression["method"]),
                    compression_level=int(compression["level"]),
                )
                if not all(
                    result[field]
                    for field in (
                        "prediction_equal",
                        "probability_equal",
                        "metadata_equal",
                        "structure_equal",
                    )
                ):
                    raise ValueError("Lossless compression changed the accepted RF bundle")
                compression_rows.append(
                    {
                        "random_state": int(seed),
                        "compression_id": compression["compression_id"],
                        **result,
                        "compressed_mib": int(result["compressed_bytes"]) / 1048576,
                        "model_sha256": sha256_file(destination),
                    }
                )
        _write_csv(phase_a / "artifact_compression_screen.csv", compression_rows)
        _write_json(
            phase_a / "artifact_compression_screen.json",
            {"screen_version": COMPACT_CONFIG_VERSION, "runs": compression_rows},
        )
        structural_records: list[dict[str, object]] = []
        structural_seed = int(config["structural_seed"])
        for candidate in config["structural_candidates"]:
            candidate_id = str(candidate["candidate_id"])
            run = phase_b / candidate_id
            train_compact_candidate(
                feature_root=feature_root,
                config_path=config_path,
                candidate_id=candidate_id,
                random_state=structural_seed,
                output_dir=run,
                preflight_only=False,
            )
            structural_records.append(_candidate_record(candidate_id, structural_seed, run))
        baseline = _baseline_metrics(_baseline_run(baseline_root, structural_seed))
        finalists = select_compact_finalists(
            structural_records, baseline, config["selection_thresholds"]
        )
        _write_csv(phase_b / "structural_screen.csv", structural_records)
        _write_json(
            phase_b / "selection.json",
            {
                "baseline": baseline,
                "thresholds": config["selection_thresholds"],
                "finalists": finalists,
                "records": structural_records,
            },
        )
        selected = [value for value in finalists.values() if value is not None]
        selected = list(dict.fromkeys(selected))
        for candidate_id in selected:
            candidate_root = phase_c / candidate_id
            candidate_root.mkdir()
            for seed in config["random_states"]:
                train_compact_candidate(
                    feature_root=feature_root,
                    config_path=config_path,
                    candidate_id=candidate_id,
                    random_state=int(seed),
                    output_dir=candidate_root / f"seed{seed}",
                    preflight_only=False,
                )
        multiseed_records, class_rows, user_rows = _multiseed_rows(
            phase_c_root=phase_c,
            selected=selected,
            random_states=[int(value) for value in config["random_states"]],
        ) if selected else ([], [], [])
        comparison_rows: list[dict[str, object]] = []
        for record in multiseed_records:
            baseline_seed = _baseline_metrics(
                _baseline_run(baseline_root, int(record["random_state"]))
            )
            comparison_rows.append(
                {
                    **record,
                    "accuracy_delta_vs_balanced_rf": float(record["validation_accuracy"])
                    - float(baseline_seed["validation_accuracy"]),
                    "macro_f1_delta_vs_balanced_rf": float(record["macro_f1"])
                    - float(baseline_seed["macro_f1"]),
                    "weighted_f1_delta_vs_balanced_rf": float(record["weighted_f1"])
                    - float(baseline_seed["weighted_f1"]),
                    "zero_f1_delta_vs_balanced_rf": int(record["zero_f1_class_count"])
                    - int(baseline_seed["zero_f1_class_count"]),
                }
            )
        if comparison_rows:
            _write_csv(staging / "compact_rf_comparison.csv", comparison_rows)
        else:
            _write_csv(staging / "compact_rf_comparison.csv", structural_records)
        comparison_payload = {
            "comparison_version": "imu-rf-compact-comparison-v1",
            "accepted_balanced_rf_three_seed_mean": {
                "validation_accuracy": 0.410704,
                "macro_f1": 0.322327,
                "weighted_f1": 0.378831,
                "zero_f1_class_count": 9.0,
            },
            "formal_tcn_reference": {
                "validation_accuracy": 0.287958,
                "macro_f1": 0.223470,
                "weighted_f1": 0.290659,
                "zero_f1_class_count": 13,
            },
            "ce_supcon_reference": {
                "validation_accuracy_mean": 0.30657,
                "macro_f1_mean": 0.24758,
            },
            "finalists": finalists,
            "runs": comparison_rows,
        }
        _write_json(staging / "compact_rf_comparison.json", comparison_payload)
        pareto = compact_pareto_frontier(structural_records, config["budgets_mib"])
        _write_csv(staging / "compact_rf_pareto.csv", pareto["frontier"])
        _write_json(staging / "compact_rf_pareto.json", pareto)
        if class_rows:
            _write_csv(
                staging / "per_class_compact_summary.csv",
                _aggregate_rows(class_rows, ["candidate_id", "label_index"], "f1"),
            )
            _write_csv(
                staging / "per_user_compact_summary.csv",
                _aggregate_rows(user_rows, ["candidate_id", "user_id"], "accuracy"),
            )
        else:
            _write_csv(
                staging / "per_class_compact_summary.csv",
                [{"status": "no_qualified_finalist"}],
            )
            _write_csv(
                staging / "per_user_compact_summary.csv",
                [{"status": "no_qualified_finalist"}],
            )
        after = {
            "feature_root": _directory_snapshot(feature_root),
            "baseline_root": _directory_snapshot(baseline_root),
        }
        if after != before:
            raise RuntimeError("Compact RF experiment modified an accepted input")
        _write_json(
            staging / "input_snapshot.json",
            {"before": before, "after": after, "identical": True},
        )
        result = {
            "status": "success",
            "phase_a_runs": len(compression_rows),
            "phase_b_runs": len(structural_records),
            "phase_c_runs": len(multiseed_records),
            "finalists": finalists,
            "budget_winners": pareto["budget_winners"],
            "input_snapshot_identical": True,
        }
        _write_json(staging / "compact_rf_summary.json", result)
    return result
