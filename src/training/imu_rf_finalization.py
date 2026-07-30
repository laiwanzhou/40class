from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src.data.imu_stage2_contracts import canonical_json_bytes, sha256_file
from src.training.imu_rf_compact import train_compact_candidate
from src.training.imu_rf_trainer import _load_features, validate_rf_run


FROZEN_ESTIMATOR_PARAMS: dict[str, object] = {
    "n_estimators": 150,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "bootstrap": True,
    "max_samples": None,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 20260725,
}

FINAL_CONFIG: dict[str, object] = {
    "config_version": "imu-rf-finalization-v1",
    "candidate_id": "trees_150_leaf4",
    "fold": 0,
    "num_classes": 40,
    "feature_schema_version": "imu-rf-summary-v1",
    "feature_count": 2310,
    "random_state": 20260725,
    "estimator_params": FROZEN_ESTIMATOR_PARAMS,
    "model_compression": {"method": "lzma", "level": 3},
    "production_package_version": "imu-rf-final-v1",
    "max_package_mib": 8,
}

TREE_FIELDS = (
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "impurity",
    "n_node_samples",
    "weighted_n_node_samples",
    "value",
)


def _strict_json(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key: {key}")
            output[key] = value
        return output

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


def _write_json(path: Path, payload: object) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
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
        raise FileExistsError("Unknown RF finalization staging residue exists")
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def directory_snapshot(root: Path) -> dict[str, object]:
    root = Path(root).resolve(strict=True)
    records = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix())
        if path.is_file()
    ]
    paths = "\n".join(str(record["relative_path"]) for record in records).encode("utf-8")
    return {
        "root": str(root),
        "file_count": len(records),
        "total_bytes": sum(int(record["size"]) for record in records),
        "canonical_sha256": hashlib.sha256(
            canonical_json_bytes({"files": records})
        ).hexdigest(),
        "relative_path_list_sha256": hashlib.sha256(paths).hexdigest(),
    }


def load_final_config(path: Path) -> dict[str, object]:
    payload = _strict_json(path)
    if set(payload) != set(FINAL_CONFIG):
        raise ValueError("Final RF config field set mismatch")
    if payload != FINAL_CONFIG:
        raise ValueError("Final RF frozen contract mismatch")
    return json.loads(json.dumps(payload))


def select_formal_source_run(compact_root: Path) -> Path:
    root = Path(compact_root).resolve(strict=True)
    summary = _strict_json(root / "compact_rf_summary.json")
    comparison = _strict_json(root / "compact_rf_comparison.json")
    finalists = summary.get("finalists")
    comparison_finalists = comparison.get("finalists")
    if (
        summary.get("status") != "success"
        or not isinstance(finalists, dict)
        or finalists.get("primary") != "trees_150_leaf4"
        or comparison_finalists != finalists
    ):
        raise ValueError("Compact RF primary finalist mismatch")
    runs = comparison.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Compact RF comparison runs are invalid")
    matches = [
        record
        for record in runs
        if isinstance(record, dict)
        and record.get("candidate_id") == "trees_150_leaf4"
        and record.get("random_state") == 20260725
    ]
    if len(matches) != 1:
        raise ValueError("Formal compact RF source must be unique")
    for name in ("model_sha256", "run_manifest_sha256"):
        value = matches[0].get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"Formal compact RF {name} is invalid")
    formal = root / "multiseed_confirmation" / "trees_150_leaf4" / "seed20260725"
    return formal.resolve(strict=True)


def validate_sklearn_compatibility(source_version: str, current_version: str) -> None:
    pattern = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?(?:[+.-].*)?$")
    source = pattern.fullmatch(source_version)
    current = pattern.fullmatch(current_version)
    if source is None or current is None:
        raise ValueError("Invalid scikit-learn version")
    if source.groups()[:2] != current.groups()[:2]:
        raise ValueError("scikit-learn major/minor version mismatch")


def canonical_tree_state_sha256(model: Any) -> str:
    if not isinstance(model, RandomForestClassifier) or not hasattr(model, "estimators_"):
        raise ValueError("Expected a fitted RandomForestClassifier")
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"tree_count": len(model.estimators_)}))
    for tree_index, estimator in enumerate(model.estimators_):
        digest.update(canonical_json_bytes({"tree_index": tree_index}))
        tree = estimator.tree_
        for field in TREE_FIELDS:
            array = np.ascontiguousarray(getattr(tree, field))
            digest.update(field.encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(canonical_json_bytes({"shape": list(array.shape)}))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def compare_forest_structures(source: Any, fresh: Any) -> dict[str, object]:
    if not isinstance(source, RandomForestClassifier) or not isinstance(
        fresh, RandomForestClassifier
    ):
        raise ValueError("Forest comparison requires RandomForestClassifier")
    source_digest = canonical_tree_state_sha256(source)
    fresh_digest = canonical_tree_state_sha256(fresh)
    mismatch: dict[str, object] | None = None
    if len(source.estimators_) != len(fresh.estimators_):
        mismatch = {"field": "tree_count", "tree_index": None}
    else:
        integer_fields = {
            "children_left",
            "children_right",
            "feature",
            "n_node_samples",
        }
        for tree_index, (source_estimator, fresh_estimator) in enumerate(
            zip(source.estimators_, fresh.estimators_, strict=True)
        ):
            for field in TREE_FIELDS:
                first = np.asarray(getattr(source_estimator.tree_, field))
                second = np.asarray(getattr(fresh_estimator.tree_, field))
                equal = (
                    np.array_equal(first, second)
                    if field in integer_fields
                    else first.shape == second.shape
                    and bool(np.allclose(first, second, rtol=0.0, atol=1e-15))
                )
                if not equal:
                    mismatch = {"tree_index": tree_index, "field": field}
                    break
            if mismatch is not None:
                break
    return {
        "equal": mismatch is None and source_digest == fresh_digest,
        "tree_count": len(source.estimators_),
        "source_tree_state_sha256": source_digest,
        "fresh_tree_state_sha256": fresh_digest,
        "first_mismatch": mismatch,
    }


def compare_reproduction_runs(source_run: Path, fresh_run: Path) -> dict[str, object]:
    source_run = Path(source_run).resolve(strict=True)
    fresh_run = Path(fresh_run).resolve(strict=True)
    source_arrays = validate_rf_run(source_run)
    fresh_arrays = validate_rf_run(fresh_run)
    source_bundle = joblib.load(source_run / "model.joblib")
    fresh_bundle = joblib.load(fresh_run / "model.joblib")
    if (
        not isinstance(source_bundle, dict)
        or not isinstance(fresh_bundle, dict)
        or set(source_bundle) != {"model", "metadata"}
        or set(fresh_bundle) != {"model", "metadata"}
        or not isinstance(source_bundle["model"], RandomForestClassifier)
        or not isinstance(fresh_bundle["model"], RandomForestClassifier)
    ):
        raise ValueError("Reproduction model bundle contract mismatch")
    source_model = source_bundle["model"]
    fresh_model = fresh_bundle["model"]
    probabilities = source_arrays["class_probabilities"]
    fresh_probabilities = fresh_arrays["class_probabilities"]
    probability_shape_equal = probabilities.shape == fresh_probabilities.shape
    if probability_shape_equal:
        absolute = np.abs(fresh_probabilities - probabilities)
        maximum = float(absolute.max(initial=0.0))
        nonzero = int(np.count_nonzero(absolute))
        per_class = np.max(absolute, axis=0).tolist() if absolute.size else []
        probability_equal = bool(
            np.allclose(
                fresh_probabilities,
                probabilities,
                rtol=0.0,
                atol=1e-15,
                equal_nan=False,
            )
        )
    else:
        maximum = float("inf")
        nonzero = -1
        per_class = []
        probability_equal = False
    array_gates = {
        "sample_ids": np.array_equal(source_arrays["sample_ids"], fresh_arrays["sample_ids"]),
        "labels": np.array_equal(source_arrays["labels"], fresh_arrays["labels"]),
        "class_order": np.array_equal(
            source_arrays["class_order"], fresh_arrays["class_order"]
        ),
        "predictions": np.array_equal(
            source_arrays["predictions"], fresh_arrays["predictions"]
        ),
        "probabilities": probability_equal,
    }
    source_schema = _strict_json(source_run / "feature_schema.json")
    fresh_schema = _strict_json(fresh_run / "feature_schema.json")
    source_config = _strict_json(source_run / "resolved_config.json")
    fresh_config = _strict_json(fresh_run / "resolved_config.json")
    source_summary = _strict_json(source_run / "training_summary.json")
    fresh_summary = _strict_json(fresh_run / "training_summary.json")
    metric_fields = (
        "validation_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    )
    metric_equal = all(source_summary.get(name) == fresh_summary.get(name) for name in metric_fields)
    source_per_class = pd.read_csv(source_run / "per_class_metrics.csv")
    fresh_per_class = pd.read_csv(fresh_run / "per_class_metrics.csv")
    source_zero = sorted(
        source_per_class.loc[source_per_class["f1"] == 0.0, "label_index"].astype(int).tolist()
    )
    fresh_zero = sorted(
        fresh_per_class.loc[fresh_per_class["f1"] == 0.0, "label_index"].astype(int).tolist()
    )
    source_confusion = pd.read_csv(source_run / "confusion_matrix.csv")
    fresh_confusion = pd.read_csv(fresh_run / "confusion_matrix.csv")
    tree_comparison = compare_forest_structures(source_model, fresh_model)
    gates = {
        **array_gates,
        "feature_schema": source_schema == fresh_schema,
        "resolved_config": source_config == fresh_config,
        "estimator_params": source_model.get_params(deep=False)
        == fresh_model.get_params(deep=False),
        "metrics": metric_equal,
        "confusion_matrix": source_confusion.equals(fresh_confusion),
        "zero_f1_classes": source_zero == fresh_zero,
        "tree_structure": bool(tree_comparison["equal"]),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    return {
        "comparison_version": "imu-rf-reproducibility-v1",
        "reproducibility_status": "exact_match" if not failed else "failed",
        "gates": gates,
        "failed_gates": failed,
        "prediction_mismatch_count": (
            int(
                np.count_nonzero(
                    source_arrays["predictions"] != fresh_arrays["predictions"]
                )
            )
            if source_arrays["predictions"].shape == fresh_arrays["predictions"].shape
            else -1
        ),
        "probability_maximum_absolute_difference": maximum,
        "probability_nonzero_difference_count": nonzero,
        "probability_per_class_maximum_absolute_difference": per_class,
        "source_metrics": {name: source_summary[name] for name in metric_fields},
        "fresh_metrics": {name: fresh_summary[name] for name in metric_fields},
        "source_zero_f1_classes": source_zero,
        "fresh_zero_f1_classes": fresh_zero,
        "tree_comparison": tree_comparison,
    }


def validate_frozen_estimator(model: Any) -> None:
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Expected RandomForestClassifier")
    params = model.get_params(deep=False)
    for name, expected in FROZEN_ESTIMATOR_PARAMS.items():
        if params.get(name) != expected:
            raise ValueError(f"Frozen estimator {name} mismatch")


def validate_formal_source(
    *, compact_root: Path, feature_root: Path, config_path: Path
) -> dict[str, object]:
    config = load_final_config(config_path)
    compact_root = Path(compact_root).resolve(strict=True)
    feature_root = Path(feature_root).resolve(strict=True)
    source_run = select_formal_source_run(compact_root)
    arrays = validate_rf_run(source_run)
    artifacts = _load_features(feature_root, config)
    resolved = _strict_json(source_run / "resolved_config.json")
    summary = _strict_json(source_run / "training_summary.json")
    candidate = resolved.get("candidate")
    expected_candidate = {
        "candidate_id": config["candidate_id"],
        "n_estimators": 150,
        "max_depth": None,
        "min_samples_leaf": 4,
        "min_samples_split": 2,
        "max_samples": None,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced_subsample",
    }
    if (
        resolved.get("config_version") != "imu-rf-compact-screen-v1"
        or resolved.get("fold") != config["fold"]
        or resolved.get("num_classes") != config["num_classes"]
        or resolved.get("feature_schema_version") != config["feature_schema_version"]
        or resolved.get("variant") != "balanced_compact"
        or resolved.get("candidate_id") != config["candidate_id"]
        or resolved.get("random_state") != config["random_state"]
        or candidate != expected_candidate
        or resolved.get("model_compression") != config["model_compression"]
    ):
        raise ValueError("Formal compact RF resolved config mismatch")
    if (
        summary.get("status") != "success"
        or summary.get("train_samples") != 2184
        or summary.get("validation_samples") != 573
        or summary.get("num_classes") != 40
        or summary.get("feature_count") != 2310
        or summary.get("candidate_id") != config["candidate_id"]
        or summary.get("random_state") != config["random_state"]
    ):
        raise ValueError("Formal compact RF training summary mismatch")
    bundle = joblib.load(source_run / "model.joblib")
    if not isinstance(bundle, dict) or set(bundle) != {"model", "metadata"}:
        raise ValueError("Formal compact RF model bundle mismatch")
    model = bundle["model"]
    metadata = bundle["metadata"]
    validate_frozen_estimator(model)
    if not isinstance(metadata, dict):
        raise ValueError("Formal compact RF metadata mismatch")
    validate_sklearn_compatibility(
        str(metadata.get("scikit_learn_version")), sklearn.__version__
    )
    if metadata.get("python_version") != platform.python_version():
        raise ValueError("Formal compact RF Python version mismatch")
    expected_bindings = {
        "feature_schema_sha256": sha256_file(feature_root / "feature_schema.json"),
        "imputer_sha256": sha256_file(feature_root / "imputer.json"),
        "class_order_sha256": artifacts["class_order_sha256"],
    }
    for name, expected in expected_bindings.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Formal compact RF {name} mismatch")
    comparison = _strict_json(compact_root / "compact_rf_comparison.json")
    records = [
        record
        for record in comparison["runs"]
        if record.get("candidate_id") == config["candidate_id"]
        and record.get("random_state") == config["random_state"]
    ]
    if len(records) != 1:
        raise ValueError("Formal compact RF comparison record must be unique")
    if (
        records[0].get("model_sha256") != sha256_file(source_run / "model.joblib")
        or records[0].get("run_manifest_sha256")
        != sha256_file(source_run / "run_manifest.json")
    ):
        raise ValueError("Formal compact RF comparison hashes mismatch")
    if arrays["sample_ids"].shape != (573,):
        raise ValueError("Formal compact RF validation count mismatch")
    return {
        "config": config,
        "source_run": source_run,
        "source_arrays": arrays,
        "model": model,
        "metadata": metadata,
        "resolved_config": resolved,
        "summary": summary,
        "artifacts": artifacts,
        "source_run_snapshot": directory_snapshot(source_run),
        "source_model_sha256": sha256_file(source_run / "model.joblib"),
        "feature_schema_sha256": expected_bindings["feature_schema_sha256"],
        "imputer_sha256": expected_bindings["imputer_sha256"],
        "class_order_sha256": expected_bindings["class_order_sha256"],
        "config_sha256": sha256_file(source_run / "resolved_config.json"),
        "environment": {
            "source_python": metadata["python_version"],
            "source_scikit_learn": metadata["scikit_learn_version"],
            "source_numpy": metadata.get("numpy_version"),
            "source_numpy_recorded": "numpy_version" in metadata,
            "current_python": platform.python_version(),
            "current_scikit_learn": sklearn.__version__,
            "current_numpy": np.__version__,
        },
    }


def _write_finalization_manifest(staging: Path) -> None:
    members = [
        {
            "relative_path": path.relative_to(staging).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(staging.rglob("*"), key=lambda value: value.as_posix())
        if path.is_file() and path.name != "finalization_manifest.json"
    ]
    _write_json(
        staging / "finalization_manifest.json",
        {"manifest_version": "imu-rf-finalization-manifest-v1", "files": members},
    )


def run_finalization(
    *,
    feature_root: Path,
    compact_root: Path,
    config_path: Path,
    compact_config_path: Path,
    output_root: Path | None,
    preflight_only: bool,
) -> dict[str, object]:
    source = validate_formal_source(
        compact_root=compact_root, feature_root=feature_root, config_path=config_path
    )
    preflight = {
        "status": "preflight_ok",
        "source_run": str(source["source_run"]),
        "source_run_snapshot": source["source_run_snapshot"],
        "source_model_sha256": source["source_model_sha256"],
        "feature_schema_sha256": source["feature_schema_sha256"],
        "imputer_sha256": source["imputer_sha256"],
        "class_order_sha256": source["class_order_sha256"],
        "config_sha256": source["config_sha256"],
        "environment": source["environment"],
    }
    if preflight_only:
        return preflight
    if output_root is None:
        raise ValueError("output_root is required for finalization")
    output_root = Path(output_root)
    with _staged_directory(output_root) as staging:
        fresh_run = staging / "reproducibility_seed20260725"
        train_compact_candidate(
            feature_root=feature_root,
            config_path=compact_config_path,
            candidate_id="trees_150_leaf4",
            random_state=20260725,
            output_dir=fresh_run,
            preflight_only=False,
        )
        comparison = compare_reproduction_runs(source["source_run"], fresh_run)
        comparison.update(
            {
                "source_run": str(source["source_run"]),
                "source_run_canonical_sha256": source["source_run_snapshot"][
                    "canonical_sha256"
                ],
                "fresh_run_canonical_sha256": directory_snapshot(fresh_run)[
                    "canonical_sha256"
                ],
                "source_model_sha256": source["source_model_sha256"],
                "fresh_model_sha256": sha256_file(fresh_run / "model.joblib"),
            }
        )
        _write_json(staging / "reproducibility_comparison.json", comparison)
        (staging / "reproducibility_comparison.md").write_text(
            "# Compact RF reproducibility\n\n"
            f"- status: `{comparison['reproducibility_status']}`\n"
            f"- prediction mismatches: {comparison['prediction_mismatch_count']}\n"
            f"- maximum probability difference: "
            f"{comparison['probability_maximum_absolute_difference']!r}\n"
            f"- source tree SHA-256: "
            f"`{comparison['tree_comparison']['source_tree_state_sha256']}`\n"
            f"- fresh tree SHA-256: "
            f"`{comparison['tree_comparison']['fresh_tree_state_sha256']}`\n",
            encoding="utf-8",
        )
        _write_json(staging / "input_snapshot.json", preflight)
        _write_finalization_manifest(staging)
    result = _strict_json(output_root / "reproducibility_comparison.json")
    return {"status": result["reproducibility_status"], **result}


def publish_validation_reference(
    *, finalization_root: Path, feature_root: Path, output_root: Path
) -> dict[str, object]:
    finalization_root = Path(finalization_root).resolve(strict=True)
    comparison = _strict_json(finalization_root / "reproducibility_comparison.json")
    if comparison.get("reproducibility_status") != "exact_match":
        raise ValueError("Fusion reference requires an exact reproduction")
    feature_root = Path(feature_root).resolve(strict=True)
    fresh_run = finalization_root / "reproducibility_seed20260725"
    arrays = validate_rf_run(fresh_run)
    predictions = pd.read_csv(fresh_run / "validation_predictions.csv", keep_default_na=False)
    sample_ids = arrays["sample_ids"].astype(np.str_)
    if predictions["sample_id"].astype(str).tolist() != sample_ids.tolist():
        raise ValueError("Fusion reference user IDs are misaligned")
    user_ids = predictions["user_id"].astype(str).to_numpy(dtype=np.str_)
    source_run_value = comparison.get("source_run")
    if not isinstance(source_run_value, str):
        raise ValueError("Fusion reference source run is missing")
    source_run = Path(source_run_value).resolve(strict=True)
    source_snapshot = directory_snapshot(source_run)
    fresh_snapshot = directory_snapshot(fresh_run)
    metadata = {
        "reference_version": "imu-rf-validation-reference-v1",
        "estimator": "RandomForestClassifier",
        "model_role": "imu_validation_reference",
        "source_kind": "fresh_fold0_reproduction",
        "candidate_id": "trees_150_leaf4",
        "random_state": 20260725,
        "fold": 0,
        "sample_count": len(sample_ids),
        "class_count": len(arrays["class_order"]),
        "source_compact_run_canonical_sha256": source_snapshot["canonical_sha256"],
        "fresh_reproduction_run_canonical_sha256": fresh_snapshot["canonical_sha256"],
        "source_fresh_probability_comparison": {
            "rtol": 0.0,
            "atol": 1e-15,
            "maximum_absolute_difference": comparison[
                "probability_maximum_absolute_difference"
            ],
            "nonzero_difference_count": comparison[
                "probability_nonzero_difference_count"
            ],
        },
        "feature_schema_sha256": sha256_file(feature_root / "feature_schema.json"),
        "imputer_sha256": sha256_file(feature_root / "imputer.json"),
        "class_order_sha256": joblib.load(fresh_run / "model.joblib")["metadata"][
            "class_order_sha256"
        ],
        "config_sha256": sha256_file(fresh_run / "resolved_config.json"),
        "deployment_package_version": "imu-rf-final-v1",
        "usage": "fusion_development_only_not_final_test_inference",
    }
    with _staged_directory(output_root) as staging:
        np.savez(
            staging / "validation_outputs.npz",
            sample_ids=sample_ids,
            user_ids=user_ids,
            labels=arrays["labels"],
            predictions=arrays["predictions"],
            class_probabilities=arrays["class_probabilities"],
            class_order=arrays["class_order"],
        )
        shutil.copyfile(
            fresh_run / "validation_predictions.csv",
            staging / "validation_predictions.csv",
        )
        shutil.copyfile(
            fresh_run / "per_class_metrics.csv", staging / "per_class_metrics.csv"
        )
        _write_json(staging / "reference_metadata.json", metadata)
        members = [
            {
                "relative_path": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.iterdir(), key=lambda value: value.name)
            if path.is_file()
        ]
        _write_json(
            staging / "reference_manifest.json",
            {"manifest_version": "imu-rf-validation-reference-manifest-v1", "files": members},
        )
        with np.load(staging / "validation_outputs.npz", allow_pickle=False) as archive:
            if set(archive.files) != {
                "sample_ids",
                "user_ids",
                "labels",
                "predictions",
                "class_probabilities",
                "class_order",
            }:
                raise ValueError("Fusion reference NPZ field set mismatch")
            probabilities = archive["class_probabilities"]
            if (
                probabilities.dtype != np.float64
                or probabilities.shape != (len(sample_ids), len(arrays["class_order"]))
                or not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
                or not np.array_equal(archive["predictions"], np.argmax(probabilities, axis=1))
            ):
                raise ValueError("Fusion reference probability contract mismatch")
    snapshot = directory_snapshot(output_root)
    return {"status": "success", "sample_count": len(sample_ids), **snapshot}
