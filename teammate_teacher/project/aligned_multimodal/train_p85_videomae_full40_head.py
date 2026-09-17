from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from train_p46_videomae_head import (
    fit_temperature,
    l2_normalize,
    make_model,
    row_standardize,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_TEST_FEATURES = (
    PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz"
)
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select full-40 Large VideoMAE early/late heads using the frozen "
            "three subject-disjoint folds, then refit deployable Test heads."
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--test-features", type=Path, default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def aligned_scores_40(model: Any, values: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(values), dtype=np.float64)
    classes = np.asarray(model.named_steps["ridge"].classes_, dtype=np.int64)
    if scores.ndim != 2 or not set(classes.tolist()).issubset(set(range(40))):
        raise RuntimeError(f"Invalid full-40 Ridge classes: {classes}")
    margin = np.maximum(np.ptp(scores, axis=1, keepdims=True), 1.0)
    floor = np.min(scores, axis=1, keepdims=True) - margin
    output = np.repeat(floor, 40, axis=1)
    output[:, classes] = scores
    return output


def sample_weights(labels: np.ndarray, power: float) -> np.ndarray:
    if power == 0.0:
        return np.ones(len(labels), dtype=np.float64)
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    reference = counts[counts > 0].mean()
    class_weights = np.zeros(40, dtype=np.float64)
    present = counts > 0
    class_weights[present] = np.power(reference / counts[present], power)
    weights = class_weights[labels]
    return weights / weights.mean()


def feature_sets(features: np.ndarray, kinetics_logits: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    logits = np.asarray(kinetics_logits, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (2, 3, 1024):
        raise RuntimeError(f"Unexpected VideoMAE features: {values.shape}")
    if logits.shape != (len(values), 2, 3, 400):
        raise RuntimeError(f"Unexpected Kinetics logits: {logits.shape}")
    values = l2_normalize(values)
    early = values[:, 0]
    late = values[:, 1]
    mean = l2_normalize(values.mean(axis=1))
    difference = late - early
    kinetics = row_standardize(logits.reshape(len(logits), -1))
    return {
        "early": early.reshape(len(values), -1),
        "late": late.reshape(len(values), -1),
        "window_mean": mean.reshape(len(values), -1),
        "early_late": values.reshape(len(values), -1),
        "temporal_delta": np.concatenate((mean, difference), axis=1).reshape(len(values), -1),
        "kinetics": kinetics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = load(args.features)
    test = load(args.test_features)
    p12 = load(args.p12_oof)

    sample_ids = np.asarray(train["sample_ids"]).astype(str)
    users = np.asarray(train["users"]).astype(str)
    labels = np.asarray(train["labels"], dtype=np.int64)
    if len(sample_ids) != 2914 or set(labels.tolist()) != set(range(40)) or len(set(users)) != 18:
        raise RuntimeError("Frozen full-40 feature universe changed")
    p12_ids = np.asarray(p12["sample_ids"]).astype(str)
    p12_lookup = {value: index for index, value in enumerate(p12_ids)}
    if set(sample_ids) != set(p12_lookup):
        raise RuntimeError("Full-40 VideoMAE cache and P12 OOF do not align")
    p12_order = np.asarray([p12_lookup[value] for value in sample_ids], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(p12["labels"], dtype=np.int64)[p12_order]):
        raise RuntimeError("Full-40 VideoMAE labels disagree with P12 OOF")
    folds = np.asarray(p12["folds"], dtype=np.int64)[p12_order]
    if set(folds.tolist()) != {0, 1, 2}:
        raise RuntimeError(f"Unexpected subject folds: {set(folds.tolist())}")

    matrices = feature_sets(train["features"], train["kinetics_logits"])
    test_matrices = feature_sets(test["features"], test["kinetics_logits"])
    if tuple(np.asarray(test["window_names"]).astype(str)) != ("early", "late"):
        raise RuntimeError("Test window order changed")
    if tuple(np.asarray(test["view_names"]).astype(str)) != ("scene", "person", "workspace"):
        raise RuntimeError("Test view order changed")

    rows: list[dict[str, Any]] = []
    oof_by_config: dict[tuple[str, float, float], np.ndarray] = {}
    for name, values in matrices.items():
        for power in (0.0, 0.5, 0.75):
            for alpha in (300.0, 1000.0, 3000.0, 10000.0):
                oof = np.full((len(labels), 40), np.nan, dtype=np.float64)
                for held_fold in (0, 1, 2):
                    fit_indices = np.flatnonzero(folds != held_fold)
                    held_indices = np.flatnonzero(folds == held_fold)
                    model = make_model(alpha)
                    model.fit(
                        values[fit_indices],
                        labels[fit_indices],
                        ridge__sample_weight=sample_weights(labels[fit_indices], power),
                    )
                    oof[held_indices] = aligned_scores_40(model, values[held_indices])
                if not np.isfinite(oof).all():
                    raise RuntimeError(f"Incomplete full-40 OOF: {name}, {power}, {alpha}")
                result = metrics(labels, oof.argmax(axis=1))
                row = {
                    "feature_set": name,
                    "class_weight_power": power,
                    "alpha": alpha,
                    "dimensions": int(values.shape[1]),
                    **result,
                }
                rows.append(row)
                oof_by_config[(name, power, alpha)] = oof
                print(
                    f"{name:15s} dim={values.shape[1]:5d} power={power:.2f} "
                    f"alpha={alpha:7g} acc={100*float(result['accuracy']):.2f}% "
                    f"macro={100*float(result['macro_f1']):.2f}%",
                    flush=True,
                )
    write_csv(output / "group_oof_results.csv", rows)

    selected_by_feature: dict[str, dict[str, Any]] = {}
    payload: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "labels": labels,
        "users": users,
        "folds": folds,
    }
    test_payload: dict[str, np.ndarray] = {
        "sample_ids": np.asarray(test["sample_ids"]).astype(str),
    }
    model_paths: dict[str, str] = {}
    for name in matrices:
        selected = max(
            (row for row in rows if row["feature_set"] == name),
            key=lambda row: (
                float(row["accuracy"]),
                float(row["balanced_accuracy"]),
                float(row["macro_f1"]),
                -float(row["class_weight_power"]),
                -float(row["alpha"]),
            ),
        )
        power = float(selected["class_weight_power"])
        alpha = float(selected["alpha"])
        raw_oof = oof_by_config[(name, power, alpha)]
        temperature = fit_temperature(raw_oof, labels)
        calibrated_oof = raw_oof / temperature
        model = make_model(alpha)
        model.fit(
            matrices[name],
            labels,
            ridge__sample_weight=sample_weights(labels, power),
        )
        test_logits = aligned_scores_40(model, test_matrices[name]) / temperature
        model_path = output / f"final_{name}_head.joblib"
        joblib.dump(model, model_path, compress=3)
        model_paths[name] = str(model_path)
        payload[f"{name}_logits"] = calibrated_oof.astype(np.float32)
        test_payload[f"{name}_logits"] = test_logits.astype(np.float32)
        selected_by_feature[name] = {
            "selected_oof": selected,
            "temperature": temperature,
            "calibrated_oof": metrics(labels, calibrated_oof.argmax(axis=1)),
        }

    oof_path = output / "candidate_oof_logits.npz"
    with oof_path.with_suffix(".npz.building").open("wb") as handle:
        np.savez_compressed(handle, **payload)
    oof_path.with_suffix(".npz.building").replace(oof_path)
    test_path = output / "candidate_test_logits.npz"
    with test_path.with_suffix(".npz.building").open("wb") as handle:
        np.savez_compressed(handle, **test_payload)
    test_path.with_suffix(".npz.building").replace(test_path)

    best_name = max(
        selected_by_feature,
        key=lambda value: (
            float(selected_by_feature[value]["selected_oof"]["accuracy"]),
            float(selected_by_feature[value]["selected_oof"]["balanced_accuracy"]),
            float(selected_by_feature[value]["selected_oof"]["macro_f1"]),
        ),
    )
    p12_logits = np.asarray(p12["final_logits"], dtype=np.float64)[p12_order]
    p12_prediction = p12_logits.argmax(axis=1)
    best_prediction = payload[f"{best_name}_logits"].argmax(axis=1)
    oracle = (p12_prediction == labels) | (best_prediction == labels)
    summary = {
        "protocol": "three frozen subject folds; all hyperparameters selected on full-40 OOF",
        "samples": len(labels),
        "classes": 40,
        "users": 18,
        "selected_by_feature": selected_by_feature,
        "best_feature": best_name,
        "p12_oof": metrics(labels, p12_prediction),
        "p12_visual_oracle": {
            "correct": int(oracle.sum()),
            "total": len(oracle),
            "accuracy": float(oracle.mean()),
        },
        "oof_logits": str(oof_path),
        "test_logits": str(test_path),
        "model_paths": model_paths,
        "test_rows": len(test_payload["sample_ids"]),
        "test_missing_ir_rows": 405 - len(test_payload["sample_ids"]),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
