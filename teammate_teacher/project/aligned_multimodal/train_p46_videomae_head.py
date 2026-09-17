from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs/p46_videomae_foundation_v1/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a low-capacity Detail21 head on P46-train grouped CV, then evaluate "
            "the four frozen target users once."
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int((labels == prediction).sum()),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def row_standardize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(centered.std(axis=-1, keepdims=True), 1e-6)


def feature_sets(features: np.ndarray, kinetics_logits: np.ndarray) -> dict[str, np.ndarray]:
    features = l2_normalize(features.astype(np.float32))
    kinetics = row_standardize(kinetics_logits.astype(np.float32))
    mean_feature = l2_normalize(features.mean(axis=1))
    mean_kinetics = row_standardize(kinetics.mean(axis=1))
    return {
        "scene": features[:, 0],
        "person": features[:, 1],
        "workspace": features[:, 2],
        "mean_views": mean_feature,
        "concat_views": features.reshape(len(features), -1),
        "kinetics_views": kinetics.reshape(len(kinetics), -1),
        "foundation_all": np.concatenate(
            (features.reshape(len(features), -1), kinetics.reshape(len(kinetics), -1)),
            axis=1,
        ),
        "mean_feature_kinetics": np.concatenate((mean_feature, mean_kinetics), axis=1),
    }


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "ridge",
                RidgeClassifier(
                    alpha=alpha,
                    class_weight=None,
                    solver="lsqr",
                    tol=1e-5,
                    max_iter=5000,
                ),
            ),
        )
    )


def aligned_scores(model: Pipeline, values: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(values), dtype=np.float64)
    classes = np.asarray(model.named_steps["ridge"].classes_, dtype=np.int64)
    if scores.ndim != 2 or not set(classes.tolist()).issubset(set(range(21))):
        raise RuntimeError(f"Invalid Detail21 classes learned by Ridge head: {classes}")
    # A subject-disjoint fold can legitimately have no training example for a rare
    # class.  Keep the output schema fixed at 21 classes without leaking the held
    # user back into fitting.  The absent class receives a conservative row-wise
    # floor below every score the fitted head can emit.
    margin = np.maximum(np.ptp(scores, axis=1, keepdims=True), 1.0)
    floor = np.min(scores, axis=1, keepdims=True) - margin
    output = np.repeat(floor, 21, axis=1)
    output[:, classes] = scores
    return output


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scaled = scores / temperature
        log_probability = scaled - logsumexp(scaled, axis=1, keepdims=True)
        return float(-log_probability[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(objective, bounds=(-2.302585, 2.302585), method="bounded")
    if not result.success:
        raise RuntimeError(f"temperature calibration failed: {result.message}")
    return float(np.exp(result.x))


def p12_prediction(path: Path, sample_ids: np.ndarray) -> np.ndarray:
    with np.load(path.resolve(), allow_pickle=False) as data:
        index = {str(value): number for number, value in enumerate(data["sample_ids"])}
        selected = np.asarray([index[value] for value in sample_ids.astype(str)])
        logits = np.asarray(data["sd_imu_logits"], dtype=np.float32)[selected]
    return logits[:, np.asarray(HARD_CLASS_IDS, dtype=np.int64)].argmax(axis=1)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        source_ids = np.asarray(data["source_ids"]).astype(str)
        users = np.asarray(data["users"]).astype(str)
        class_labels = np.asarray(data["labels"], dtype=np.int64)
        views = tuple(np.asarray(data["view_names"]).astype(str))
        features = np.asarray(data["features"], dtype=np.float32)
        kinetics_logits = np.asarray(data["kinetics_logits"], dtype=np.float32)
    if (
        len(sample_ids) != 1384
        or features.ndim != 3
        or features.shape[:2] != (1384, 3)
    ):
        raise RuntimeError(f"VideoMAE feature universe changed: {features.shape}")
    if views != ("scene", "person", "workspace"):
        raise RuntimeError(f"VideoMAE view order changed: {views}")
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    if not set(class_labels).issubset(class_to_index):
        raise RuntimeError("VideoMAE cache contains a non-Detail21 label")
    labels = np.asarray([class_to_index[value] for value in class_labels], dtype=np.int64)
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["detail_selected"] == "1"
        }
    if set(sample_ids) != set(manifest):
        raise RuntimeError("VideoMAE cache and P46 manifest do not align")
    split = np.asarray([manifest[value]["p46_split"] for value in sample_ids])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    if len(train_indices) != 1094 or len(val_indices) != 290:
        raise RuntimeError("Frozen P46 split counts changed")

    matrices = feature_sets(features, kinetics_logits)
    splitter = StratifiedGroupKFold(
        n_splits=args.cv_splits, shuffle=True, random_state=args.seed
    )
    folds = list(
        splitter.split(
            train_indices,
            labels[train_indices],
            groups=users[train_indices],
        )
    )
    alphas = (1.0, 10.0, 100.0, 1000.0, 10000.0)
    cv_rows: list[dict[str, Any]] = []
    cv_scores_by_config: dict[tuple[str, float], np.ndarray] = {}
    cv_fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    for fold, (_, held_local) in enumerate(folds):
        cv_fold_vector[held_local] = fold
    if np.any(cv_fold_vector < 0):
        raise RuntimeError("VideoMAE grouped CV did not cover every P46 training row")
    for feature_name, matrix in matrices.items():
        for alpha in alphas:
            oof_scores = np.full((len(train_indices), 21), np.nan, dtype=np.float64)
            for fold, (fit_local, held_local) in enumerate(folds):
                model = make_model(alpha)
                model.fit(matrix[train_indices[fit_local]], labels[train_indices[fit_local]])
                oof_scores[held_local] = aligned_scores(
                    model, matrix[train_indices[held_local]]
                )
            if not np.isfinite(oof_scores).all():
                raise RuntimeError(f"incomplete OOF scores: {feature_name}, alpha={alpha}")
            result = metrics(labels[train_indices], oof_scores.argmax(axis=1))
            row = {
                "feature_set": feature_name,
                "alpha": alpha,
                "dimensions": int(matrix.shape[1]),
                **result,
            }
            cv_rows.append(row)
            cv_scores_by_config[(feature_name, alpha)] = oof_scores
            print(
                f"CV {feature_name:24s} alpha={alpha:7g} "
                f"acc={100*float(result['accuracy']):.2f}% "
                f"macro={100*float(result['macro_f1']):.2f}%",
                flush=True,
            )
    selected = max(
        cv_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -int(row["dimensions"]),
            -float(row["alpha"]),
        ),
    )
    write_csv(output / "group_cv_results.csv", cv_rows)
    feature_name = str(selected["feature_set"])
    alpha = float(selected["alpha"])
    selected_oof_scores = cv_scores_by_config[(feature_name, alpha)]
    temperature = fit_temperature(selected_oof_scores, labels[train_indices])
    selected_oof_logits = selected_oof_scores / temperature
    print(
        f"Selected on P46-train users only: feature={feature_name} alpha={alpha:g} "
        f"temperature={temperature:.4f}",
        flush=True,
    )

    final_model = make_model(alpha)
    final_model.fit(matrices[feature_name][train_indices], labels[train_indices])
    val_scores = aligned_scores(final_model, matrices[feature_name][val_indices])
    val_logits = val_scores / temperature
    val_prediction = val_logits.argmax(axis=1)
    validation = metrics(labels[val_indices], val_prediction)
    p12 = p12_prediction(args.p12_oof, sample_ids[val_indices])
    p12_metrics = metrics(labels[val_indices], p12)
    oracle = np.logical_or(
        val_prediction == labels[val_indices], p12 == labels[val_indices]
    )
    val_folds = np.full(len(val_indices), args.cv_splits, dtype=np.int64)
    complete_logits = np.concatenate((selected_oof_logits, val_logits), axis=0)
    complete_indices = np.concatenate((train_indices, val_indices))
    complete_folds = np.concatenate((cv_fold_vector, val_folds))
    crossfit_path = output / "crossfit_logits.npz"
    temporary = crossfit_path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=sample_ids[complete_indices],
            source_ids=source_ids[complete_indices],
            labels=class_labels[complete_indices],
            users=users[complete_indices],
            folds=complete_folds,
            logits=complete_logits.astype(np.float32),
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[complete_logits.argmax(axis=1)],
            train_oof_mask=np.concatenate(
                (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
            ),
        )
    temporary.replace(crossfit_path)
    prediction_rows: list[dict[str, Any]] = []
    for local, index in enumerate(val_indices):
        true_class = int(class_labels[index])
        predicted_class = int(HARD_CLASS_IDS[val_prediction[local]])
        prediction_rows.append(
            {
                "sample_id": sample_ids[index],
                "source_id": source_ids[index],
                "user_id": users[index],
                "true_class_id": true_class,
                "predicted_class_id": predicted_class,
                "correct": int(true_class == predicted_class),
                "p12_correct": int(p12[local] == labels[index]),
            }
        )
    write_csv(output / "validation_predictions.csv", prediction_rows)
    joblib.dump(final_model, output / "final_head.joblib", compress=3)
    summary = {
        "protocol": (
            "Frozen VideoMAE features; head/config selected by four-fold grouped CV on "
            "the 14 P46 training users; target four users evaluated once after selection"
        ),
        "selected_cv": selected,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": p12_metrics,
        "p12_videomae_oracle_diagnostic": {
            "correct": int(oracle.sum()),
            "total": int(len(oracle)),
            "accuracy": float(oracle.mean()),
        },
        "required_for_70_percent": 203,
        "crossfit_logits": str(crossfit_path),
        "final_head": str((output / "final_head.joblib").resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
