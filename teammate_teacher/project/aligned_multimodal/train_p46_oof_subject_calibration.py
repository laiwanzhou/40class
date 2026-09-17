from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

import train_p46_oof_stacker as stacker
from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_70_subject_calibrated_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select unlabeled per-subject class-bias correction on P46 training-user OOF."
    )
    parser.add_argument("--manifest", type=Path, default=stacker.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--cv-splits", type=int, default=4)
    return parser.parse_args()


def statistic(probabilities: np.ndarray, name: str) -> np.ndarray:
    log_probabilities = np.log(np.clip(probabilities, 1e-9, 1.0))
    if name == "mean_log_probability":
        return log_probabilities.mean(axis=0)
    if name == "median_log_probability":
        return np.median(log_probabilities, axis=0)
    if name == "log_mean_probability":
        return np.log(np.clip(probabilities.mean(axis=0), 1e-9, 1.0))
    raise ValueError(name)


def adjusted_predictions(
    probabilities: np.ndarray,
    groups: np.ndarray,
    reference_by_row: dict[str, np.ndarray] | np.ndarray,
    estimator: str,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty_like(probabilities, dtype=np.float64)
    for user in np.unique(groups):
        mask = groups == user
        reference = (
            reference_by_row[estimator][mask][0]
            if isinstance(reference_by_row, dict)
            else reference_by_row
        )
        bias = statistic(probabilities[mask], estimator) - reference
        scores[mask] = np.log(np.clip(probabilities[mask], 1e-9, 1.0)) - strength * bias
    return scores.argmax(axis=1), scores


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = stacker.load_rows(args.manifest.resolve())
    stacker.LABELS = np.asarray([int(row["detail_index"]) for row in rows], dtype=np.int64)
    labels = stacker.LABELS
    groups = np.asarray([row["user_id"] for row in rows])
    train_indices = np.flatnonzero(np.asarray([row["p46_split"] == "train" for row in rows]))
    val_indices = np.flatnonzero(np.asarray([row["p46_split"] == "val" for row in rows]))
    expert_logits = np.stack(
        [stacker.load_expert(spec, rows) for spec in stacker.EXPERT_SPECS], axis=0
    )
    estimators = (
        "mean_log_probability",
        "median_log_probability",
        "log_mean_probability",
    )
    oof_probabilities = np.zeros((len(train_indices), 21), dtype=np.float64)
    references = {
        name: np.zeros((len(train_indices), 21), dtype=np.float64) for name in estimators
    }
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    folds = list(
        GroupKFold(n_splits=args.cv_splits).split(
            train_indices, labels[train_indices], groups[train_indices]
        )
    )
    for fold, (fit_local, held_local) in enumerate(folds):
        fold_vector[held_local] = fold
        fit_indices = train_indices[fit_local]
        held_indices = train_indices[held_local]
        _, temperatures = stacker.calibrated_features(expert_logits, fit_indices, fit_indices)
        fit_experts = stacker.calibrated_probabilities(expert_logits, temperatures, fit_indices)
        held_experts = stacker.calibrated_probabilities(expert_logits, temperatures, held_indices)
        weights = stacker.fit_mixture_weights(
            fit_experts, labels[fit_indices], args.regularization
        )
        fit_probabilities = np.einsum("e,ned->nd", weights, fit_experts)
        oof_probabilities[held_local] = np.einsum("e,ned->nd", weights, held_experts)
        for name in estimators:
            references[name][held_local] = statistic(fit_probabilities, name)
    if not np.isfinite(oof_probabilities).all() or np.any(fold_vector < 0):
        raise RuntimeError("Incomplete subject-calibration OOF predictions")

    cv_rows: list[dict[str, Any]] = []
    scores_by_config: dict[tuple[str, float], np.ndarray] = {}
    for estimator in estimators:
        for strength in np.round(np.arange(0.0, 1.61, 0.1), 2):
            prediction, scores = adjusted_predictions(
                oof_probabilities,
                groups[train_indices],
                references,
                estimator,
                float(strength),
            )
            result = stacker.metrics(labels[train_indices], prediction)
            row = {"estimator": estimator, "strength": float(strength), **result}
            cv_rows.append(row)
            scores_by_config[(estimator, float(strength))] = scores
            print(
                f"CV estimator={estimator:24s} strength={strength:.1f} "
                f"acc={100*float(result['accuracy']):.2f}% macro={100*float(result['macro_f1']):.2f}%",
                flush=True,
            )
    selected = max(
        cv_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -float(row["strength"]),
            row["estimator"] == "median_log_probability",
        ),
    )
    stacker.write_csv(output / "group_cv_results.csv", cv_rows)
    _, temperatures = stacker.calibrated_features(
        expert_logits, train_indices, train_indices
    )
    train_experts = stacker.calibrated_probabilities(
        expert_logits, temperatures, train_indices
    )
    val_experts = stacker.calibrated_probabilities(
        expert_logits, temperatures, val_indices
    )
    weights = stacker.fit_mixture_weights(
        train_experts, labels[train_indices], args.regularization
    )
    train_probabilities = np.einsum("e,ned->nd", weights, train_experts)
    val_probabilities = np.einsum("e,ned->nd", weights, val_experts)
    estimator = str(selected["estimator"])
    strength = float(selected["strength"])
    reference = statistic(train_probabilities, estimator)
    val_prediction, val_scores = adjusted_predictions(
        val_probabilities,
        groups[val_indices],
        reference,
        estimator,
        strength,
    )
    validation = stacker.metrics(labels[val_indices], val_prediction)
    complete_indices = np.concatenate((train_indices, val_indices))
    complete_scores = np.concatenate(
        (scores_by_config[(estimator, strength)], val_scores), axis=0
    )
    path = output / "crossfit_logits.npz"
    temporary = path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray([rows[index]["sample_id"] for index in complete_indices]),
            source_ids=np.asarray([rows[index]["source_id"] for index in complete_indices]),
            labels=np.asarray([int(rows[index]["class_id"]) for index in complete_indices]),
            users=groups[complete_indices],
            folds=np.concatenate((fold_vector, np.full(len(val_indices), args.cv_splits, dtype=np.int64))),
            logits=complete_scores.astype(np.float32),
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[complete_scores.argmax(axis=1)],
            train_oof_mask=np.concatenate(
                (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
            ),
        )
    temporary.replace(path)
    prediction_rows = []
    for local, index in enumerate(val_indices):
        prediction_rows.append(
            {
                "sample_id": rows[index]["sample_id"],
                "source_id": rows[index]["source_id"],
                "user_id": rows[index]["user_id"],
                "class_id": int(rows[index]["class_id"]),
                "prediction": int(HARD_CLASS_IDS[val_prediction[local]]),
                "correct": int(labels[index] == val_prediction[local]),
            }
        )
    stacker.write_csv(output / "validation_predictions.csv", prediction_rows)
    summary = {
        "protocol": "Unlabeled per-subject class-bias correction selected on training-user OOF only",
        "expert_count": len(stacker.EXPERT_SPECS),
        "mixture_regularization": args.regularization,
        "selected_cv": selected,
        "validation": validation,
        "temperatures": temperatures.tolist(),
        "weights": weights.tolist(),
        "reference": reference.tolist(),
        "crossfit_logits": str(path),
    }
    stacker.write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
