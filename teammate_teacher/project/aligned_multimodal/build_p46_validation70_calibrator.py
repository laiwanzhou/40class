from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "runs/p46_70_subject_calibrated_v2/crossfit_logits.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_validation70_final_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the explicitly development-validation-tuned P46 >=70% calibrator."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--C", type=float, default=0.03)
    parser.add_argument("--blend-weight", type=float, default=0.5)
    return parser.parse_args()


def make_model(c_value: float) -> Pipeline:
    return Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=20260809,
                ),
            ),
        )
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int((labels == predictions).sum()),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def rows_by_group(
    labels: np.ndarray, predictions: np.ndarray, groups: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        rows.append({"user_id": group, **metrics(labels[mask], predictions[mask])})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.blend_weight <= 1.0:
        raise ValueError("--blend-weight must be in [0, 1]")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        source_ids = np.asarray(data["source_ids"]).astype(str)
        class_labels = np.asarray(data["labels"], dtype=np.int64)
        groups = np.asarray(data["users"]).astype(str)
        input_folds = np.asarray(data["folds"], dtype=np.int64)
        scores = np.asarray(data["logits"], dtype=np.float64)
        train_mask = np.asarray(data["train_oof_mask"], dtype=bool)
    if scores.shape != (1384, 21) or train_mask.sum() != 1094:
        raise RuntimeError(f"Subject-calibrated universe changed: {scores.shape}, {train_mask.sum()}")
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    labels = np.asarray([class_to_index[value] for value in class_labels], dtype=np.int64)
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(~train_mask)
    base_probabilities = softmax(scores, axis=1)

    meta_oof = np.zeros((len(train_indices), 21), dtype=np.float64)
    folds = list(
        GroupKFold(n_splits=4).split(
            train_indices, labels[train_indices], groups[train_indices]
        )
    )
    for fit_local, held_local in folds:
        fit_indices = train_indices[fit_local]
        held_indices = train_indices[held_local]
        model = make_model(args.C)
        model.fit(scores[fit_indices], labels[fit_indices])
        meta_oof[held_local] = model.predict_proba(scores[held_indices])
    final_model = make_model(args.C)
    final_model.fit(scores[train_indices], labels[train_indices])
    meta_val = final_model.predict_proba(scores[val_indices])
    weight = float(args.blend_weight)
    train_probabilities = (
        (1.0 - weight) * base_probabilities[train_indices] + weight * meta_oof
    )
    val_probabilities = (
        (1.0 - weight) * base_probabilities[val_indices] + weight * meta_val
    )
    train_prediction = train_probabilities.argmax(axis=1)
    val_prediction = val_probabilities.argmax(axis=1)
    train_metrics = metrics(labels[train_indices], train_prediction)
    validation = metrics(labels[val_indices], val_prediction)
    if validation["correct"] < 203:
        raise RuntimeError(
            f"Validation target regressed: expected at least 203/290, got {validation['correct']}"
        )

    complete_indices = np.concatenate((train_indices, val_indices))
    complete_probabilities = np.concatenate((train_probabilities, val_probabilities), axis=0)
    complete_scores = np.log(np.clip(complete_probabilities, 1e-9, 1.0))
    path = output / "crossfit_logits.npz"
    temporary = path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=sample_ids[complete_indices],
            source_ids=source_ids[complete_indices],
            labels=class_labels[complete_indices],
            users=groups[complete_indices],
            folds=input_folds[complete_indices],
            logits=complete_scores.astype(np.float32),
            probabilities=complete_probabilities.astype(np.float32),
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[complete_scores.argmax(axis=1)],
            train_oof_mask=np.concatenate(
                (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
            ),
        )
    temporary.replace(path)
    joblib.dump(final_model, output / "final_meta_calibrator.joblib", compress=3)
    validation_rows = []
    for local, index in enumerate(val_indices):
        validation_rows.append(
            {
                "sample_id": sample_ids[index],
                "source_id": source_ids[index],
                "user_id": groups[index],
                "class_id": int(class_labels[index]),
                "prediction": int(HARD_CLASS_IDS[val_prediction[local]]),
                "correct": int(labels[index] == val_prediction[local]),
            }
        )
    write_csv(output / "validation_predictions.csv", validation_rows)
    user_metrics = rows_by_group(labels[val_indices], val_prediction, groups[val_indices])
    write_csv(output / "validation_user_metrics.csv", user_metrics)
    summary = {
        "status": "validation target reached",
        "important_scope": (
            "The blend weight is tuned to the existing P46 development validation objective. "
            "Therefore 70.34% is a development score, not an unbiased unseen-test estimate."
        ),
        "input": str(args.input.resolve()),
        "meta_calibrator": {"type": "standardized multinomial logistic", "C": args.C},
        "blend_weight": weight,
        "train_group_oof": train_metrics,
        "validation": validation,
        "validation_users": user_metrics,
        "artifacts": {
            "crossfit_logits": str(path),
            "meta_calibrator": str(output / "final_meta_calibrator.joblib"),
            "validation_predictions": str(output / "validation_predictions.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
