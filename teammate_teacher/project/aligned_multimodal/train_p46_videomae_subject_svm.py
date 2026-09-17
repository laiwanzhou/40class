from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from p46_protocol import HARD_CLASS_IDS
from train_p46_videomae_head import fit_temperature, p12_prediction, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs/p46_videomae_foundation_v1/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_subject_svm_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select an RBF decision boundary and label-free per-subject centering "
            "using only held-user CV on P46 training users."
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


def l2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def adjust_subjects(
    values: np.ndarray,
    fit_indices: np.ndarray,
    output_indices: np.ndarray,
    users: np.ndarray,
    strength: float,
) -> np.ndarray:
    if strength == 0.0:
        return values[output_indices]
    global_mean = values[fit_indices].mean(axis=0, keepdims=True)
    output = values[output_indices].copy()
    output_users = users[output_indices]
    for user in np.unique(output_users):
        mask = output_users == user
        user_mean = output[mask].mean(axis=0, keepdims=True)
        output[mask] -= strength * (user_mean - global_mean)
    return output


def make_model(c_value: float, gamma_multiplier: float, dimensions: int) -> Pipeline:
    return Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    C=c_value,
                    gamma=gamma_multiplier / dimensions,
                    kernel="rbf",
                    class_weight=None,
                    decision_function_shape="ovr",
                    probability=False,
                    cache_size=2048,
                    tol=1e-3,
                ),
            ),
        )
    )


def aligned_scores(model: Pipeline, values: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(values), dtype=np.float64)
    classes = np.asarray(model.named_steps["svc"].classes_, dtype=np.int64)
    if scores.ndim != 2 or not set(classes.tolist()).issubset(set(range(21))):
        raise RuntimeError(f"invalid SVC classes: {classes}")
    margin = np.maximum(np.ptp(scores, axis=1, keepdims=True), 1.0)
    output = np.repeat(np.min(scores, axis=1, keepdims=True) - margin, 21, axis=1)
    output[:, classes] = scores
    return output


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        source_ids = np.asarray(data["source_ids"]).astype(str)
        users = np.asarray(data["users"]).astype(str)
        class_labels = np.asarray(data["labels"], dtype=np.int64)
        features = np.asarray(data["features"], dtype=np.float32)
    if features.shape != (1384, 3, 768):
        raise RuntimeError(f"VideoMAE feature contract changed: {features.shape}")
    values = l2(l2(features).mean(axis=1))
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    labels = np.asarray([class_to_index[value] for value in class_labels], dtype=np.int64)
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["detail_selected"] == "1"
        }
    split = np.asarray([manifest[value]["p46_split"] for value in sample_ids])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    folds = list(
        StratifiedGroupKFold(
            n_splits=args.cv_splits, shuffle=True, random_state=args.seed
        ).split(train_indices, labels[train_indices], groups=users[train_indices])
    )
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    for fold, (_, held_local) in enumerate(folds):
        fold_vector[held_local] = fold
    configs = [
        (subject_strength, c_value, gamma_multiplier)
        for subject_strength in (0.0, 0.25, 0.5, 0.75, 1.0)
        for c_value in (1.0, 10.0, 100.0)
        for gamma_multiplier in (0.25, 0.5, 1.0, 2.0)
    ]
    cv_rows: list[dict[str, Any]] = []
    oof_by_config: dict[tuple[float, float, float], np.ndarray] = {}
    for number, (subject_strength, c_value, gamma_multiplier) in enumerate(configs, start=1):
        oof_scores = np.full((len(train_indices), 21), np.nan, dtype=np.float64)
        for fit_local, held_local in folds:
            fit_indices = train_indices[fit_local]
            held_indices = train_indices[held_local]
            fit_values = adjust_subjects(
                values, fit_indices, fit_indices, users, subject_strength
            )
            held_values = adjust_subjects(
                values, fit_indices, held_indices, users, subject_strength
            )
            model = make_model(c_value, gamma_multiplier, values.shape[1])
            model.fit(fit_values, labels[fit_indices])
            oof_scores[held_local] = aligned_scores(model, held_values)
        result = metrics(labels[train_indices], oof_scores.argmax(axis=1))
        row = {
            "subject_centering": subject_strength,
            "C": c_value,
            "gamma_multiplier": gamma_multiplier,
            **result,
        }
        cv_rows.append(row)
        oof_by_config[(subject_strength, c_value, gamma_multiplier)] = oof_scores
        print(
            f"[{number:02d}/{len(configs)}] center={subject_strength:g} C={c_value:g} "
            f"gamma={gamma_multiplier:g} acc={100*float(result['accuracy']):.2f}%",
            flush=True,
        )
    selected = max(
        cv_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -float(row["subject_centering"]),
            -float(row["C"]),
        ),
    )
    write_csv(output / "group_cv_results.csv", cv_rows)
    key = (
        float(selected["subject_centering"]),
        float(selected["C"]),
        float(selected["gamma_multiplier"]),
    )
    oof_scores = oof_by_config[key]
    temperature = fit_temperature(oof_scores, labels[train_indices])
    oof_logits = oof_scores / temperature
    print(f"Selected on training users only: {selected}, T={temperature:.4f}", flush=True)
    fit_values = adjust_subjects(
        values, train_indices, train_indices, users, float(selected["subject_centering"])
    )
    val_values = adjust_subjects(
        values, train_indices, val_indices, users, float(selected["subject_centering"])
    )
    final_model = make_model(
        float(selected["C"]), float(selected["gamma_multiplier"]), values.shape[1]
    )
    final_model.fit(fit_values, labels[train_indices])
    val_logits = aligned_scores(final_model, val_values) / temperature
    val_prediction = val_logits.argmax(axis=1)
    validation = metrics(labels[val_indices], val_prediction)
    p12 = p12_prediction(args.p12_oof, sample_ids[val_indices])
    oracle = np.logical_or(val_prediction == labels[val_indices], p12 == labels[val_indices])
    complete_indices = np.concatenate((train_indices, val_indices))
    complete_logits = np.concatenate((oof_logits, val_logits), axis=0)
    crossfit_path = output / "crossfit_logits.npz"
    temporary = crossfit_path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=sample_ids[complete_indices],
            source_ids=source_ids[complete_indices],
            labels=class_labels[complete_indices],
            users=users[complete_indices],
            folds=np.concatenate(
                (fold_vector, np.full(len(val_indices), args.cv_splits, dtype=np.int64))
            ),
            logits=complete_logits.astype(np.float32),
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[complete_logits.argmax(axis=1)],
            train_oof_mask=np.concatenate(
                (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
            ),
        )
    temporary.replace(crossfit_path)
    joblib.dump(
        {
            "model": final_model,
            "subject_centering": float(selected["subject_centering"]),
            "training_global_mean": values[train_indices].mean(axis=0),
        },
        output / "final_model.joblib",
        compress=3,
    )
    summary = {
        "protocol": (
            "RBF-SVM and optional label-free per-user centering selected only by "
            "held-training-user CV; target users evaluated after selection"
        ),
        "transductive_note": (
            "Nonzero centering uses the unlabeled batch mean of each target user; "
            "no target label enters the transformation."
        ),
        "selected_cv": selected,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_svm_oracle_diagnostic": {
            "correct": int(oracle.sum()),
            "total": int(len(oracle)),
            "accuracy": float(oracle.mean()),
        },
        "crossfit_logits": str(crossfit_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
