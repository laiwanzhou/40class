from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

import train_p46_oof_stacker as stacker
from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_70_oof_router_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a held-user reliability router over independently OOF experts."
    )
    parser.add_argument("--manifest", type=Path, default=stacker.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int((labels == predictions).sum()),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def probability_features(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return [sample*expert, features] and each expert's proposed class."""
    sample_count, expert_count, class_count = probabilities.shape
    predictions = probabilities.argmax(axis=2)
    sorted_probability = np.sort(probabilities, axis=2)
    confidence = sorted_probability[:, :, -1]
    margin = sorted_probability[:, :, -1] - sorted_probability[:, :, -2]
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=2
    ) / np.log(class_count)
    mean_probability = probabilities.mean(axis=1)
    std_probability = probabilities.std(axis=1)
    one_hot_prediction = np.eye(class_count, dtype=np.float64)[predictions]
    consensus = one_hot_prediction.mean(axis=1)

    sample_index = np.arange(sample_count)[:, None]
    probability_at_vote = mean_probability[sample_index, predictions]
    std_at_vote = std_probability[sample_index, predictions]
    consensus_at_vote = consensus[sample_index, predictions]
    mean_top = mean_probability.max(axis=1, keepdims=True).repeat(expert_count, axis=1)
    expert_ids = np.arange(expert_count, dtype=np.float64)[None, :].repeat(sample_count, axis=0)

    features = np.stack(
        (
            expert_ids,
            predictions.astype(np.float64),
            confidence,
            margin,
            entropy,
            consensus_at_vote,
            probability_at_vote,
            std_at_vote,
            probability_at_vote - mean_top,
            confidence - probability_at_vote,
        ),
        axis=2,
    )
    return features.reshape(sample_count * expert_count, -1), predictions


def correctness_targets(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return (predictions == labels[:, None]).reshape(-1).astype(np.int64)


def fit_hgb(
    fit_probabilities: np.ndarray,
    fit_labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> HistGradientBoostingClassifier:
    features, predictions = probability_features(fit_probabilities)
    targets = correctness_targets(predictions, fit_labels)
    model = HistGradientBoostingClassifier(
        learning_rate=float(config["learning_rate"]),
        max_iter=int(config["max_iter"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        categorical_features=np.asarray([True, True] + [False] * 8),
        early_stopping=False,
        random_state=seed,
    )
    model.fit(features, targets)
    return model


def route_hgb(
    model: HistGradientBoostingClassifier,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count, expert_count, _ = probabilities.shape
    features, predictions = probability_features(probabilities)
    reliability = model.predict_proba(features)[:, 1].reshape(sample_count, expert_count)
    selected_expert = reliability.argmax(axis=1)
    selected_prediction = predictions[np.arange(sample_count), selected_expert]
    return selected_prediction, selected_expert


def reliability_table(
    fit_predictions: np.ndarray,
    fit_labels: np.ndarray,
    alpha: float,
) -> np.ndarray:
    expert_count = fit_predictions.shape[1]
    table = np.zeros((expert_count, 21), dtype=np.float64)
    global_rate = (fit_predictions == fit_labels[:, None]).mean(axis=0)
    for expert in range(expert_count):
        for prediction in range(21):
            mask = fit_predictions[:, expert] == prediction
            count = int(mask.sum())
            correct = int((fit_labels[mask] == prediction).sum())
            table[expert, prediction] = (
                correct + alpha * global_rate[expert]
            ) / (count + alpha)
    return table


def route_table(
    fit_probabilities: np.ndarray,
    fit_labels: np.ndarray,
    held_probabilities: np.ndarray,
    alpha: float,
    confidence_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    fit_predictions = fit_probabilities.argmax(axis=2)
    held_predictions = held_probabilities.argmax(axis=2)
    table = reliability_table(fit_predictions, fit_labels, alpha)
    expert_ids = np.arange(held_predictions.shape[1])[None, :]
    score = table[expert_ids, held_predictions]
    confidence = held_probabilities.max(axis=2)
    score = score + confidence_weight * (confidence - 1.0 / 21.0)
    selected_expert = score.argmax(axis=1)
    prediction = held_predictions[np.arange(len(held_predictions)), selected_expert]
    return prediction, selected_expert


def choose_by_cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -float(row["complexity"]),
        ),
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = stacker.load_rows(args.manifest.resolve())
    labels = np.asarray([int(row["detail_index"]) for row in rows], dtype=np.int64)
    stacker.LABELS = labels
    groups = np.asarray([row["user_id"] for row in rows])
    train_indices = np.flatnonzero(np.asarray([row["p46_split"] == "train" for row in rows]))
    val_indices = np.flatnonzero(np.asarray([row["p46_split"] == "val" for row in rows]))
    print(f"Loading {len(stacker.EXPERT_SPECS)} subject-OOF experts...", flush=True)
    expert_logits = np.stack(
        [stacker.load_expert(spec, rows) for spec in stacker.EXPERT_SPECS], axis=0
    )
    cv_folds = list(
        GroupKFold(n_splits=args.cv_splits).split(
            train_indices, labels[train_indices], groups[train_indices]
        )
    )

    table_configs = [
        {
            "model": "table",
            "alpha": alpha,
            "confidence_weight": confidence_weight,
            "complexity": 0,
        }
        for alpha in (5.0, 20.0, 50.0, 100.0)
        for confidence_weight in (0.0, 0.25, 0.5, 1.0)
    ]
    hgb_configs = [
        {
            "model": "hgb",
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": leaves,
            "min_samples_leaf": min_leaf,
            "l2_regularization": l2,
            "complexity": leaves,
        }
        for leaves, min_leaf, l2 in (
            (7, 100, 10.0),
            (15, 100, 10.0),
            (15, 50, 3.0),
            (31, 50, 10.0),
        )
    ]
    configs = table_configs + hgb_configs
    result_rows: list[dict[str, Any]] = []
    for config_number, config in enumerate(configs, start=1):
        predictions = np.full(len(train_indices), -1, dtype=np.int64)
        for fold, (fit_local, held_local) in enumerate(cv_folds):
            fit_indices = train_indices[fit_local]
            held_indices = train_indices[held_local]
            _, temperatures = stacker.calibrated_features(
                expert_logits, fit_indices, fit_indices
            )
            fit_probabilities = stacker.calibrated_probabilities(
                expert_logits, temperatures, fit_indices
            )
            held_probabilities = stacker.calibrated_probabilities(
                expert_logits, temperatures, held_indices
            )
            if config["model"] == "table":
                held_prediction, _ = route_table(
                    fit_probabilities,
                    labels[fit_indices],
                    held_probabilities,
                    float(config["alpha"]),
                    float(config["confidence_weight"]),
                )
            else:
                model = fit_hgb(
                    fit_probabilities, labels[fit_indices], config, args.seed + fold
                )
                held_prediction, _ = route_hgb(model, held_probabilities)
            predictions[held_local] = held_prediction
        row = {**config, **metrics(labels[train_indices], predictions)}
        result_rows.append(row)
        print(
            f"[{config_number:02d}/{len(configs)}] {config['model']} "
            f"acc={100*float(row['accuracy']):.2f}% macro={100*float(row['macro_f1']):.2f}% "
            f"config={config}",
            flush=True,
        )

    selected = choose_by_cv(result_rows)
    fields = sorted({key for row in result_rows for key in row})
    stacker.write_csv(output / "router_cv_results.csv", result_rows, fields)
    print(f"Selected on training-user CV only: {selected}", flush=True)

    _, temperatures = stacker.calibrated_features(
        expert_logits, train_indices, train_indices
    )
    train_probabilities = stacker.calibrated_probabilities(
        expert_logits, temperatures, train_indices
    )
    val_probabilities = stacker.calibrated_probabilities(
        expert_logits, temperatures, val_indices
    )
    if selected["model"] == "table":
        val_prediction, selected_expert = route_table(
            train_probabilities,
            labels[train_indices],
            val_probabilities,
            float(selected["alpha"]),
            float(selected["confidence_weight"]),
        )
    else:
        final_model = fit_hgb(
            train_probabilities, labels[train_indices], selected, args.seed
        )
        val_prediction, selected_expert = route_hgb(final_model, val_probabilities)

    validation = metrics(labels[val_indices], val_prediction)
    class_predictions = np.asarray(HARD_CLASS_IDS, dtype=np.int64)[val_prediction]
    prediction_rows: list[dict[str, Any]] = []
    for local, global_index in enumerate(val_indices):
        row = rows[global_index]
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "user_id": row["user_id"],
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "prediction": int(class_predictions[local]),
                "selected_expert": stacker.EXPERT_SPECS[int(selected_expert[local])].name,
                "correct": int(class_predictions[local] == int(row["class_id"])),
            }
        )
    stacker.write_csv(output / "validation_predictions.csv", prediction_rows)
    expert_usage = {
        stacker.EXPERT_SPECS[index].name: int((selected_expert == index).sum())
        for index in np.unique(selected_expert)
    }
    summary = {
        "protocol": "P46 Detail21 held-user reliability routing over subject-OOF experts",
        "leakage_control": (
            "Router configurations are selected by GroupKFold across the 14 P46-train "
            "users. Temperatures and reliability targets are fit inside each fold. "
            "The four P46 validation users are evaluated only after selection."
        ),
        "selected_cv": selected,
        "validation": validation,
        "validation_expert_usage": expert_usage,
        "temperatures": temperatures.tolist(),
        "expert_names": [spec.name for spec in stacker.EXPERT_SPECS],
    }
    stacker.write_json(output / "summary.json", summary)
    print(json.dumps({"selected_cv": selected, "validation": validation, "expert_usage": expert_usage}, indent=2), flush=True)


if __name__ == "__main__":
    main()
