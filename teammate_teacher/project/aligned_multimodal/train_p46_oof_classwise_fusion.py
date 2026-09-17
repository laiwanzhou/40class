from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

import train_p46_oof_stacker as stacker
from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_70_oof_classwise_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a low-capacity class-conditional reliability mixture from subject-OOF "
            "experts. All model selection is restricted to P46 training users."
        )
    )
    parser.add_argument("--manifest", type=Path, default=stacker.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def reliability(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    mode: str,
) -> np.ndarray:
    predictions = probabilities.argmax(axis=2)
    expert_count = probabilities.shape[1]
    result = np.empty((expert_count, 21), dtype=np.float64)
    global_accuracy = (predictions == labels[:, None]).mean(axis=0)
    for expert in range(expert_count):
        for class_index in range(21):
            if mode == "precision":
                mask = predictions[:, expert] == class_index
                successes = int((labels[mask] == class_index).sum())
            elif mode == "recall":
                mask = labels == class_index
                successes = int((predictions[mask, expert] == class_index).sum())
            else:
                raise ValueError(mode)
            result[expert, class_index] = (
                successes + alpha * global_accuracy[expert]
            ) / (int(mask.sum()) + alpha)
    return result


def predict(
    fit_probabilities: np.ndarray,
    fit_labels: np.ndarray,
    output_probabilities: np.ndarray,
    global_weights: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    mode = str(config["mode"])
    alpha = float(config["alpha"])
    gamma = float(config["gamma"])
    if mode == "geometric":
        precision = reliability(fit_probabilities, fit_labels, alpha, "precision")
        recall = reliability(fit_probabilities, fit_labels, alpha, "recall")
        class_reliability = np.sqrt(np.maximum(precision * recall, 1e-12))
    else:
        class_reliability = reliability(fit_probabilities, fit_labels, alpha, mode)
    class_weights = global_weights[:, None] * np.power(
        np.maximum(class_reliability, 1e-8), gamma
    )
    # Normalize separately for every candidate class.  This changes which experts
    # provide evidence for a class without injecting the training class frequency.
    class_weights /= np.maximum(class_weights.sum(axis=0, keepdims=True), 1e-12)
    scores = np.einsum("ec,nec->nc", class_weights, output_probabilities)
    return scores.argmax(axis=1)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = stacker.load_rows(args.manifest.resolve())
    labels = np.asarray([int(row["detail_index"]) for row in rows], dtype=np.int64)
    groups = np.asarray([row["user_id"] for row in rows])
    stacker.LABELS = labels
    train_indices = np.flatnonzero(
        np.asarray([row["p46_split"] == "train" for row in rows])
    )
    val_indices = np.flatnonzero(
        np.asarray([row["p46_split"] == "val" for row in rows])
    )
    print(f"Loading {len(stacker.EXPERT_SPECS)} subject-OOF experts...", flush=True)
    expert_logits = np.stack(
        [stacker.load_expert(spec, rows) for spec in stacker.EXPERT_SPECS], axis=0
    )
    folds = list(
        GroupKFold(n_splits=args.cv_splits).split(
            train_indices, labels[train_indices], groups[train_indices]
        )
    )
    mixture_regularizations = (0.01, 0.1)
    fold_cache: list[dict[str, Any]] = []
    for fit_local, held_local in folds:
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
        weights = {
            regularization: stacker.fit_mixture_weights(
                fit_probabilities, labels[fit_indices], regularization
            )
            for regularization in mixture_regularizations
        }
        fold_cache.append(
            {
                "fit_local": fit_local,
                "held_local": held_local,
                "fit_indices": fit_indices,
                "fit_probabilities": fit_probabilities,
                "held_probabilities": held_probabilities,
                "weights": weights,
            }
        )
    configs = [
        {
            "mode": mode,
            "alpha": alpha,
            "gamma": gamma,
            "mixture_regularization": mixture_regularization,
        }
        for mode in ("precision", "recall", "geometric")
        for alpha in (10.0, 30.0, 100.0)
        for gamma in (0.5, 1.0, 2.0, 4.0)
        for mixture_regularization in mixture_regularizations
    ]
    result_rows: list[dict[str, Any]] = []
    for config_number, config in enumerate(configs, start=1):
        predictions = np.full(len(train_indices), -1, dtype=np.int64)
        for cached in fold_cache:
            fit_local = cached["fit_local"]
            held_local = cached["held_local"]
            fit_indices = cached["fit_indices"]
            fit_probabilities = cached["fit_probabilities"]
            held_probabilities = cached["held_probabilities"]
            global_weights = cached["weights"][
                float(config["mixture_regularization"])
            ]
            predictions[held_local] = predict(
                fit_probabilities,
                labels[fit_indices],
                held_probabilities,
                global_weights,
                config,
            )
        result = stacker.metrics(labels[train_indices], predictions)
        result_rows.append({**config, **result})
        print(
            f"[{config_number:02d}/{len(configs)}] mode={config['mode']:9s} "
            f"alpha={config['alpha']:g} gamma={config['gamma']:g} "
            f"reg={config['mixture_regularization']:g} "
            f"acc={100*float(result['accuracy']):.2f}%",
            flush=True,
        )
    selected = max(
        result_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -float(row["gamma"]),
        ),
    )
    stacker.write_csv(output / "group_cv_results.csv", result_rows)
    print(f"Selected on P46-train users only: {selected}", flush=True)

    _, temperatures = stacker.calibrated_features(
        expert_logits, train_indices, train_indices
    )
    train_probabilities = stacker.calibrated_probabilities(
        expert_logits, temperatures, train_indices
    )
    val_probabilities = stacker.calibrated_probabilities(
        expert_logits, temperatures, val_indices
    )
    global_weights = stacker.fit_mixture_weights(
        train_probabilities,
        labels[train_indices],
        float(selected["mixture_regularization"]),
    )
    val_prediction = predict(
        train_probabilities,
        labels[train_indices],
        val_probabilities,
        global_weights,
        selected,
    )
    validation = stacker.metrics(labels[val_indices], val_prediction)
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
                "correct": int(class_predictions[local] == int(row["class_id"])),
            }
        )
    stacker.write_csv(output / "validation_predictions.csv", prediction_rows)
    summary = {
        "protocol": "Subject-OOF class-conditional reliability fusion",
        "leakage_control": (
            "Configuration selection and all reliability estimates use only grouped "
            "P46 training-user folds; target users are evaluated once after selection."
        ),
        "selected_cv": selected,
        "validation": validation,
        "expert_names": [spec.name for spec in stacker.EXPERT_SPECS],
        "global_weights": global_weights.tolist(),
        "temperatures": temperatures.tolist(),
    }
    stacker.write_json(output / "summary.json", summary)
    print(json.dumps({"selected_cv": selected, "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
