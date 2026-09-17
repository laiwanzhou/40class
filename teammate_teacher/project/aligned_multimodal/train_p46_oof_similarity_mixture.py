from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

import train_p46_oof_stacker as stacker
from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs/p46_videomae_foundation_v1/complete_features.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_70_similarity_mixture_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt expert mixture weights to each unlabeled target user's nearest "
            "training-user VideoMAE distributions."
        )
    )
    parser.add_argument("--manifest", type=Path, default=stacker.DEFAULT_MANIFEST)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    return parser.parse_args()


def fit_weighted_mixture(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    regularization: float = 0.1,
) -> np.ndarray:
    expert_count = probabilities.shape[1]
    sample_weights = sample_weights.astype(np.float64)
    sample_weights /= np.maximum(sample_weights.sum(), 1e-12)

    def objective(raw: np.ndarray) -> float:
        weights = np.exp(raw - raw.max())
        weights /= weights.sum()
        mixture = np.einsum("e,nec->nc", weights, probabilities)
        losses = -np.log(np.clip(mixture[np.arange(len(labels)), labels], 1e-12, 1.0))
        penalty = regularization * expert_count * np.square(
            weights - 1.0 / expert_count
        ).mean()
        return float(np.sum(sample_weights * losses) + penalty)

    result = minimize(
        objective,
        np.zeros(expert_count, dtype=np.float64),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"weighted mixture fit failed: {result.message}")
    weights = np.exp(result.x - result.x.max())
    return weights / weights.sum()


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def user_centroid(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return l2(values[indices].mean(axis=0, keepdims=True))[0]


def nearest_user_sample_weights(
    features: np.ndarray,
    fit_indices: np.ndarray,
    target_indices: np.ndarray,
    users: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, list[str]]:
    target = user_centroid(features, target_indices)
    fit_users = np.unique(users[fit_indices])
    similarity = []
    for user in fit_users:
        indices = fit_indices[users[fit_indices] == user]
        similarity.append(float(user_centroid(features, indices) @ target))
    order = np.argsort(similarity)[::-1]
    selected = fit_users[order[: min(neighbors, len(fit_users))]]
    sample_weights = np.zeros(len(fit_indices), dtype=np.float64)
    for user in selected:
        mask = users[fit_indices] == user
        sample_weights[mask] = 1.0 / max(1, int(mask.sum()))
    return sample_weights, selected.tolist()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = stacker.load_rows(args.manifest.resolve())
    labels = np.asarray([int(row["detail_index"]) for row in rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in rows])
    stacker.LABELS = labels
    train_indices = np.flatnonzero(
        np.asarray([row["p46_split"] == "train" for row in rows])
    )
    val_indices = np.flatnonzero(
        np.asarray([row["p46_split"] == "val" for row in rows])
    )
    expert_logits = np.stack(
        [stacker.load_expert(spec, rows) for spec in stacker.EXPERT_SPECS], axis=0
    )
    with np.load(args.features.resolve(), allow_pickle=False) as data:
        feature_ids = np.asarray(data["sample_ids"]).astype(str)
        raw_features = np.asarray(data["features"], dtype=np.float32)
    index = {value: number for number, value in enumerate(feature_ids)}
    aligned = np.asarray([index[row["sample_id"]] for row in rows])
    features = l2(l2(raw_features[aligned]).mean(axis=1))

    folds = list(
        GroupKFold(n_splits=args.cv_splits).split(
            train_indices, labels[train_indices], groups=users[train_indices]
        )
    )
    configs = [
        {"neighbors": neighbors, "local_fraction": local_fraction}
        for neighbors in (1, 2, 3, 5, 8)
        for local_fraction in (0.25, 0.5, 0.75, 1.0)
    ]
    cv_cache: list[dict[str, Any]] = []
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
        global_weights = stacker.fit_mixture_weights(
            fit_probabilities, labels[fit_indices], 0.1
        )
        local_weights: dict[tuple[str, int], np.ndarray] = {}
        for held_user in np.unique(users[held_indices]):
            user_held = held_indices[users[held_indices] == held_user]
            for neighbors in (1, 2, 3, 5, 8):
                sample_weights, _ = nearest_user_sample_weights(
                    features, fit_indices, user_held, users, neighbors
                )
                local_weights[(held_user, neighbors)] = fit_weighted_mixture(
                    fit_probabilities,
                    labels[fit_indices],
                    sample_weights,
                    regularization=0.1,
                )
        cv_cache.append(
            {
                "held_local": held_local,
                "held_indices": held_indices,
                "held_probabilities": held_probabilities,
                "global_weights": global_weights,
                "local_weights": local_weights,
            }
        )
    result_rows: list[dict[str, Any]] = []
    for config in configs:
        predictions = np.full(len(train_indices), -1, dtype=np.int64)
        for cache in cv_cache:
            held_local = cache["held_local"]
            held_indices = cache["held_indices"]
            held_probabilities = cache["held_probabilities"]
            for held_user in np.unique(users[held_indices]):
                user_mask = users[held_indices] == held_user
                global_weights = cache["global_weights"]
                local_weights = cache["local_weights"][(held_user, config["neighbors"])]
                fraction = float(config["local_fraction"])
                weights = (1.0 - fraction) * global_weights + fraction * local_weights
                predictions[held_local[user_mask]] = stacker.mixture_predictions(
                    held_probabilities[user_mask], weights
                )
        result = stacker.metrics(labels[train_indices], predictions)
        row = {**config, **result}
        result_rows.append(row)
        print(
            f"neighbors={config['neighbors']} local={config['local_fraction']:.2f} "
            f"acc={100*float(result['accuracy']):.2f}%",
            flush=True,
        )
    selected = max(
        result_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -float(row["local_fraction"]),
        ),
    )
    stacker.write_csv(output / "group_cv_results.csv", result_rows)
    print(f"Selected on training users only: {selected}", flush=True)

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
        train_probabilities, labels[train_indices], 0.1
    )
    val_prediction = np.full(len(val_indices), -1, dtype=np.int64)
    nearest_report: dict[str, list[str]] = {}
    user_weights: dict[str, list[float]] = {}
    for val_user in np.unique(users[val_indices]):
        user_mask = users[val_indices] == val_user
        user_indices = val_indices[user_mask]
        sample_weights, nearest = nearest_user_sample_weights(
            features,
            train_indices,
            user_indices,
            users,
            int(selected["neighbors"]),
        )
        local_weights = fit_weighted_mixture(
            train_probabilities,
            labels[train_indices],
            sample_weights,
            regularization=0.1,
        )
        fraction = float(selected["local_fraction"])
        weights = (1.0 - fraction) * global_weights + fraction * local_weights
        val_prediction[user_mask] = stacker.mixture_predictions(
            val_probabilities[user_mask], weights
        )
        nearest_report[val_user] = nearest
        user_weights[val_user] = weights.tolist()
    validation = stacker.metrics(labels[val_indices], val_prediction)
    class_prediction = np.asarray(HARD_CLASS_IDS, dtype=np.int64)[val_prediction]
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
                "prediction": int(class_prediction[local]),
                "correct": int(class_prediction[local] == int(row["class_id"])),
            }
        )
    stacker.write_csv(output / "validation_predictions.csv", prediction_rows)
    summary = {
        "protocol": (
            "Per-target-user expert mixture adapted from nearest unlabeled VideoMAE "
            "user centroids; configuration selected by training-user GroupCV only"
        ),
        "selected_cv": selected,
        "validation": validation,
        "nearest_training_users": nearest_report,
        "expert_names": [spec.name for spec in stacker.EXPERT_SPECS],
        "user_weights": user_weights,
    }
    stacker.write_json(output / "summary.json", summary)
    print(json.dumps({"selected_cv": selected, "validation": validation, "nearest": nearest_report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
