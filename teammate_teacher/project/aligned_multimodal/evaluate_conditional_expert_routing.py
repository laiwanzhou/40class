from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SMALL_ACTION_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fitted low-capacity routing between a base and an expert candidate"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-key", default="base_logits")
    parser.add_argument("--candidate-key", default="candidate_logits")
    parser.add_argument("--present-key", default="expert_present")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
            "macro_f1": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
        }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-9, 1.0)), axis=1
    ) / np.log(probabilities.shape[1])


def top_margin(probabilities: np.ndarray) -> np.ndarray:
    top = np.partition(probabilities, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def scalar_features(
    base_logits: np.ndarray, candidate_logits: np.ndarray
) -> np.ndarray:
    base_prob = softmax(base_logits)
    candidate_prob = softmax(candidate_logits)
    base_pred = base_prob.argmax(1)
    candidate_pred = candidate_prob.argmax(1)
    midpoint = 0.5 * (base_prob + candidate_prob)
    js = 0.5 * np.sum(
        base_prob
        * (
            np.log(np.clip(base_prob, 1e-9, 1.0))
            - np.log(np.clip(midpoint, 1e-9, 1.0))
        ),
        axis=1,
    ) + 0.5 * np.sum(
        candidate_prob
        * (
            np.log(np.clip(candidate_prob, 1e-9, 1.0))
            - np.log(np.clip(midpoint, 1e-9, 1.0))
        ),
        axis=1,
    )
    base_top = base_prob.max(axis=1)
    candidate_top = candidate_prob.max(axis=1)
    return np.column_stack(
        [
            base_top,
            top_margin(base_prob),
            entropy(base_prob),
            candidate_top,
            top_margin(candidate_prob),
            entropy(candidate_prob),
            candidate_top - base_top,
            (base_pred == candidate_pred).astype(np.float64),
            js,
        ]
    )


def build_features(
    base_logits: np.ndarray,
    candidate_logits: np.ndarray,
    include_class: bool,
) -> np.ndarray:
    values = scalar_features(base_logits, candidate_logits)
    if not include_class:
        return values
    base_pred = base_logits.argmax(1)
    candidate_pred = candidate_logits.argmax(1)
    one_hot = np.zeros((len(values), 80), dtype=np.float64)
    one_hot[np.arange(len(values)), base_pred] = 1.0
    one_hot[np.arange(len(values)), 40 + candidate_pred] = 1.0
    return np.concatenate([values, one_hot], axis=1)


def route_cross_fitted(
    labels: np.ndarray,
    folds: np.ndarray,
    base_logits: np.ndarray,
    candidate_logits: np.ndarray,
    present: np.ndarray,
    include_class: bool,
) -> dict[str, object]:
    features = build_features(base_logits, candidate_logits, include_class)
    base_predictions = base_logits.argmax(1)
    candidate_predictions = candidate_logits.argmax(1)
    routed_predictions = base_predictions.copy()
    route_probability = np.zeros(len(labels), dtype=np.float64)
    route_candidate = np.zeros(len(labels), dtype=bool)
    protocols = []
    for held_fold in range(3):
        calibration = (folds != held_fold) & present
        target = (folds == held_fold) & present
        base_correct = base_predictions == labels
        candidate_correct = candidate_predictions == labels
        sensitive = calibration & (base_correct != candidate_correct)
        target_values = candidate_correct[sensitive].astype(np.int64)
        if len(np.unique(target_values)) < 2:
            probability = np.full(
                int(target.sum()), float(target_values[0]) if len(target_values) else 0.0
            )
            coefficients = 0
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.05,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=20260724 + held_fold,
                ),
            )
            model.fit(features[sensitive], target_values)
            probability = model.predict_proba(features[target])[:, 1]
            coefficients = int(features.shape[1])
        route_probability[target] = probability
        selected = probability >= 0.5
        target_indices = np.flatnonzero(target)
        route_candidate[target_indices[selected]] = True
        routed_predictions[target_indices[selected]] = candidate_predictions[
            target_indices[selected]
        ]
        protocols.append(
            {
                "held_fold": held_fold,
                "calibration_present": int(calibration.sum()),
                "calibration_sensitive": int(sensitive.sum()),
                "calibration_candidate_beneficial": int(target_values.sum()),
                "calibration_candidate_harmful": int(
                    len(target_values) - target_values.sum()
                ),
                "target_present": int(target.sum()),
                "target_routed_to_candidate": int(selected.sum()),
                "feature_count": coefficients,
                "threshold": 0.5,
            }
        )
    return {
        "predictions": routed_predictions,
        "probability": route_probability,
        "route_candidate": route_candidate,
        "protocols": protocols,
    }


def bootstrap_delta(
    labels: np.ndarray,
    primary: np.ndarray,
    baseline: np.ndarray,
    sample_ids: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    users = np.asarray([sample_id.split("__")[2] for sample_id in sample_ids])
    unique_users = np.asarray(
        sorted(set(users.tolist()), key=lambda value: int(value[4:]))
    )
    user_indices = {
        user: np.flatnonzero(users == user) for user in unique_users
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        indices = np.concatenate([user_indices[user] for user in sampled])
        deltas[repeat] = (
            accuracy_score(labels[indices], primary[indices])
            - accuracy_score(labels[indices], baseline[indices])
        )
    return {
        "delta_pp": float(
            100
            * (
                accuracy_score(labels, primary)
                - accuracy_score(labels, baseline)
            )
        ),
        "subject_cluster_bootstrap_95_ci_pp": [
            float(100 * np.percentile(deltas, 2.5)),
            float(100 * np.percentile(deltas, 97.5)),
        ],
        "probability_delta_positive": float(np.mean(deltas > 0)),
        "subjects": int(len(unique_users)),
        "repeats": int(repeats),
    }


def summarize_variant(
    name: str,
    result: dict[str, object],
    labels: np.ndarray,
    folds: np.ndarray,
    sample_ids: np.ndarray,
    present: np.ndarray,
    base_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    predictions = result["predictions"]
    small = np.isin(labels, SMALL_ACTION_IDS)
    return {
        "name": name,
        "metrics": metrics(labels, predictions),
        "small_action_metrics": metrics(labels[small], predictions[small]),
        "route_to_candidate": int(result["route_candidate"].sum()),
        "route_rate_among_present": float(
            result["route_candidate"].sum() / max(1, present.sum())
        ),
        "vs_full_candidate": bootstrap_delta(
            labels,
            predictions,
            candidate_predictions,
            sample_ids,
            repeats,
            seed,
        ),
        "vs_base": bootstrap_delta(
            labels,
            predictions,
            base_predictions,
            sample_ids,
            repeats,
            seed + 100,
        ),
        "per_fold": {
            str(fold): {
                "router": metrics(
                    labels[folds == fold], predictions[folds == fold]
                ),
                "base": metrics(
                    labels[folds == fold],
                    base_predictions[folds == fold],
                ),
                "candidate": metrics(
                    labels[folds == fold],
                    candidate_predictions[folds == fold],
                ),
                "routed_to_candidate": int(
                    np.sum((folds == fold) & result["route_candidate"])
                ),
            }
            for fold in range(3)
        },
        "protocols": result["protocols"],
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    with np.load(input_path, allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        base_logits = data[args.base_key].astype(np.float64)
        candidate_logits = data[args.candidate_key].astype(np.float64)
        present = (
            data[args.present_key].astype(bool)
            if args.present_key in data.files
            else np.ones(len(labels), dtype=bool)
        )
    base_predictions = base_logits.argmax(1)
    candidate_predictions = candidate_logits.argmax(1)
    variants = {
        "scalar_only": route_cross_fitted(
            labels,
            folds,
            base_logits,
            candidate_logits,
            present,
            include_class=False,
        ),
        "scalar_plus_predicted_class": route_cross_fitted(
            labels,
            folds,
            base_logits,
            candidate_logits,
            present,
            include_class=True,
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": args.name,
        "protocol": (
            "For each held subject fold, train a regularized logistic gate only on "
            "the other two OOF folds. The gate sees confidence, entropy, margin, "
            "expert disagreement and optionally predicted-class one-hot features. "
            "It never sees the held-fold labels. Expert-missing samples use base."
        ),
        "samples": int(len(labels)),
        "expert_present": int(present.sum()),
        "base": {
            "metrics": metrics(labels, base_predictions),
            "small_action_metrics": metrics(
                labels[np.isin(labels, SMALL_ACTION_IDS)],
                base_predictions[np.isin(labels, SMALL_ACTION_IDS)],
            ),
        },
        "full_candidate": {
            "metrics": metrics(labels, candidate_predictions),
            "small_action_metrics": metrics(
                labels[np.isin(labels, SMALL_ACTION_IDS)],
                candidate_predictions[np.isin(labels, SMALL_ACTION_IDS)],
            ),
        },
        "oracle_choose_base_or_candidate": {
            "accuracy": float(
                np.mean(
                    (base_predictions == labels)
                    | (candidate_predictions == labels)
                )
            )
        },
        "variants": {
            name: summarize_variant(
                name,
                result,
                labels,
                folds,
                sample_ids,
                present,
                base_predictions,
                candidate_predictions,
                int(args.bootstrap_repeats),
                int(args.seed) + index * 1000,
            )
            for index, (name, result) in enumerate(variants.items())
        },
        "source": str(input_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, result in variants.items():
        with (output_dir / f"{name}_oof.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sample_id",
                    "fold",
                    "label",
                    "base_prediction",
                    "candidate_prediction",
                    "router_prediction",
                    "expert_present",
                    "route_probability",
                    "route_to_candidate",
                ]
            )
            writer.writerows(
                zip(
                    sample_ids,
                    folds,
                    labels,
                    base_predictions,
                    candidate_predictions,
                    result["predictions"],
                    present.astype(np.int64),
                    result["probability"],
                    result["route_candidate"].astype(np.int64),
                )
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
