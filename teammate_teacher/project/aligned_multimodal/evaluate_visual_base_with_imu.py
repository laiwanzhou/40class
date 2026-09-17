from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


SMALL_ACTION_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit an existing visual-base OOF expert with IMU logits"
    )
    parser.add_argument("--visual-oof", type=Path, required=True)
    parser.add_argument("--visual-key", default="candidate_logits")
    parser.add_argument("--imu-oof", type=Path, required=True)
    parser.add_argument("--reference-oof", type=Path, default=None)
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


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    maximum = logits.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(
        np.exp(logits - maximum).sum(axis=1)
    )
    return float(np.mean(logsumexp - logits[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.geomspace(0.25, 4.0, 81)
    losses = [nll(logits / value, labels) for value in values]
    return float(values[int(np.argmin(losses))])


def align_visual(
    visual_ids: np.ndarray,
    visual_labels: np.ndarray,
    visual_logits: np.ndarray,
    imu_ids: np.ndarray,
    imu_labels: np.ndarray,
) -> np.ndarray:
    lookup = {
        sample_id: index for index, sample_id in enumerate(visual_ids.tolist())
    }
    missing = [sample_id for sample_id in imu_ids if sample_id not in lookup]
    if missing:
        raise KeyError(f"Visual OOF missing {len(missing)} IMU reference IDs: {missing[:3]}")
    indices = np.asarray([lookup[sample_id] for sample_id in imu_ids], dtype=np.int64)
    if not np.array_equal(visual_labels[indices], imu_labels):
        raise ValueError("Visual/IMU labels differ after sample-ID alignment")
    return visual_logits[indices]


def cross_fitted_fusion(
    labels: np.ndarray,
    folds: np.ndarray,
    visual_logits: np.ndarray,
    imu_logits: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    result = np.zeros_like(visual_logits)
    protocols = []
    weights = np.linspace(0.0, 0.5, 11)
    for held_fold in range(3):
        tune = folds != held_fold
        target = folds == held_fold
        visual_temperature = fit_temperature(visual_logits[tune], labels[tune])
        imu_temperature = fit_temperature(imu_logits[tune], labels[tune])
        candidates = []
        for weight in weights:
            logits = (
                (1.0 - weight) * visual_logits[tune] / visual_temperature
                + weight * imu_logits[tune] / imu_temperature
            )
            prediction = logits.argmax(1)
            candidates.append(
                (
                    float(accuracy_score(labels[tune], prediction)),
                    float(
                        f1_score(
                            labels[tune],
                            prediction,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    -float(weight),
                    float(weight),
                )
            )
        _, _, _, selected_weight = max(candidates)
        result[target] = (
            (1.0 - selected_weight)
            * visual_logits[target]
            / visual_temperature
            + selected_weight * imu_logits[target] / imu_temperature
        )
        protocols.append(
            {
                "held_fold": held_fold,
                "tune_samples": int(tune.sum()),
                "target_samples": int(target.sum()),
                "visual_temperature": visual_temperature,
                "imu_temperature": imu_temperature,
                "selected_imu_weight": selected_weight,
            }
        )
    return result, protocols


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
    indices = {user: np.flatnonzero(users == user) for user in unique_users}
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        selected = np.concatenate([indices[user] for user in sampled])
        deltas[repeat] = (
            accuracy_score(labels[selected], primary[selected])
            - accuracy_score(labels[selected], baseline[selected])
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


def main() -> None:
    args = parse_args()
    with np.load(args.visual_oof.resolve(), allow_pickle=False) as data:
        visual_ids = data["sample_ids"].astype(str)
        visual_labels = data["labels"].astype(np.int64)
        visual_logits = data[args.visual_key].astype(np.float64)
    with np.load(args.imu_oof.resolve(), allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        imu_logits = data["imu_logits"].astype(np.float64)
        current_sd_imu_logits = data["fused_logits"].astype(np.float64)
    aligned_visual_logits = align_visual(
        visual_ids, visual_labels, visual_logits, sample_ids, labels
    )
    fused_logits, protocols = cross_fitted_fusion(
        labels, folds, aligned_visual_logits, imu_logits
    )
    visual_prediction = aligned_visual_logits.argmax(1)
    fused_prediction = fused_logits.argmax(1)
    current_prediction = current_sd_imu_logits.argmax(1)
    small = np.isin(labels, SMALL_ACTION_IDS)
    summary = {
        "protocol": (
            "The alternative visual base is already cross-fitted OOF. For each "
            "held subject fold, visual/IMU temperatures and IMU weight are selected "
            "only on the other two folds."
        ),
        "samples": int(len(labels)),
        "visual_base": {
            "metrics": metrics(labels, visual_prediction),
            "small_action_metrics": metrics(
                labels[small], visual_prediction[small]
            ),
        },
        "visual_plus_imu": {
            "metrics": metrics(labels, fused_prediction),
            "small_action_metrics": metrics(
                labels[small], fused_prediction[small]
            ),
        },
        "current_sd_plus_imu": {
            "metrics": metrics(labels, current_prediction),
            "small_action_metrics": metrics(
                labels[small], current_prediction[small]
            ),
        },
        "visual_plus_imu_vs_current_sd_plus_imu": bootstrap_delta(
            labels,
            fused_prediction,
            current_prediction,
            sample_ids,
            int(args.bootstrap_repeats),
            int(args.seed),
        ),
        "per_fold": {
            str(fold): {
                "visual_base": metrics(
                    labels[folds == fold], visual_prediction[folds == fold]
                ),
                "visual_plus_imu": metrics(
                    labels[folds == fold], fused_prediction[folds == fold]
                ),
                "current_sd_plus_imu": metrics(
                    labels[folds == fold], current_prediction[folds == fold]
                ),
            }
            for fold in range(3)
        },
        "cross_fit_protocols": protocols,
        "sources": {
            "visual_oof": str(args.visual_oof.resolve()),
            "imu_oof": str(args.imu_oof.resolve()),
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "aligned_oof.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        visual_logits=aligned_visual_logits,
        imu_logits=imu_logits,
        fused_logits=fused_logits,
    )
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "fold",
                "label",
                "visual_prediction",
                "visual_imu_prediction",
                "current_sd_imu_prediction",
            ]
        )
        writer.writerows(
            zip(
                sample_ids,
                folds,
                labels,
                visual_prediction,
                fused_prediction,
                current_prediction,
            )
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
