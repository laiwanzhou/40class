from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_THERMAL_ROOT = (
    REPO_DIR / "thermal_baseline" / "runs" / "p11_thermal_imagenet"
)
DEFAULT_SD_ROOT = PROJECT_DIR / "runs" / "p5_oof_fusion"
DEFAULT_IMU_OOF = (
    PROJECT_DIR
    / "runs"
    / "p3_imu_oof"
    / "stat_random_forest_device_dropout_aligned_oof.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p11_thermal_fusion"
SMALL_ACTION_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fitted missing-aware Thermal OOF fusion audit"
    )
    parser.add_argument("--thermal-root", type=Path, default=DEFAULT_THERMAL_ROOT)
    parser.add_argument("--sd-root", type=Path, default=DEFAULT_SD_ROOT)
    parser.add_argument(
        "--thermal-logits-name",
        default="val_logits_best_accuracy.npz",
        help="Per-fold Thermal logit filename under fold_N.",
    )
    parser.add_argument(
        "--sd-logits-name",
        default="logits.npz",
        help="Per-fold Skeleton+Depth expert logit filename under fold_N.",
    )
    parser.add_argument("--imu-oof", type=Path, default=DEFAULT_IMU_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
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
    candidates = np.geomspace(0.25, 4.0, 81)
    losses = [nll(logits / temperature, labels) for temperature in candidates]
    return float(candidates[int(np.argmin(losses))])


def load_thermal(root: Path, logits_name: str) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "labels": [],
        "logits": [],
        "folds": [],
    }
    for fold in range(3):
        path = root / f"fold_{fold}" / logits_name
        with np.load(path, allow_pickle=False) as data:
            count = len(data["labels"])
            parts["sample_ids"].append(data["sample_ids"].astype(str))
            parts["labels"].append(data["labels"].astype(np.int64))
            parts["logits"].append(data["logits"].astype(np.float32))
            parts["folds"].append(np.full(count, fold, dtype=np.int64))
    result = {
        key: np.concatenate(values) for key, values in parts.items()
    }
    if len(set(result["sample_ids"].tolist())) != len(result["sample_ids"]):
        raise ValueError("Duplicate Thermal OOF sample IDs")
    return result


def load_sd(root: Path, logits_name: str) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "labels": [],
        "skeleton_logits": [],
        "depth_logits": [],
        "folds": [],
    }
    for fold in range(3):
        path = root / f"fold_{fold}" / logits_name
        with np.load(path, allow_pickle=False) as data:
            count = len(data["labels"])
            for key in (
                "sample_ids",
                "labels",
                "skeleton_logits",
                "depth_logits",
            ):
                parts[key].append(data[key])
            parts["folds"].append(np.full(count, fold, dtype=np.int64))
    result = {
        key: np.concatenate(values) for key, values in parts.items()
    }
    result["sample_ids"] = result["sample_ids"].astype(str)
    result["labels"] = result["labels"].astype(np.int64)
    result["folds"] = result["folds"].astype(np.int64)
    if len(set(result["sample_ids"].tolist())) != len(result["sample_ids"]):
        raise ValueError("Duplicate S+D OOF sample IDs")
    return result


def align_thermal(
    reference_ids: np.ndarray,
    reference_labels: np.ndarray,
    thermal: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        sample_id: index
        for index, sample_id in enumerate(thermal["sample_ids"].tolist())
    }
    present = np.asarray(
        [sample_id in lookup for sample_id in reference_ids], dtype=bool
    )
    logits = np.zeros((len(reference_ids), 40), dtype=np.float32)
    for reference_index, sample_id in enumerate(reference_ids):
        if not present[reference_index]:
            continue
        thermal_index = lookup[sample_id]
        if int(thermal["labels"][thermal_index]) != int(
            reference_labels[reference_index]
        ):
            raise ValueError(f"Label mismatch for {sample_id}")
        logits[reference_index] = thermal["logits"][thermal_index]
    return present, logits


def fused_logits(
    base_logits: np.ndarray,
    thermal_logits: np.ndarray,
    present: np.ndarray,
    weight: float,
    base_temperature: float,
    thermal_temperature: float,
) -> np.ndarray:
    result = base_logits.copy()
    result[present] = (
        (1.0 - weight) * base_logits[present] / base_temperature
        + weight * thermal_logits[present] / thermal_temperature
    )
    return result


def cross_fitted_thermal_residual(
    name: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    base_logits: np.ndarray,
    thermal: dict[str, np.ndarray],
) -> dict[str, object]:
    present, thermal_logits = align_thermal(sample_ids, labels, thermal)
    result_logits = base_logits.copy()
    protocols = []
    weights = np.linspace(0.0, 1.0, 21)
    for held_fold in range(3):
        calibration = folds != held_fold
        calibration_common = calibration & present
        target = folds == held_fold
        base_temperature = fit_temperature(
            base_logits[calibration_common], labels[calibration_common]
        )
        thermal_temperature = fit_temperature(
            thermal_logits[calibration_common], labels[calibration_common]
        )
        candidates = []
        for weight in weights:
            candidate_logits = fused_logits(
                base_logits,
                thermal_logits,
                present,
                float(weight),
                base_temperature,
                thermal_temperature,
            )
            predictions = candidate_logits[calibration].argmax(1)
            accuracy = float(
                accuracy_score(labels[calibration], predictions)
            )
            macro_f1 = float(
                f1_score(
                    labels[calibration],
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )
            candidates.append((accuracy, macro_f1, -float(weight), float(weight)))
        _, _, _, selected_weight = max(candidates)
        target_logits = fused_logits(
            base_logits[target],
            thermal_logits[target],
            present[target],
            selected_weight,
            base_temperature,
            thermal_temperature,
        )
        result_logits[target] = target_logits
        protocols.append(
            {
                "held_fold": held_fold,
                "calibration_samples": int(calibration.sum()),
                "calibration_thermal_present": int(
                    calibration_common.sum()
                ),
                "target_samples": int(target.sum()),
                "target_thermal_present": int((target & present).sum()),
                "base_temperature": base_temperature,
                "thermal_temperature": thermal_temperature,
                "selected_thermal_weight": selected_weight,
            }
        )
    predictions = result_logits.argmax(1)
    base_predictions = base_logits.argmax(1)
    common = present
    return {
        "name": name,
        "sample_ids": sample_ids,
        "labels": labels,
        "folds": folds,
        "present": present,
        "logits": result_logits,
        "predictions": predictions,
        "base_predictions": base_predictions,
        "metrics": metric_dict(labels, predictions),
        "base_metrics": metric_dict(labels, base_predictions),
        "complete_case_metrics": metric_dict(
            labels[common], predictions[common]
        ),
        "complete_case_base_metrics": metric_dict(
            labels[common], base_predictions[common]
        ),
        "small_action_metrics": metric_dict(
            labels[np.isin(labels, SMALL_ACTION_IDS)],
            predictions[np.isin(labels, SMALL_ACTION_IDS)],
        ),
        "small_action_base_metrics": metric_dict(
            labels[np.isin(labels, SMALL_ACTION_IDS)],
            base_predictions[np.isin(labels, SMALL_ACTION_IDS)],
        ),
        "thermal_present": int(present.sum()),
        "thermal_missing_fallback": int((~present).sum()),
        "base_wrong_thermal_fused_correct": int(
            np.sum(common & (base_predictions != labels) & (predictions == labels))
        ),
        "base_correct_thermal_fused_wrong": int(
            np.sum(common & (base_predictions == labels) & (predictions != labels))
        ),
        "protocols": protocols,
    }


def cluster_bootstrap_delta(
    labels: np.ndarray,
    predictions: np.ndarray,
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
            accuracy_score(labels[indices], predictions[indices])
            - accuracy_score(labels[indices], baseline[indices])
        )
    return {
        "delta_pp": float(
            100
            * (
                accuracy_score(labels, predictions)
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


def serialize_result(
    result: dict[str, object], repeats: int, seed: int
) -> dict[str, object]:
    folds = result["folds"]
    labels = result["labels"]
    predictions = result["predictions"]
    baseline = result["base_predictions"]
    return {
        "metrics": result["metrics"],
        "base_metrics": result["base_metrics"],
        "complete_case_metrics": result["complete_case_metrics"],
        "complete_case_base_metrics": result["complete_case_base_metrics"],
        "small_action_metrics": result["small_action_metrics"],
        "small_action_base_metrics": result["small_action_base_metrics"],
        "thermal_present": result["thermal_present"],
        "thermal_missing_fallback": result["thermal_missing_fallback"],
        "base_wrong_thermal_fused_correct": result[
            "base_wrong_thermal_fused_correct"
        ],
        "base_correct_thermal_fused_wrong": result[
            "base_correct_thermal_fused_wrong"
        ],
        "per_fold": {
            str(fold): {
                "fused": metric_dict(
                    labels[folds == fold], predictions[folds == fold]
                ),
                "base": metric_dict(
                    labels[folds == fold], baseline[folds == fold]
                ),
                "thermal_present": int(
                    np.sum((folds == fold) & result["present"])
                ),
            }
            for fold in range(3)
        },
        "vs_base_bootstrap": cluster_bootstrap_delta(
            labels,
            predictions,
            baseline,
            result["sample_ids"],
            repeats,
            seed,
        ),
        "cross_fit_protocols": result["protocols"],
    }


def write_predictions(path: Path, result: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "fold",
                "label",
                "base_prediction",
                "fused_prediction",
                "thermal_present",
            ]
        )
        writer.writerows(
            zip(
                result["sample_ids"],
                result["folds"],
                result["labels"],
                result["base_predictions"],
                result["predictions"],
                result["present"].astype(np.int64),
            )
        )


def per_class_rows(
    name: str, result: dict[str, object]
) -> list[dict[str, object]]:
    rows = []
    for class_id in range(40):
        mask = result["labels"] == class_id
        rows.append(
            {
                "method": name,
                "class_id": class_id,
                "samples": int(mask.sum()),
                "thermal_present": int(np.sum(mask & result["present"])),
                "base_recall": float(
                    np.mean(result["base_predictions"][mask] == class_id)
                ),
                "fused_recall": float(
                    np.mean(result["predictions"][mask] == class_id)
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    thermal = load_thermal(
        args.thermal_root.resolve(), str(args.thermal_logits_name)
    )
    sd = load_sd(args.sd_root.resolve(), str(args.sd_logits_name))
    skeleton_logits = sd["skeleton_logits"].astype(np.float32)
    sd_logits = (
        0.6 * skeleton_logits
        + 0.4 * sd["depth_logits"].astype(np.float32)
    )
    thermal_predictions = thermal["logits"].argmax(1)

    st = cross_fitted_thermal_residual(
        "s_plus_thermal",
        sd["sample_ids"],
        sd["labels"],
        sd["folds"],
        skeleton_logits,
        thermal,
    )
    sdt = cross_fitted_thermal_residual(
        "sd_plus_thermal",
        sd["sample_ids"],
        sd["labels"],
        sd["folds"],
        sd_logits,
        thermal,
    )

    with np.load(args.imu_oof.resolve(), allow_pickle=False) as data:
        imu_ids = data["sample_ids"].astype(str)
        imu_labels = data["labels"].astype(np.int64)
        imu_folds = data["folds"].astype(np.int64)
        imu_base_logits = data["fused_logits"].astype(np.float32)
    sdit = cross_fitted_thermal_residual(
        "sd_imu_plus_thermal",
        imu_ids,
        imu_labels,
        imu_folds,
        imu_base_logits,
        thermal,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"s_plus_thermal": st, "sd_plus_thermal": sdt, "sd_imu_plus_thermal": sdit}
    for name, result in results.items():
        write_predictions(output_dir / f"{name}_oof.csv", result)
        np.savez_compressed(
            output_dir / f"{name}_logits.npz",
            sample_ids=result["sample_ids"],
            labels=result["labels"],
            folds=result["folds"],
            base_logits=result["logits"] * 0.0 + (
                skeleton_logits
                if name == "s_plus_thermal"
                else sd_logits
                if name == "sd_plus_thermal"
                else imu_base_logits
            ),
            candidate_logits=result["logits"],
            expert_present=result["present"].astype(np.int64),
        )

    per_class = []
    for name, result in results.items():
        per_class.extend(per_class_rows(name, result))
    with (output_dir / "per_class.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)

    summary = {
        "protocol": {
            "thermal": (
                "Three subject-disjoint ImageNet Thermal models. Each trial uses "
                "normalized-progress sampling; Thermal has no claim of exact "
                "Depth timestamp alignment."
            ),
            "fusion": (
                "For each held fold, temperatures and the Thermal weight are "
                "selected only on the other two OOF folds. Missing Thermal falls "
                "back exactly to the base logits."
            ),
            "weight_grid": np.linspace(0.0, 1.0, 21).tolist(),
        },
        "thermal_only": {
            "samples": int(len(thermal["labels"])),
            "metrics": metric_dict(
                thermal["labels"], thermal_predictions
            ),
            "small_action_metrics": metric_dict(
                thermal["labels"][
                    np.isin(thermal["labels"], SMALL_ACTION_IDS)
                ],
                thermal_predictions[
                    np.isin(thermal["labels"], SMALL_ACTION_IDS)
                ],
            ),
            "per_fold": {
                str(fold): metric_dict(
                    thermal["labels"][thermal["folds"] == fold],
                    thermal_predictions[thermal["folds"] == fold],
                )
                for fold in range(3)
            },
        },
        "methods": {
            name: serialize_result(
                result,
                int(args.bootstrap_repeats),
                int(args.seed) + index,
            )
            for index, (name, result) in enumerate(results.items())
        },
        "sources": {
            "thermal_root": str(args.thermal_root.resolve()),
            "thermal_logits_name": str(args.thermal_logits_name),
            "sd_root": str(args.sd_root.resolve()),
            "sd_logits_name": str(args.sd_logits_name),
            "imu_oof": str(args.imu_oof.resolve()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
