from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
SMALL_ACTION_IDS = (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IMU OOF and cross-fitted S+D+IMU fusion")
    parser.add_argument("--tcn-root", type=Path, default=PROJECT_DIR / "runs" / "p3_imu_tcn_accgyro")
    parser.add_argument("--tcn-quat-root", type=Path, default=PROJECT_DIR / "runs" / "p3_imu_tcn_accgyroquat")
    parser.add_argument("--stat-root", type=Path, default=PROJECT_DIR / "runs" / "p3_imu_stat")
    parser.add_argument("--sd-root", type=Path, default=PROJECT_DIR / "runs" / "p5_oof_fusion")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "p3_imu_oof")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    return parser.parse_args()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probabilities = softmax(logits / temperature)
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    result = minimize_scalar(
        lambda value: nll(logits, labels, float(value)),
        bounds=(0.2, 8.0),
        method="bounded",
        options={"xatol": 1e-3},
    )
    return float(result.x)


def load_tcn(root: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "sample_ids": [], "labels": [], "logits": [], "folds": []
    }
    for fold in range(3):
        data = np.load(root / f"fold_{fold}" / "val_logits.npz")
        arrays["sample_ids"].append(data["sample_ids"].astype(str))
        arrays["labels"].append(data["labels"].astype(np.int64))
        arrays["logits"].append(data["logits"].astype(np.float64))
        arrays["folds"].append(np.full(len(data["labels"]), fold, dtype=np.int64))
    return {key: np.concatenate(value) for key, value in arrays.items()}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    result = {
        "sample_ids": data["sample_ids"].astype(str),
        "labels": data["labels"].astype(np.int64),
        "logits": data["logits"].astype(np.float64),
        "folds": data["folds"].astype(np.int64),
    }
    if "drop_one_device_logits" in data:
        result["drop_one_device_logits"] = data[
            "drop_one_device_logits"
        ].astype(np.float64)
    return result


def load_sd(root: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "sample_ids": [], "labels": [], "logits": [], "folds": []
    }
    for fold in range(3):
        data = np.load(root / f"fold_{fold}" / "logits.npz")
        arrays["sample_ids"].append(data["sample_ids"].astype(str))
        arrays["labels"].append(data["labels"].astype(np.int64))
        arrays["logits"].append(
            0.6 * data["skeleton_logits"].astype(np.float64)
            + 0.4 * data["depth_logits"].astype(np.float64)
        )
        arrays["folds"].append(np.full(len(data["labels"]), fold, dtype=np.int64))
    return {key: np.concatenate(value) for key, value in arrays.items()}


def align(
    imu: dict[str, np.ndarray], sd: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    imu_index = {sample_id: index for index, sample_id in enumerate(imu["sample_ids"])}
    sd_index = {sample_id: index for index, sample_id in enumerate(sd["sample_ids"])}
    common = sorted(set(imu_index) & set(sd_index))
    ii = np.asarray([imu_index[sample_id] for sample_id in common], dtype=np.int64)
    si = np.asarray([sd_index[sample_id] for sample_id in common], dtype=np.int64)
    if not np.array_equal(imu["labels"][ii], sd["labels"][si]):
        raise RuntimeError("IMU and S+D labels disagree after canonical alignment")
    if not np.array_equal(imu["folds"][ii], sd["folds"][si]):
        raise RuntimeError("IMU and S+D folds disagree after canonical alignment")
    output = {
        "sample_ids": np.asarray(common),
        "labels": imu["labels"][ii],
        "folds": imu["folds"][ii],
        "imu_logits": imu["logits"][ii],
        "sd_logits": sd["logits"][si],
    }
    if "drop_one_device_logits" in imu:
        output["drop_one_device_logits"] = imu["drop_one_device_logits"][ii]
    return output


def source_summary(source: dict[str, np.ndarray]) -> dict[str, object]:
    labels = source["labels"]
    predictions = source["logits"].argmax(1)
    small = np.isin(labels, SMALL_ACTION_IDS)
    folds = [
        {"fold": fold, "samples": int(np.sum(source["folds"] == fold)), **metrics(labels[source["folds"] == fold], predictions[source["folds"] == fold])}
        for fold in range(3)
    ]
    return {
        "samples": len(labels),
        "overall": metrics(labels, predictions),
        "fixed_small_actions": {"samples": int(small.sum()), **metrics(labels[small], predictions[small])},
        "folds": folds,
    }


def cross_fitted_fusion(common: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels = common["labels"]
    folds = common["folds"]
    output = np.zeros_like(common["sd_logits"])
    protocols = []
    for target_fold in range(3):
        tune = folds != target_fold
        target = folds == target_fold
        sd_temperature = fit_temperature(common["sd_logits"][tune], labels[tune])
        imu_temperature = fit_temperature(common["imu_logits"][tune], labels[tune])
        sd_tune = common["sd_logits"][tune] / sd_temperature
        imu_tune = common["imu_logits"][tune] / imu_temperature
        candidates = np.linspace(0.0, 0.5, 11)
        tune_accuracy = [
            accuracy_score(labels[tune], ((1.0 - weight) * sd_tune + weight * imu_tune).argmax(1))
            for weight in candidates
        ]
        best_index = max(range(len(candidates)), key=lambda index: (tune_accuracy[index], -candidates[index]))
        weight = float(candidates[best_index])
        output[target] = (
            (1.0 - weight) * common["sd_logits"][target] / sd_temperature
            + weight * common["imu_logits"][target] / imu_temperature
        )
        protocols.append(
            {
                "target_fold": target_fold,
                "tuning_samples": int(tune.sum()),
                "target_samples": int(target.sum()),
                "sd_temperature": sd_temperature,
                "imu_temperature": imu_temperature,
                "selected_imu_weight": weight,
                "tuning_accuracy": float(tune_accuracy[best_index]),
            }
        )
    return output, protocols


def missing_device_fusion_stress(
    common: dict[str, np.ndarray],
    protocols: list[dict[str, object]],
) -> dict[str, object] | None:
    if "drop_one_device_logits" not in common:
        return None
    labels = common["labels"]
    sd_prediction = common["sd_logits"].argmax(1)
    results = []
    for device_index, device_name in enumerate(("WTC", "WTLA", "WTRA", "WTLL", "WTRL")):
        fused_logits = np.zeros_like(common["sd_logits"])
        for protocol in protocols:
            fold = int(protocol["target_fold"])
            target = common["folds"] == fold
            normal_weight = float(protocol["selected_imu_weight"])
            # One of five body locations is removed, so reduce the residual
            # expert weight in direct proportion to retained device coverage.
            stress_weight = normal_weight * 0.8
            fused_logits[target] = (
                (1.0 - stress_weight)
                * common["sd_logits"][target]
                / float(protocol["sd_temperature"])
                + stress_weight
                * common["drop_one_device_logits"][target, device_index]
                / float(protocol["imu_temperature"])
            )
        prediction = fused_logits.argmax(1)
        results.append(
            {
                "dropped_device": device_name,
                "imu_weight_scale": 0.8,
                **metrics(labels, prediction),
                "fusion_minus_sd_pp": float(
                    100
                    * (
                        accuracy_score(labels, prediction)
                        - accuracy_score(labels, sd_prediction)
                    )
                ),
            }
        )
    return {
        "protocol": "drop one named device for every OOF sample and scale the normal IMU residual weight by 4/5",
        "devices": results,
        "worst_fusion_minus_sd_pp": float(
            min(row["fusion_minus_sd_pp"] for row in results)
        ),
    }


def subject_from_id(sample_id: str) -> str:
    parts = sample_id.split("__")
    if len(parts) < 4:
        raise ValueError(sample_id)
    return parts[2]


def cluster_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    sample_ids: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    users = np.asarray([subject_from_id(value) for value in sample_ids])
    unique_users = np.asarray(sorted(set(users.tolist()), key=lambda value: int(value[4:])))
    indices = {user: np.flatnonzero(users == user) for user in unique_users}
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        selected = np.concatenate([indices[user] for user in sampled])
        deltas[repeat] = (
            accuracy_score(labels[selected], candidate[selected])
            - accuracy_score(labels[selected], baseline[selected])
        )
    return {
        "delta_pp": float(100 * (accuracy_score(labels, candidate) - accuracy_score(labels, baseline))),
        "subject_cluster_bootstrap_95_ci_pp": [
            float(100 * np.percentile(deltas, 2.5)),
            float(100 * np.percentile(deltas, 97.5)),
        ],
    }


def complementarity(
    common: dict[str, np.ndarray],
    fused_logits: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, object]:
    labels = common["labels"]
    sd_prediction = common["sd_logits"].argmax(1)
    imu_prediction = common["imu_logits"].argmax(1)
    fused_prediction = fused_logits.argmax(1)
    small = np.isin(labels, SMALL_ACTION_IDS)
    per_class = []
    per_fold = []
    for fold in range(3):
        fold_mask = common["folds"] == fold
        per_fold.append(
            {
                "fold": fold,
                "samples": int(fold_mask.sum()),
                "sd": metrics(labels[fold_mask], sd_prediction[fold_mask]),
                "imu": metrics(labels[fold_mask], imu_prediction[fold_mask]),
                "cross_fitted_fusion": metrics(
                    labels[fold_mask], fused_prediction[fold_mask]
                ),
                "fusion_minus_sd_pp": float(
                    100
                    * (
                        accuracy_score(labels[fold_mask], fused_prediction[fold_mask])
                        - accuracy_score(labels[fold_mask], sd_prediction[fold_mask])
                    )
                ),
            }
        )
    for class_id in range(40):
        mask = labels == class_id
        per_class.append(
            {
                "class_id": class_id,
                "samples": int(mask.sum()),
                "sd_recall": float(np.mean(sd_prediction[mask] == labels[mask])),
                "imu_recall": float(np.mean(imu_prediction[mask] == labels[mask])),
                "fused_recall": float(np.mean(fused_prediction[mask] == labels[mask])),
                "sd_wrong_imu_correct": int(np.sum((sd_prediction[mask] != labels[mask]) & (imu_prediction[mask] == labels[mask]))),
            }
        )
    return {
        "samples": len(labels),
        "sd": metrics(labels, sd_prediction),
        "imu": metrics(labels, imu_prediction),
        "cross_fitted_fusion": metrics(labels, fused_prediction),
        "fixed_small_actions": {
            "samples": int(small.sum()),
            "sd": metrics(labels[small], sd_prediction[small]),
            "imu": metrics(labels[small], imu_prediction[small]),
            "cross_fitted_fusion": metrics(labels[small], fused_prediction[small]),
        },
        "sd_wrong_imu_correct": int(np.sum((sd_prediction != labels) & (imu_prediction == labels))),
        "oracle_accuracy": float(np.mean((sd_prediction == labels) | (imu_prediction == labels))),
        "per_fold": per_fold,
        "fusion_vs_sd_bootstrap": cluster_bootstrap(
            labels, sd_prediction, fused_prediction, common["sample_ids"], bootstrap_repeats, seed
        ),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "tcn_accgyro": load_tcn(args.tcn_root.resolve()),
        "tcn_accgyroquat": load_tcn(args.tcn_quat_root.resolve()),
        "stat_logistic": load_npz(args.stat_root.resolve() / "logistic_oof.npz"),
        "stat_random_forest": load_npz(args.stat_root.resolve() / "random_forest_oof.npz"),
        "stat_random_forest_device_dropout": load_npz(
            args.stat_root.resolve() / "random_forest_device_dropout_oof.npz"
        ),
    }
    sd = load_sd(args.sd_root.resolve())
    summary: dict[str, object] = {
        "protocol": {
            "imu": "three fixed subject-disjoint OOF folds",
            "fusion": "for each target fold, temperatures and IMU weight are selected using only the other two OOF folds",
            "weights": [round(float(value), 2) for value in np.linspace(0.0, 0.5, 11)],
        },
        "sources": {},
    }
    for name, source in sources.items():
        common = align(source, sd)
        fused_logits, protocols = cross_fitted_fusion(common)
        detail = {
            "imu_only": source_summary(source),
            "common_with_sd": complementarity(
                common, fused_logits, args.bootstrap_repeats, args.seed
            ),
            "cross_fitted_protocols": protocols,
            "missing_device_fusion_stress": missing_device_fusion_stress(
                common, protocols
            ),
        }
        summary["sources"][name] = detail
        np.savez_compressed(
            output / f"{name}_aligned_oof.npz",
            sample_ids=common["sample_ids"],
            labels=common["labels"],
            folds=common["folds"],
            sd_logits=common["sd_logits"],
            imu_logits=common["imu_logits"],
            fused_logits=fused_logits,
        )
        with (output / f"{name}_per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            rows = detail["common_with_sd"]["per_class"]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(name, json.dumps({k: v for k, v in detail["common_with_sd"].items() if k != "per_class"}, ensure_ascii=False), flush=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
