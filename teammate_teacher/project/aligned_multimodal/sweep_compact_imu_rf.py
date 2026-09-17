from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from imu_data import read_index
from run_imu_stat_baseline import (
    drop_devices,
    feature_vector,
    random_present_devices,
)


PROJECT_DIR = Path(__file__).resolve().parent
SMALL_ACTION_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep compact device-dropout IMU Random Forests"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_DIR / "cache" / "imu_32"
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json",
    )
    parser.add_argument(
        "--sd-root",
        type=Path,
        default=PROJECT_DIR / "runs" / "p5_oof_fusion",
    )
    parser.add_argument(
        "--current-oof",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p3_imu_oof"
            / "stat_random_forest_device_dropout_aligned_oof.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p11_compact_imu_rf",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260723)
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


def load_sd(root: Path) -> dict[str, np.ndarray]:
    parts = {
        "sample_ids": [],
        "labels": [],
        "folds": [],
        "logits": [],
    }
    for fold in range(3):
        with np.load(root / f"fold_{fold}" / "logits.npz", allow_pickle=False) as data:
            count = len(data["labels"])
            parts["sample_ids"].append(data["sample_ids"].astype(str))
            parts["labels"].append(data["labels"].astype(np.int64))
            parts["folds"].append(np.full(count, fold, dtype=np.int64))
            parts["logits"].append(
                0.6 * data["skeleton_logits"].astype(np.float64)
                + 0.4 * data["depth_logits"].astype(np.float64)
            )
    return {key: np.concatenate(value) for key, value in parts.items()}


def align_sd(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    imu_logits: np.ndarray,
    sd: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    lookup = {
        sample_id: index for index, sample_id in enumerate(sd["sample_ids"].tolist())
    }
    keep = np.asarray([sample_id in lookup for sample_id in sample_ids], dtype=bool)
    indices = np.asarray(
        [lookup[sample_id] for sample_id in sample_ids[keep]], dtype=np.int64
    )
    if not np.array_equal(labels[keep], sd["labels"][indices]):
        raise ValueError("IMU/S+D labels differ after sample-ID alignment")
    if not np.array_equal(folds[keep], sd["folds"][indices]):
        raise ValueError("IMU/S+D fold IDs differ after sample-ID alignment")
    return {
        "sample_ids": sample_ids[keep],
        "labels": labels[keep],
        "folds": folds[keep],
        "sd_logits": sd["logits"][indices],
        "imu_logits": imu_logits[keep],
    }


def cross_fitted_fusion(
    common: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels = common["labels"]
    folds = common["folds"]
    result = np.zeros_like(common["sd_logits"])
    protocols = []
    weights = np.linspace(0.0, 0.5, 11)
    for held_fold in range(3):
        tune = folds != held_fold
        target = folds == held_fold
        sd_temperature = fit_temperature(common["sd_logits"][tune], labels[tune])
        imu_temperature = fit_temperature(common["imu_logits"][tune], labels[tune])
        candidates = []
        for weight in weights:
            logits = (
                (1.0 - weight) * common["sd_logits"][tune] / sd_temperature
                + weight * common["imu_logits"][tune] / imu_temperature
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
            * common["sd_logits"][target]
            / sd_temperature
            + selected_weight
            * common["imu_logits"][target]
            / imu_temperature
        )
        protocols.append(
            {
                "held_fold": held_fold,
                "sd_temperature": sd_temperature,
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


def model_from(candidate: dict[str, int], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(candidate["n_estimators"]),
        max_depth=int(candidate["max_depth"]),
        min_samples_leaf=int(candidate["min_samples_leaf"]),
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def probability_logits(
    model: RandomForestClassifier, source: np.ndarray
) -> np.ndarray:
    probabilities = model.predict_proba(source)
    logits = np.full((len(source), 40), np.log(1e-12), dtype=np.float64)
    logits[:, model.classes_.astype(np.int64)] = np.log(
        np.clip(probabilities, 1e-12, 1.0)
    )
    return logits


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve()
    rows = [
        row
        for row in read_index(cache / "index.csv")
        if row.split == "train" and row.usable
    ]
    values = np.load(cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    time_mask = np.load(
        cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False
    )
    device_mask = np.load(
        cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False
    )
    features = []
    masks = []
    for row in rows:
        feature, mask = feature_vector(
            values[row.cache_index],
            time_mask[row.cache_index],
            device_mask[row.cache_index],
        )
        features.append(feature)
        masks.append(mask)
    features = np.stack(features)
    masks = np.stack(masks)
    source = np.concatenate([features, masks], axis=1)
    labels = np.asarray([row.class_id for row in rows], dtype=np.int64)
    users = np.asarray([row.user_id for row in rows])
    sample_ids = np.asarray([row.sample_id for row in rows])
    fold_summary = json.loads(
        args.fold_summary.resolve().read_text(encoding="utf-8")
    )
    folds = np.full(len(labels), -1, dtype=np.int64)
    for fold_info in fold_summary["folds"]:
        folds[np.isin(users, fold_info["val_users"])] = int(fold_info["fold"])
    if np.any(folds < 0):
        raise ValueError("Some IMU samples have no subject fold")
    sd = load_sd(args.sd_root.resolve())
    with np.load(args.current_oof.resolve(), allow_pickle=False) as current:
        current_ids = current["sample_ids"].astype(str)
        current_labels = current["labels"].astype(np.int64)
        current_logits = current["fused_logits"].astype(np.float64)
    current_lookup = {
        sample_id: index for index, sample_id in enumerate(current_ids.tolist())
    }

    candidates = [
        {"name": "rf080_d12_l2", "n_estimators": 80, "max_depth": 12, "min_samples_leaf": 2},
        {"name": "rf100_d14_l2", "n_estimators": 100, "max_depth": 14, "min_samples_leaf": 2},
        {"name": "rf120_d16_l2", "n_estimators": 120, "max_depth": 16, "min_samples_leaf": 2},
        {"name": "rf160_d14_l2", "n_estimators": 160, "max_depth": 14, "min_samples_leaf": 2},
        {"name": "rf200_d16_l2", "n_estimators": 200, "max_depth": 16, "min_samples_leaf": 2},
    ]
    summaries = []
    for candidate_index, candidate in enumerate(candidates):
        started = time.time()
        oof_logits = np.zeros((len(labels), 40), dtype=np.float64)
        for fold_info in fold_summary["folds"]:
            fold = int(fold_info["fold"])
            train = np.isin(users, fold_info["train_users"])
            val = np.isin(users, fold_info["val_users"])
            rng = np.random.default_rng(int(args.seed) + fold)
            dropped_features, dropped_masks = drop_devices(
                features[train],
                masks[train],
                random_present_devices(masks[train], rng),
            )
            fit_source = np.concatenate(
                [
                    source[train],
                    np.concatenate([dropped_features, dropped_masks], axis=1),
                ],
                axis=0,
            )
            fit_labels = np.concatenate([labels[train], labels[train]])
            model = model_from(candidate, int(args.seed))
            model.fit(fit_source, fit_labels)
            oof_logits[val] = probability_logits(model, source[val])

        common = align_sd(sample_ids, labels, folds, oof_logits, sd)
        fused_logits, protocols = cross_fitted_fusion(common)
        prediction = fused_logits.argmax(1)
        sd_prediction = common["sd_logits"].argmax(1)
        current_indices = np.asarray(
            [current_lookup[sample_id] for sample_id in common["sample_ids"]],
            dtype=np.int64,
        )
        if not np.array_equal(current_labels[current_indices], common["labels"]):
            raise ValueError("Current OOF labels differ from compact sweep reference")
        current_prediction = current_logits[current_indices].argmax(1)

        rng = np.random.default_rng(int(args.seed))
        dropped_features, dropped_masks = drop_devices(
            features,
            masks,
            random_present_devices(masks, rng),
        )
        full_source = np.concatenate(
            [source, np.concatenate([dropped_features, dropped_masks], axis=1)],
            axis=0,
        )
        full_labels = np.concatenate([labels, labels])
        final_model = model_from(candidate, int(args.seed))
        final_model.fit(full_source, full_labels)
        model_path = output_dir / f"{candidate['name']}.joblib"
        joblib.dump(final_model, model_path, compress=3)
        small = np.isin(common["labels"], SMALL_ACTION_IDS)
        summary = {
            **candidate,
            "model_path": str(model_path),
            "model_size_mib": model_path.stat().st_size / 1024**2,
            "combined_sd_thermal_model_mib": (
                46.33 + 45.07754898071289 + model_path.stat().st_size / 1024**2
            ),
            "oof_samples": int(len(common["labels"])),
            "sd": metrics(common["labels"], sd_prediction),
            "compact_sd_imu": metrics(common["labels"], prediction),
            "current_sd_imu": metrics(common["labels"], current_prediction),
            "compact_small_actions": metrics(
                common["labels"][small], prediction[small]
            ),
            "current_small_actions": metrics(
                common["labels"][small], current_prediction[small]
            ),
            "compact_vs_current": bootstrap_delta(
                common["labels"],
                prediction,
                current_prediction,
                common["sample_ids"],
                int(args.bootstrap_repeats),
                int(args.seed) + candidate_index,
            ),
            "per_fold": {
                str(fold): {
                    "compact": metrics(
                        common["labels"][common["folds"] == fold],
                        prediction[common["folds"] == fold],
                    ),
                    "current": metrics(
                        common["labels"][common["folds"] == fold],
                        current_prediction[common["folds"] == fold],
                    ),
                }
                for fold in range(3)
            },
            "cross_fit_protocols": protocols,
            "seconds": round(time.time() - started, 2),
        }
        np.savez_compressed(
            output_dir / f"{candidate['name']}_aligned_oof.npz",
            sample_ids=common["sample_ids"],
            labels=common["labels"],
            folds=common["folds"],
            sd_logits=common["sd_logits"],
            imu_logits=common["imu_logits"],
            fused_logits=fused_logits,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    result = {
        "protocol": (
            "Same fixed subject folds, same 250 statistical/mask features, same "
            "one-device-dropout augmentation and cross-fitted temperature/weight "
            "selection as the existing 400-tree RF. Only tree count/depth changes."
        ),
        "candidates": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
