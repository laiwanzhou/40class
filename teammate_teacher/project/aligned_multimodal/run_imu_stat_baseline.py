from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imu_data import DEVICES, read_index


PROJECT_DIR = Path(__file__).resolve().parent
SMALL_ACTION_IDS = (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-disjoint IMU statistical baselines")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "cache" / "imu_32")
    parser.add_argument("--fold-summary", type=Path, default=PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "p3_imu_stat")
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def feature_vector(
    values: np.ndarray,
    time_mask: np.ndarray,
    device_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[float] = []
    mask_features: list[float] = []
    for device in range(len(DEVICES)):
        valid = time_mask[device] > 0
        mask_features.extend([float(device_mask[device]), float(valid.mean())])
        for channel in range(6):
            signal = values[device, valid, channel]
            if not len(signal):
                features.extend([0.0] * 8)
                continue
            difference = np.diff(signal)
            features.extend(
                [
                    float(signal.mean()),
                    float(signal.std()),
                    float(np.sqrt(np.mean(signal**2))),
                    float(signal.min()),
                    float(signal.max()),
                    float(np.ptp(signal)),
                    float(np.mean(np.abs(difference))) if len(difference) else 0.0,
                    float(np.mean(difference**2)) if len(difference) else 0.0,
                ]
            )
    return np.asarray(features, dtype=np.float32), np.asarray(mask_features, dtype=np.float32)


def drop_devices(
    features: np.ndarray,
    masks: np.ndarray,
    device_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output_features = features.copy()
    output_masks = masks.copy()
    features_per_device = features.shape[1] // len(DEVICES)
    masks_per_device = masks.shape[1] // len(DEVICES)
    for row_index, device_index in enumerate(device_indices):
        if device_index < 0:
            continue
        output_features[
            row_index,
            device_index * features_per_device : (device_index + 1) * features_per_device,
        ] = 0.0
        output_masks[
            row_index,
            device_index * masks_per_device : (device_index + 1) * masks_per_device,
        ] = 0.0
    return output_features, output_masks


def random_present_devices(
    masks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    selected = np.full(len(masks), -1, dtype=np.int64)
    for index, row in enumerate(masks):
        present = np.flatnonzero(row[::2] > 0)
        if len(present):
            selected[index] = int(rng.choice(present))
    return selected


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve()
    index_rows = [row for row in read_index(cache / "index.csv") if row.split == "train" and row.usable]
    values = np.load(cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    time_mask = np.load(cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    device_mask = np.load(cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    features = []
    mask_only = []
    for row in index_rows:
        feature, mask_feature = feature_vector(
            values[row.cache_index],
            time_mask[row.cache_index],
            device_mask[row.cache_index],
        )
        features.append(feature)
        mask_only.append(mask_feature)
    features_array = np.stack(features)
    mask_array = np.stack(mask_only)
    labels = np.asarray([row.class_id for row in index_rows], dtype=np.int64)
    users = np.asarray([row.user_id for row in index_rows])
    sample_ids = np.asarray([row.sample_id for row in index_rows])
    fold_summary = json.loads(args.fold_summary.resolve().read_text(encoding="utf-8"))

    model_factories = {
        "logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=args.seed,
            ),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        ),
        "random_forest_device_dropout": lambda: RandomForestClassifier(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        ),
        "mask_only_logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=args.seed,
            ),
        ),
    }
    all_results: dict[str, object] = {}
    for model_name, factory in model_factories.items():
        source = (
            mask_array
            if model_name.startswith("mask_only")
            else np.concatenate([features_array, mask_array], axis=1)
        )
        predictions = np.full(len(labels), -1, dtype=np.int64)
        logits = np.full((len(labels), 40), -1e6, dtype=np.float64)
        stress_logits = (
            np.full((len(labels), len(DEVICES), 40), -1e6, dtype=np.float64)
            if model_name.startswith("random_forest")
            else None
        )
        fold_ids = np.full(len(labels), -1, dtype=np.int64)
        fold_metrics = []
        for fold_info in fold_summary["folds"]:
            fold = int(fold_info["fold"])
            train = np.isin(users, fold_info["train_users"])
            val = np.isin(users, fold_info["val_users"])
            model = factory()
            fit_source = source[train]
            fit_labels = labels[train]
            if model_name == "random_forest_device_dropout":
                rng = np.random.default_rng(args.seed + fold)
                train_features, train_masks = drop_devices(
                    features_array[train],
                    mask_array[train],
                    random_present_devices(mask_array[train], rng),
                )
                fit_source = np.concatenate(
                    [
                        np.concatenate([features_array[train], mask_array[train]], axis=1),
                        np.concatenate([train_features, train_masks], axis=1),
                    ],
                    axis=0,
                )
                fit_labels = np.concatenate([labels[train], labels[train]])
            model.fit(fit_source, fit_labels)
            predictions[val] = model.predict(source[val])
            probabilities = model.predict_proba(source[val])
            classes = model.classes_.astype(np.int64)
            val_indices = np.flatnonzero(val)
            logits[np.ix_(val_indices, classes)] = np.log(
                np.clip(probabilities, 1e-12, 1.0)
            )
            fold_ids[val] = fold
            fold_result = {
                "fold": fold,
                "samples": int(val.sum()),
                **metrics(labels[val], predictions[val]),
            }
            if model_name.startswith("random_forest"):
                per_device = {}
                for device_index, device_name in enumerate(DEVICES):
                    dropped_features, dropped_masks = drop_devices(
                        features_array[val],
                        mask_array[val],
                        np.full(int(val.sum()), device_index, dtype=np.int64),
                    )
                    dropped_source = np.concatenate(
                        [dropped_features, dropped_masks], axis=1
                    )
                    dropped_probabilities = model.predict_proba(dropped_source)
                    dropped_predictions = model.classes_[
                        dropped_probabilities.argmax(axis=1)
                    ]
                    per_device[device_name] = metrics(
                        labels[val], dropped_predictions
                    )
                    assert stress_logits is not None
                    stress_logits[
                        np.ix_(
                            val_indices,
                            np.asarray([device_index]),
                            model.classes_.astype(np.int64),
                        )
                    ] = np.log(
                        np.clip(dropped_probabilities[:, None, :], 1e-12, 1.0)
                    )
                fold_result["drop_one_device"] = per_device
            fold_metrics.append(fold_result)
        if np.any(predictions < 0) or np.any(fold_ids < 0):
            raise RuntimeError(f"Incomplete OOF coverage for {model_name}")
        small = np.isin(labels, SMALL_ACTION_IDS)
        majority = int(np.bincount(labels).argmax())
        result = {
            "folds": fold_metrics,
            "overall": metrics(labels, predictions),
            "fixed_small_actions": {"samples": int(small.sum()), **metrics(labels[small], predictions[small])},
            "majority_class_baseline_accuracy": float(np.mean(labels == majority)),
            "majority_class": majority,
        }
        all_results[model_name] = result
        save_arrays = {
            "sample_ids": sample_ids,
            "labels": labels,
            "logits": logits,
            "predictions": predictions,
            "folds": fold_ids,
        }
        if stress_logits is not None:
            save_arrays["drop_one_device_logits"] = stress_logits
        np.savez_compressed(output / f"{model_name}_oof.npz", **save_arrays)
        print(model_name, json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "protocol": "three fixed subject-disjoint folds; fold-local preprocessing; usable IMU trials only",
        "channels": "acc XYZ + gyro XYZ",
        "samples": len(labels),
        "feature_dimension": int(features_array.shape[1] + mask_array.shape[1]),
        "mask_feature_dimension": int(mask_array.shape[1]),
        "models": all_results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "oof_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "user_id"])
        writer.writerows(zip(sample_ids, labels.tolist(), users))


if __name__ == "__main__":
    main()
