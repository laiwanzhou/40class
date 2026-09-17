from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from imu_data import read_index
from run_imu_stat_baseline import (
    drop_devices,
    feature_vector,
    random_present_devices,
)


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refit the robust IMU RF and create an S+D+IMU Test candidate")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "cache" / "imu_32")
    parser.add_argument("--oof-summary", type=Path, default=PROJECT_DIR / "runs" / "p3_imu_oof" / "summary.json")
    parser.add_argument(
        "--sd-test-logits",
        type=Path,
        default=PROJECT_DIR / "runs" / "p0_six_modality_audit" / "refit_all18_w040" / "test_logits.npz",
    )
    parser.add_argument(
        "--sd-test-predictions",
        type=Path,
        default=PROJECT_DIR / "runs" / "p0_six_modality_audit" / "refit_all18_w040" / "test_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p3_sd_imu_rf_full18",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve()
    rows = read_index(cache / "index.csv")
    values = np.load(cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    time_mask = np.load(cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    device_mask = np.load(cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False)

    features = []
    masks = []
    for row in rows:
        feature, mask = feature_vector(
            values[row.cache_index], time_mask[row.cache_index], device_mask[row.cache_index]
        )
        features.append(feature)
        masks.append(mask)
    features_array = np.stack(features)
    masks_array = np.stack(masks)
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.split == "train" and row.usable],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.split == "test"],
        dtype=np.int64,
    )
    labels = np.asarray([row.class_id for row in rows], dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    dropped_features, dropped_masks = drop_devices(
        features_array[train_indices],
        masks_array[train_indices],
        random_present_devices(masks_array[train_indices], rng),
    )
    train_source = np.concatenate(
        [
            np.concatenate(
                [features_array[train_indices], masks_array[train_indices]], axis=1
            ),
            np.concatenate([dropped_features, dropped_masks], axis=1),
        ],
        axis=0,
    )
    train_labels = np.concatenate([labels[train_indices], labels[train_indices]])
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    started = time.time()
    model.fit(train_source, train_labels)
    model_path = output / "imu_random_forest.joblib"
    joblib.dump(model, model_path, compress=3)

    test_source = np.concatenate(
        [features_array[test_indices], masks_array[test_indices]], axis=1
    )
    probabilities = model.predict_proba(test_source)
    imu_logits = np.full((len(test_indices), 40), np.log(1e-12), dtype=np.float64)
    imu_logits[:, model.classes_.astype(np.int64)] = np.log(
        np.clip(probabilities, 1e-12, 1.0)
    )
    test_rows = [rows[index] for index in test_indices]
    test_sample_ids = np.asarray([row.sample_id for row in test_rows])
    test_device_counts = np.asarray([row.device_count for row in test_rows], dtype=np.int64)

    sd = np.load(args.sd_test_logits.resolve())
    if not np.array_equal(sd["sample_ids"].astype(str), test_sample_ids):
        raise RuntimeError("IMU Test order does not match S+D Test logits")
    sd_logits = (
        0.6 * sd["skeleton_logits"].astype(np.float64)
        + 0.4 * sd["depth_logits"].astype(np.float64)
    )
    oof_summary = json.loads(args.oof_summary.resolve().read_text(encoding="utf-8"))
    protocols = oof_summary["sources"]["stat_random_forest_device_dropout"][
        "cross_fitted_protocols"
    ]
    sd_temperature = float(np.median([row["sd_temperature"] for row in protocols]))
    imu_temperature = float(np.median([row["imu_temperature"] for row in protocols]))
    base_imu_weight = float(np.median([row["selected_imu_weight"] for row in protocols]))
    imu_weights = base_imu_weight * np.clip(test_device_counts / 5.0, 0.0, 1.0)
    fused_logits = (
        (1.0 - imu_weights[:, None]) * sd_logits / sd_temperature
        + imu_weights[:, None] * imu_logits / imu_temperature
    )
    predictions = fused_logits.argmax(axis=1)
    fused_probabilities = softmax(fused_logits)
    confidence = fused_probabilities.max(axis=1)
    entropy = -(
        fused_probabilities * np.log(np.clip(fused_probabilities, 1e-12, 1.0))
    ).sum(axis=1)

    with args.sd_test_predictions.resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        submission_rows = list(csv.DictReader(handle))
    if len(submission_rows) != len(predictions):
        raise RuntimeError("Submission row count does not match fused predictions")
    with (output / "test_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "prediction"])
        writer.writerows(
            (row["path"], int(prediction))
            for row, prediction in zip(submission_rows, predictions)
        )
    with (output / "test_predictions_detailed.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "prediction",
                "confidence",
                "entropy",
                "imu_device_count",
                "imu_weight",
                "sd_prediction",
                "imu_prediction",
            ]
        )
        writer.writerows(
            zip(
                test_sample_ids,
                predictions.tolist(),
                confidence.tolist(),
                entropy.tolist(),
                test_device_counts.tolist(),
                imu_weights.tolist(),
                sd_logits.argmax(1).tolist(),
                imu_logits.argmax(1).tolist(),
            )
        )
    np.savez_compressed(
        output / "test_logits.npz",
        sample_ids=test_sample_ids,
        sd_logits=sd_logits,
        imu_logits=imu_logits,
        fused_logits=fused_logits,
        imu_device_counts=test_device_counts,
        imu_weights=imu_weights,
    )
    summary = {
        "training_samples_usable": len(train_indices),
        "training_samples_after_device_dropout_augmentation": len(train_source),
        "test_samples": len(test_indices),
        "test_imu_temporally_usable": int(np.sum(test_device_counts > 0)),
        "test_all_imu_missing": int(np.sum(test_device_counts == 0)),
        "test_complete_five_devices": int(np.sum(test_device_counts == 5)),
        "sd_temperature_oof_median": sd_temperature,
        "imu_temperature_oof_median": imu_temperature,
        "base_imu_weight_oof_median": base_imu_weight,
        "per_sample_weight_protocol": "base IMU weight multiplied by device_count/5; all-missing falls back exactly to S+D",
        "predicted_classes": int(len(set(predictions.tolist()))),
        "class_counts": {
            str(class_id): int(np.sum(predictions == class_id)) for class_id in range(40)
        },
        "model_path": str(model_path),
        "model_file_size_mb": model_path.stat().st_size / (1024**2),
        "combined_with_existing_sd_model_mb": model_path.stat().st_size / (1024**2) + 46.33,
        "fit_seconds": round(time.time() - started, 2),
        "oof_reference": oof_summary["sources"]["stat_random_forest_device_dropout"][
            "common_with_sd"
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "oof_reference"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
