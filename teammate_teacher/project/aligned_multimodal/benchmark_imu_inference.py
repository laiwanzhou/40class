from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np

from imu_data import read_index
from run_imu_stat_baseline import feature_vector


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cached IMU RF feature extraction and inference")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "cache" / "imu_32")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "runs" / "p3_sd_imu_rf_full18" / "imu_random_forest.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "runs" / "p3_sd_imu_rf_full18" / "inference_benchmark.json",
    )
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = args.cache_dir.resolve()
    rows = read_index(cache / "index.csv")
    test_indices = [index for index, row in enumerate(rows) if row.split == "test"]
    values = np.load(cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    time_mask = np.load(cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    device_mask = np.load(cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False)

    feature_started = time.perf_counter()
    features = []
    masks = []
    for index in test_indices:
        feature, mask = feature_vector(values[index], time_mask[index], device_mask[index])
        features.append(feature)
        masks.append(mask)
    source = np.concatenate([np.stack(features), np.stack(masks)], axis=1)
    feature_seconds = time.perf_counter() - feature_started

    load_started = time.perf_counter()
    model = joblib.load(args.model.resolve())
    load_seconds = time.perf_counter() - load_started
    model.predict_proba(source)
    predict_started = time.perf_counter()
    for _ in range(args.repeats):
        model.predict_proba(source)
    predict_seconds = (time.perf_counter() - predict_started) / args.repeats

    result = {
        "protocol": (
            "CPU benchmark from the committed 32-step cache; excludes raw CSV parsing/cache build "
            "and the existing S+D neural forward pass"
        ),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "test_samples": len(test_indices),
        "feature_dimension_with_masks": int(source.shape[1]),
        "feature_extraction_batch_ms": 1000.0 * feature_seconds,
        "feature_extraction_ms_per_trial": 1000.0 * feature_seconds / len(test_indices),
        "model_load_ms": 1000.0 * load_seconds,
        "rf_predict_batch_ms": 1000.0 * predict_seconds,
        "rf_predict_ms_per_trial": 1000.0 * predict_seconds / len(test_indices),
        "prediction_repeats": args.repeats,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
