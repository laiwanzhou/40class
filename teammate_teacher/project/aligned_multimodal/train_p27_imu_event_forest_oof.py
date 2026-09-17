from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from imu_data import DEVICES, read_index
from probe_p27r3_incremental_information import metric_bundle, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = PROJECT_DIR / "cache" / "imu_32"
DEFAULT_MANIFEST_DIR = PROJECT_DIR / "data" / "p27_strong_inner"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "runs" / "p27_strong_inner" / "imu_event_forest"
)
STATS_PER_SIGNAL = 23
SIGNALS_PER_DEVICE = 8
FEATURES_PER_DEVICE = STATS_PER_SIGNAL * SIGNALS_PER_DEVICE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train richer raw-temporal IMU forest on P27 inner folds"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=27111)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def signal_features(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) == 0:
        return np.zeros(STATS_PER_SIGNAL, dtype=np.float32)
    difference = np.diff(signal)
    centered = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    spectrum_sum = float(spectrum.sum())
    normalized_spectrum = (
        spectrum / spectrum_sum if spectrum_sum > 1e-12 else np.zeros_like(spectrum)
    )
    bands = np.array_split(normalized_spectrum[1:], 3)
    entropy = float(
        -np.sum(
            normalized_spectrum[normalized_spectrum > 1e-12]
            * np.log(normalized_spectrum[normalized_spectrum > 1e-12])
        )
    )
    correlations = []
    variance = float(np.dot(centered, centered))
    for lag in (1, 2, 4, 8):
        if len(signal) <= lag or variance < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(
                float(np.dot(centered[:-lag], centered[lag:]) / variance)
            )
    if len(signal) > 1:
        slope = float(
            np.polyfit(np.linspace(-1.0, 1.0, len(signal)), signal, 1)[0]
        )
        derivative_threshold = float(
            np.median(np.abs(difference))
            + 2.5 * np.median(np.abs(np.abs(difference) - np.median(np.abs(difference))))
        )
        sparse_events = float(np.mean(np.abs(difference) > max(derivative_threshold, 1e-6)))
    else:
        slope = 0.0
        sparse_events = 0.0
    values = [
        float(signal.mean()),
        float(signal.std()),
        float(np.sqrt(np.mean(signal**2))),
        float(signal.min()),
        float(signal.max()),
        float(np.ptp(signal)),
        float(np.quantile(signal, 0.1)),
        float(np.quantile(signal, 0.25)),
        float(np.median(signal)),
        float(np.quantile(signal, 0.75)),
        float(np.quantile(signal, 0.9)),
        float(np.mean(np.abs(difference))) if len(difference) else 0.0,
        float(np.std(difference)) if len(difference) else 0.0,
        slope,
        *[float(band.sum()) if len(band) else 0.0 for band in bands],
        entropy,
        *correlations,
        sparse_events,
    ]
    if len(values) != STATS_PER_SIGNAL:
        raise AssertionError(len(values))
    return np.nan_to_num(np.asarray(values, dtype=np.float32))


def feature_vector(
    values: np.ndarray,
    time_mask: np.ndarray,
    device_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device_features: list[np.ndarray] = []
    mask_features: list[float] = []
    magnitude_sequences: list[tuple[np.ndarray, np.ndarray]] = []
    for device in range(len(DEVICES)):
        valid = time_mask[device] > 0
        signals = [values[device, valid, channel] for channel in range(6)]
        acceleration = np.linalg.norm(values[device, valid, :3], axis=1)
        gyroscope = np.linalg.norm(values[device, valid, 3:6], axis=1)
        signals.extend([acceleration, gyroscope])
        device_features.append(
            np.concatenate([signal_features(signal) for signal in signals])
        )
        mask_features.extend([float(device_mask[device]), float(valid.mean())])
        magnitude_sequences.append((acceleration, gyroscope))
    cross_features: list[float] = []
    for left, right in itertools.combinations(range(len(DEVICES)), 2):
        for signal_index in range(2):
            a = magnitude_sequences[left][signal_index]
            b = magnitude_sequences[right][signal_index]
            length = min(len(a), len(b))
            if length < 3 or a[:length].std() < 1e-6 or b[:length].std() < 1e-6:
                cross_features.append(0.0)
            else:
                cross_features.append(
                    float(np.corrcoef(a[:length], b[:length])[0, 1])
                )
    return (
        np.concatenate(
            [
                *device_features,
                np.asarray(mask_features, dtype=np.float32),
                np.nan_to_num(np.asarray(cross_features, dtype=np.float32)),
            ]
        ),
        np.asarray(mask_features, dtype=np.float32),
    )


def drop_one_device(
    features: np.ndarray,
    masks: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = features.copy()
    mask_offset = FEATURES_PER_DEVICE * len(DEVICES)
    for row_index, mask in enumerate(masks):
        present = np.flatnonzero(mask[::2] > 0)
        if not len(present):
            continue
        device = int(rng.choice(present))
        output[
            row_index,
            device * FEATURES_PER_DEVICE : (device + 1) * FEATURES_PER_DEVICE,
        ] = 0.0
        output[row_index, mask_offset + 2 * device : mask_offset + 2 * device + 2] = 0.0
    return output


def dense_logits(model: ExtraTreesClassifier, source: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(source)
    dense = np.full((len(source), 40), 1e-7, dtype=np.float64)
    dense[:, model.classes_.astype(np.int64)] = probabilities
    dense /= dense.sum(axis=1, keepdims=True)
    return np.log(np.clip(dense, 1e-12, 1.0)).astype(np.float32)


def flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve()
    index_rows = [row for row in read_index(cache / "index.csv") if row.split == "train"]
    values = np.load(cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    time_mask = np.load(cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    device_mask = np.load(cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    feature_by_id: dict[str, np.ndarray] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    usable_by_id = {row.sample_id: row.usable for row in index_rows}
    started = time.perf_counter()
    for row in index_rows:
        feature, mask = feature_vector(
            values[row.cache_index],
            time_mask[row.cache_index],
            device_mask[row.cache_index],
        )
        feature_by_id[row.sample_id] = feature
        mask_by_id[row.sample_id] = mask
    extraction_seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    fold_info: dict[str, Any] = {}
    for fold in range(3):
        manifest = read_csv(args.manifest_dir.resolve() / f"fold_{fold}.csv")
        train = [
            row for row in manifest
            if row["split"] == "train" and usable_by_id.get(row["sample_id"], False)
        ]
        held = [row for row in manifest if row["split"] == "val"]
        fit_features = np.stack([feature_by_id[row["sample_id"]] for row in train])
        fit_masks = np.stack([mask_by_id[row["sample_id"]] for row in train])
        fit_labels = np.asarray([int(row["class_id"]) for row in train], dtype=np.int64)
        dropped = drop_one_device(
            fit_features, fit_masks, np.random.default_rng(int(args.seed) + fold)
        )
        source = np.concatenate([fit_features, dropped])
        source_labels = np.concatenate([fit_labels, fit_labels])
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(args.seed) + fold,
        )
        fit_started = time.perf_counter()
        model.fit(source, source_labels)
        fit_seconds = time.perf_counter() - fit_started
        held_logits = np.zeros((len(held), 40), dtype=np.float32)
        present = np.asarray(
            [usable_by_id.get(row["sample_id"], False) for row in held],
            dtype=bool,
        )
        if present.any():
            held_source = np.stack(
                [
                    feature_by_id[held[index]["sample_id"]]
                    for index in np.flatnonzero(present)
                ]
            )
            held_logits[present] = dense_logits(model, held_source)
        held_logits[~present] = np.log(1.0 / 40.0)
        labels = np.asarray([int(row["class_id"]) for row in held], dtype=np.int64)
        predictions = held_logits.argmax(axis=1)
        metrics = metric_bundle(labels[present], predictions[present])
        rows.append(
            {
                "inner_fold": fold,
                "method": "imu_event_forest_present_only",
                "feature_dim": int(source.shape[1]),
                **flatten_metrics(metrics),
            }
        )
        model_path = output / f"fold_{fold}_imu_event_forest.joblib"
        joblib.dump(model, model_path, compress=3)
        np.savez_compressed(
            output / f"fold_{fold}_logits.npz",
            protocol=np.asarray("p27-imu-event-forest-inner-v1"),
            sample_ids=np.asarray([row["sample_id"] for row in held]),
            labels=labels,
            subjects=np.asarray([row["user_id"] for row in held]),
            logits=held_logits,
            present=present,
            outer_held_predictions_generated=np.asarray(False),
        )
        fold_info[str(fold)] = {
            "train_samples": len(train),
            "held_samples": len(held),
            "held_present": int(present.sum()),
            "fit_seconds": float(fit_seconds),
            "model_bytes": int(model_path.stat().st_size),
            "outer_held_predictions_generated": False,
        }
        print(
            f"fold={fold} imu_event={np.mean(predictions[present] == labels[present]):.4f}",
            flush=True,
        )
    write_csv(output / "fold_metrics.csv", rows)
    summary = {
        "protocol": "p27-imu-event-forest-inner-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "feature_definition": "per-device raw-axis and magnitude distribution, derivative, trend, sparse-event, FFT-band, entropy and autocorrelation features plus cross-device magnitude correlations",
        "feature_extraction_seconds": float(extraction_seconds),
        "folds": fold_info,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
