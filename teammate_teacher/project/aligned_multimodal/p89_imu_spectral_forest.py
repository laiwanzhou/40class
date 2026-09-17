from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from imu_data import DEVICES, read_index


PROJECT_DIR = Path(__file__).resolve().parent
CACHE = PROJECT_DIR / "cache/imu_32"
OUTPUT = PROJECT_DIR / "runs/p89_imu_spectral_forest_v1"
SEED = 20260816
SIGNALS_PER_DEVICE = 13
STATS_PER_SIGNAL = 25
CORRELATIONS_PER_DEVICE = 16
FEATURES_PER_DEVICE = SIGNALS_PER_DEVICE * STATS_PER_SIGNAL + CORRELATIONS_PER_DEVICE


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.std(first) < 1e-8 or np.std(second) < 1e-8:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def signal_statistics(signal: np.ndarray) -> list[float]:
    value = np.asarray(signal, dtype=np.float64)
    if not len(value):
        return [0.0] * STATS_PER_SIGNAL
    centered = value - np.mean(value)
    difference = np.diff(value)
    quantiles = np.quantile(value, (0.10, 0.25, 0.50, 0.75, 0.90))
    mad = np.median(np.abs(value - quantiles[2]))
    if len(value) >= 3:
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        spectrum = spectrum[1:]
    else:
        spectrum = np.empty(0)
    if spectrum.size and float(spectrum.sum()) > 1e-12:
        normalized = spectrum / spectrum.sum()
        frequencies = np.arange(1, len(spectrum) + 1, dtype=np.float64)
        centroid = float(np.sum(normalized * frequencies) / max(len(spectrum), 1))
        spectral_entropy = float(
            -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)))
            / np.log(max(len(normalized), 2))
        )
        bands = np.array_split(normalized, 3)
        band_energy = [float(part.sum()) for part in bands]
    else:
        centroid = spectral_entropy = 0.0
        band_energy = [0.0, 0.0, 0.0]
    slope = (
        float(np.polyfit(np.linspace(-1.0, 1.0, len(value)), value, 1)[0])
        if len(value) >= 2
        else 0.0
    )
    zero_crossing = (
        float(np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1])))
        if len(centered) >= 2
        else 0.0
    )
    return [
        float(np.mean(value)),
        float(np.std(value)),
        float(np.sqrt(np.mean(value**2))),
        float(np.min(value)),
        float(np.max(value)),
        float(np.ptp(value)),
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
        float(quantiles[3]),
        float(quantiles[4]),
        float(quantiles[3] - quantiles[1]),
        float(mad),
        float(np.mean(np.abs(difference))) if len(difference) else 0.0,
        float(np.sqrt(np.mean(difference**2))) if len(difference) else 0.0,
        float(np.max(np.abs(difference))) if len(difference) else 0.0,
        zero_crossing,
        slope,
        safe_correlation(value[:-1], value[1:]) if len(value) >= 3 else 0.0,
        float(np.nan_to_num(skew(value), nan=0.0)) if len(value) >= 3 else 0.0,
        float(np.nan_to_num(kurtosis(value), nan=0.0)) if len(value) >= 4 else 0.0,
        centroid,
        spectral_entropy,
        band_energy[0],
        band_energy[2],
    ]


def device_features(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return np.zeros(FEATURES_PER_DEVICE, dtype=np.float32)
    source = np.asarray(values[valid], dtype=np.float64)
    acc_magnitude = np.linalg.norm(source[:, :3], axis=1)
    gyro_magnitude = np.linalg.norm(source[:, 3:6], axis=1)
    quaternion_angle = 2.0 * np.arccos(np.clip(np.abs(source[:, 6]), 0.0, 1.0))
    signals = [source[:, channel] for channel in range(10)] + [
        acc_magnitude,
        gyro_magnitude,
        quaternion_angle,
    ]
    features = [value for signal in signals for value in signal_statistics(signal)]
    correlations = []
    for start in (0, 3):
        for first, second in ((0, 1), (0, 2), (1, 2)):
            correlations.append(
                safe_correlation(source[:, start + first], source[:, start + second])
            )
    for axis in range(3):
        correlations.append(safe_correlation(source[:, axis], source[:, axis + 3]))
    correlations.extend(
        [
            safe_correlation(acc_magnitude, gyro_magnitude),
            safe_correlation(acc_magnitude, quaternion_angle),
            safe_correlation(gyro_magnitude, quaternion_angle),
            float(valid.mean()),
            float(len(source) / 32.0),
            float(np.mean(np.abs(np.diff(quaternion_angle))))
            if len(quaternion_angle) >= 2
            else 0.0,
            float(np.std(np.diff(quaternion_angle)))
            if len(quaternion_angle) >= 2
            else 0.0,
        ]
    )
    # Keep the contract explicit; this catches accidental feature drift.
    if len(correlations) != CORRELATIONS_PER_DEVICE:
        raise RuntimeError(len(correlations))
    return np.asarray(features + correlations, dtype=np.float32)


def feature_vector(
    values: np.ndarray,
    time_mask: np.ndarray,
    device_mask: np.ndarray,
    duration_seconds: float,
) -> np.ndarray:
    per_device = [
        device_features(values[index], time_mask[index])
        for index in range(len(DEVICES))
    ]
    device_summary = []
    for index in range(len(DEVICES)):
        present = float(device_mask[index] > 0)
        device_summary.extend((present, float(np.mean(time_mask[index] > 0))))
    magnitude_correlations = []
    for first in range(len(DEVICES)):
        for second in range(first + 1, len(DEVICES)):
            valid = (time_mask[first] > 0) & (time_mask[second] > 0)
            if np.sum(valid) < 3:
                magnitude_correlations.extend((0.0, 0.0))
                continue
            first_acc = np.linalg.norm(values[first, valid, :3], axis=1)
            second_acc = np.linalg.norm(values[second, valid, :3], axis=1)
            first_gyro = np.linalg.norm(values[first, valid, 3:6], axis=1)
            second_gyro = np.linalg.norm(values[second, valid, 3:6], axis=1)
            magnitude_correlations.extend(
                (
                    safe_correlation(first_acc, second_acc),
                    safe_correlation(first_gyro, second_gyro),
                )
            )
    return np.concatenate(
        (
            *per_device,
            np.asarray(device_summary, dtype=np.float32),
            np.asarray(magnitude_correlations, dtype=np.float32),
            np.asarray(
                (
                    np.log1p(max(duration_seconds, 0.0)),
                    float(np.sum(device_mask > 0)) / len(DEVICES),
                ),
                dtype=np.float32,
            ),
        )
    )


def dropped_copy(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = features.copy()
    for row in range(len(result)):
        present = [
            device
            for device in range(len(DEVICES))
            if result[row, len(DEVICES) * FEATURES_PER_DEVICE + 2 * device] > 0
        ]
        if len(present) <= 1:
            continue
        device = int(rng.choice(present))
        start = device * FEATURES_PER_DEVICE
        result[row, start : start + FEATURES_PER_DEVICE] = 0.0
        mask_start = len(DEVICES) * FEATURES_PER_DEVICE + 2 * device
        result[row, mask_start : mask_start + 2] = 0.0
    return result


def make_model(name: str):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
    raise ValueError(name)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    rows = read_index(CACHE / "index.csv")
    values = np.load(CACHE / "imu_float32.npy", mmap_mode="r")
    time_mask = np.load(CACHE / "time_mask_uint8.npy", mmap_mode="r")
    device_mask = np.load(CACHE / "device_mask_uint8.npy", mmap_mode="r")
    features = np.stack(
        [
            feature_vector(
                values[row.cache_index],
                time_mask[row.cache_index],
                device_mask[row.cache_index],
                row.duration_seconds,
            )
            for row in rows
        ]
    )
    train_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row.split == "train" and row.usable
        ],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.split == "test"],
        dtype=np.int64,
    )
    labels = np.asarray([row.class_id for row in rows], dtype=np.int64)
    users = np.asarray([row.user_id for row in rows]).astype(str)
    fold_source = json.loads(
        (PROJECT_DIR / "data/subject_folds/folds_summary.json").read_text(
            encoding="utf-8"
        )
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    model_oof = {}
    for model_name in ("extra_trees", "random_forest"):
        probability = np.zeros((len(train_indices), 40), dtype=np.float64)
        train_users = users[train_indices]
        train_labels = labels[train_indices]
        per_fold = []
        for fold in fold_source["folds"]:
            fit = np.isin(train_users, fold["train_users"])
            held = np.isin(train_users, fold["val_users"])
            rng = np.random.default_rng(SEED + int(fold["fold"]))
            fit_features = np.concatenate(
                (features[train_indices[fit]], dropped_copy(features[train_indices[fit]], rng))
            )
            fit_labels = np.concatenate((train_labels[fit], train_labels[fit]))
            model = make_model(model_name)
            model.fit(fit_features, fit_labels)
            held_probability = model.predict_proba(features[train_indices[held]])
            held_indices = np.flatnonzero(held)
            probability[
                np.ix_(held_indices, model.classes_.astype(np.int64))
            ] = held_probability
            held_prediction = model.classes_[held_probability.argmax(axis=1)]
            per_fold.append(
                {"fold": int(fold["fold"]), **metrics(train_labels[held], held_prediction)}
            )
            print(model_name, per_fold[-1], flush=True)
        prediction = np.argmax(probability, axis=1)
        result = {"overall": metrics(train_labels, prediction), "folds": per_fold}
        all_results[model_name] = result
        model_oof[model_name] = probability
        print(model_name, result["overall"], flush=True)

    selected_name = max(
        all_results,
        key=lambda name: (
            all_results[name]["overall"]["accuracy"],
            all_results[name]["overall"]["balanced_accuracy"],
        ),
    )
    rng = np.random.default_rng(SEED + 100)
    full_features = np.concatenate(
        (features[train_indices], dropped_copy(features[train_indices], rng))
    )
    full_labels = np.concatenate((labels[train_indices], labels[train_indices]))
    model = make_model(selected_name)
    model.fit(full_features, full_labels)
    test_probability = model.predict_proba(features[test_indices])
    expanded_test = np.zeros((len(test_indices), 40), dtype=np.float64)
    expanded_test[:, model.classes_.astype(np.int64)] = test_probability
    np.savez_compressed(
        OUTPUT / "oof_logits.npz",
        sample_ids=np.asarray([rows[index].sample_id for index in train_indices]),
        labels=labels[train_indices],
        imu_logits=np.log(np.maximum(model_oof[selected_name], 1e-12)),
    )
    np.savez_compressed(
        OUTPUT / "test_logits.npz",
        sample_ids=np.asarray([rows[index].sample_id for index in test_indices]),
        imu_logits=np.log(np.maximum(expanded_test, 1e-12)),
    )
    np.save(OUTPUT / "features.npy", features.astype(np.float32))
    report = {
        "stage": "P89_IMU_time_frequency_rotation_robust_forest_v1",
        "protocol": (
            "Fixed subject-disjoint folds; time statistics, centered dynamics, "
            "relative FFT bands, quaternion rotation angle, magnitude and "
            "cross-device correlations; fold-local model fitting and device dropout."
        ),
        "feature_dimension": int(features.shape[1]),
        "selected_model": selected_name,
        "models": all_results,
        "train_usable": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
