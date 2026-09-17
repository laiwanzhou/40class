from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from imu_data import DEVICES, read_index


PROJECT_DIR = Path(__file__).resolve().parent
CACHE = PROJECT_DIR / "cache/imu_32"
OUTPUT = PROJECT_DIR / "runs/p89_imu_orientation_expert_v1"
SEED = 20260816
STATISTICS = 25


def normalize_quaternion(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-8)
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result[:, 1:] *= -1.0
    return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    cross = 2.0 * np.cross(q[:, 1:], vector)
    return vector + q[:, :1] * cross + np.cross(q[:, 1:], cross)


def rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    q[q[:, 0] < 0.0] *= -1.0
    vector = q[:, 1:]
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(q[:, :1], 1e-8, None))
    axis = vector / np.maximum(norm, 1e-8)
    return axis * angle


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.std(first) < 1e-8 or np.std(second) < 1e-8:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def signal_statistics_matrix(signal: np.ndarray) -> np.ndarray:
    """Return the same 25 statistics for every signal channel in one pass."""

    value = np.asarray(signal, dtype=np.float64)
    if value.ndim != 2 or not len(value):
        raise ValueError("expected a non-empty [time, channels] array")
    mean = value.mean(axis=0)
    centered = value - mean
    std = value.std(axis=0)
    difference = np.diff(value, axis=0)
    quantiles = np.quantile(value, (0.10, 0.25, 0.50, 0.75, 0.90), axis=0)
    if len(value) >= 2:
        x = np.linspace(-1.0, 1.0, len(value))
        slope = np.sum((x - x.mean())[:, None] * centered, axis=0) / np.sum(
            (x - x.mean()) ** 2
        )
        zero_crossing = np.mean(
            np.signbit(centered[1:]) != np.signbit(centered[:-1]), axis=0
        )
    else:
        slope = zero_crossing = np.zeros(value.shape[1])
    if len(value) >= 3:
        first, second = value[:-1], value[1:]
        first = first - first.mean(axis=0)
        second = second - second.mean(axis=0)
        autocorrelation = np.sum(first * second, axis=0) / np.maximum(
            np.sqrt(np.sum(first**2, axis=0) * np.sum(second**2, axis=0)), 1e-8
        )
        skewness = np.mean(centered**3, axis=0) / np.maximum(std**3, 1e-8)
    else:
        autocorrelation = skewness = np.zeros(value.shape[1])
    if len(value) >= 4:
        excess_kurtosis = np.mean(centered**4, axis=0) / np.maximum(std**4, 1e-8) - 3.0
    else:
        excess_kurtosis = np.zeros(value.shape[1])

    spectrum = np.abs(np.fft.rfft(centered, axis=0))[1:] ** 2
    spectral_sum = spectrum.sum(axis=0)
    normalized = spectrum / np.maximum(spectral_sum[None], 1e-12)
    if len(spectrum):
        frequencies = np.arange(1, len(spectrum) + 1, dtype=np.float64)[:, None]
        centroid = np.sum(normalized * frequencies, axis=0) / len(spectrum)
        entropy = -np.sum(
            normalized * np.log(np.maximum(normalized, 1e-12)), axis=0
        ) / np.log(max(len(spectrum), 2))
        bands = np.array_split(normalized, 3, axis=0)
        low_band = bands[0].sum(axis=0)
        high_band = bands[2].sum(axis=0)
    else:
        centroid = entropy = low_band = high_band = np.zeros(value.shape[1])
    invalid_spectrum = spectral_sum <= 1e-12
    for block in (centroid, entropy, low_band, high_band):
        block[invalid_spectrum] = 0.0

    if len(difference):
        mean_abs_difference = np.mean(np.abs(difference), axis=0)
        rms_difference = np.sqrt(np.mean(difference**2, axis=0))
        maximum_difference = np.max(np.abs(difference), axis=0)
    else:
        mean_abs_difference = rms_difference = maximum_difference = np.zeros(
            value.shape[1]
        )
    blocks = (
        mean,
        std,
        np.sqrt(np.mean(value**2, axis=0)),
        value.min(axis=0),
        value.max(axis=0),
        np.ptp(value, axis=0),
        *quantiles,
        quantiles[3] - quantiles[1],
        np.median(np.abs(value - quantiles[2]), axis=0),
        mean_abs_difference,
        rms_difference,
        maximum_difference,
        zero_crossing,
        slope,
        autocorrelation,
        skewness,
        excess_kurtosis,
        centroid,
        entropy,
        low_band,
        high_band,
    )
    result = np.stack(blocks, axis=1)
    if result.shape[1] != STATISTICS:
        raise RuntimeError(f"unexpected statistic count {result.shape[1]}")
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def device_signals(value: np.ndarray) -> np.ndarray:
    acc = value[:, :3]
    gyro = value[:, 3:6]
    quaternion = normalize_quaternion(value[:, 6:10])
    rotvec = rotation_vector(quaternion)
    if len(quaternion) >= 2:
        delta_q = quaternion_multiply(
            quaternion_conjugate(quaternion[:-1]), quaternion[1:]
        )
        delta_rotvec = rotation_vector(delta_q)
        delta_rotvec = np.concatenate((np.zeros((1, 3)), delta_rotvec), axis=0)
    else:
        delta_rotvec = np.zeros((len(quaternion), 3), dtype=np.float64)
    acc_reference = quaternion_rotate(quaternion, acc)
    gyro_reference = quaternion_rotate(quaternion, gyro)
    inverse = quaternion_conjugate(quaternion)
    acc_inverse = quaternion_rotate(inverse, acc)
    gyro_inverse = quaternion_rotate(inverse, gyro)
    acc_centered = acc - np.mean(acc, axis=0, keepdims=True)
    blocks = (
        acc,
        gyro,
        quaternion,
        rotvec,
        delta_rotvec,
        acc_reference,
        gyro_reference,
        acc_inverse,
        gyro_inverse,
        acc_centered,
        np.stack(
            (
                np.linalg.norm(acc, axis=1),
                np.linalg.norm(gyro, axis=1),
                np.linalg.norm(acc_centered, axis=1),
                np.linalg.norm(rotvec, axis=1),
                np.linalg.norm(delta_rotvec, axis=1),
            ),
            axis=1,
        ),
    )
    return np.concatenate(blocks, axis=1)


def phase_features(signals: np.ndarray) -> list[float]:
    # Preserve coarse action phase while staying insensitive to source frame rate.
    selected = np.concatenate((signals[:, :6], signals[:, 16:19], signals[:, -5:]), axis=1)
    output: list[float] = []
    for part in np.array_split(selected, 4, axis=0):
        output.extend(np.mean(part, axis=0).tolist())
        output.extend(np.std(part, axis=0).tolist())
    return output


def feature_vector(
    values: np.ndarray,
    time_mask: np.ndarray,
    device_mask: np.ndarray,
    duration_seconds: float,
) -> np.ndarray:
    features: list[float] = []
    magnitude_series: list[np.ndarray | None] = []
    for device in range(len(DEVICES)):
        valid = np.asarray(time_mask[device], dtype=bool)
        if not np.any(valid):
            # device_signals has 36 channels; phase block has 14*2*4 values.
            features.extend([0.0] * (36 * STATISTICS + 112 + 2))
            magnitude_series.append(None)
            continue
        signals = device_signals(np.asarray(values[device, valid], dtype=np.float64))
        if signals.shape[1] != 36:
            raise RuntimeError(f"unexpected orientation signal count {signals.shape[1]}")
        features.extend(signal_statistics_matrix(signals).reshape(-1).tolist())
        features.extend(phase_features(signals))
        features.extend((float(device_mask[device] > 0), float(valid.mean())))
        magnitude_series.append(signals[:, -5:])

    for first in range(len(DEVICES)):
        for second in range(first + 1, len(DEVICES)):
            left, right = magnitude_series[first], magnitude_series[second]
            if left is None or right is None:
                features.extend([0.0] * 15)
                continue
            length = min(len(left), len(right))
            left = left[:length]
            right = right[:length]
            for channel in range(5):
                features.extend(
                    (
                        safe_correlation(left[:, channel], right[:, channel]),
                        float(np.mean(np.abs(left[:, channel] - right[:, channel]))),
                        float(np.std(left[:, channel] - right[:, channel])),
                    )
                )
    features.extend(
        (
            float(np.log1p(max(duration_seconds, 0.0))),
            float(np.sum(device_mask > 0)) / len(DEVICES),
        )
    )
    result = np.nan_to_num(
        np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    return result


def model(name: str):
    if name == "lightgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=40,
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=6,
            min_child_samples=18,
            subsample=0.85,
            colsample_bytree=0.35,
            max_bin=63,
            reg_alpha=1.0,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_depth=24,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
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
    feature_path = OUTPUT / "features.npy"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if feature_path.is_file():
        features = np.load(feature_path, mmap_mode="r")
        if len(features) != len(rows):
            raise RuntimeError("cached feature row count changed")
    else:
        extracted = []
        for index, row in enumerate(rows, start=1):
            extracted.append(
                feature_vector(
                    values[row.cache_index],
                    time_mask[row.cache_index],
                    device_mask[row.cache_index],
                    row.duration_seconds,
                )
            )
            if index % 300 == 0 or index == len(rows):
                print(f"orientation features {index}/{len(rows)}", flush=True)
        features = np.stack(extracted)
        np.save(feature_path, features.astype(np.float32))

    train_indices = np.asarray(
        [i for i, row in enumerate(rows) if row.split == "train" and row.usable],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [i for i, row in enumerate(rows) if row.split == "test"], dtype=np.int64
    )
    labels = np.asarray([row.class_id for row in rows], dtype=np.int64)
    users = np.asarray([row.user_id for row in rows]).astype(str)
    folds = json.loads(
        (PROJECT_DIR / "data/subject_folds/folds_summary.json").read_text(
            encoding="utf-8"
        )
    )["folds"]
    all_results = {}
    all_oof = {}
    # Run the cheap randomized-tree control first; it gives an early signal
    # before the more expensive multiclass boosting fit.
    for name in ("extra_trees", "lightgbm"):
        probability = np.zeros((len(train_indices), 40), dtype=np.float64)
        per_fold = []
        for fold in folds:
            fit = np.isin(users[train_indices], fold["train_users"])
            held = np.isin(users[train_indices], fold["val_users"])
            estimator = model(name)
            estimator.fit(features[train_indices[fit]], labels[train_indices[fit]])
            held_probability = estimator.predict_proba(features[train_indices[held]])
            target = np.flatnonzero(held)
            probability[np.ix_(target, estimator.classes_.astype(np.int64))] = held_probability
            item = {
                "fold": int(fold["fold"]),
                **metrics(labels[train_indices[held]], probability[target].argmax(axis=1)),
            }
            per_fold.append(item)
            print(name, item, flush=True)
        prediction = probability.argmax(axis=1)
        all_results[name] = {
            "overall": metrics(labels[train_indices], prediction),
            "folds": per_fold,
        }
        all_oof[name] = probability
        print(name, all_results[name]["overall"], flush=True)

    selected = max(
        all_results,
        key=lambda name: (
            all_results[name]["overall"]["accuracy"],
            all_results[name]["overall"]["balanced_accuracy"],
        ),
    )
    final = model(selected)
    final.fit(features[train_indices], labels[train_indices])
    test_probability = final.predict_proba(features[test_indices])
    expanded = np.zeros((len(test_indices), 40), dtype=np.float64)
    expanded[:, final.classes_.astype(np.int64)] = test_probability
    np.savez_compressed(
        OUTPUT / "oof_logits.npz",
        sample_ids=np.asarray([rows[i].sample_id for i in train_indices]),
        labels=labels[train_indices],
        imu_logits=np.log(np.maximum(all_oof[selected], 1e-12)),
    )
    np.savez_compressed(
        OUTPUT / "test_logits.npz",
        sample_ids=np.asarray([rows[i].sample_id for i in test_indices]),
        imu_logits=np.log(np.maximum(expanded, 1e-12)),
    )
    report = {
        "stage": "P89_orientation_aware_multiview_IMU_expert_v1",
        "protocol": (
            "Fixed subject-disjoint folds. Preserve local XYZ, quaternion axis, "
            "relative rotation, forward/inverse reference-frame signals, action "
            "phase, and cross-device coordination. No Test labels or Test tuning."
        ),
        "feature_dimension": int(features.shape[1]),
        "selected_model": selected,
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
