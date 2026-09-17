from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_UNION_DIR = PROJECT_DIR / "data" / "six_modality_audit"
EXPECTED_IMU_DEVICES = ["WTC", "WTLA", "WTRA", "WTLL", "WTRL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 IMU/Radar 格式、时间轴与 Radar 低成本可分性")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_UNION_DIR / "train_union_manifest.csv")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_UNION_DIR / "test_union_manifest.csv")
    parser.add_argument("--fold-summary", type=Path, default=PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "p0_six_modality_audit")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def device_prefix(name: str) -> str:
    return name.split("(", 1)[0].strip()


def finite_stats(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {f"{prefix}_{name}": math.nan for name in ["mean", "std", "p10", "p50", "p90", "min", "max"]}
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std()),
        f"{prefix}_p10": float(np.quantile(array, 0.1)),
        f"{prefix}_p50": float(np.quantile(array, 0.5)),
        f"{prefix}_p90": float(np.quantile(array, 0.9)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
    }


def parse_imu_trial(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, list[tuple[datetime, np.ndarray]]]]:
    base: dict[str, Any] = {
        "split": row["split"],
        "sample_id": row["sample_id"],
        "class_id": int(row["class_id"]),
        "user_id": row["user_id"],
        "trial_id": row["trial_id"],
        "imu_present": int(row["imu_present"]),
        "imu_usable": int(row["imu_usable"]),
    }
    path = Path(row["imu_path"]) if row["imu_path"] else None
    by_device: dict[str, list[tuple[datetime, np.ndarray]]] = defaultdict(list)
    original_last_time: dict[str, datetime] = {}
    ordering_violations = 0
    malformed_rows = 0
    headers: Counter[tuple[str, ...]] = Counter()
    file_names: list[str] = []
    versions: Counter[str] = Counter()
    batteries: list[float] = []
    temperatures: list[float] = []
    if path is not None and path.is_dir():
        for file_path in sorted(path.glob("*.csv")):
            file_names.append(file_path.name)
            try:
                with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, None)
                    if header:
                        headers[tuple(header)] += 1
                    for fields in reader:
                        if len(fields) < 21:
                            malformed_rows += 1
                            continue
                        try:
                            timestamp = parse_time(fields[0])
                            device = device_prefix(fields[1])
                            values = np.asarray([float(fields[index]) for index in range(2, 19)], dtype=np.float64)
                            battery = float(fields[20])
                        except (ValueError, IndexError):
                            malformed_rows += 1
                            continue
                        if device in original_last_time and timestamp < original_last_time[device]:
                            ordering_violations += 1
                        original_last_time[device] = timestamp
                        by_device[device].append((timestamp, values))
                        versions[fields[19]] += 1
                        batteries.append(battery)
                        temperatures.append(float(values[16]))
            except (OSError, UnicodeError):
                malformed_rows += 1

    starts: list[datetime] = []
    ends: list[datetime] = []
    gaps: list[float] = []
    device_counts = {}
    acceleration_magnitudes: list[float] = []
    gyro_magnitudes: list[float] = []
    for device, samples in by_device.items():
        samples.sort(key=lambda item: item[0])
        timestamps = [item[0] for item in samples]
        values = np.stack([item[1] for item in samples])
        starts.append(timestamps[0])
        ends.append(timestamps[-1])
        device_counts[device] = len(samples)
        if len(timestamps) >= 2:
            gaps.extend((timestamps[index] - timestamps[index - 1]).total_seconds() for index in range(1, len(timestamps)))
        acceleration_magnitudes.extend(np.linalg.norm(values[:, 0:3], axis=1).tolist())
        gyro_magnitudes.extend(np.linalg.norm(values[:, 3:6], axis=1).tolist())

    base.update(
        {
            "file_count": len(file_names),
            "file_names": "|".join(file_names),
            "row_count": int(sum(device_counts.values())),
            "malformed_rows": malformed_rows,
            "device_count": len(by_device),
            "devices": "|".join(sorted(by_device)),
            "expected_five_devices": int(set(by_device) == set(EXPECTED_IMU_DEVICES)),
            "min_rows_per_device": min(device_counts.values()) if device_counts else 0,
            "max_rows_per_device": max(device_counts.values()) if device_counts else 0,
            "ordering_violations_before_sort": ordering_violations,
            "ordering_violation_fraction": ordering_violations / max(sum(device_counts.values()) - len(device_counts), 1),
            "trial_duration_seconds": (max(ends) - min(starts)).total_seconds() if starts else math.nan,
            "device_start_spread_seconds": (max(starts) - min(starts)).total_seconds() if starts else math.nan,
            "device_end_spread_seconds": (max(ends) - min(ends)).total_seconds() if ends else math.nan,
            "sampling_gap_median_seconds": float(np.median(gaps)) if gaps else math.nan,
            "sampling_gap_p90_seconds": float(np.quantile(gaps, 0.9)) if gaps else math.nan,
            "header_variants": len(headers),
            "firmware_versions": "|".join(sorted(versions)),
            "battery_mean": float(np.mean(batteries)) if batteries else math.nan,
            "temperature_mean": float(np.mean(temperatures)) if temperatures else math.nan,
        }
    )
    base.update(finite_stats(acceleration_magnitudes, "acceleration_magnitude"))
    base.update(finite_stats(gyro_magnitudes, "gyro_magnitude"))
    return base, by_device


def parse_radar_trial(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": row["split"],
        "sample_id": row["sample_id"],
        "class_id": int(row["class_id"]),
        "user_id": row["user_id"],
        "trial_id": row["trial_id"],
        "radar_present": int(row["radar_present"]),
        "radar_usable": int(row["radar_usable"]),
    }
    timestamps: list[datetime] = []
    frames: list[int] = []
    values: dict[str, list[float]] = {name: [] for name in ["x", "y", "z", "v", "snr", "noise"]}
    malformed_rows = 0
    path = Path(row["radar_path"]) if row["radar_path"] else None
    file_names: list[str] = []
    if path is not None and path.is_dir():
        for file_path in sorted(path.glob("*.csv")):
            file_names.append(file_path.name)
            try:
                with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for fields in reader:
                        try:
                            timestamps.append(parse_time(fields["timestamp"]))
                            frames.append(int(fields["frame"]))
                            for name in values:
                                values[name].append(float(fields[name]))
                        except (ValueError, TypeError, KeyError):
                            malformed_rows += 1
            except (OSError, UnicodeError):
                malformed_rows += 1
    unique_frames, frame_counts = np.unique(frames, return_counts=True) if frames else (np.asarray([]), np.asarray([]))
    if values["x"]:
        xyz = np.column_stack([values["x"], values["y"], values["z"]])
        ranges = np.linalg.norm(xyz, axis=1).tolist()
    else:
        ranges = []
    result.update(
        {
            "file_count": len(file_names),
            "file_names": "|".join(file_names),
            "point_count": len(frames),
            "frame_count": len(unique_frames),
            "points_per_frame_mean": float(frame_counts.mean()) if len(frame_counts) else math.nan,
            "points_per_frame_std": float(frame_counts.std()) if len(frame_counts) else math.nan,
            "duration_seconds": (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else math.nan,
            "malformed_rows": malformed_rows,
        }
    )
    for name, channel_values in values.items():
        result.update(finite_stats(channel_values, name))
        if channel_values:
            result[f"{name}_abs_mean"] = float(np.mean(np.abs(channel_values)))
            result[f"{name}_energy"] = float(np.mean(np.square(channel_values)))
        else:
            result[f"{name}_abs_mean"] = math.nan
            result[f"{name}_energy"] = math.nan
    result.update(finite_stats(ranges, "range"))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution_summary(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p10": float(np.quantile(values, 0.1)) if len(values) else None,
        "p90": float(np.quantile(values, 0.9)) if len(values) else None,
    }


def imu_typical_timeseries(
    rows: list[dict[str, Any]],
    raw_cache: dict[str, dict[str, list[tuple[datetime, np.ndarray]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [row for row in rows if int(row["expected_five_devices"]) == 1 and int(row["row_count"]) > 0]
    target_duration = float(np.median([row["trial_duration_seconds"] for row in candidates]))
    selected = min(candidates, key=lambda row: abs(float(row["trial_duration_seconds"]) - target_duration))
    by_device = raw_cache[selected["sample_id"]]
    output_rows: list[dict[str, Any]] = []
    frequency = {}
    global_start = min(samples[0][0] for samples in by_device.values())
    for device in sorted(by_device):
        samples = by_device[device]
        times = np.asarray([(timestamp - global_start).total_seconds() for timestamp, _ in samples])
        values = np.stack([value for _, value in samples])
        acc_magnitude = np.linalg.norm(values[:, 0:3], axis=1)
        gyro_magnitude = np.linalg.norm(values[:, 3:6], axis=1)
        for time_value, acc_value, gyro_value in zip(times, acc_magnitude, gyro_magnitude):
            output_rows.append(
                {
                    "sample_id": selected["sample_id"],
                    "device": device,
                    "seconds_from_trial_start": float(time_value),
                    "acceleration_magnitude_g": float(acc_value),
                    "gyro_magnitude_deg_s": float(gyro_value),
                }
            )
        if len(times) >= 4:
            gap = float(np.median(np.diff(times)))
            signal = acc_magnitude - acc_magnitude.mean()
            frequencies = np.fft.rfftfreq(len(signal), d=gap)
            spectrum = np.abs(np.fft.rfft(signal))
            if len(spectrum) > 1:
                peak_index = int(np.argmax(spectrum[1:]) + 1)
                frequency[device] = {
                    "samples": len(signal),
                    "median_gap_seconds": gap,
                    "acceleration_magnitude_peak_frequency_hz": float(frequencies[peak_index]),
                    "peak_amplitude": float(spectrum[peak_index]),
                }
    return {
        "sample_id": selected["sample_id"],
        "class_id": selected["class_id"],
        "user_id": selected["user_id"],
        "trial_id": selected["trial_id"],
        "duration_seconds": selected["trial_duration_seconds"],
        "devices": sorted(by_device),
        "frequency": frequency,
    }, output_rows


def radar_feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {
        "split", "sample_id", "class_id", "user_id", "trial_id", "radar_present", "radar_usable",
        "file_count", "file_names", "malformed_rows",
    }
    columns = []
    for name in rows[0]:
        if name in excluded:
            continue
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if np.isfinite(values).all() and np.std(values) > 0:
            columns.append(name)
    return columns


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def radar_baselines(
    rows: list[dict[str, Any]], fold_summary: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    usable = [row for row in rows if int(row["class_id"]) >= 0 and int(row["radar_usable"]) == 1 and int(row["point_count"]) > 0]
    columns = radar_feature_columns(usable)
    x = np.asarray([[float(row[column]) for column in columns] for row in usable], dtype=np.float64)
    y = np.asarray([int(row["class_id"]) for row in usable], dtype=np.int64)
    users = np.asarray([str(row["user_id"]) for row in usable])
    predictions = {
        "logistic_regression": np.full(len(usable), -1, dtype=np.int64),
        "random_forest": np.full(len(usable), -1, dtype=np.int64),
        "majority": np.full(len(usable), -1, dtype=np.int64),
    }
    folds = []
    for fold_info in fold_summary["folds"]:
        fold = int(fold_info["fold"])
        val_users = set(fold_info["val_users"])
        val_mask = np.asarray([user in val_users for user in users])
        train_mask = ~val_mask
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        )
        forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=20260723 + fold,
        )
        logistic.fit(x[train_indices], y[train_indices])
        forest.fit(x[train_indices], y[train_indices])
        predictions["logistic_regression"][val_indices] = logistic.predict(x[val_indices])
        predictions["random_forest"][val_indices] = forest.predict(x[val_indices])
        majority = Counter(y[train_indices].tolist()).most_common(1)[0][0]
        predictions["majority"][val_indices] = majority
        folds.append(
            {
                "fold": fold,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "val_users": sorted(val_users),
                "metrics": {name: metric_dict(y[val_indices], values[val_indices]) for name, values in predictions.items()},
            }
        )
    if any(np.any(values < 0) for values in predictions.values()):
        raise RuntimeError("Radar OOF did not cover every usable sample")
    summary = {
        "protocol": (
            "Only trials with actual Radar points; exact existing subject folds; no missingness mask, absolute timestamp, "
            "subject ID, file name, or availability feature is used. Diagnostic models only."
        ),
        "usable_samples": len(usable),
        "subjects": len(set(users.tolist())),
        "feature_columns": columns,
        "pooled_oof": {name: metric_dict(y, values) for name, values in predictions.items()},
        "folds": folds,
    }
    prediction_rows = [
        {
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
            "user_id": row["user_id"],
            "trial_id": row["trial_id"],
            "logistic_regression_prediction": int(predictions["logistic_regression"][index]),
            "random_forest_prediction": int(predictions["random_forest"][index]),
        }
        for index, row in enumerate(usable)
    ]
    return summary, prediction_rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = read_csv(args.train_manifest.resolve())
    test_manifest = read_csv(args.test_manifest.resolve())
    fold_summary = json.loads(args.fold_summary.resolve().read_text(encoding="utf-8"))

    print("解析 IMU 设备与时间轴……", flush=True)
    imu_train: list[dict[str, Any]] = []
    imu_test: list[dict[str, Any]] = []
    imu_raw_train: dict[str, dict[str, list[tuple[datetime, np.ndarray]]]] = {}
    for source, target in [(train_manifest, imu_train), (test_manifest, imu_test)]:
        for row in source:
            features, raw = parse_imu_trial(row)
            target.append(features)
            if raw and row["split"] == "train":
                imu_raw_train[row["sample_id"]] = raw
    write_csv(output_dir / "imu_trial_features.csv", imu_train + imu_test)
    typical_summary, typical_rows = imu_typical_timeseries(imu_train, imu_raw_train)
    write_csv(output_dir / "imu_typical_trial_timeseries.csv", typical_rows)

    print("解析 Radar 点与帧统计……", flush=True)
    radar_train = [parse_radar_trial(row) for row in train_manifest]
    radar_test = [parse_radar_trial(row) for row in test_manifest]
    write_csv(output_dir / "radar_trial_features.csv", radar_train + radar_test)
    radar_model_audit, radar_prediction_rows = radar_baselines(radar_train, fold_summary)
    write_csv(output_dir / "radar_oof_predictions.csv", radar_prediction_rows)

    imu_summary = {
        "schema": {
            "files_per_trial_expected": ["up(LA+RA+C).csv", "down(LL+RL).csv"],
            "device_mapping_from_file_names": {
                "WTC": "center/torso",
                "WTLA": "left arm",
                "WTRA": "right arm",
                "WTLL": "left leg",
                "WTRL": "right leg",
            },
            "columns_by_index": {
                "0": "timestamp",
                "1": "device name and hardware identifier",
                "2-4": "acceleration XYZ (g)",
                "5-7": "angular velocity XYZ (deg/s)",
                "8-10": "Euler angles XYZ (deg)",
                "11-13": "magnetic field XYZ (uT)",
                "14-17": "quaternion 0-3",
                "18": "temperature (deg C)",
                "19": "firmware version",
                "20": "battery percent",
            },
            "training_exclusion": (
                "Absolute timestamp, hardware identifier, firmware, battery and temperature are nuisance/shortcut fields; "
                "do not feed them to an action classifier. Magnetometer should be an explicit ablation, not a default assumption."
            ),
        },
        "train": {
            "trials": len(imu_train),
            "usable": int(sum(int(row["imu_usable"]) for row in imu_train)),
            "expected_five_devices": int(sum(int(row["expected_five_devices"]) for row in imu_train)),
            "trials_with_ordering_violations": int(sum(int(row["ordering_violations_before_sort"]) > 0 for row in imu_train)),
            "ordering_violations_total": int(sum(int(row["ordering_violations_before_sort"]) for row in imu_train)),
            "row_count": distribution_summary(imu_train, "row_count"),
            "duration_seconds": distribution_summary(imu_train, "trial_duration_seconds"),
            "sampling_gap_seconds": distribution_summary(imu_train, "sampling_gap_median_seconds"),
            "device_start_spread_seconds": distribution_summary(imu_train, "device_start_spread_seconds"),
        },
        "test": {
            "trials": len(imu_test),
            "usable": int(sum(int(row["imu_usable"]) for row in imu_test)),
            "expected_five_devices": int(sum(int(row["expected_five_devices"]) for row in imu_test)),
            "trials_with_ordering_violations": int(sum(int(row["ordering_violations_before_sort"]) > 0 for row in imu_test)),
            "ordering_violations_total": int(sum(int(row["ordering_violations_before_sort"]) for row in imu_test)),
            "row_count": distribution_summary(imu_test, "row_count"),
            "duration_seconds": distribution_summary(imu_test, "trial_duration_seconds"),
            "sampling_gap_seconds": distribution_summary(imu_test, "sampling_gap_median_seconds"),
            "device_start_spread_seconds": distribution_summary(imu_test, "device_start_spread_seconds"),
        },
        "typical_trial": typical_summary,
        "scope_note": "IMU classifier training is intentionally left to the teammate owning IMU; this audit defines the safe schema and time-order interface.",
    }
    radar_summary = {
        "schema": {
            "columns": ["timestamp", "frame", "DetObj#", "x", "y", "z", "v", "snr", "noise"],
            "row_meaning": "one detected Radar point; point count varies by frame and may be zero for the whole trial",
            "empty_file_definition": "header present but no data row",
        },
        "train": {
            "trials": len(radar_train),
            "present": int(sum(int(row["radar_present"]) for row in radar_train)),
            "usable_with_points": int(sum(int(row["radar_usable"]) for row in radar_train)),
            "point_count": distribution_summary([row for row in radar_train if int(row["radar_usable"])], "point_count"),
            "frame_count": distribution_summary([row for row in radar_train if int(row["radar_usable"])], "frame_count"),
            "duration_seconds": distribution_summary([row for row in radar_train if int(row["radar_usable"])], "duration_seconds"),
        },
        "test": {
            "trials": len(radar_test),
            "present": int(sum(int(row["radar_present"]) for row in radar_test)),
            "usable_with_points": int(sum(int(row["radar_usable"]) for row in radar_test)),
            "point_count": distribution_summary([row for row in radar_test if int(row["radar_usable"])], "point_count"),
            "frame_count": distribution_summary([row for row in radar_test if int(row["radar_usable"])], "frame_count"),
            "duration_seconds": distribution_summary([row for row in radar_test if int(row["radar_usable"])], "duration_seconds"),
        },
        "usable_only_subject_disjoint_baseline": radar_model_audit,
        "oof_predictions": str((output_dir / "radar_oof_predictions.csv").resolve()),
    }
    (output_dir / "imu_schema_time_audit.json").write_text(
        json.dumps(imu_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "radar_schema_baseline_audit.json").write_text(
        json.dumps(radar_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"imu": imu_summary, "radar": radar_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
