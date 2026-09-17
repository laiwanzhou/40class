from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from imu_data import CHANNEL_NAMES, DEVICES, DEVICE_TO_INDEX


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_UNION = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_TEST_UNION = PROJECT_DIR / "data" / "six_modality_audit" / "test_union_manifest.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "cache" / "imu_32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a canonical, mask-aware IMU cache")
    parser.add_argument("--train-union", type=Path, default=DEFAULT_TRAIN_UNION)
    parser.add_argument("--test-union", type=Path, default=DEFAULT_TEST_UNION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--time-steps", type=int, default=32)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_sample_id(row: dict[str, str]) -> str:
    if row["split"] == "test":
        return row["sample_id"]
    return f"train__c{int(row['class_id']):02d}__{row['user_id']}__{row['trial_id']}"


def device_name(value: str) -> str:
    return value.split("(", 1)[0].strip()


def parse_time(value: str) -> float:
    return datetime.fromisoformat(value.strip()).timestamp()


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def relative_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = quaternion / np.maximum(norms, 1e-8)
    for index in range(1, len(quaternion)):
        if float(np.dot(quaternion[index - 1], quaternion[index])) < 0:
            quaternion[index] *= -1
    inverse_first = quaternion[0].copy()
    inverse_first[1:] *= -1
    relative = quaternion_multiply(
        np.repeat(inverse_first[None], len(quaternion), axis=0), quaternion
    )
    relative /= np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-8)
    relative[relative[:, 0] < 0] *= -1
    return relative


def load_trial(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    if not path.is_dir():
        return {}
    for file_path in sorted(path.glob("*.csv")):
        try:
            with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for fields in reader:
                    if len(fields) < 21:
                        continue
                    try:
                        timestamp = parse_time(fields[0])
                        device = device_name(fields[1])
                        acc_gyro = np.asarray(
                            [float(fields[index]) for index in range(2, 8)],
                            dtype=np.float64,
                        )
                        quaternion = np.asarray(
                            [float(fields[index]) for index in range(14, 18)],
                            dtype=np.float64,
                        )
                    except (ValueError, IndexError):
                        continue
                    if device in DEVICE_TO_INDEX and np.isfinite(acc_gyro).all() and np.isfinite(quaternion).all():
                        raw[device].append((timestamp, np.concatenate([acc_gyro, quaternion])))
        except (OSError, UnicodeError):
            continue

    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device, samples in raw.items():
        by_time: dict[float, list[np.ndarray]] = defaultdict(list)
        for timestamp, values in samples:
            by_time[timestamp].append(values)
        timestamps = np.asarray(sorted(by_time), dtype=np.float64)
        if len(timestamps) < 2:
            continue
        values = np.stack(
            [np.mean(np.stack(by_time[timestamp]), axis=0) for timestamp in timestamps]
        )
        values[:, 6:10] = relative_quaternion(values[:, 6:10])
        output[device] = (timestamps, values)
    return output


def resample_trial(
    by_device: dict[str, tuple[np.ndarray, np.ndarray]], time_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    values = np.zeros((len(DEVICES), time_steps, len(CHANNEL_NAMES)), dtype=np.float32)
    time_mask = np.zeros((len(DEVICES), time_steps), dtype=np.uint8)
    device_mask = np.zeros(len(DEVICES), dtype=np.uint8)
    if not by_device:
        return values, time_mask, device_mask, 0.0

    global_start = min(timestamps[0] for timestamps, _ in by_device.values())
    global_end = max(timestamps[-1] for timestamps, _ in by_device.values())
    duration = max(float(global_end - global_start), 1e-3)
    target = np.linspace(global_start, global_end, time_steps, dtype=np.float64)

    for device, (timestamps, source) in by_device.items():
        device_index = DEVICE_TO_INDEX[device]
        valid = (target >= timestamps[0]) & (target <= timestamps[-1])
        if not valid.any():
            continue
        interpolated = np.stack(
            [np.interp(target, timestamps, source[:, channel]) for channel in range(source.shape[1])],
            axis=1,
        )
        quaternion = interpolated[:, 6:10]
        quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-8)
        interpolated[:, 6:10] = quaternion
        interpolated[~valid] = 0.0
        values[device_index] = interpolated.astype(np.float32)
        time_mask[device_index] = valid.astype(np.uint8)
        device_mask[device_index] = 1
    return values, time_mask, device_mask, duration


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.train_union.resolve()) + read_csv(args.test_union.resolve())
    count = len(rows)
    shapes = {
        "imu_float32.npy": (count, len(DEVICES), args.time_steps, len(CHANNEL_NAMES)),
        "time_mask_uint8.npy": (count, len(DEVICES), args.time_steps),
        "device_mask_uint8.npy": (count, len(DEVICES)),
    }
    temporary = {name: output / f"{name}.building" for name in shapes}
    arrays = {
        "values": np.lib.format.open_memmap(
            temporary["imu_float32.npy"], mode="w+", dtype=np.float32, shape=shapes["imu_float32.npy"]
        ),
        "time_mask": np.lib.format.open_memmap(
            temporary["time_mask_uint8.npy"], mode="w+", dtype=np.uint8, shape=shapes["time_mask_uint8.npy"]
        ),
        "device_mask": np.lib.format.open_memmap(
            temporary["device_mask_uint8.npy"], mode="w+", dtype=np.uint8, shape=shapes["device_mask_uint8.npy"]
        ),
    }

    index_rows: list[dict[str, object]] = []
    started = time.time()
    for index, row in enumerate(rows):
        path = Path(row["imu_path"]) if row["imu_path"] else Path("__missing__")
        by_device = load_trial(path)
        values, time_mask, device_mask, duration = resample_trial(by_device, args.time_steps)
        arrays["values"][index] = values
        arrays["time_mask"][index] = time_mask
        arrays["device_mask"][index] = device_mask
        index_rows.append(
            {
                "cache_index": index,
                "split": row["split"],
                "sample_id": canonical_sample_id(row),
                "class_id": int(row["class_id"]),
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "source_sample_id": row["sample_id"],
                "source_path": row["imu_path"],
                "usable": int(device_mask.sum() > 0),
                "device_count": int(device_mask.sum()),
                "duration_seconds": duration,
            }
        )
        if (index + 1) % 400 == 0 or index + 1 == count:
            print(f"IMU cache {index + 1}/{count}", flush=True)

    for array in arrays.values():
        array.flush()
    del array, arrays
    for name, path in temporary.items():
        os.replace(path, output / name)

    with (output / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    train_rows = [row for row in index_rows if row["split"] == "train"]
    test_rows = [row for row in index_rows if row["split"] == "test"]
    metadata = {
        "version": 1,
        "time_steps": args.time_steps,
        "devices": list(DEVICES),
        "channels": list(CHANNEL_NAMES),
        "quaternion_protocol": "source order is wxyz; sign-continuous, normalized, relative to each device first sample",
        "time_protocol": "all devices share a trial-global linear time grid; per-device outside-range positions are zero with time_mask=0",
        "train_samples": len(train_rows),
        "train_usable": sum(int(row["usable"]) for row in train_rows),
        "train_complete_five_devices": sum(int(row["device_count"]) == 5 for row in train_rows),
        "test_samples": len(test_rows),
        "test_usable": sum(int(row["usable"]) for row in test_rows),
        "test_complete_five_devices": sum(int(row["device_count"]) == 5 for row in test_rows),
        "files": list(shapes) + ["index.csv"],
        "build_seconds": round(time.time() - started, 2),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

