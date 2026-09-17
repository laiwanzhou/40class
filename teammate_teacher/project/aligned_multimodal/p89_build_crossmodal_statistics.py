from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = PROJECT_DIR / "runs" / "p86_motion_window_cache_t16_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p89_crossmodal_statistics_v1"
BODY_JOINTS = ((0, 7, 8), (11, 12, 13), (14, 15, 16), (4, 5, 6), (1, 2, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P89 Skeleton/IMU temporal and synchronization statistics.")
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def temporal_statistics(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    difference = np.diff(values, axis=1)
    midpoint = max(1, values.shape[1] // 2)
    statistics = (
        values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1),
        np.ptp(values, axis=1), np.quantile(values, 0.25, axis=1), np.median(values, axis=1),
        np.quantile(values, 0.75, axis=1),
        np.mean(np.abs(difference), axis=1) if difference.shape[1] else np.zeros_like(values[:, 0]),
        np.sqrt(np.mean(np.square(difference), axis=1)) if difference.shape[1] else np.zeros_like(values[:, 0]),
        values[:, midpoint:].mean(axis=1) - values[:, :midpoint].mean(axis=1),
    )
    return np.stack(statistics, axis=-1).reshape(len(values), -1).astype(np.float32)


def safe_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(first * first, axis=1) * np.sum(second * second, axis=1))
    return np.sum(first * second, axis=1) / np.maximum(denominator, 1e-6)


def synchronization_features(skeleton: np.ndarray, imu: np.ndarray) -> np.ndarray:
    # skeleton [N,32,17,13], imu statistics [N,32,5,52]
    velocity = np.linalg.norm(skeleton[..., 6:9], axis=-1)
    acceleration = np.linalg.norm(skeleton[..., 9:12], axis=-1)
    skeleton_signals = []
    for joints in BODY_JOINTS:
        skeleton_signals.append(np.stack((velocity[:, :, list(joints)].mean(axis=2), acceleration[:, :, list(joints)].mean(axis=2)), axis=-1))
    skeleton_signals = np.stack(skeleton_signals, axis=2)  # [N,T,device,2]
    raw_acc = np.linalg.norm(imu[..., 0:3], axis=-1)
    raw_gyro = np.linalg.norm(imu[..., 3:6], axis=-1)
    compensated_acc = np.linalg.norm(imu[..., 6:9], axis=-1)
    compensated_gyro = np.linalg.norm(imu[..., 9:12], axis=-1)
    imu_signals = np.stack((raw_acc, raw_gyro, compensated_acc, compensated_gyro), axis=-1)

    result: list[np.ndarray] = []
    for device in range(5):
        for skeleton_channel in range(2):
            first = skeleton_signals[:, :, device, skeleton_channel]
            for imu_channel in range(4):
                second = imu_signals[:, :, device, imu_channel]
                lag_correlations = []
                for lag in range(-3, 4):
                    if lag < 0:
                        lag_correlations.append(safe_correlation(first[:, :lag], second[:, -lag:]))
                    elif lag > 0:
                        lag_correlations.append(safe_correlation(first[:, lag:], second[:, :-lag]))
                    else:
                        lag_correlations.append(safe_correlation(first, second))
                correlations = np.stack(lag_correlations, axis=1)
                result.extend((
                    correlations,
                    correlations.max(axis=1, keepdims=True),
                    correlations.min(axis=1, keepdims=True),
                    (np.argmax(correlations, axis=1, keepdims=True).astype(np.float32) - 3.0) / 3.0,
                ))
    return np.concatenate(result, axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    source = args.motion_cache.resolve()
    skeleton = np.asarray(np.load(source / "skeleton_features.npy", mmap_mode="r"), dtype=np.float32).reshape(-1, 32, 17, 13)
    relations = np.asarray(np.load(source / "skeleton_relations.npy", mmap_mode="r"), dtype=np.float32).reshape(-1, 32, 18)
    imu = np.asarray(np.load(source / "imu_bin_statistics.npy", mmap_mode="r"), dtype=np.float32).reshape(-1, 32, 5, 52)
    imu_global = np.asarray(np.load(source / "imu_global_statistics.npy", mmap_mode="r"), dtype=np.float32).reshape(len(skeleton), -1)
    skeleton_quality = np.asarray(np.load(source / "skeleton_frame_quality.npy", mmap_mode="r"), dtype=np.float32).reshape(-1, 32, 1)
    imu_mask = np.asarray(np.load(source / "imu_bin_mask.npy", mmap_mode="r"), dtype=np.float32).reshape(-1, 32, 5)

    blocks = [
        temporal_statistics(skeleton.reshape(len(skeleton), 32, -1)),
        temporal_statistics(relations),
        temporal_statistics(imu.reshape(len(imu), 32, -1)),
        np.nan_to_num(imu_global),
        temporal_statistics(skeleton_quality),
        temporal_statistics(imu_mask),
        synchronization_features(skeleton, imu),
    ]
    features = np.concatenate(blocks, axis=1).astype(np.float32)
    rows = list(csv.DictReader((source / "rows.csv").open("r", encoding="utf-8-sig", newline="")))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "crossmodal_statistics.npz",
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        users=np.asarray([row["user_id"] for row in rows]),
        labels=np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64),
        # Raw accelerometer extrema can exceed float16.  Keep this compact cache
        # finite; downstream scalers are fit strictly on training subjects.
        features=features,
    )
    summary = {
        "stage": "P89_skeleton_IMU_crossmodal_statistics_v1", "status": "complete",
        "samples": len(features), "feature_dim": int(features.shape[1]),
        "blocks": {
            "skeleton_temporal": int(blocks[0].shape[1]), "relation_temporal": int(blocks[1].shape[1]),
            "imu_bin_temporal": int(blocks[2].shape[1]), "imu_global": int(blocks[3].shape[1]),
            "skeleton_quality": int(blocks[4].shape[1]), "imu_mask": int(blocks[5].shape[1]),
            "skeleton_IMU_lag_correlation": int(blocks[6].shape[1]),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
