from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from p31_skeleton_imu_preprocessing import (
    COMMON_PART_NAMES,
    H36M_JOINT_NAMES,
    IMU_CHANNEL_NAMES,
    IMU_DEVICE_NAMES,
    SKELETON_FEATURE_NAMES,
    SKELETON_RELATION_NAMES,
    assign_points_to_frame_intervals,
    build_skeleton_features,
    frame_times_from_ids,
    load_full_imu_trial,
    safe_trial_path,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_P28 = PROJECT_DIR / "runs" / "p28_adaptive_ir_pose_skeleton_full"
DEFAULT_P30 = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_P31 = PROJECT_DIR / "runs" / "p31_skeleton_imu_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify every P31 cache against P28, P30 and raw IMU")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p28-run", type=Path, default=DEFAULT_P28)
    parser.add_argument("--p30-run", type=Path, default=DEFAULT_P30)
    parser.add_argument("--p31-run", type=Path, default=DEFAULT_P31)
    parser.add_argument("--skip-source-imu-reparse", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    manifest = {row["sample_id"]: row for row in read_csv(args.manifest.resolve())}
    p28 = args.p28_run.resolve()
    p30 = args.p30_run.resolve()
    p31 = args.p31_run.resolve()
    rows28 = read_csv(p28 / "trial_summary.csv")
    rows30 = read_csv(p30 / "trial_summary.csv")
    rows31 = read_csv(p31 / "trial_summary.csv")
    ids28 = [row["sample_id"] for row in rows28]
    ids30 = [row["sample_id"] for row in rows30]
    ids31 = [row["sample_id"] for row in rows31]
    require(set(ids31) == set(ids28), "P31 sample ids differ from P28")
    require(set(ids31) == set(ids30), "P31 sample ids differ from P30")

    total_frames = 0
    total_points = 0
    total_source_points = 0
    full_imu_trials = 0
    usable_imu_trials = 0
    counter_trials = 0
    maximum_frames = 0
    maximum_points_per_device = 0
    maximum_point_trial = ""
    maximum_frame_trial = ""
    frame_id_mismatches = 0
    recomputed_skeleton_mismatches = 0
    source_imu_mismatches = 0

    for index, row in enumerate(rows31, 1):
        sample_id = row["sample_id"]
        relative = safe_trial_path(sample_id)
        path28 = p28 / "trial_cache" / relative.with_suffix(".npz")
        path30 = p30 / "trial_feature_cache" / relative.with_suffix(".npz")
        path31 = p31 / "trial_motion_cache" / relative.with_suffix(".npz")
        require(path28.is_file() and path30.is_file() and path31.is_file(), f"missing cache: {sample_id}")
        with np.load(path28, allow_pickle=False) as cache28, np.load(
            path30, allow_pickle=False
        ) as cache30, np.load(path31, allow_pickle=False) as cache:
            frame_ids = np.asarray(cache["frame_ids"])
            if not (
                np.array_equal(frame_ids, cache28["frame_ids"])
                and np.array_equal(frame_ids, cache30["frame_ids"])
            ):
                frame_id_mismatches += 1
            time_steps = len(frame_ids)
            require(cache["frame_time_seconds"].shape == (time_steps,), f"frame time shape: {sample_id}")
            require(cache["skeleton_features"].shape == (time_steps, 17, 13), f"Skeleton shape: {sample_id}")
            require(cache["skeleton_feature_mask"].shape == (time_steps, 17, 13), f"Skeleton mask: {sample_id}")
            require(cache["skeleton_relations"].shape == (time_steps, 18), f"relations: {sample_id}")
            require(cache["skeleton_relation_mask"].shape == (time_steps, 18), f"relation mask: {sample_id}")
            require(np.isfinite(cache["skeleton_features"]).all(), f"nonfinite Skeleton: {sample_id}")
            require(np.isfinite(cache["skeleton_relations"]).all(), f"nonfinite relations: {sample_id}")
            require(
                np.all(cache["skeleton_features"][~cache["skeleton_feature_mask"]] == 0),
                f"masked Skeleton values are not zero: {sample_id}",
            )
            require(
                np.all(cache["skeleton_relations"][~cache["skeleton_relation_mask"]] == 0),
                f"masked Skeleton relations are not zero: {sample_id}",
            )
            contracts = (
                ("joint_names", H36M_JOINT_NAMES),
                ("common_part_names", COMMON_PART_NAMES),
                ("skeleton_feature_names", SKELETON_FEATURE_NAMES),
                ("skeleton_relation_names", SKELETON_RELATION_NAMES),
                ("imu_device_names", IMU_DEVICE_NAMES),
                ("imu_channel_names", IMU_CHANNEL_NAMES),
            )
            for key, expected in contracts:
                require(tuple(str(value) for value in cache[key]) == expected, f"{key}: {sample_id}")

            absolute_frame_times = frame_times_from_ids(frame_ids)
            recomputed = build_skeleton_features(
                cache28["skeleton_h36m_xyz_conf_raw"].astype(np.float32),
                absolute_frame_times,
            )
            if not (
                np.allclose(cache["skeleton_features"], recomputed["features"], atol=1e-6)
                and np.array_equal(cache["skeleton_feature_mask"], recomputed["feature_mask"])
                and np.allclose(cache["skeleton_relations"], recomputed["relations"], atol=1e-6)
                and np.array_equal(cache["skeleton_relation_mask"], recomputed["relation_mask"])
            ):
                recomputed_skeleton_mismatches += 1

            values = cache["imu_values"]
            point_times = cache["imu_time_seconds"]
            point_frame_index = cache["imu_frame_index"].astype(np.int64)
            offsets = cache["imu_device_offsets"].astype(np.int64)
            counts = cache["imu_interval_counts"].astype(np.int64)
            device_mask = cache["imu_device_mask"].astype(bool)
            point_count = len(values)
            require(values.shape == (point_count, 10), f"IMU value shape: {sample_id}")
            require(point_times.shape == (point_count,), f"IMU time shape: {sample_id}")
            require(point_frame_index.shape == (point_count,), f"IMU bin shape: {sample_id}")
            require(offsets.shape == (6,), f"IMU offsets shape: {sample_id}")
            require(counts.shape == (time_steps, 5), f"IMU count shape: {sample_id}")
            require(np.isfinite(values).all() and np.isfinite(point_times).all(), f"nonfinite IMU: {sample_id}")
            require(np.all(np.diff(offsets) >= 0) and offsets[0] == 0 and offsets[-1] == point_count, f"bad offsets: {sample_id}")
            require(np.array_equal(device_mask, np.diff(offsets) > 0), f"device mask: {sample_id}")
            require(int(counts.sum()) == point_count, f"IMU total count: {sample_id}")
            if point_count:
                require(point_frame_index.min() >= 0 and point_frame_index.max() < time_steps, f"IMU bin range: {sample_id}")
            for device_index in range(5):
                start, end = offsets[device_index : device_index + 2]
                require(np.all(np.diff(point_times[start:end]) >= 0), f"IMU time order: {sample_id}")
                actual = np.bincount(point_frame_index[start:end], minlength=time_steps)
                require(np.array_equal(actual, counts[:, device_index]), f"IMU interval count: {sample_id}")

            if not args.skip_source_imu_reparse:
                imu_text = manifest[sample_id].get("imu_path", "").strip()
                by_device, audit = load_full_imu_trial(
                    Path(imu_text) if imu_text else Path("__missing_imu_path__")
                )
                source_values: list[np.ndarray] = []
                source_times: list[np.ndarray] = []
                source_bins: list[np.ndarray] = []
                for device in IMU_DEVICE_NAMES:
                    if device not in by_device:
                        continue
                    times, device_values = by_device[device]
                    source_values.append(device_values)
                    source_times.append((times - absolute_frame_times[0]).astype(np.float32))
                    source_bins.append(assign_points_to_frame_intervals(times, absolute_frame_times))
                expected_values = np.concatenate(source_values) if source_values else np.empty((0, 10), np.float32)
                expected_times = np.concatenate(source_times) if source_times else np.empty(0, np.float32)
                expected_bins = np.concatenate(source_bins) if source_bins else np.empty(0, np.int32)
                total_source_points += audit.accepted_rows
                if not (
                    np.array_equal(values, expected_values)
                    and np.array_equal(point_times, expected_times)
                    and np.array_equal(point_frame_index, expected_bins)
                ):
                    source_imu_mismatches += 1

            total_frames += time_steps
            total_points += point_count
            usable_imu_trials += int(device_mask.any())
            full_imu_trials += int(device_mask.all())
            counter_trials += int(str(frame_ids[0]).isdigit())
            if time_steps > maximum_frames:
                maximum_frames = time_steps
                maximum_frame_trial = sample_id
            trial_maximum = int(np.diff(offsets).max(initial=0))
            if trial_maximum > maximum_points_per_device:
                maximum_points_per_device = trial_maximum
                maximum_point_trial = sample_id
        if index % 400 == 0 or index == len(rows31):
            print(f"verify P31 {index}/{len(rows31)}", flush=True)

    require(frame_id_mismatches == 0, f"frame id mismatches: {frame_id_mismatches}")
    require(recomputed_skeleton_mismatches == 0, f"Skeleton recompute mismatches: {recomputed_skeleton_mismatches}")
    require(source_imu_mismatches == 0, f"raw IMU mismatches: {source_imu_mismatches}")
    if not args.skip_source_imu_reparse:
        require(total_source_points == total_points, "raw valid IMU total differs from cache")
    report: dict[str, Any] = {
        "verified": True,
        "trials": len(rows31),
        "frames": total_frames,
        "imu_points": total_points,
        "imu_usable_trials": usable_imu_trials,
        "imu_complete_five_device_trials": full_imu_trials,
        "counter_only_no_imu_trials": counter_trials,
        "maximum_frames": maximum_frames,
        "maximum_frame_trial": maximum_frame_trial,
        "maximum_points_per_device": maximum_points_per_device,
        "maximum_point_trial": maximum_point_trial,
        "frame_id_mismatches": frame_id_mismatches,
        "recomputed_skeleton_mismatches": recomputed_skeleton_mismatches,
        "source_imu_mismatches": source_imu_mismatches,
        "source_imu_reparsed": not args.skip_source_imu_reparse,
    }
    (p31 / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
