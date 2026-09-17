from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from p31_skeleton_imu_preprocessing import (
    COMMON_PART_NAMES,
    H36M_JOINT_NAMES,
    IMU_CHANNEL_NAMES,
    IMU_DEVICE_BODY_PARTS,
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
DEFAULT_P28_RUN = PROJECT_DIR / "runs" / "p28_adaptive_ir_pose_skeleton_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p31_skeleton_imu_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Step 8/9 all-frame Skeleton inputs and lossless valid-row IMU-to-frame "
            "interval mappings for every synchronized P28 trial."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p28-run", type=Path, default=DEFAULT_P28_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def cache_path(run: Path, sample_id: str) -> Path:
    return run / "trial_motion_cache" / safe_trial_path(sample_id).with_suffix(".npz")


def load_existing_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_trial(
    row: dict[str, str],
    p28_run: Path,
    output: Path,
) -> dict[str, Any]:
    sample_id = row["sample_id"]
    relative = safe_trial_path(sample_id)
    p28_path = p28_run / "trial_cache" / relative.with_suffix(".npz")
    with np.load(p28_path, allow_pickle=False) as p28:
        frame_ids = np.asarray(p28["frame_ids"])
        skeleton_raw = p28["skeleton_h36m_xyz_conf_raw"].astype(np.float32)
        skeleton_people = p28["skeleton_person_count"].astype(np.uint8)

    absolute_frame_times = frame_times_from_ids(frame_ids)
    frame_time_mode = "counter_10hz_no_imu" if str(frame_ids[0]).isdigit() else "absolute_timestamp"
    frame_times = (absolute_frame_times - absolute_frame_times[0]).astype(np.float32)
    skeleton = build_skeleton_features(skeleton_raw, absolute_frame_times)
    skeleton_quality = np.asarray(skeleton["frame_quality"], dtype=np.float32)
    skeleton_quality *= (skeleton_people > 0).astype(np.float32)

    imu_text = row.get("imu_path", "").strip()
    imu_path = Path(imu_text) if imu_text else Path("__missing_imu_path__")
    by_device, imu_audit = load_full_imu_trial(imu_path)
    imu_values_parts: list[np.ndarray] = []
    imu_time_parts: list[np.ndarray] = []
    imu_frame_parts: list[np.ndarray] = []
    device_offsets = [0]
    interval_counts = np.zeros((len(frame_ids), len(IMU_DEVICE_NAMES)), dtype=np.uint32)
    device_mask = np.zeros(len(IMU_DEVICE_NAMES), dtype=bool)
    before_points = 0
    after_points = 0
    absolute_offsets: list[np.ndarray] = []

    for device_index, device in enumerate(IMU_DEVICE_NAMES):
        if device not in by_device:
            device_offsets.append(device_offsets[-1])
            continue
        point_times, values = by_device[device]
        frame_index = assign_points_to_frame_intervals(point_times, absolute_frame_times)
        counts = np.bincount(frame_index, minlength=len(frame_ids)).astype(np.uint32)
        if int(counts.sum()) != len(values):
            raise AssertionError(f"IMU interval assignment lost points: {sample_id}/{device}")
        interval_counts[:, device_index] = counts
        device_mask[device_index] = len(values) > 0
        imu_values_parts.append(values.astype(np.float32))
        imu_time_parts.append((point_times - absolute_frame_times[0]).astype(np.float32))
        imu_frame_parts.append(frame_index.astype(np.int16))
        device_offsets.append(device_offsets[-1] + len(values))
        before_points += int((point_times < absolute_frame_times[0]).sum())
        after_points += int((point_times > absolute_frame_times[-1]).sum())
        absolute_offsets.append(np.abs(point_times - absolute_frame_times[frame_index]))

    if imu_values_parts:
        imu_values = np.concatenate(imu_values_parts, axis=0)
        imu_times = np.concatenate(imu_time_parts, axis=0)
        imu_frame_index = np.concatenate(imu_frame_parts, axis=0)
        alignment_offsets = np.concatenate(absolute_offsets)
    else:
        imu_values = np.empty((0, len(IMU_CHANNEL_NAMES)), dtype=np.float32)
        imu_times = np.empty(0, dtype=np.float32)
        imu_frame_index = np.empty(0, dtype=np.int16)
        alignment_offsets = np.empty(0, dtype=np.float64)
    device_offsets_array = np.asarray(device_offsets, dtype=np.int32)
    if int(interval_counts.sum()) != len(imu_values):
        raise AssertionError(f"trial-level IMU interval counts differ: {sample_id}")
    if int(device_offsets_array[-1]) != len(imu_values):
        raise AssertionError(f"trial-level IMU device offsets differ: {sample_id}")
    if int(interval_counts.max(initial=0)) > np.iinfo(np.uint16).max:
        raise OverflowError(f"too many IMU points in one frame interval: {sample_id}")

    destination = cache_path(output, sample_id)
    atomic_npz(
        destination,
        frame_ids=frame_ids,
        frame_time_seconds=frame_times,
        skeleton_features=np.asarray(skeleton["features"], dtype=np.float32),
        skeleton_feature_mask=np.asarray(skeleton["feature_mask"], dtype=bool),
        skeleton_joint_mask=np.asarray(skeleton["joint_mask"], dtype=bool),
        skeleton_relations=np.asarray(skeleton["relations"], dtype=np.float32),
        skeleton_relation_mask=np.asarray(skeleton["relation_mask"], dtype=bool),
        skeleton_frame_quality=skeleton_quality,
        skeleton_person_count=skeleton_people,
        skeleton_clip_scale=np.asarray(float(skeleton["clip_scale"]), dtype=np.float32),
        imu_values=imu_values,
        imu_time_seconds=imu_times,
        imu_frame_index=imu_frame_index,
        imu_device_offsets=device_offsets_array,
        imu_interval_counts=interval_counts.astype(np.uint16),
        imu_device_mask=device_mask,
        joint_names=np.asarray(H36M_JOINT_NAMES),
        common_part_names=np.asarray(COMMON_PART_NAMES),
        skeleton_feature_names=np.asarray(SKELETON_FEATURE_NAMES),
        skeleton_relation_names=np.asarray(SKELETON_RELATION_NAMES),
        imu_device_names=np.asarray(IMU_DEVICE_NAMES),
        imu_channel_names=np.asarray(IMU_CHANNEL_NAMES),
    )
    covered_frames = (interval_counts.sum(axis=1) > 0)
    return {
        "sample_id": sample_id,
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "user_id": row["user_id"],
        "trial_id": row["trial_id"],
        "frames": len(frame_ids),
        "frame_time_mode": frame_time_mode,
        "skeleton_valid_frames": int(np.asarray(skeleton["joint_mask"]).any(axis=1).sum()),
        "skeleton_mean_joint_valid_rate": float(np.asarray(skeleton["joint_mask"]).mean()),
        "skeleton_clip_scale": float(skeleton["clip_scale"]),
        "imu_path_present": int(bool(imu_text)),
        "imu_csv_files": imu_audit.csv_files,
        "imu_data_rows": imu_audit.data_rows,
        "imu_accepted_points": len(imu_values),
        "imu_rejected_rows": imu_audit.rejected_rows,
        "imu_unknown_device_rows": imu_audit.unknown_device_rows,
        "imu_device_count": int(device_mask.sum()),
        "imu_complete_five_devices": int(device_mask.all()),
        "imu_covered_frames": int(covered_frames.sum()),
        "imu_frame_coverage_rate": float(covered_frames.mean()),
        "imu_points_before_camera_span": before_points,
        "imu_points_after_camera_span": after_points,
        "imu_mean_abs_frame_offset_seconds": (
            float(alignment_offsets.mean()) if len(alignment_offsets) else 0.0
        ),
        "imu_p95_abs_frame_offset_seconds": (
            float(np.percentile(alignment_offsets, 95)) if len(alignment_offsets) else 0.0
        ),
        "cache_bytes": destination.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    p28_run = args.p28_run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest_rows = {row["sample_id"]: row for row in read_csv(manifest_path)}
    p28_rows = read_csv(p28_run / "trial_summary.csv")
    selected = [manifest_rows[row["sample_id"]] for row in p28_rows]
    if args.sample_id:
        wanted = set(args.sample_id)
        selected = [row for row in selected if row["sample_id"] in wanted]
    if args.max_trials > 0:
        selected = selected[: args.max_trials]
    if not selected:
        raise RuntimeError("no synchronized P28 trials selected")

    config = {
        "stage": "36_steps_8_9_skeleton_and_imu_part_inputs",
        "version": 2,
        "manifest": str(manifest_path),
        "p28_run": str(p28_run),
        "selected_trials": len(selected),
        "skeleton_policy": (
            "all synchronized frames; one clip-level body scale; root-centered XYZ plus "
            "bones, true-time velocity/acceleration, confidence and explicit relations"
        ),
        "imu_policy": (
            "every finite known-device CSV row retained; shared absolute timestamps; each "
            "point assigned exactly once by visual-frame midpoint boundaries; no 32-step resampling"
        ),
        "boundary_policy": (
            "first/last visual intervals are open-ended so same-trial IMU points outside the "
            "camera span remain in boundary intervals and are auditable"
        ),
        "legacy_counter_policy": (
            "four user1 trials have counter-only frame ids and no IMU; use 10 Hz only for "
            "Skeleton derivatives and mark frame_time_mode=counter_10hz_no_imu"
        ),
        "legacy_imu_timestamp_policy": (
            "accept both zero-padded ISO timestamps and legacy non-zero-padded month/day; "
            "finite known-device rows are never rejected only because of this spelling"
        ),
        "joint_names": list(H36M_JOINT_NAMES),
        "common_part_names": list(COMMON_PART_NAMES),
        "skeleton_feature_names": list(SKELETON_FEATURE_NAMES),
        "skeleton_relation_names": list(SKELETON_RELATION_NAMES),
        "imu_devices": list(IMU_DEVICE_NAMES),
        "imu_device_body_parts": list(IMU_DEVICE_BODY_PARTS),
        "imu_channels": list(IMU_CHANNEL_NAMES),
        "label_free_cache": True,
    }
    atomic_json(output / "config.json", config)

    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(selected, 1):
        relative = safe_trial_path(row["sample_id"])
        destination = cache_path(output, row["sample_id"])
        summary_path = output / "trial_summary" / relative.with_suffix(".json")
        existing = load_existing_summary(summary_path)
        if destination.is_file() and existing is not None and not args.overwrite:
            summaries.append(existing)
        else:
            summary = build_trial(row, p28_run, output)
            atomic_json(summary_path, summary)
            summaries.append(summary)
        if index % 200 == 0 or index == len(selected):
            print(
                f"P31 Skeleton/IMU cache {index}/{len(selected)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    write_csv(output / "trial_summary.csv", summaries)
    total_frames = int(sum(int(row["frames"]) for row in summaries))
    total_points = int(sum(int(row["imu_accepted_points"]) for row in summaries))
    final = {
        **config,
        "completed_trials": len(summaries),
        "completed_frames": total_frames,
        "skeleton_valid_frames": int(
            sum(int(row["skeleton_valid_frames"]) for row in summaries)
        ),
        "imu_usable_trials": int(sum(int(row["imu_device_count"]) > 0 for row in summaries)),
        "imu_complete_five_device_trials": int(
            sum(int(row["imu_complete_five_devices"]) for row in summaries)
        ),
        "imu_accepted_points": total_points,
        "imu_rejected_rows": int(sum(int(row["imu_rejected_rows"]) for row in summaries)),
        "imu_unknown_device_rows": int(
            sum(int(row["imu_unknown_device_rows"]) for row in summaries)
        ),
        "imu_points_before_camera_span": int(
            sum(int(row["imu_points_before_camera_span"]) for row in summaries)
        ),
        "imu_points_after_camera_span": int(
            sum(int(row["imu_points_after_camera_span"]) for row in summaries)
        ),
        "cache_bytes": int(sum(int(row["cache_bytes"]) for row in summaries)),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    atomic_json(output / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
