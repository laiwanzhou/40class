from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_CACHE = PROJECT_DIR / "cache" / "skeleton_raw"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "skeleton_normalization_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 Skeleton root、scale 与时间间隔")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def describe(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    cache = args.cache_dir.resolve()
    output = args.output.resolve()
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    raw = np.load(cache / "skeleton_raw_float32.npy", mmap_mode="r")
    times = np.load(cache / "frame_time_float32.npy", mmap_mode="r")
    people = np.load(cache / "person_count_uint8.npy", mmap_mode="r")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}

    root_z: list[float] = []
    scales: list[float] = []
    trial_root_z_std: list[float] = []
    trial_root_z_range: list[float] = []
    trial_scale_cv: list[float] = []
    time_gaps: list[float] = []
    per_user: dict[str, list[float]] = defaultdict(list)
    short_trials = 0
    gap_outlier_trials = 0

    for sample_id, (offset, length) in zip(metadata["sample_ids"], metadata["offsets"]):
        offset, length = int(offset), int(length)
        clip = np.asarray(raw[offset : offset + length, :, :3], dtype=np.float32)
        present = np.asarray(people[offset : offset + length]) > 0
        if length < 12:
            short_trials += 1
        clip = clip[present]
        if len(clip):
            roots = clip[:, 0]
            centered = clip - roots[:, None]
            scale = np.linalg.norm(centered, axis=2).max(axis=1)
            valid_scale = np.isfinite(scale) & (scale > 1e-6)
            roots = roots[valid_scale]
            scale = scale[valid_scale]
            if len(scale):
                root_z.extend(roots[:, 2].tolist())
                scales.extend(scale.tolist())
                trial_root_z_std.append(float(roots[:, 2].std()))
                trial_root_z_range.append(float(np.ptp(roots[:, 2])))
                trial_scale_cv.append(float(scale.std() / max(scale.mean(), 1e-6)))
                per_user[rows[sample_id]["user_id"]].extend(roots[:, 2].tolist())
        clip_times = np.asarray(times[offset : offset + length], dtype=np.float64)
        gaps = np.diff(clip_times)
        gaps = gaps[gaps > 0]
        if len(gaps):
            time_gaps.extend(gaps.tolist())
            gap_outlier_trials += int(np.any(np.abs(gaps - 0.1) > 0.002))

    result = {
        "samples": len(metadata["sample_ids"]),
        "frames": int(metadata["total_frames"]),
        "shorter_than_12_frames": short_trials,
        "shorter_than_12_fraction": short_trials / len(metadata["sample_ids"]),
        "multi_person_frame_fraction": float(np.mean(np.asarray(people) > 1)),
        "root_z": describe(root_z),
        "scale": describe(scales),
        "trial_root_z_std": describe(trial_root_z_std),
        "trial_root_z_range": describe(trial_root_z_range),
        "trial_scale_cv": describe(trial_scale_cv),
        "positive_frame_gap_seconds": describe(time_gaps),
        "frame_gap_outlier_trials": gap_outlier_trials,
        "per_user_root_z_median": {
            user: float(np.median(values))
            for user, values in sorted(per_user.items(), key=lambda item: int(item[0][4:]))
        },
        "recommendation": {
            "pose": "per-frame root center + one robust clip-level scale",
            "scale_estimator": "median per-frame max joint radius; later compare median bone length",
            "root_z": "retain only as a separate optional channel because it contains subject/camera bias",
            "velocity": "compute on full sequence with actual timestamps before 12-frame sampling",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
