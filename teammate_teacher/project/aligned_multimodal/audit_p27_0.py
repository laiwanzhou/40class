from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from p27_data import (
    DEFAULT_ALIGNED_CACHE,
    DEFAULT_FOLD_SUMMARY,
    DEFAULT_IMU_CACHE,
    DEFAULT_LOCATOR_DIR,
    DEFAULT_UNION_MANIFEST,
    build_p27_manifest_rows,
    frame_map,
    read_csv,
    write_manifest,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_0_audit"
VISUAL_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)")
TARGET_CLASSES = {9, 10, 19, 21, 22, 24, 25, 26, 37}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P27-0 read-only data/protocol audit")
    parser.add_argument("--union-manifest", type=Path, default=DEFAULT_UNION_MANIFEST)
    parser.add_argument("--fold-summary", type=Path, default=DEFAULT_FOLD_SUMMARY)
    parser.add_argument("--aligned-cache", type=Path, default=DEFAULT_ALIGNED_CACHE)
    parser.add_argument("--imu-cache", type=Path, default=DEFAULT_IMU_CACHE)
    parser.add_argument("--locator-dir", type=Path, default=DEFAULT_LOCATOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spatial-pairs", type=int, default=240)
    parser.add_argument("--weak-label-samples", type=int, default=180)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def visual_times(row: dict[str, str]) -> tuple[float, float] | None:
    candidates = [
        Path(row["depth_dir"]),
        Path(row["skeleton_dir"]) / "predictions",
    ]
    values: list[float] = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        values.clear()
        for path in directory.iterdir():
            match = VISUAL_TIME_RE.search(path.stem)
            if not match:
                continue
            try:
                values.append(
                    datetime.strptime(
                        match.group(1), "%Y-%m-%d_%H-%M-%S.%f"
                    ).timestamp()
                )
            except ValueError:
                continue
        if values:
            return min(values), max(values)
    return None


def imu_times(path: Path) -> tuple[float, float] | None:
    values: list[float] = []
    if not path.is_dir():
        return None
    for file_path in path.glob("*.csv"):
        try:
            with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for fields in reader:
                    if not fields:
                        continue
                    try:
                        values.append(datetime.fromisoformat(fields[0].strip()).timestamp())
                    except ValueError:
                        continue
        except (OSError, UnicodeError):
            continue
    return (min(values), max(values)) if values else None


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
    }


def audit_time(rows: list[dict[str, str]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for row in rows:
        if int(row["imu_usable"]) != 1:
            continue
        visual = visual_times(row)
        imu = imu_times(Path(row["imu_dir"]))
        if visual is None or imu is None or visual[1] <= visual[0] or imu[1] <= imu[0]:
            continue
        records.append(
            {
                "sample_id": row["sample_id"],
                "class_id": int(row["class_id"]),
                "user_id": row["user_id"],
                "visual_duration_seconds": visual[1] - visual[0],
                "imu_duration_seconds": imu[1] - imu[0],
                "start_offset_seconds": imu[0] - visual[0],
                "end_offset_seconds": imu[1] - visual[1],
                "duration_ratio": (imu[1] - imu[0]) / (visual[1] - visual[0]),
            }
        )
    starts = np.asarray([float(row["start_offset_seconds"]) for row in records])
    ends = np.asarray([float(row["end_offset_seconds"]) for row in records])
    ratios = np.asarray([float(row["duration_ratio"]) for row in records])
    summary = {
        "paired_trials": len(records),
        "start_offset_seconds": percentile_summary(starts),
        "end_offset_seconds": percentile_summary(ends),
        "duration_ratio": percentile_summary(ratios),
        "both_endpoints_within_0_05": float(np.mean((np.abs(starts) <= 0.05) & (np.abs(ends) <= 0.05))),
        "both_endpoints_within_0_10": float(np.mean((np.abs(starts) <= 0.10) & (np.abs(ends) <= 0.10))),
        "both_endpoints_within_0_20": float(np.mean((np.abs(starts) <= 0.20) & (np.abs(ends) <= 0.20))),
    }
    return summary, records


def audit_spatial(
    rows: list[dict[str, str]], pair_count: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates = [row for row in rows if int(row["depth_usable"]) and int(row["ir_usable"])]
    random.Random(2701).shuffle(candidates)
    records: list[dict[str, object]] = []
    for row in candidates:
        if len(records) >= pair_count:
            break
        depth = frame_map(row["depth_dir"], "depth")
        ir = frame_map(row["ir_dir"], "ir")
        common = sorted(set(depth) & set(ir))
        if not common:
            continue
        key = common[len(common) // 2]
        depth_image = cv2.imread(str(depth[key]), cv2.IMREAD_GRAYSCALE)
        ir_image = cv2.imread(str(ir[key]), cv2.IMREAD_GRAYSCALE)
        if depth_image is None or ir_image is None:
            continue
        same_shape = depth_image.shape == ir_image.shape
        if not same_shape:
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "same_shape": 0,
                    "dx_pixels": np.nan,
                    "dy_pixels": np.nan,
                    "phase_response": 0.0,
                }
            )
            continue
        depth_small = cv2.resize(depth_image, (160, 120)).astype(np.float32)
        ir_small = cv2.resize(ir_image, (160, 120)).astype(np.float32)
        depth_edge = cv2.magnitude(
            cv2.Sobel(depth_small, cv2.CV_32F, 1, 0),
            cv2.Sobel(depth_small, cv2.CV_32F, 0, 1),
        )
        ir_edge = cv2.magnitude(
            cv2.Sobel(ir_small, cv2.CV_32F, 1, 0),
            cv2.Sobel(ir_small, cv2.CV_32F, 0, 1),
        )
        depth_edge = (depth_edge - depth_edge.mean()) / (depth_edge.std() + 1e-6)
        ir_edge = (ir_edge - ir_edge.mean()) / (ir_edge.std() + 1e-6)
        shift, response = cv2.phaseCorrelate(depth_edge, ir_edge)
        records.append(
            {
                "sample_id": row["sample_id"],
                "same_shape": 1,
                "dx_pixels": float(shift[0] * 4.0),
                "dy_pixels": float(shift[1] * 4.0),
                "phase_response": float(response),
            }
        )
    valid = [row for row in records if int(row["same_shape"]) == 1]
    dx = np.asarray([float(row["dx_pixels"]) for row in valid])
    dy = np.asarray([float(row["dy_pixels"]) for row in valid])
    response = np.asarray([float(row["phase_response"]) for row in valid])
    summary = {
        "pairs": len(records),
        "same_shape": sum(int(row["same_shape"]) for row in records),
        "dx_pixels": percentile_summary(dx),
        "dy_pixels": percentile_summary(dy),
        "phase_response": percentile_summary(response),
        "within_4_pixels": float(np.mean((np.abs(dx) <= 4) & (np.abs(dy) <= 4))),
        "within_8_pixels": float(np.mean((np.abs(dx) <= 8) & (np.abs(dy) <= 8))),
        "within_16_pixels": float(np.mean((np.abs(dx) <= 16) & (np.abs(dy) <= 16))),
        "interpretation": (
            "Same raster geometry and predominantly small edge shift support broad shared ROI coordinates; "
            "low-response outliers forbid tight pixel-level registration claims."
        ),
    }
    return summary, records


def load_locator(locator_dir: Path, fold: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for row in read_csv(locator_dir / f"fold_{fold}_locator_predictions.csv"):
        output[row["sample_id"]] = np.asarray(
            [
                float(row["x0"]) / float(row["raw_width"]),
                float(row["y0"]) / float(row["raw_height"]),
                float(row["x1"]) / float(row["raw_width"]),
                float(row["y1"]) / float(row["raw_height"]),
            ],
            dtype=np.float32,
        )
    return output


def weak_label_audit(
    rows: list[dict[str, str]],
    aligned_cache: Path,
    imu_cache: Path,
    locator_dir: Path,
    sample_count: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    metadata = json.loads((aligned_cache / "metadata.json").read_text(encoding="utf-8"))
    locations = {
        sample_id: (int(offset), int(length))
        for sample_id, (offset, length) in zip(
            metadata["sample_ids"], metadata["offsets"], strict=True
        )
    }
    depth = np.load(aligned_cache / "depth_uint8.npy", mmap_mode="r", allow_pickle=False)
    ir = np.load(aligned_cache / "ir_uint8.npy", mmap_mode="r", allow_pickle=False)
    skeleton = np.load(aligned_cache / "skeleton_float32.npy", mmap_mode="r", allow_pickle=False)
    imu = np.load(imu_cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    imu_time_mask = np.load(imu_cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    locators = {fold: load_locator(locator_dir, fold) for fold in range(3)}
    target = [row for row in rows if int(row["class_id"]) in TARGET_CLASSES and row["sample_id"] in locations]
    other = [row for row in rows if int(row["class_id"]) not in TARGET_CLASSES and row["sample_id"] in locations]
    rng = random.Random(2702)
    rng.shuffle(target)
    rng.shuffle(other)
    chosen = target[: sample_count // 2] + other[: sample_count - sample_count // 2]
    records: list[dict[str, object]] = []
    for row in chosen:
        offset, length = locations[row["sample_id"]]
        positions = np.rint(np.linspace(0, length - 1, 12)).astype(np.int64) + offset
        sk = np.asarray(skeleton[positions], dtype=np.float32)
        confidence = sk[:, :, 3]
        left = sk[:, 13, :3]
        right = sk[:, 16, :3]
        head = sk[:, 10, :3]
        hand_head = np.stack(
            [np.linalg.norm(left - head, axis=1), np.linalg.norm(right - head, axis=1)],
            axis=1,
        )
        hand_speed = np.stack(
            [
                np.linalg.norm(np.diff(left, axis=0), axis=1),
                np.linalg.norm(np.diff(right, axis=0), axis=1),
            ],
            axis=1,
        )
        depth_clip = np.asarray(depth[positions], dtype=np.float32).mean(axis=3)
        ir_clip = np.asarray(ir[positions], dtype=np.float32)
        global_motion = 0.5 * (
            np.abs(np.diff(depth_clip, axis=0)).mean(axis=(1, 2))
            + np.abs(np.diff(ir_clip, axis=0)).mean(axis=(1, 2))
        )
        fold = int(row["subject_fold"])
        box = locators[fold].get(row["sample_id"], np.asarray([0, 0, 1, 1], dtype=np.float32))
        y0, y1 = int(box[1] * 144), max(int(box[3] * 144), int(box[1] * 144) + 1)
        x0, x1 = int(box[0] * 192), max(int(box[2] * 192), int(box[0] * 192) + 1)
        local_motion = 0.5 * (
            np.abs(np.diff(depth_clip[:, y0:y1, x0:x1], axis=0)).mean(axis=(1, 2))
            + np.abs(np.diff(ir_clip[:, y0:y1, x0:x1], axis=0)).mean(axis=(1, 2))
        )
        imu_index = int(row["imu_cache_index"])
        if imu_index >= 0:
            imu_values = np.asarray(imu[imu_index], dtype=np.float32)
            time_mask = np.asarray(imu_time_mask[imu_index], dtype=bool)
            gyro = np.linalg.norm(imu_values[:, :, 3:6], axis=2)
            gyro_valid = gyro[time_mask]
            imu_gyro_mean = float(gyro_valid.mean()) if len(gyro_valid) else 0.0
            imu_mask_fraction = float(time_mask.mean())
        else:
            imu_gyro_mean = 0.0
            imu_mask_fraction = 0.0
        records.append(
            {
                "sample_id": row["sample_id"],
                "class_id": int(row["class_id"]),
                "user_id": row["user_id"],
                "skeleton_confidence_mean": float(confidence.mean()),
                "hand_head_mean": float(hand_head.mean()),
                "hand_head_range": float(np.ptp(hand_head, axis=0).mean()),
                "hand_speed_mean": float(hand_speed.mean()),
                "visual_global_motion_mean": float(global_motion.mean()),
                "visual_local_motion_mean": float(local_motion.mean()),
                "local_global_motion_ratio": float(
                    local_motion.mean() / max(global_motion.mean(), 1e-6)
                ),
                "imu_time_mask_fraction": imu_mask_fraction,
                "imu_gyro_mean": imu_gyro_mean,
            }
        )
    numeric = [
        key
        for key in records[0]
        if key not in {"sample_id", "class_id", "user_id"}
    ]
    summary = {key: percentile_summary(np.asarray([float(row[key]) for row in records])) for key in numeric}
    summary["samples"] = len(records)
    summary["interpretation"] = (
        "Continuous geometry, motion, and IMU targets have non-degenerate ranges. "
        "They are retained as continuous fold-normalized targets; no class-derived thresholds are used."
    )
    return summary, records


def build_montage(
    rows: list[dict[str, str]],
    locator_dir: Path,
    output_path: Path,
) -> None:
    locators = {fold: load_locator(locator_dir, fold) for fold in range(3)}
    chosen: list[dict[str, str]] = []
    for class_id in sorted(TARGET_CLASSES):
        match = next(
            (
                row
                for row in rows
                if int(row["class_id"]) == class_id
                and int(row["depth_usable"])
                and int(row["ir_usable"])
                and row["sample_id"] in locators[int(row["subject_fold"])]
            ),
            None,
        )
        if match is not None:
            chosen.append(match)
    width, panel_height = 640, 250
    canvas = Image.new("RGB", (width, panel_height * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(chosen):
        depth = frame_map(row["depth_dir"], "depth")
        ir = frame_map(row["ir_dir"], "ir")
        keys = sorted(set(depth) & set(ir))
        key = keys[len(keys) // 2]
        with Image.open(depth[key]) as image:
            depth_image = image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
        with Image.open(ir[key]) as image:
            ir_image = image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
        box = locators[int(row["subject_fold"])][row["sample_id"]]
        for image in (depth_image, ir_image):
            panel_draw = ImageDraw.Draw(image)
            panel_draw.rectangle(
                [box[0] * 320, box[1] * 240, box[2] * 320, box[3] * 240],
                outline=(0, 255, 0),
                width=3,
            )
        y = row_index * panel_height
        canvas.paste(depth_image, (0, y))
        canvas.paste(ir_image, (320, y))
        draw.text((4, y + 2), f"D c{row['class_id']} {row['user_id']}", fill=(255, 255, 255))
        draw.text((324, y + 2), "IR / same broad ROI", fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def checkpoint_audit() -> dict[str, object]:
    entries: dict[str, object] = {}
    for fold in range(3):
        fold_entries: dict[str, object] = {}
        paths = {
            "skeleton": PROJECT_DIR / "runs" / "p0_tracking" / f"fold_{fold}_skeleton_first" / "last.pt",
            "depth": PROJECT_DIR / "runs" / "p5_depth_imagenet" / f"fold_{fold}" / "last.pt",
            "ir": PROJECT_DIR / "runs" / "p8_ir_imagenet_sum" / f"fold_{fold}" / "last.pt",
        }
        for name, path in paths.items():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            fold_entries[name] = {
                "path": str(path.relative_to(REPO_DIR)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "epoch": int(checkpoint["epoch"]),
                "fixed_last": path.name == "last.pt",
            }
        entries[str(fold)] = fold_entries
    return entries


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, manifest_summary = build_p27_manifest_rows(
        args.union_manifest.resolve(),
        args.imu_cache.resolve(),
        args.fold_summary.resolve(),
    )
    manifest_path = output / "p27_train_manifest.csv"
    write_manifest(manifest_path, rows)
    csv_rows = [{key: str(value) for key, value in row.items()} for row in rows]

    fold_summary: dict[str, object] = {}
    for fold in range(3):
        train_rows = [row for row in rows if int(row["subject_fold"]) != fold]
        held_rows = [row for row in rows if int(row["subject_fold"]) == fold]
        fold_summary[str(fold)] = {
            "train": len(train_rows),
            "held": len(held_rows),
            "held_subjects": sorted({str(row["user_id"]) for row in held_rows}),
            "train_imu_usable": sum(int(row["imu_usable"]) for row in train_rows),
            "held_imu_usable": sum(int(row["imu_usable"]) for row in held_rows),
        }

    time_summary, time_rows = audit_time(csv_rows)
    spatial_summary, spatial_rows = audit_spatial(csv_rows, args.spatial_pairs)
    weak_summary, weak_rows = weak_label_audit(
        csv_rows,
        args.aligned_cache.resolve(),
        args.imu_cache.resolve(),
        args.locator_dir.resolve(),
        args.weak_label_samples,
    )
    write_csv(output / "time_alignment_audit.csv", time_rows)
    write_csv(output / "spatial_registration_audit.csv", spatial_rows)
    write_csv(output / "weak_label_numeric_audit.csv", weak_rows)
    build_montage(csv_rows, args.locator_dir.resolve(), output / "weak_label_roi_montage.jpg")

    summary = {
        "protocol": "p27-0-v1-fixed-before-training",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": manifest_summary,
        "folds": fold_summary,
        "time_alignment": time_summary,
        "spatial_registration": spatial_summary,
        "weak_labels": weak_summary,
        "checkpoints": checkpoint_audit(),
        "hard_blockers": [],
        "fixed_adjustments": [
            "Use terminal trial-local frame counter to recover 17 valid timestamp-less visual trials.",
            "Define the P27-A union from actual P27 parser usability (2933), not the six-modality directory union (3036).",
            "Use fold-pure broad Depth ROI coordinates; transfer only broad coordinates to IR, never claim tight pixel registration.",
            "Use visual/Skeleton/IR as 12-bin main axis and IMU local attention window of plus/minus one bin.",
            "Exclude P12 distillation from A0/A1/A2.",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps(manifest_hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
