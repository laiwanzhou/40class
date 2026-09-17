from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Depth/IR spatial consistency from label-free temporal "
            "motion maps on outer-train development subjects."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_DIR / "cache" / "aligned_192x144",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p27_strong_inner"
        / "depth_ir_registration_audit",
    )
    parser.add_argument("--train-samples", type=int, default=240)
    parser.add_argument("--val-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=27083)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def downsample4(array: np.ndarray) -> np.ndarray:
    height = array.shape[-2] // 4 * 4
    width = array.shape[-1] // 4 * 4
    return array[..., :height, :width].reshape(
        *array.shape[:-2], height // 4, 4, width // 4, 4
    ).mean(axis=(-3, -1))


def motion_map(frames: np.ndarray) -> np.ndarray:
    if len(frames) < 2:
        return np.zeros(frames.shape[-2:], dtype=np.float32)
    differences = np.abs(np.diff(frames.astype(np.float32), axis=0))
    # The 80th percentile captures repeated arm/body contours without letting
    # one corrupt/noisy frame dominate the registration estimate.
    return np.quantile(differences, 0.8, axis=0).astype(np.float32)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first.reshape(-1).astype(np.float64)
    second = second.reshape(-1).astype(np.float64)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first.dot(second) / denominator) if denominator > 1e-8 else 0.0


def shifted_overlap(
    first: np.ndarray, second: np.ndarray, dy: int, dx: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = first.shape
    first_y0, first_y1 = max(0, dy), min(height, height + dy)
    first_x0, first_x1 = max(0, dx), min(width, width + dx)
    second_y0, second_y1 = max(0, -dy), min(height, height - dy)
    second_x0, second_x1 = max(0, -dx), min(width, width - dx)
    return (
        first[first_y0:first_y1, first_x0:first_x1],
        second[second_y0:second_y1, second_x0:second_x1],
    )


def best_shift(
    depth_motion: np.ndarray,
    ir_motion: np.ndarray,
    maximum: int = 6,
) -> tuple[int, int, float, float]:
    zero = correlation(depth_motion, ir_motion)
    best = (0, 0, zero)
    for dy in range(-maximum, maximum + 1):
        for dx in range(-maximum, maximum + 1):
            first, second = shifted_overlap(depth_motion, ir_motion, dy, dx)
            score = correlation(first, second)
            if score > best[2]:
                best = (dy, dx, score)
    return best[0], best[1], best[2], zero


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shifts_y = np.asarray([row["shift_y_lowres"] for row in rows], dtype=float)
    shifts_x = np.asarray([row["shift_x_lowres"] for row in rows], dtype=float)
    best = np.asarray([row["best_correlation"] for row in rows], dtype=float)
    zero = np.asarray([row["zero_correlation"] for row in rows], dtype=float)
    near_zero = (np.abs(shifts_y) <= 1) & (np.abs(shifts_x) <= 1)
    return {
        "samples": int(len(rows)),
        "median_shift_pixels_192x144": [
            float(np.median(shifts_y) * 4),
            float(np.median(shifts_x) * 4),
        ],
        "near_zero_within_4px_fraction": float(near_zero.mean()),
        "median_zero_correlation": float(np.median(zero)),
        "median_best_correlation": float(np.median(best)),
        "median_search_gain": float(np.median(best - zero)),
        "positive_zero_correlation_fraction": float((zero > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    cache = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    locations = {
        str(sample_id): (int(offset), int(length))
        for sample_id, (offset, length) in zip(
            metadata["sample_ids"], metadata["offsets"]
        )
    }
    depth = np.load(cache / "depth_uint8.npy", mmap_mode="r")
    ir = np.load(cache / "ir_uint8.npy", mmap_mode="r")
    rows = read_manifest(manifest)
    rng = np.random.default_rng(args.seed)
    audit_rows: list[dict[str, Any]] = []
    for split, count in (("train", args.train_samples), ("val", args.val_samples)):
        candidates = [
            row
            for row in rows
            if row["split"] == split and row["sample_id"] in locations
        ]
        selected = rng.choice(
            len(candidates), size=min(count, len(candidates)), replace=False
        )
        for candidate_index in selected:
            row = candidates[int(candidate_index)]
            offset, length = locations[row["sample_id"]]
            depth_clip = np.asarray(depth[offset : offset + length])
            depth_gray = (
                0.299 * depth_clip[..., 0]
                + 0.587 * depth_clip[..., 1]
                + 0.114 * depth_clip[..., 2]
            )
            ir_clip = np.asarray(ir[offset : offset + length])
            depth_motion = downsample4(motion_map(depth_gray))
            ir_motion = downsample4(motion_map(ir_clip))
            dy, dx, best, zero = best_shift(depth_motion, ir_motion)
            audit_rows.append(
                {
                    "split": split,
                    "sample_id": row["sample_id"],
                    "subject": row["user_id"],
                    "class_id": int(row["class_id"]),
                    "frames": int(length),
                    "shift_y_lowres": int(dy),
                    "shift_x_lowres": int(dx),
                    "shift_y_pixels_192x144": int(dy * 4),
                    "shift_x_pixels_192x144": int(dx * 4),
                    "zero_correlation": float(zero),
                    "best_correlation": float(best),
                    "search_gain": float(best - zero),
                }
            )
    with (output / "per_sample.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "protocol": "p27-depth-ir-motion-registration-audit-v1",
        "manifest": str(manifest),
        "cache": str(cache),
        "outer_fold": 0,
        "outer_held_predictions_generated": False,
        "method": (
            "80th-percentile temporal motion maps at 48x36; exhaustive relative "
            "shift search within +/-24 pixels of the 192x144 cache"
        ),
        "train": summarize(
            [row for row in audit_rows if row["split"] == "train"]
        ),
        "inner_held_validation": summarize(
            [row for row in audit_rows if row["split"] == "val"]
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
