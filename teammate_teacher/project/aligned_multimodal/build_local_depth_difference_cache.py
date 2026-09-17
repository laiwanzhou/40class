from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from aligned_data import frame_map
from depth_encoding import decode_jet_rgb, resize_decoded_depth
from local_roi_data import sample_positions, standardize_box


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_LOCATOR = (
    PROJECT_DIR
    / "runs"
    / "p16_oracle_assisted_locator_predictions"
    / "fold_0_locator_predictions.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p19_local_depth_difference_cache"
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 144
NUM_FRAMES = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build decoded Local Depth difference and valid-intersection "
            "channels for the fixed 12-midpoint sampling experiment."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--locator", type=Path, default=DEFAULT_LOCATOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def locator_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[field]) for field in ("x0", "y0", "x1", "y1"))


def build_trial(
    item: tuple[dict[str, str], tuple[float, float, float, float]],
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    row, raw_box = item
    maps = {
        "depth": frame_map(Path(row["depth_dir"]), "depth"),
        "ir": frame_map(Path(row["ir_dir"]), "ir"),
        "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
    }
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    if len(common) != int(row["num_aligned_frames"]):
        raise ValueError(
            f"{row['sample_id']}: aligned length {len(common)} differs from manifest"
        )
    positions = sample_positions(len(common), NUM_FRAMES, augment=False)
    crop_box = standardize_box(raw_box, 640, 480, context=0.15)
    decoded_frames: list[np.ndarray] = []
    valid_frames: list[np.ndarray] = []
    for position in positions:
        with Image.open(maps["depth"][common[position]]) as image:
            rgb = np.asarray(
                image.convert("RGB").crop(crop_box),
                dtype=np.uint8,
            )
        decoded, valid, _ = decode_jet_rgb(rgb, repair_unmatched=False)
        decoded, valid = resize_decoded_depth(
            decoded,
            valid,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )
        decoded_frames.append(decoded)
        valid_frames.append(valid.astype(bool))
    depth = np.stack(decoded_frames)
    valid = np.stack(valid_frames)
    difference = np.zeros_like(depth, dtype=np.uint8)
    valid_intersection = np.zeros_like(depth, dtype=np.uint8)
    valid_intersection[0] = valid[0].astype(np.uint8) * 255
    if NUM_FRAMES > 1:
        both = valid[1:] & valid[:-1]
        raw_difference = np.abs(
            depth[1:].astype(np.int16) - depth[:-1].astype(np.int16)
        )
        difference[1:] = np.where(both, raw_difference, 0).astype(np.uint8)
        valid_intersection[1:] = both.astype(np.uint8) * 255
    extra = np.stack([difference, valid_intersection], axis=-1)
    active = difference[1:] > 0
    valid_pairs = valid_intersection[1:] > 0
    stats: dict[str, float | int | str] = {
        "sample_id": row["sample_id"],
        "length": len(common),
        "positions": " ".join(str(value) for value in positions),
        "valid_fraction": float((valid_intersection > 0).mean()),
        "nonzero_difference_fraction": float(active.mean()),
        "mean_difference_all_pixels": float(difference[1:].mean()),
        "mean_difference_valid_pixels": (
            float(difference[1:][valid_pairs].mean())
            if valid_pairs.any()
            else 0.0
        ),
        "p95_difference_valid_pixels": (
            float(np.quantile(difference[1:][valid_pairs], 0.95))
            if valid_pairs.any()
            else 0.0
        ),
    }
    return extra, stats


def write_stats(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if int(args.workers) < 1:
        raise ValueError("--workers must be positive")
    manifest = read_csv(args.manifest.resolve())
    manifest.sort(key=lambda row: row["sample_id"])
    locator = {
        row["sample_id"]: row for row in read_csv(args.locator.resolve())
    }
    sample_ids = [row["sample_id"] for row in manifest]
    if len(manifest) != 2914 or set(locator) != set(sample_ids):
        raise ValueError("Manifest and locator must cover the same 2914 samples")
    items = [
        (row, locator_box(locator[row["sample_id"]]))
        for row in manifest
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "extra_channels_uint8.npy"
    partial_path = output_dir / "extra_channels_uint8.partial.npy"
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite {final_path}")
    started = time.perf_counter()
    cache = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(items), NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 2),
    )
    stats: list[dict[str, float | int | str]] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        for index, (extra, row_stats) in enumerate(
            executor.map(build_trial, items)
        ):
            cache[index] = extra
            stats.append(row_stats)
            completed = index + 1
            if completed % 50 == 0 or completed == len(items):
                print(
                    f"Difference cache {completed}/{len(items)} "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    cache.flush()
    del cache
    partial_path.replace(final_path)
    np.save(output_dir / "sample_ids.npy", np.asarray(sample_ids))
    write_stats(output_dir / "per_sample_stats.csv", stats)
    summary = {
        "status": "exploratory_oracle_assisted",
        "deployable_oof": False,
        "experiment": "fixed12_local_decoded_depth_difference",
        "samples": len(items),
        "shape": [len(items), NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 2],
        "dtype": "uint8",
        "channels": [
            "absolute decoded-JET index difference to previous selected frame",
            "valid-pixel intersection mask (0 or 255)",
        ],
        "first_frame": (
            "difference=0; valid channel=current decoded-depth valid mask"
        ),
        "sampling": "same fixed midpoint from each of 12 segments as p16 baseline",
        "roi": "same all286 oracle-assisted ROI and 15% context as p16",
        "normalization_at_training": (
            "difference clipped after division by 16 JET indices; "
            "valid mask divided by 255"
        ),
        "bytes": final_path.stat().st_size,
        "build_seconds": time.perf_counter() - started,
        "aggregate": {
            "valid_fraction": float(
                np.mean([float(row["valid_fraction"]) for row in stats])
            ),
            "nonzero_difference_fraction": float(
                np.mean(
                    [float(row["nonzero_difference_fraction"]) for row in stats]
                )
            ),
            "mean_difference_valid_pixels": float(
                np.mean(
                    [float(row["mean_difference_valid_pixels"]) for row in stats]
                )
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
