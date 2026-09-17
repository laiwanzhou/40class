from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from aligned_data import frame_map
from local_roi_data import sample_positions, standardize_box


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_LOCATOR_DIR = PROJECT_DIR / "runs" / "p12_fold_pure_locator_predictions"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p12_local_depth_cache"
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 144
NUM_FRAMES = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache Local Depth uint8 tensors after cropping original 640x480 "
            "frames. Non-fallback samples are shared; fallback samples are fold-specific."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--locator-dir", type=Path, default=DEFAULT_LOCATOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[field]) for field in ("x0", "y0", "x1", "y1"))


def crop_trial(
    item: tuple[dict[str, str], tuple[float, float, float, float]],
) -> np.ndarray:
    row, raw_box = item
    maps = {
        "depth": frame_map(Path(row["depth_dir"]), "depth"),
        "ir": frame_map(Path(row["ir_dir"]), "ir"),
        "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
    }
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    positions = sample_positions(len(common), NUM_FRAMES, augment=False)
    crop_box = standardize_box(raw_box, 640, 480, context=0.15)
    frames: list[np.ndarray] = []
    for position in positions:
        with Image.open(maps["depth"][common[position]]) as image:
            image = image.convert("RGB")
            if image.size != (640, 480):
                raise ValueError(f"{row['sample_id']}: unexpected Depth {image.size}")
            local = image.crop(crop_box).resize(
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                Image.Resampling.BILINEAR,
            )
            frames.append(np.asarray(local, dtype=np.uint8).copy())
    return np.stack(frames)


def build_memmap(
    output_path: Path,
    items: list[tuple[dict[str, str], tuple[float, float, float, float]]],
    workers: int,
    label: str,
) -> None:
    if output_path.exists():
        array = np.load(output_path, mmap_mode="r")
        expected = (len(items), NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        if array.shape != expected or array.dtype != np.uint8:
            raise ValueError(f"Existing cache mismatch: {output_path}")
        print(f"{label}: reuse {output_path}", flush=True)
        return
    partial = output_path.with_suffix(".partial.npy")
    array = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.uint8,
        shape=(len(items), NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, frames in enumerate(
            executor.map(crop_trial, items),
            start=1,
        ):
            array[index - 1] = frames
            if index % 50 == 0 or index == len(items):
                print(f"{label} {index}/{len(items)}", flush=True)
    array.flush()
    del array
    partial.replace(output_path)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    manifest_rows = read_csv(args.manifest.resolve())
    manifest_rows.sort(key=lambda row: row["sample_id"])
    if len(manifest_rows) != 2914:
        raise ValueError("Expected 2914 manifest rows")
    predictions = {
        fold: {
            row["sample_id"]: row
            for row in read_csv(
                args.locator_dir.resolve()
                / f"fold_{fold}_locator_predictions.csv"
            )
        }
        for fold in range(3)
    }
    sample_ids = [row["sample_id"] for row in manifest_rows]
    for held_fold in range(3):
        if set(predictions[held_fold]) != set(sample_ids):
            raise ValueError(f"Fold {held_fold} locator coverage mismatch")
    fallback_ids = [
        sample_id
        for sample_id in sample_ids
        if int(predictions[0][sample_id]["motion_fallback"]) == 1
    ]
    fallback_id_set = set(fallback_ids)
    non_fallback_rows = [
        row
        for row in manifest_rows
        if row["sample_id"] not in fallback_id_set
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_items = [
        (row, box(predictions[0][row["sample_id"]]))
        for row in non_fallback_rows
    ]
    build_memmap(
        output_dir / "nonfallback_uint8.npy",
        base_items,
        args.workers,
        "Non-fallback cache",
    )
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    for held_fold in range(3):
        fallback_items = [
            (
                manifest_by_id[sample_id],
                box(predictions[held_fold][sample_id]),
            )
            for sample_id in fallback_ids
        ]
        build_memmap(
            output_dir / f"fallback_fold_{held_fold}_uint8.npy",
            fallback_items,
            args.workers,
            f"Fallback fold {held_fold} cache",
        )
    np.save(output_dir / "nonfallback_sample_ids.npy", np.asarray(
        [row["sample_id"] for row in non_fallback_rows]
    ))
    np.save(output_dir / "fallback_sample_ids.npy", np.asarray(fallback_ids))
    summary = {
        "samples": len(manifest_rows),
        "nonfallback": len(non_fallback_rows),
        "fallback": len(fallback_ids),
        "shape_per_sample": [NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "dtype": "uint8",
        "source": (
            "Original 640x480 Depth -> fold-pure ROI +15% context/4:3 "
            "-> resize 192x144"
        ),
        "bytes": {
            path.name: path.stat().st_size
            for path in sorted(output_dir.glob("*_uint8.npy"))
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
