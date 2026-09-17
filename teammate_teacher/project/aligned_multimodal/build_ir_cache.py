from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from aligned_data import frame_map


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an IR-only aligned cache without duplicating Depth."
    )
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_DIR / "data" / "manifest.csv"
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_trial(
    work: tuple[dict[str, str], int, int],
) -> np.ndarray:
    row, height, width = work
    maps = {
        "depth": frame_map(Path(row["depth_dir"]), "depth"),
        "ir": frame_map(Path(row["ir_dir"]), "ir"),
        "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
    }
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    frames: list[np.ndarray] = []
    for key in common:
        with Image.open(maps["ir"][key]) as image:
            image = image.convert("L").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames)


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    output = args.cache_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(manifest)
    lengths = [int(row["num_aligned_frames"]) for row in rows]
    total_frames = int(sum(lengths))
    shape = (total_frames, int(args.image_height), int(args.image_width))
    temporary = output / "ir_uint8.npy.building"
    final = output / "ir_uint8.npy"
    if final.exists() or temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing cache: {final} / {temporary}"
        )
    array = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=shape
    )
    offsets: list[list[int]] = []
    cursor = 0
    started = time.time()
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        work = (
            (row, int(args.image_height), int(args.image_width)) for row in rows
        )
        for index, frames in enumerate(executor.map(load_trial, work)):
            expected = lengths[index]
            if len(frames) != expected:
                raise RuntimeError(
                    f"{rows[index]['sample_id']}: {len(frames)} != {expected}"
                )
            end = cursor + expected
            array[cursor:end] = frames
            offsets.append([cursor, expected])
            cursor = end
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                elapsed = time.time() - started
                print(
                    f"{index + 1}/{len(rows)} trials; "
                    f"{cursor}/{total_frames} frames; {elapsed / 60:.1f} min",
                    flush=True,
                )
    array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()
    del array
    os.replace(temporary, final)
    metadata = {
        "version": 1,
        "manifest": str(manifest),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "num_samples": len(rows),
        "total_frames": total_frames,
        "sample_ids": [row["sample_id"] for row in rows],
        "offsets": offsets,
        "files": ["ir_uint8.npy"],
        "build_seconds": round(time.time() - started, 2),
        "modalities": ["ir"],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
