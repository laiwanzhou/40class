from __future__ import annotations

import argparse
import csv
import json
import math
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
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p18_peak_local_depth_cache"
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 144
ENERGY_WIDTH = 64
ENERGY_HEIGHT = 48
UNIFORM_FRAMES = 8
PEAK_FRAMES = 4
PEAK_CANDIDATES = (-1, 0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an oracle-assisted Local Depth cache for the preregistered "
            "8-uniform + 4 motion-peak temporal sampling experiment."
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


def raw_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[field]) for field in ("x0", "y0", "x1", "y1"))


def robust_motion_energy(
    previous_depth: np.ndarray,
    previous_valid: np.ndarray,
    current_depth: np.ndarray,
    current_valid: np.ndarray,
) -> float:
    both = previous_valid.astype(bool) & current_valid.astype(bool)
    valid_count = int(both.sum())
    if valid_count < max(16, int(round(both.size * 0.05))):
        return 0.0
    difference = np.abs(
        current_depth.astype(np.int16) - previous_depth.astype(np.int16)
    )[both].astype(np.float32)
    difference[difference < 2.0] = 0.0
    top_count = max(1, int(math.ceil(valid_count * 0.10)))
    if top_count >= len(difference):
        return float(difference.mean())
    split = len(difference) - top_count
    return float(np.partition(difference, split)[split:].mean())


def select_peak_positions(
    length: int,
    transition_energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        raise ValueError("length must be positive")
    uniform = np.asarray(
        sample_positions(length, UNIFORM_FRAMES, augment=False),
        dtype=np.int16,
    )
    if length <= UNIFORM_FRAMES + PEAK_FRAMES:
        selected = np.asarray(
            sample_positions(length, UNIFORM_FRAMES + PEAK_FRAMES, augment=False),
            dtype=np.int16,
        )
        peaks = selected[-PEAK_FRAMES:].copy()
        return uniform, peaks

    candidates = list(range(1, length))
    candidates.sort(key=lambda position: (-float(transition_energy[position - 1]), position))
    used = set(int(value) for value in uniform)
    peaks: list[int] = []
    radius = max(1, int(round(length / 24.0)))
    for position in candidates:
        if position in used:
            continue
        if any(abs(position - existing) <= radius for existing in peaks):
            continue
        peaks.append(position)
        used.add(position)
        if len(peaks) == PEAK_FRAMES:
            break
    if len(peaks) < PEAK_FRAMES:
        for position in candidates:
            if position in used:
                continue
            peaks.append(position)
            used.add(position)
            if len(peaks) == PEAK_FRAMES:
                break
    if len(peaks) < PEAK_FRAMES:
        for position in range(length):
            if position in used:
                continue
            peaks.append(position)
            used.add(position)
            if len(peaks) == PEAK_FRAMES:
                break
    if len(peaks) != PEAK_FRAMES:
        raise RuntimeError(f"Could not select {PEAK_FRAMES} peaks from length={length}")
    return uniform, np.asarray(peaks, dtype=np.int16)


def unique_peak_candidates(
    length: int,
    uniform: np.ndarray,
    peaks: np.ndarray,
) -> np.ndarray:
    result = np.zeros((PEAK_FRAMES, len(PEAK_CANDIDATES)), dtype=np.int16)
    for peak_index, peak in enumerate(peaks.astype(int).tolist()):
        for candidate_index, delta in enumerate(PEAK_CANDIDATES):
            result[peak_index, candidate_index] = min(
                length - 1,
                max(0, peak + delta),
            )
    return result


def analyse_and_crop(
    item: tuple[dict[str, str], tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row, box = item
    maps = {
        "depth": frame_map(Path(row["depth_dir"]), "depth"),
        "ir": frame_map(Path(row["ir_dir"]), "ir"),
        "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
    }
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    length = len(common)
    if length != int(row["num_aligned_frames"]):
        raise ValueError(
            f"{row['sample_id']}: aligned length {length} != manifest "
            f"{row['num_aligned_frames']}"
        )
    crop_box = standardize_box(box, 640, 480, context=0.15)
    transition_energy = np.zeros(max(0, length - 1), dtype=np.float32)
    previous_depth: np.ndarray | None = None
    previous_valid: np.ndarray | None = None
    for frame_index, frame_id in enumerate(common):
        with Image.open(maps["depth"][frame_id]) as image:
            rgb = np.asarray(
                image.convert("RGB").crop(crop_box),
                dtype=np.uint8,
            )
        decoded, valid, _ = decode_jet_rgb(rgb, repair_unmatched=False)
        decoded, valid = resize_decoded_depth(
            decoded,
            valid,
            ENERGY_HEIGHT,
            ENERGY_WIDTH,
        )
        if previous_depth is not None and previous_valid is not None:
            transition_energy[frame_index - 1] = robust_motion_energy(
                previous_depth,
                previous_valid,
                decoded,
                valid,
            )
        previous_depth, previous_valid = decoded, valid

    uniform, peaks = select_peak_positions(length, transition_energy)
    peak_candidates = unique_peak_candidates(length, uniform, peaks)
    cache_positions = np.concatenate([uniform, peak_candidates.reshape(-1)])
    frames: list[np.ndarray] = []
    for position in cache_positions.astype(int).tolist():
        with Image.open(maps["depth"][common[position]]) as image:
            local = image.convert("RGB").crop(crop_box).resize(
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                Image.Resampling.BILINEAR,
            )
            frames.append(np.asarray(local, dtype=np.uint8).copy())
    return (
        np.stack(frames),
        uniform,
        peaks,
        peak_candidates,
        transition_energy,
    )


def main() -> None:
    args = parse_args()
    if int(args.workers) < 1:
        raise ValueError("--workers must be positive")
    started = time.perf_counter()
    manifest_rows = read_csv(args.manifest.resolve())
    manifest_rows.sort(key=lambda row: row["sample_id"])
    locator_rows = {
        row["sample_id"]: row for row in read_csv(args.locator.resolve())
    }
    sample_ids = [row["sample_id"] for row in manifest_rows]
    if len(manifest_rows) != 2914 or set(locator_rows) != set(sample_ids):
        raise ValueError("Manifest and locator must cover the same 2914 samples")
    items = [
        (row, raw_box(locator_rows[row["sample_id"]]))
        for row in manifest_rows
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_cache = output_dir / "local_peak_candidates_uint8.npy"
    partial_cache = output_dir / "local_peak_candidates_uint8.partial.npy"
    if final_cache.exists():
        raise FileExistsError(
            f"{final_cache} already exists; refusing to overwrite an experiment cache"
        )
    cached_frames = UNIFORM_FRAMES + PEAK_FRAMES * len(PEAK_CANDIDATES)
    cache = np.lib.format.open_memmap(
        partial_cache,
        mode="w+",
        dtype=np.uint8,
        shape=(len(items), cached_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )
    uniform_positions = np.zeros((len(items), UNIFORM_FRAMES), dtype=np.int16)
    peak_positions = np.zeros((len(items), PEAK_FRAMES), dtype=np.int16)
    peak_candidates = np.zeros(
        (len(items), PEAK_FRAMES, len(PEAK_CANDIDATES)),
        dtype=np.int16,
    )
    lengths = np.asarray(
        [int(row["num_aligned_frames"]) for row in manifest_rows],
        dtype=np.int16,
    )
    transition_offsets = np.zeros((len(items), 2), dtype=np.int64)
    flat_energy = np.zeros(
        int(np.maximum(lengths.astype(np.int64) - 1, 0).sum()),
        dtype=np.float32,
    )
    energy_offset = 0
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        for index, result in enumerate(executor.map(analyse_and_crop, items)):
            frames, uniform, peaks, candidates, energy = result
            cache[index] = frames
            uniform_positions[index] = uniform
            peak_positions[index] = peaks
            peak_candidates[index] = candidates
            transition_offsets[index] = (energy_offset, len(energy))
            flat_energy[energy_offset : energy_offset + len(energy)] = energy
            energy_offset += len(energy)
            completed = index + 1
            if completed % 25 == 0 or completed == len(items):
                print(
                    f"Peak Local cache {completed}/{len(items)} "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    if energy_offset != len(flat_energy):
        raise RuntimeError("Transition energy offset mismatch")
    cache.flush()
    del cache
    partial_cache.replace(final_cache)
    np.save(output_dir / "sample_ids.npy", np.asarray(sample_ids))
    np.save(output_dir / "lengths.npy", lengths)
    np.save(output_dir / "uniform_positions.npy", uniform_positions)
    np.save(output_dir / "peak_positions.npy", peak_positions)
    np.save(output_dir / "peak_candidates.npy", peak_candidates)
    np.save(output_dir / "transition_offsets.npy", transition_offsets)
    np.save(output_dir / "transition_energy.npy", flat_energy)
    summary = {
        "status": "exploratory_oracle_assisted",
        "deployable_oof": False,
        "experiment": "8_uniform_plus_4_local_depth_motion_peaks",
        "samples": len(items),
        "cached_frames_per_sample": cached_frames,
        "cache_layout": (
            "8 uniform midpoint frames followed by 4 peaks x [-1,0,+1] "
            "candidate frames"
        ),
        "motion_energy": (
            "mean of the largest 10% valid-intersection absolute decoded-JET "
            "index differences after suppressing differences below 2 indices"
        ),
        "energy_resolution": [ENERGY_HEIGHT, ENERGY_WIDTH],
        "nms_radius": "max(1, round(sequence_length/24))",
        "validation": "8 fixed uniform + 4 fixed peak centers, sorted in time",
        "training": (
            "8 fixed uniform + independently jittered peak candidate in "
            "[-1,0,+1], de-duplicated where possible, sorted in time"
        ),
        "roi": "all286 oracle-assisted locator, 15% context, 4:3",
        "bytes": final_cache.stat().st_size,
        "build_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
