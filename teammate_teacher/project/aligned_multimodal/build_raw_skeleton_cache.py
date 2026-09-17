from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from aligned_data import frame_map


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_ALIGNED_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
DEFAULT_OUTPUT = PROJECT_DIR / "cache" / "skeleton_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="缓存未经 root-center/scale 的原始 Skeleton")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--aligned-cache", type=Path, default=DEFAULT_ALIGNED_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_first_person(path: Path) -> tuple[np.ndarray, int]:
    try:
        people = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        people = []
    person_count = len(people) if isinstance(people, list) else 0
    if not isinstance(people, list) or not people or not isinstance(people[0], dict):
        return np.zeros((17, 4), dtype=np.float32), person_count
    keypoints = np.asarray(people[0].get("keypoints", []), dtype=np.float32)
    scores = np.asarray(people[0].get("keypoint_scores", []), dtype=np.float32)
    if keypoints.shape != (17, 3):
        return np.zeros((17, 4), dtype=np.float32), person_count
    if scores.shape != (17,):
        scores = np.ones(17, dtype=np.float32)
    raw = np.concatenate([keypoints, scores[:, None]], axis=1)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), person_count


def relative_times(frame_ids: list[str]) -> np.ndarray:
    parsed: list[datetime | None] = []
    for frame_id in frame_ids:
        timestamp = frame_id.rsplit("_", 1)[0]
        try:
            parsed.append(datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S.%f"))
        except ValueError:
            parsed.append(None)
    if parsed and all(value is not None for value in parsed):
        first = parsed[0]
        assert first is not None
        return np.asarray([(value - first).total_seconds() for value in parsed], dtype=np.float32)
    return np.arange(len(frame_ids), dtype=np.float32) * 0.1


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    aligned_cache = args.aligned_cache.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    aligned_metadata = json.loads((aligned_cache / "metadata.json").read_text(encoding="utf-8"))
    sample_ids = list(aligned_metadata["sample_ids"])
    offsets = [tuple(map(int, value)) for value in aligned_metadata["offsets"]]
    total_frames = int(aligned_metadata["total_frames"])
    final_paths = {
        "skeleton": output / "skeleton_raw_float32.npy",
        "time": output / "frame_time_float32.npy",
        "people": output / "person_count_uint8.npy",
    }

    recoverable_people = output / "person_count_uint8.npy.building"
    if final_paths["skeleton"].is_file() and final_paths["time"].is_file() and (
        final_paths["people"].is_file() or recoverable_people.is_file()
    ):
        if not final_paths["people"].is_file():
            recoverable_people.replace(final_paths["people"])
        metadata = {
            "version": 1,
            "manifest": str(manifest),
            "source_aligned_cache": str(aligned_cache),
            "num_samples": len(sample_ids),
            "total_frames": total_frames,
            "sample_ids": sample_ids,
            "offsets": offsets,
            "files": [path.name for path in final_paths.values()],
            "description": "first-person raw 3D keypoints + score; no root centering or scale normalization",
            "build_seconds": None,
            "recovered_after_completed_build": True,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"recovered": True, "frames": total_frames}, ensure_ascii=False))
        return

    temporary = {
        "skeleton": output / "skeleton_raw_float32.npy.building",
        "time": output / "frame_time_float32.npy.building",
        "people": output / "person_count_uint8.npy.building",
    }
    arrays = {
        "skeleton": np.lib.format.open_memmap(
            temporary["skeleton"], mode="w+", dtype=np.float32, shape=(total_frames, 17, 4)
        ),
        "time": np.lib.format.open_memmap(
            temporary["time"], mode="w+", dtype=np.float32, shape=(total_frames,)
        ),
        "people": np.lib.format.open_memmap(
            temporary["people"], mode="w+", dtype=np.uint8, shape=(total_frames,)
        ),
    }

    started = time.time()
    for sample_index, (sample_id, (offset, expected_length)) in enumerate(zip(sample_ids, offsets), 1):
        row = rows.get(sample_id)
        if row is None:
            raise KeyError(f"manifest 缺少缓存样本：{sample_id}")
        maps = {
            "depth": frame_map(Path(row["depth_dir"]), "depth"),
            "ir": frame_map(Path(row["ir_dir"]), "ir"),
            "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
        }
        common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
        if len(common_ids) != expected_length:
            raise RuntimeError(f"{sample_id} 对齐帧数变化：{len(common_ids)} != {expected_length}")
        for local_index, frame_id in enumerate(common_ids):
            raw, person_count = load_first_person(maps["skeleton"][frame_id])
            arrays["skeleton"][offset + local_index] = raw
            arrays["people"][offset + local_index] = min(person_count, 255)
        arrays["time"][offset : offset + expected_length] = relative_times(common_ids)
        if sample_index % 300 == 0 or sample_index == len(sample_ids):
            print(f"raw Skeleton {sample_index}/{len(sample_ids)}", flush=True)

    for array in arrays.values():
        array.flush()
    del array, arrays
    for key, source in temporary.items():
        source.replace(final_paths[key])
    metadata = {
        "version": 1,
        "manifest": str(manifest),
        "source_aligned_cache": str(aligned_cache),
        "num_samples": len(sample_ids),
        "total_frames": total_frames,
        "sample_ids": sample_ids,
        "offsets": offsets,
        "files": [path.name for path in final_paths.values()],
        "description": "first-person raw 3D keypoints + score; no root centering or scale normalization",
        "build_seconds": round(time.time() - started, 2),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: metadata[key] for key in ("num_samples", "total_frames", "build_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
