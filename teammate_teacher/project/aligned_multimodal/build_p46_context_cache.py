from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from p30_shared_dir_roi_model import MODALITY_NAMES, REGION_NAMES
from p46_event_data import CONTEXT_REGIONS, safe_path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_RUN = PROJECT_DIR / "runs" / "p46_event_inputs_full"
DEFAULT_P30_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_context_uncompressed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only P46 full-body/global P30 context into lossless uncompressed caches."
    )
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--p30-run", type=Path, default=DEFAULT_P30_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard arguments")
    event_run = args.event_run.resolve()
    p30_run = args.p30_run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(event_run / "trial_summary.csv")
    rows.sort(
        key=lambda row: (
            0 if row["p46_split"] == "train" else 1,
            int(row["class_id"]),
            row["user_id"],
            row["trial_id"],
        )
    )
    rows = rows[args.shard_index :: args.num_shards]
    region_indices = [REGION_NAMES.index(value) for value in CONTEXT_REGIONS]
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        source_id = row["source_id"]
        relative = safe_path(source_id).with_suffix(".npz")
        source = p30_run / "trial_feature_cache" / relative
        target = output / "trial_context_cache" / relative
        status = "cached"
        trial_started = time.perf_counter()
        if args.overwrite or not target.is_file():
            with np.load(source, allow_pickle=False) as cache:
                modalities = tuple(str(value) for value in cache["modality_names"])
                regions = tuple(str(value) for value in cache["region_names"])
                if modalities != MODALITY_NAMES or regions != REGION_NAMES:
                    raise RuntimeError(f"P30 contract changed: {source_id}")
                arrays = {
                    "frame_ids": np.asarray(cache["frame_ids"]),
                    "modality_names": np.asarray(MODALITY_NAMES),
                    "context_region_names": np.asarray(CONTEXT_REGIONS),
                    "context_features": np.asarray(
                        cache["features"][:, :, region_indices], dtype=np.float16
                    ),
                    "context_valid": np.asarray(
                        cache["roi_valid"][:, region_indices], dtype=bool
                    ),
                    "context_quality": np.asarray(
                        cache["roi_quality"][:, region_indices], dtype=np.float32
                    ),
                }
            atomic_npz(target, arrays)
            status = "built"
        with np.load(target, allow_pickle=False) as cache:
            frames = len(cache["frame_ids"])
            shape = list(cache["context_features"].shape)
        if frames != int(row["frames"]) or shape[1:] != [2, 2, 896]:
            raise RuntimeError(f"P46 context shape/frame mismatch: {source_id} {shape}")
        elapsed = time.perf_counter() - trial_started
        summaries.append(
            {
                "source_id": source_id,
                "p46_split": row["p46_split"],
                "frames": frames,
                "status": status,
                "seconds": elapsed,
                "bytes": target.stat().st_size,
            }
        )
        print(
            f"P46 context [{index}/{len(rows)}] {source_id} {status} "
            f"frames={frames} {elapsed:.2f}s",
            flush=True,
        )
    suffix = "" if args.num_shards == 1 else f".shard{args.shard_index}-of-{args.num_shards}"
    summary = {
        "stage": "P46_training_context_cache",
        "version": 1,
        "selected_regions": list(CONTEXT_REGIONS),
        "source_p30": str(p30_run),
        "trials": len(summaries),
        "frames": sum(value["frames"] for value in summaries),
        "bytes": sum(value["bytes"] for value in summaries),
        "built": sum(value["status"] == "built" for value in summaries),
        "npz_compressed": False,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / f"summary{suffix}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

