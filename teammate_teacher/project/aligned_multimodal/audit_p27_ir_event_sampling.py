from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aligned_data import AlignedMultimodalDataset


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit label-free IR event sampling on outer-train clips."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def make_dataset(manifest: Path, split: str, mode: str) -> AlignedMultimodalDataset:
    return AlignedMultimodalDataset(
        manifest_path=manifest,
        split=split,
        modalities=["ir"],
        num_frames=16,
        image_height=144,
        image_width=192,
        augment=False,
        cache_dir="cache/aligned_192x144",
        visual_normalization="imagenet",
        temporal_sampling=mode,
    )


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    datasets = {
        mode: make_dataset(manifest, args.split, mode)
        for mode in (
            "uniform",
            "uniform_plus_ir_motion_peak",
            "uniform_plus_ir_motion_pairs",
        )
    }
    reference = datasets["uniform"]
    ir_cache = reference._cache_array("ir")
    rows: dict[str, list[dict[str, float]]] = {mode: [] for mode in datasets}
    for sample in reference.samples[: args.limit]:
        offset, length = reference.cache_locations[sample.sample_id]
        sparse = np.asarray(
            ir_cache[offset : offset + length, ::8, ::8], dtype=np.float32
        )
        full_differences = np.abs(np.diff(sparse, axis=0)).mean(axis=(1, 2))
        for mode, dataset in datasets.items():
            if mode == "uniform":
                positions = dataset._sample_positions(length)
            elif mode == "uniform_plus_ir_motion_peak":
                positions = dataset._uniform_plus_ir_motion_peak_positions(
                    ir_cache, offset, length
                )
            else:
                positions = dataset._uniform_plus_ir_motion_pair_positions(
                    ir_cache, offset, length
                )
            positions_array = np.asarray(positions, dtype=np.int64)
            consecutive = positions_array[1:] - positions_array[:-1] == 1
            consecutive_before = positions_array[:-1][consecutive]
            local_energy = (
                float(full_differences[consecutive_before].mean())
                if consecutive_before.size
                else 0.0
            )
            rows[mode].append(
                {
                    "unique_frames": float(np.unique(positions_array).size),
                    "adjacent_pairs": float(consecutive.sum()),
                    "adjacent_pair_energy": local_energy,
                    "maximum_gap": float(np.diff(positions_array).max()),
                }
            )

    summary = {
        "protocol": "outer-train only; label-free sampling audit",
        "manifest": str(manifest),
        "split": args.split,
        "samples": min(args.limit, len(reference.samples)),
        "modes": {
            mode: {
                key: float(np.mean([row[key] for row in values]))
                for key in values[0]
            }
            for mode, values in rows.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
