from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from build_local_depth_cache import build_memmap, box


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_LOCATOR = (
    PROJECT_DIR
    / "runs"
    / "p16_oracle_assisted_locator_predictions"
    / "fold_0_locator_predictions.csv"
)
DEFAULT_BASE_CACHE = PROJECT_DIR / "runs" / "p12_local_depth_cache"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p16_oracle_local_depth_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the standard Local Depth cache for the oracle-assisted "
            "locator while hard-linking unchanged non-fallback data."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--locator", type=Path, default=DEFAULT_LOCATOR)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def hardlink(source: Path, target: Path) -> None:
    if target.exists():
        return
    os.link(source, target)


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.manifest.resolve())
    manifest_rows.sort(key=lambda row: row["sample_id"])
    predictions = {
        row["sample_id"]: row for row in read_csv(args.locator.resolve())
    }
    if len(manifest_rows) != 2914 or set(predictions) != {
        row["sample_id"] for row in manifest_rows
    }:
        raise ValueError("Manifest/locator must cover the same 2914 samples")
    fallback_rows = [
        row
        for row in manifest_rows
        if int(predictions[row["sample_id"]]["motion_fallback"]) == 1
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cache = args.base_cache.resolve()
    for name in ("nonfallback_uint8.npy", "nonfallback_sample_ids.npy"):
        hardlink(base_cache / name, output_dir / name)

    fallback_items = [
        (row, box(predictions[row["sample_id"]])) for row in fallback_rows
    ]
    fallback_shared = output_dir / "fallback_oracle_all286_uint8.npy"
    build_memmap(
        fallback_shared,
        fallback_items,
        int(args.workers),
        "Oracle-assisted fallback cache",
    )
    for held_fold in range(3):
        hardlink(
            fallback_shared,
            output_dir / f"fallback_fold_{held_fold}_uint8.npy",
        )
    fallback_ids = output_dir / "fallback_sample_ids.npy"
    if not fallback_ids.exists():
        import numpy as np

        np.save(
            fallback_ids,
            np.asarray([row["sample_id"] for row in fallback_rows]),
        )
    summary = {
        "status": "exploratory_oracle_assisted",
        "deployable_oof": False,
        "samples": len(manifest_rows),
        "nonfallback": len(manifest_rows) - len(fallback_rows),
        "fallback": len(fallback_rows),
        "shape_per_sample": [12, 144, 192, 3],
        "crop": (
            "Original 640x480 Depth -> all286 hybrid ROI -> 15% context "
            "per side -> 4:3 -> resize 192x144"
        ),
        "wide_local": False,
        "storage": (
            "Unchanged non-fallback cache and the three identical fallback "
            "views are hard-linked to avoid duplicate disk use."
        ),
        "warning": (
            "All available human ROI supervision was used. Downstream metrics "
            "must be labelled exploratory/oracle-assisted."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
