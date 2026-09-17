from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aligned_data import frame_map


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_UNION_DIR = PROJECT_DIR / "data" / "six_modality_audit"
FULL_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 Depth/IR/Skeleton/Thermal 的时间键、分辨率与空间映射前提")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_UNION_DIR / "train_union_manifest.csv")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_UNION_DIR / "test_union_manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "runs" / "p0_six_modality_audit" / "visual_alignment_audit.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def image_files(path: Path, modality: str) -> list[Path]:
    if modality == "depth_color":
        return sorted(path.glob("*.png"))
    if modality == "ir":
        return sorted(path.glob("*.png"))
    if modality == "thermal":
        return sorted(path.glob("*.jpg"))
    raise ValueError(modality)


def spatial_shape(path: Path, modality: str) -> str:
    files = image_files(path, modality)
    if not files:
        return ""
    try:
        with Image.open(files[len(files) // 2]) as image:
            return f"{image.width}x{image.height}:{image.mode}"
    except OSError:
        return "unreadable"


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "count": len(array),
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "p10": float(np.quantile(array, 0.1)) if len(array) else None,
        "p90": float(np.quantile(array, 0.9)) if len(array) else None,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def split_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    shape_counts: dict[str, Counter[str]] = {name: Counter() for name in ["depth_color", "ir", "thermal"]}
    frame_counts: dict[str, list[float]] = {name: [] for name in ["depth_color", "ir", "thermal", "skeleton"]}
    timestamp_name_counts: dict[str, list[int]] = {name: [] for name in ["depth_color", "ir", "thermal", "skeleton"]}
    aligned_rows = []
    thermal_depth_ratios = []

    for row in rows:
        for modality in ["depth_color", "ir", "thermal"]:
            if int(row[f"{modality}_usable"]) == 1:
                path = Path(row[f"{modality}_path"])
                files = image_files(path, modality)
                shape_counts[modality][spatial_shape(path, modality)] += 1
                frame_counts[modality].append(len(files))
                timestamp_name_counts[modality].extend(int(bool(FULL_TIMESTAMP.search(file.name))) for file in files)
        if int(row["skeleton_usable"]) == 1:
            skeleton_files = sorted(Path(row["skeleton_path"]).rglob("*.json"))
            frame_counts["skeleton"].append(len(skeleton_files))
            timestamp_name_counts["skeleton"].extend(int(bool(FULL_TIMESTAMP.search(file.name))) for file in skeleton_files)

        if int(row["depth_color_usable"]) and int(row["thermal_usable"]):
            depth_count = int(row["depth_color_file_count"])
            thermal_count = int(row["thermal_file_count"])
            if depth_count:
                thermal_depth_ratios.append(thermal_count / depth_count)

        if all(int(row[f"{name}_usable"]) for name in ["depth_color", "ir", "skeleton"]):
            depth = frame_map(Path(row["depth_color_path"]), "depth")
            ir = frame_map(Path(row["ir_path"]), "ir")
            skeleton = frame_map(Path(row["skeleton_path"]), "skeleton")
            sets = {"depth": set(depth), "ir": set(ir), "skeleton": set(skeleton)}
            common = set.intersection(*sets.values())
            union = set.union(*sets.values())
            aligned_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "depth_frames": len(depth),
                    "ir_frames": len(ir),
                    "skeleton_frames": len(skeleton),
                    "common_frames": len(common),
                    "union_frames": len(union),
                    "common_over_union": len(common) / len(union) if union else 0.0,
                    "depth_ir_exact": int(sets["depth"] == sets["ir"]),
                    "depth_skeleton_exact": int(sets["depth"] == sets["skeleton"]),
                    "all_three_exact": int(len(common) == len(union)),
                }
            )

    return {
        "trials": len(rows),
        "shapes_by_trial": {modality: dict(counts) for modality, counts in shape_counts.items()},
        "frame_counts": {modality: distribution(values) for modality, values in frame_counts.items()},
        "filename_has_absolute_timestamp_fraction": {
            modality: float(np.mean(values)) if values else None for modality, values in timestamp_name_counts.items()
        },
        "depth_ir_skeleton_alignment": {
            "eligible_trials": len(aligned_rows),
            "depth_ir_exact_frame_set_trials": int(sum(row["depth_ir_exact"] for row in aligned_rows)),
            "depth_skeleton_exact_frame_set_trials": int(sum(row["depth_skeleton_exact"] for row in aligned_rows)),
            "all_three_exact_frame_set_trials": int(sum(row["all_three_exact"] for row in aligned_rows)),
            "common_over_union": distribution([row["common_over_union"] for row in aligned_rows]),
            "worst_trials": sorted(aligned_rows, key=lambda row: row["common_over_union"])[:20],
        },
        "thermal_frames_per_depth_frame": distribution(thermal_depth_ratios),
    }


def main() -> None:
    args = parse_args()
    train = split_audit(read_csv(args.train_manifest.resolve()))
    test = split_audit(read_csv(args.test_manifest.resolve()))
    summary = {
        "train": train,
        "test": test,
        "conclusions": {
            "depth_ir": (
                "Depth and IR share 640x480 resolution and canonical frame IDs. Exact frame-set coverage is quantified above. "
                "This supports same-frame feature alignment, but pixel-level calibration should still be treated as near-alignment rather than assumed perfect geometry."
            ),
            "skeleton": (
                "Skeleton JSON uses the same canonical frame IDs as Depth/IR, so temporal alignment is direct. "
                "The stored 17x3 keypoints are not documented 2D camera pixels; direct visual ROI sampling is not currently justified."
            ),
            "thermal": (
                "Thermal is 320x240 and uses independent frame_NNNNNN names with a different frame count. "
                "There is no direct timestamp key or pixel mapping in the files; a Thermal bbox must not be copied to Depth/IR without calibration."
            ),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
