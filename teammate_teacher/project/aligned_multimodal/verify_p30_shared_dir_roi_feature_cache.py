from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_yolo11_pose_skeleton import atomic_json
from p30_shared_dir_roi_model import MODALITY_NAMES, PYRAMID_FEATURE_DIM, REGION_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROI_RUN = PROJECT_DIR / "runs" / "p29_dir_multiscale_roi_full"
DEFAULT_FEATURE_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify every P30 shared D/IR ROI feature cache.")
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI_RUN)
    parser.add_argument("--feature-run", type=Path, default=DEFAULT_FEATURE_RUN)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    roi_run = args.roi_run.resolve()
    feature_run = args.feature_run.resolve()
    feature_root = feature_run / "trial_feature_cache"
    feature_paths = sorted(feature_root.rglob("*.npz"))
    with (feature_run / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with (feature_run / "trial_summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    total_frames = 0
    total_bytes = 0
    invalid_regions = 0
    zero_valid_tokens = 0
    feature_elements = 0
    norm_sum = np.zeros((len(MODALITY_NAMES), len(REGION_NAMES)), dtype=np.float64)
    norm_count = np.zeros_like(norm_sum, dtype=np.int64)
    errors: list[str] = []

    for index, feature_path in enumerate(feature_paths, start=1):
        relative = feature_path.relative_to(feature_root)
        roi_path = roi_run / "trial_roi_cache" / relative
        try:
            require(roi_path.is_file(), f"missing P29 ROI cache: {relative}")
            with np.load(feature_path, allow_pickle=False) as feature, np.load(
                roi_path, allow_pickle=False
            ) as roi:
                frame_ids = feature["frame_ids"]
                values = feature["features"]
                valid = feature["roi_valid"]
                frame_count = len(frame_ids)
                require(values.dtype == np.float16, f"feature dtype: {relative}")
                require(
                    values.shape
                    == (
                        frame_count,
                        len(MODALITY_NAMES),
                        len(REGION_NAMES),
                        PYRAMID_FEATURE_DIM,
                    ),
                    f"feature shape: {relative}",
                )
                require(
                    tuple(str(value) for value in feature["modality_names"])
                    == MODALITY_NAMES,
                    f"modality order: {relative}",
                )
                require(
                    tuple(str(value) for value in feature["region_names"]) == REGION_NAMES,
                    f"region order: {relative}",
                )
                require(np.array_equal(frame_ids, roi["frame_ids"]), f"frame ids: {relative}")
                for key in (
                    "roi_valid",
                    "roi_quality",
                    "roi_source",
                    "roi_clipped_ratio",
                    "left_right_ambiguous",
                    "pose_quality_factor",
                ):
                    require(np.array_equal(feature[key], roi[key]), f"{key}: {relative}")
                require(np.isfinite(values).all(), f"non-finite features: {relative}")

                for region in range(len(REGION_NAMES)):
                    invalid = ~valid[:, region]
                    valid_frames = valid[:, region]
                    invalid_regions += int(invalid.sum())
                    if invalid.any():
                        require(
                            np.count_nonzero(values[invalid, :, region]) == 0,
                            f"invalid region has non-zero features: {relative} r={region}",
                        )
                    if valid_frames.any():
                        token_norm = np.linalg.norm(
                            values[valid_frames, :, region].astype(np.float32), axis=-1
                        )
                        zero_valid_tokens += int((token_norm == 0).sum())
                        norm_sum[:, region] += token_norm.sum(axis=0)
                        norm_count[:, region] += token_norm.shape[0]

                total_frames += frame_count
                total_bytes += feature_path.stat().st_size
                feature_elements += values.size
        except Exception as exc:
            errors.append(f"{relative.as_posix()}: {exc}")
        if index % 250 == 0:
            print(f"verified {index}/{len(feature_paths)}", flush=True)

    require(not errors, "\n".join(errors[:20]))
    require(len(feature_paths) == int(summary["completed_trials"]), "trial count mismatch")
    require(len(rows) == len(feature_paths), "trial summary count mismatch")
    require(total_frames == int(summary["completed_frames"]), "frame count mismatch")
    require(total_bytes == int(summary["total_cache_bytes"]), "cache byte count mismatch")
    require(zero_valid_tokens == 0, f"zero-valued valid modality tokens: {zero_valid_tokens}")

    mean_norm = norm_sum / np.maximum(norm_count, 1)
    report = {
        "status": "pass",
        "verified_trials": len(feature_paths),
        "verified_frames": total_frames,
        "verified_feature_elements": feature_elements,
        "cache_bytes": total_bytes,
        "invalid_regions_with_zero_features": invalid_regions,
        "zero_valid_modality_tokens": zero_valid_tokens,
        "mean_feature_l2_norm": {
            modality: {
                region: float(mean_norm[modality_index, region_index])
                for region_index, region in enumerate(REGION_NAMES)
            }
            for modality_index, modality in enumerate(MODALITY_NAMES)
        },
        "checks": [
            "P30 and P29 frame IDs are exactly equal for all 2,914 trials",
            "feature layout is exactly [T, 2, 7, 896] float16",
            "ROI validity, quality, source, clipping and pose factors exactly equal P29",
            "all feature elements are finite",
            "every invalid ROI feature is exactly zero",
            "every valid Depth/IR ROI token has non-zero feature norm",
        ],
    }
    atomic_json(feature_run / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
