from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_yolo11_pose_skeleton import atomic_json
from build_multiscale_dir_rois import REGION_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_POSE_RUN = PROJECT_DIR / "runs" / "p28_adaptive_ir_pose_skeleton_full"
DEFAULT_ROI_RUN = PROJECT_DIR / "runs" / "p29_dir_multiscale_roi_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify every Step-07 D/IR ROI cache.")
    parser.add_argument("--pose-run", type=Path, default=DEFAULT_POSE_RUN)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI_RUN)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    pose_run = args.pose_run.resolve()
    roi_run = args.roi_run.resolve()
    roi_root = roi_run / "trial_roi_cache"
    paths = sorted(roi_root.rglob("*.npz"))
    require(bool(paths), f"No ROI caches under {roi_root}")

    with (roi_run / "summary.json").open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    with (roi_run / "trial_summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    source_counts: Counter[int] = Counter()
    quality_factor_counts: Counter[str] = Counter()
    total_frames = 0
    valid_boxes = 0
    invalid_boxes = 0
    finite_invalid_boxes = 0
    clipped_boxes = 0
    errors: list[str] = []

    for index, roi_path in enumerate(paths, start=1):
        relative = roi_path.relative_to(roi_root)
        pose_path = pose_run / "trial_cache" / relative
        try:
            require(pose_path.is_file(), f"missing pose cache: {relative}")
            with np.load(roi_path, allow_pickle=False) as roi, np.load(
                pose_path, allow_pickle=False
            ) as pose:
                frame_ids = roi["frame_ids"]
                boxes = roi["roi_boxes_xyxy"]
                valid = roi["roi_valid"]
                quality = roi["roi_quality"]
                source = roi["roi_source"]
                clipped = roi["roi_clipped_ratio"]
                factor = roi["pose_quality_factor"]
                region_names = tuple(str(value) for value in roi["region_names"])
                frame_count = len(frame_ids)

                require(region_names == REGION_NAMES, f"region order: {relative}")
                require(np.array_equal(frame_ids, pose["frame_ids"]), f"frame ids: {relative}")
                require(boxes.shape == (frame_count, len(REGION_NAMES), 4), f"boxes: {relative}")
                for name, array in (
                    ("valid", valid),
                    ("quality", quality),
                    ("source", source),
                    ("clipped", clipped),
                ):
                    require(array.shape == (frame_count, len(REGION_NAMES)), f"{name}: {relative}")
                require(factor.shape == (frame_count,), f"quality factor: {relative}")
                require(np.isfinite(quality).all(), f"non-finite quality: {relative}")
                require(((quality >= 0.0) & (quality <= 1.0)).all(), f"quality range: {relative}")
                require(np.isfinite(factor).all(), f"non-finite factor: {relative}")
                require(((factor >= 0.0) & (factor <= 1.0)).all(), f"factor range: {relative}")
                require(((source >= 0) & (source <= 6)).all(), f"source range: {relative}")

                valid_values = boxes[valid]
                require(np.isfinite(valid_values).all(), f"non-finite valid box: {relative}")
                require((valid_values[:, 2] > valid_values[:, 0]).all(), f"invalid x extent: {relative}")
                require((valid_values[:, 3] > valid_values[:, 1]).all(), f"invalid y extent: {relative}")
                require((valid_values[:, 0] >= 0.0).all(), f"x0 out of bounds: {relative}")
                require((valid_values[:, 1] >= 0.0).all(), f"y0 out of bounds: {relative}")
                require((valid_values[:, 2] <= 639.0).all(), f"x1 out of bounds: {relative}")
                require((valid_values[:, 3] <= 479.0).all(), f"y1 out of bounds: {relative}")
                require((quality[~valid] == 0.0).all(), f"invalid ROI has quality: {relative}")
                require((source[~valid] == 0).all(), f"invalid ROI has source: {relative}")

                global_index = REGION_NAMES.index("global_fallback")
                expected_global = np.asarray((0.0, 0.0, 639.0, 479.0), dtype=np.float32)
                require(valid[:, global_index].all(), f"global validity: {relative}")
                require(
                    np.array_equal(boxes[:, global_index], np.broadcast_to(expected_global, (frame_count, 4))),
                    f"global geometry: {relative}",
                )
                require((source[:, global_index] == 6).all(), f"global source: {relative}")
                require((quality[:, global_index] == 1.0).all(), f"global quality: {relative}")

                total_frames += frame_count
                valid_boxes += int(valid.sum())
                invalid_boxes += int((~valid).sum())
                finite_invalid_boxes += int(np.isfinite(boxes[~valid]).all(axis=1).sum())
                clipped_boxes += int((clipped >= 0.25).sum())
                source_counts.update(int(value) for value in source.ravel())
                for value in factor:
                    quality_factor_counts[f"{float(value):.2f}"] += 1
        except Exception as exc:  # collect several failures in one audit pass
            errors.append(f"{relative.as_posix()}: {exc}")
        if index % 500 == 0:
            print(f"verified {index}/{len(paths)}")

    require(not errors, "\n".join(errors[:20]))
    require(len(paths) == int(run_summary["completed_trials"]), "trial count mismatch")
    require(len(summary_rows) == len(paths), "trial summary count mismatch")
    require(total_frames == int(run_summary["completed_frames"]), "frame count mismatch")

    report = {
        "status": "pass",
        "verified_trials": len(paths),
        "verified_frames": total_frames,
        "verified_region_boxes": valid_boxes + invalid_boxes,
        "valid_region_boxes": valid_boxes,
        "invalid_region_boxes": invalid_boxes,
        "finite_invalid_region_boxes": finite_invalid_boxes,
        "boxes_clipped_ge_025": clipped_boxes,
        "roi_source_counts": {str(key): value for key, value in sorted(source_counts.items())},
        "pose_quality_factor_frame_counts": dict(sorted(quality_factor_counts.items())),
        "checks": [
            "ROI/P28 frame IDs are exactly equal for every trial",
            "all arrays have the declared T x 7 geometry",
            "every valid XYXY box is finite, positive-area, and inside 640 x 480",
            "invalid regions carry zero quality and source code 0",
            "global fallback is exactly [0, 0, 639, 479] on every frame",
            "quality values and P28 reliability factors are finite and bounded",
        ],
    }
    atomic_json(roi_run / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
