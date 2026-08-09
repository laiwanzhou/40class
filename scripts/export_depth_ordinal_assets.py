from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ordinal_depth import (  # noqa: E402
    crop_and_letterbox_ordinal,
    load_depth_color_ordinal,
)
from src.data.pose_roi_dataset import paired_frame_key, paired_frame_paths  # noqa: E402


DEFAULT_BASELINE_ROOT = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_full_inputs_256"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256"
)
DEFAULT_EFFECTIVE_AUDIT = (
    PROJECT_ROOT / "reports/roi640_stage2_temporal_identity/effective_view_audit.csv"
)
DEFAULT_SOURCE_DEPTH = (
    PROJECT_ROOT / "reports/roi640_stage2_temporal_identity/source_depth_jet_integrity.csv"
)
IR_VIEWS = ("ir_context", "ir_left", "ir_right", "ir_relation")
DEPTH_VIEWS = ("depth_context", "depth_relation")
ALL_VIEWS = (*IR_VIEWS, *DEPTH_VIEWS)
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ordinal Depth values/masks and a combined manifest reusing ROI640 IR files.",
    )
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--effective-audit", type=Path, default=DEFAULT_EFFECTIVE_AUDIT)
    parser.add_argument("--source-depth-audit", type=Path, default=DEFAULT_SOURCE_DEPTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--png-compress-level", type=int, default=3, choices=range(10))
    return parser.parse_args()


def content_metrics(values: np.ndarray) -> dict[str, float | int | bool]:
    image = np.asarray(values, dtype=np.uint8)
    thumb = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    histogram = np.bincount(thumb.reshape(-1), minlength=256).astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probability * np.log2(probability)).sum())
    dynamic_range = int(image.max()) - int(image.min())
    standard_deviation = float(image.std())
    low_information_tests = int(dynamic_range <= 8) + int(standard_deviation <= 2.0) + int(entropy <= 1.0)
    return {
        "dynamic_range": dynamic_range,
        "std": standard_deviation,
        "entropy_32": entropy,
        "content_invalid_2of3": bool(low_information_tests >= 2),
    }


def deterministic_reliability(effective_valid: bool, pose_score: float, pixel_coverage: float) -> float:
    normalized_pose = float(np.clip((float(pose_score) - 0.25) / 0.75, 0.0, 1.0))
    return float(bool(effective_valid) * normalized_pose * float(np.clip(pixel_coverage, 0.0, 1.0)))


def _load_tables(
    baseline_root: Path,
    effective_audit_path: Path,
    source_depth_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = pd.read_csv(baseline_root / "all_frame_inputs.csv", encoding="utf-8-sig")
    roi = pd.read_csv(
        baseline_root / "roi_frame_audit.csv",
        encoding="utf-8-sig",
        usecols=[
            "split", "class_id", "action_name", "sample_id", "user_id", "source_frame_index",
            "view_name", "valid", "source", "confidence", "x1", "y1", "x2", "y2",
        ],
    )
    effective = pd.read_csv(
        effective_audit_path,
        encoding="utf-8-sig",
        usecols=[
            "sample_id", "source_frame_index", "view_name", "valid_flag",
            "content_invalid", "effective_valid",
        ],
    )
    source = pd.read_csv(source_depth_path, encoding="utf-8-sig")
    if not source.readable.astype(bool).all() or int(source.unexpected_pixels.sum()) != 0:
        raise ValueError("Stage 2 source Depth audit is not clean")
    source["source_frame_index"] = source.groupby("sample_id", sort=False).cumcount()

    key = ["sample_id", "source_frame_index"]
    if frames.duplicated(key).any() or source.duplicated(key).any():
        raise ValueError("Frame manifest keys are not unique")
    if set(map(tuple, frames[key].to_numpy())) != set(map(tuple, source[key].to_numpy())):
        raise ValueError("Baseline and source Depth frame keys do not match")
    if roi.duplicated([*key, "view_name"]).any() or effective.duplicated([*key, "view_name"]).any():
        raise ValueError("View audit keys are not unique")
    return frames, roi, effective.merge(source[key + ["path"]], on=key, how="right", validate="many_to_one")


def _source_pairs(source_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sample_id, group in source_rows.groupby("sample_id", sort=False):
        ordered = group.sort_values("source_frame_index")
        depth_paths = tuple(Path(value) for value in ordered.path)
        depth_trial = depth_paths[0].parent
        parts = list(depth_trial.parts)
        try:
            modality_index = parts.index("Depth_Color")
        except ValueError as error:
            raise ValueError(f"Depth_Color segment absent from {depth_trial}") from error
        parts[modality_index] = "IR"
        paired_depth, paired_ir = paired_frame_paths(depth_trial, Path(*parts))
        if paired_depth != depth_paths:
            raise ValueError(f"Source Depth order differs from paired order for {sample_id}")
        previous: datetime | None = None
        for frame_index, (depth_path, ir_path) in enumerate(zip(paired_depth, paired_ir, strict=True)):
            timestamp, frame_id = paired_frame_key(depth_path, "Depth")
            instant = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
            delta = 0.0 if previous is None else (instant - previous).total_seconds()
            records.append({
                "sample_id": sample_id,
                "source_frame_index": frame_index,
                "source_depth_path": str(depth_path),
                "source_ir_path": str(ir_path),
                "timestamp": timestamp,
                "frame_id": frame_id,
                "inter_frame_delta_seconds": delta,
            })
            previous = instant
    return pd.DataFrame.from_records(records)


def _atomic_png(image: np.ndarray, path: Path, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]):
        raise OSError(f"Could not write {temporary}")
    os.replace(temporary, path)


def export_depth_ordinal_assets(
    *,
    baseline_root: Path,
    effective_audit_path: Path,
    source_depth_path: Path,
    output_root: Path,
    image_size: int = 256,
    limit_samples: int | None = None,
    sample_ids: set[str] | None = None,
    png_compress_level: int = 3,
) -> dict[str, object]:
    baseline_root = baseline_root.resolve()
    output_root = output_root.resolve()
    if (output_root / "_SUCCESS").exists():
        raise FileExistsError(f"Completed export already exists: {output_root}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    frames, roi, effective_source = _load_tables(
        baseline_root,
        effective_audit_path.resolve(),
        source_depth_path.resolve(),
    )
    if limit_samples is not None and sample_ids is not None:
        raise ValueError("limit_samples and sample_ids are mutually exclusive")
    selected_ids: set[str] | None = sample_ids
    if limit_samples is not None:
        selected_ids = set(frames.sample_id.drop_duplicates().iloc[:limit_samples].astype(str))
    if selected_ids is not None:
        missing_ids = selected_ids - set(frames.sample_id.astype(str))
        if missing_ids:
            raise ValueError(f"Requested sample IDs are absent: {sorted(missing_ids)}")
        frames = frames[frames.sample_id.isin(selected_ids)].copy()
        roi = roi[roi.sample_id.isin(selected_ids)].copy()
        effective_source = effective_source[effective_source.sample_id.isin(selected_ids)].copy()
    source_rows = effective_source[["sample_id", "source_frame_index", "path"]].drop_duplicates()
    source_pairs = _source_pairs(source_rows)
    keys = ["sample_id", "source_frame_index"]
    frames = frames.merge(source_pairs, on=keys, how="left", validate="one_to_one")
    if frames[["source_depth_path", "source_ir_path"]].isna().any().any():
        raise ValueError("Combined manifest has missing source paths")

    roi_lookup = roi.set_index([*keys, "view_name"])
    effective_lookup = effective_source.set_index([*keys, "view_name"])
    manifest_partial = output_root / "combined_frame_manifest.csv.partial"
    depth_audit_partial = output_root / "depth_ordinal_view_audit.csv.partial"
    started = time.perf_counter()
    manifest_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for row in frames.itertuples(index=False):
        decoded = load_depth_color_ordinal(row.source_depth_path)
        base = {
            "split": row.split,
            "class_id": int(row.class_id),
            "action_name": row.action_name,
            "sample_id": row.sample_id,
            "user_id": row.user_id,
            "source_frame_index": int(row.source_frame_index),
            "timestamp": row.timestamp,
            "frame_id": int(row.frame_id),
            "inter_frame_delta_seconds": float(row.inter_frame_delta_seconds),
            "source_ir_path": row.source_ir_path,
            "source_depth_path": row.source_depth_path,
            "temporal_valid": 1,
        }
        for view in IR_VIEWS:
            baseline_path = baseline_root / getattr(row, f"{view}_path")
            evidence = effective_lookup.loc[(row.sample_id, int(row.source_frame_index), view)]
            pose_valid = bool(evidence.valid_flag)
            content_invalid = bool(evidence.content_invalid)
            effective_valid = bool(evidence.effective_valid)
            pose_score = float(getattr(row, f"{view}_confidence"))
            base.update({
                f"{view}_path": str(baseline_path.resolve()),
                f"{view}_pose_valid": int(pose_valid),
                f"{view}_content_valid": int(not content_invalid),
                f"{view}_effective_valid": int(effective_valid),
                f"{view}_reliability": deterministic_reliability(effective_valid, pose_score, 1.0),
            })

        for view in DEPTH_VIEWS:
            evidence = roi_lookup.loc[(row.sample_id, int(row.source_frame_index), view)]
            old_effective = effective_lookup.loc[(row.sample_id, int(row.source_frame_index), view)]
            pose_valid = bool(evidence.valid)
            value_path = (
                output_root / row.split / f"c{int(row.class_id):02d}" / row.sample_id
                / f"{view}_ordinal" / f"f{int(row.source_frame_index):04d}.png"
            )
            mask_path = value_path.parent.parent / f"{view}_pixel_valid" / value_path.name
            if pose_valid:
                values, pixel_valid = crop_and_letterbox_ordinal(
                    decoded.values,
                    decoded.pixel_valid,
                    np.asarray([evidence.x1, evidence.y1, evidence.x2, evidence.y2]),
                    (image_size, image_size),
                )
            else:
                values = np.zeros((image_size, image_size), dtype=np.uint8)
                pixel_valid = np.zeros((image_size, image_size), dtype=bool)
            metrics = content_metrics(values)
            retained_invalid = bool(old_effective.content_invalid)
            content_invalid = bool(pose_valid and (metrics["content_invalid_2of3"] or retained_invalid))
            effective_valid = bool(pose_valid and not content_invalid)
            pixel_coverage = float(pixel_valid.mean())
            pose_score = 1.0 if view == "depth_context" else float(row.ir_relation_confidence)
            reliability = deterministic_reliability(effective_valid, pose_score, pixel_coverage)
            _atomic_png(values, value_path, png_compress_level)
            _atomic_png(pixel_valid.astype(np.uint8) * 255, mask_path, png_compress_level)
            base.update({
                f"{view}_ordinal_path": str(value_path),
                f"{view}_pixel_valid_path": str(mask_path),
                f"{view}_pose_valid": int(pose_valid),
                f"{view}_content_valid": int(not content_invalid),
                f"{view}_effective_valid": int(effective_valid),
                f"{view}_pixel_coverage": pixel_coverage,
                f"{view}_reliability": reliability,
            })
            audit_rows.append({
                **{key: base[key] for key in (
                    "split", "class_id", "action_name", "sample_id", "user_id", "source_frame_index",
                )},
                "view_name": view,
                "pose_valid": int(pose_valid),
                "baseline_content_invalid_retained": int(retained_invalid),
                "ordinal_content_invalid_2of3": int(metrics["content_invalid_2of3"]),
                "content_invalid": int(content_invalid),
                "effective_valid": int(effective_valid),
                "pixel_coverage": pixel_coverage,
                "reliability": reliability,
                **metrics,
                "ordinal_path": str(value_path),
                "pixel_valid_path": str(mask_path),
            })
        manifest_rows.append(base)

    pd.DataFrame(manifest_rows).to_csv(manifest_partial, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(depth_audit_partial, index=False, encoding="utf-8-sig")
    os.replace(manifest_partial, output_root / "combined_frame_manifest.csv")
    os.replace(depth_audit_partial, output_root / "depth_ordinal_view_audit.csv")
    metadata = {
        "status": "complete",
        "baseline_ir_root": str(baseline_root),
        "depth_ordinal_root": str(output_root),
        "image_size": image_size,
        "samples": int(frames.sample_id.nunique()),
        "frames": len(frames),
        "ir_images_duplicated": False,
        "depth_value_images": len(frames) * len(DEPTH_VIEWS),
        "depth_mask_images": len(frames) * len(DEPTH_VIEWS),
        "all_selected_sample_frames_preserved": True,
        "full_dataset_export": limit_samples is None and sample_ids is None,
        "temporal_sampling_applied": False,
        "competition_test_read": False,
        "training_run": False,
        "seconds": round(time.perf_counter() - started, 3),
    }
    (output_root / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_root / "_SUCCESS").write_text("complete\n", encoding="ascii")
    return metadata


def main() -> None:
    args = parse_args()
    sample_ids = None
    if args.sample_ids_file is not None:
        selection = pd.read_csv(args.sample_ids_file, encoding="utf-8-sig")
        if "sample_id" not in selection:
            raise ValueError("sample_ids_file must contain a sample_id column")
        sample_ids = set(selection.sample_id.astype(str))
    result = export_depth_ordinal_assets(
        baseline_root=args.baseline_root,
        effective_audit_path=args.effective_audit,
        source_depth_path=args.source_depth_audit,
        output_root=args.output,
        image_size=args.image_size,
        limit_samples=args.limit_samples,
        sample_ids=sample_ids,
        png_compress_level=args.png_compress_level,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
