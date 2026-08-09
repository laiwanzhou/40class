from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_depth_ordinal_assets import content_metrics  # noqa: E402
from src.data.ordinal_depth import crop_and_letterbox_ordinal, load_depth_color_ordinal  # noqa: E402
from src.data.pose_roi_dataset import paired_frame_key  # noqa: E402


DEFAULT_EXPORT = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256_smoke_stage5"
)
DEFAULT_BASELINE = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_full_inputs_256"
)
DEFAULT_STAGE2 = PROJECT_ROOT / "reports/roi640_stage2_temporal_identity/effective_view_audit.csv"
DEFAULT_JSON = PROJECT_ROOT / "reports/depth_ordinal_stage5_audit.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports/depth_ordinal_stage5_audit.md"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"
DEPTH_VIEWS = ("depth_context", "depth_relation")
IR_VIEWS = ("ir_context", "ir_left", "ir_right", "ir_relation")
PREVIEW_CLASS_IDS = (1, 2, 7, 11, 17, 18, 20, 22, 24, 37)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--stage2-effective-audit", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--expected-samples", type=int, default=82)
    parser.add_argument("--expected-frames", type=int, default=3562)
    parser.add_argument("--expect-full-export", action="store_true")
    return parser.parse_args()


def _tile(image: np.ndarray, label: str, size: int = 180) -> np.ndarray:
    if image.ndim == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display = image.copy()
    display = cv2.resize(display, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(display, (0, 0), (size, 24), (0, 0, 0), -1)
    cv2.putText(display, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return display


def _write_previews(manifest: pd.DataFrame, export_root: Path) -> int:
    known = manifest[manifest.sample_id.isin({
        "train__c36__user1__2-1-2",
        "train__c36__user8__2-2-1",
    })].sample_id.drop_duplicates().tolist()
    selected = list(known)
    for class_id in PREVIEW_CLASS_IDS:
        candidate = manifest[manifest.class_id == class_id].sample_id.drop_duplicates()
        if len(candidate):
            selected.append(str(candidate.iloc[0]))
    selected = list(dict.fromkeys(selected))
    review = export_root / "_stage5_review"
    review.mkdir(exist_ok=True)
    for sample_id in selected:
        group = manifest[manifest.sample_id == sample_id].sort_values("source_frame_index")
        center = len(group) // 2
        positions = sorted(set(np.clip([center - 1, center, center + 1], 0, len(group) - 1)))
        columns: list[np.ndarray] = []
        for position in positions:
            row = group.iloc[int(position)]
            source = cv2.imread(str(row.source_depth_path), cv2.IMREAD_COLOR)
            panels = [_tile(source, f"f{int(row.source_frame_index)} source JET")]
            for view in DEPTH_VIEWS:
                value = cv2.imread(str(row[f"{view}_ordinal_path"]), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(str(row[f"{view}_pixel_valid_path"]), cv2.IMREAD_UNCHANGED)
                panels.append(_tile(value, f"{view} ordinal"))
                panels.append(_tile(mask, f"{view} mask"))
            columns.append(np.vstack(panels))
        sheet = np.hstack(columns)
        cv2.imwrite(str(review / f"{sample_id}.png"), sheet)
    return len(selected)


def main() -> None:
    args = parse_args()
    export_root = args.export_root.resolve()
    metadata = json.loads((export_root / "export_metadata.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(export_root / "combined_frame_manifest.csv", encoding="utf-8-sig")
    exported_audit = pd.read_csv(export_root / "depth_ordinal_view_audit.csv", encoding="utf-8-sig")
    roi = pd.read_csv(
        args.baseline_root / "roi_frame_audit.csv",
        encoding="utf-8-sig",
        usecols=["sample_id", "source_frame_index", "view_name", "valid", "x1", "y1", "x2", "y2"],
    )
    roi = roi[roi.view_name.isin(DEPTH_VIEWS)].set_index(["sample_id", "source_frame_index", "view_name"])
    stage2 = pd.read_csv(
        args.stage2_effective_audit,
        encoding="utf-8-sig",
        usecols=["sample_id", "source_frame_index", "view_name", "content_invalid"],
    )
    stage2 = stage2[stage2.view_name.isin(DEPTH_VIEWS)]
    selected_ids = set(manifest.sample_id.astype(str))
    stage2 = stage2[stage2.sample_id.isin(selected_ids)]

    missing_files = wrong_shape = wrong_dtype = mask_nonbinary = 0
    invalid_nonzero = value_mismatches = mask_mismatches = content_mismatches = 0
    timestamp_mismatches = delta_mismatches = 0
    ir_missing = ir_inside_export = 0
    previous_time: dict[str, datetime] = {}
    previous_index: dict[str, int] = {}
    audit_lookup = exported_audit.set_index(["sample_id", "source_frame_index", "view_name"])

    for row in manifest.sort_values(["sample_id", "source_frame_index"]).itertuples(index=False):
        timestamp, frame_id = paired_frame_key(Path(row.source_depth_path), "Depth")
        instant = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        expected_delta = 0.0 if row.sample_id not in previous_time else (instant - previous_time[row.sample_id]).total_seconds()
        timestamp_mismatches += int(timestamp != row.timestamp or int(frame_id) != int(row.frame_id))
        delta_mismatches += int(abs(expected_delta - float(row.inter_frame_delta_seconds)) > 1e-9)
        if row.sample_id in previous_index and int(row.source_frame_index) != previous_index[row.sample_id] + 1:
            delta_mismatches += 1
        previous_time[row.sample_id] = instant
        previous_index[row.sample_id] = int(row.source_frame_index)
        for view in IR_VIEWS:
            path = Path(getattr(row, f"{view}_path"))
            ir_missing += int(not path.is_file())
            ir_inside_export += int(export_root in path.parents)

        decoded = load_depth_color_ordinal(row.source_depth_path)
        for view in DEPTH_VIEWS:
            value_path = Path(getattr(row, f"{view}_ordinal_path"))
            mask_path = Path(getattr(row, f"{view}_pixel_valid_path"))
            if not value_path.is_file() or not mask_path.is_file():
                missing_files += int(not value_path.is_file()) + int(not mask_path.is_file())
                continue
            value = cv2.imread(str(value_path), cv2.IMREAD_UNCHANGED)
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if value is None or mask_image is None:
                missing_files += int(value is None) + int(mask_image is None)
                continue
            wrong_shape += int(value.shape != (256, 256)) + int(mask_image.shape != (256, 256))
            wrong_dtype += int(value.dtype != np.uint8) + int(mask_image.dtype != np.uint8)
            mask_nonbinary += int(not set(np.unique(mask_image)).issubset({0, 255}))
            stored_valid = mask_image > 0
            invalid_nonzero += int(np.count_nonzero(value[~stored_valid]))
            evidence = roi.loc[(row.sample_id, int(row.source_frame_index), view)]
            if bool(evidence.valid):
                expected_value, expected_valid = crop_and_letterbox_ordinal(
                    decoded.values,
                    decoded.pixel_valid,
                    np.asarray([evidence.x1, evidence.y1, evidence.x2, evidence.y2]),
                    (256, 256),
                )
            else:
                expected_value = np.zeros((256, 256), dtype=np.uint8)
                expected_valid = np.zeros((256, 256), dtype=bool)
            value_mismatches += int(np.count_nonzero(value != expected_value))
            mask_mismatches += int(np.count_nonzero(stored_valid != expected_valid))
            metrics = content_metrics(value)
            recorded = audit_lookup.loc[(row.sample_id, int(row.source_frame_index), view)]
            content_mismatches += int(bool(metrics["content_invalid_2of3"]) != bool(recorded.ordinal_content_invalid_2of3))

    retained = stage2[stage2.content_invalid.astype(bool)]
    retained_failures = 0
    for row in retained.itertuples(index=False):
        value = audit_lookup.loc[(row.sample_id, int(row.source_frame_index), row.view_name)]
        retained_failures += int(not bool(value.content_invalid) or bool(value.effective_valid))

    duplicate_frame_keys = int(manifest.duplicated(["sample_id", "source_frame_index"]).sum())
    frame_order_failures = int(sum(
        not np.array_equal(group.source_frame_index.to_numpy(), np.arange(len(group)))
        for _, group in manifest.sort_values("source_frame_index").groupby("sample_id")
    ))
    depth_png_count = sum(1 for path in export_root.rglob("*.png") if "_stage5_review" not in path.parts)
    expected_depth_pngs = len(manifest) * 4
    preview_count = _write_previews(manifest, export_root)
    checks = {
        "success_marker": (export_root / "_SUCCESS").is_file(),
        "sample_count_expected": manifest.sample_id.nunique() == args.expected_samples,
        "frame_count_expected": len(manifest) == args.expected_frames,
        "all_40_classes": manifest.class_id.nunique() == 40,
        "both_splits": set(manifest.split) == {"train", "val"},
        "metadata_frame_count": int(metadata["frames"]) == len(manifest),
        "full_dataset_export_flag": bool(metadata.get("full_dataset_export")) == args.expect_full_export,
        "duplicate_frame_keys_zero": duplicate_frame_keys == 0,
        "frame_order_failures_zero": frame_order_failures == 0,
        "depth_png_count_exact": depth_png_count == expected_depth_pngs,
        "missing_files_zero": missing_files == 0,
        "wrong_shape_zero": wrong_shape == 0,
        "wrong_dtype_zero": wrong_dtype == 0,
        "mask_nonbinary_zero": mask_nonbinary == 0,
        "invalid_nonzero_zero": invalid_nonzero == 0,
        "value_mismatches_zero": value_mismatches == 0,
        "mask_mismatches_zero": mask_mismatches == 0,
        "content_mismatches_zero": content_mismatches == 0,
        "timestamps_and_deltas_match": timestamp_mismatches == 0 and delta_mismatches == 0,
        "ir_missing_zero": ir_missing == 0,
        "ir_not_duplicated": ir_inside_export == 0,
        "stage2_depth_invalid_retained": retained_failures == 0 and len(retained) == 2,
    }
    result = {
        "stage": args.stage,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "values": {
            "samples": int(manifest.sample_id.nunique()),
            "frames": len(manifest),
            "classes": int(manifest.class_id.nunique()),
            "train_frames": int((manifest.split == "train").sum()),
            "val_frames": int((manifest.split == "val").sum()),
            "depth_pngs": depth_png_count,
            "expected_depth_pngs": expected_depth_pngs,
            "value_mismatched_pixels": value_mismatches,
            "mask_mismatched_pixels": mask_mismatches,
            "invalid_nonzero_pixels": invalid_nonzero,
            "known_depth_content_invalid_retained": len(retained),
            "ordinal_low_information_flags_total": int(exported_audit.ordinal_content_invalid_2of3.astype(bool).sum()),
            "pose_valid_ordinal_content_invalid_views": int(
                (exported_audit.pose_valid.astype(bool) & exported_audit.ordinal_content_invalid_2of3.astype(bool)).sum()
            ),
            "pose_invalid_placeholder_low_information_views": int(
                (~exported_audit.pose_valid.astype(bool) & exported_audit.ordinal_content_invalid_2of3.astype(bool)).sum()
            ),
            "final_content_invalid_views": int(exported_audit.content_invalid.astype(bool).sum()),
            "preview_contact_sheets": preview_count,
            "depth_context_pixel_coverage_mean": float(exported_audit.loc[exported_audit.view_name == "depth_context", "pixel_coverage"].mean()),
            "depth_relation_pixel_coverage_mean": float(exported_audit.loc[exported_audit.view_name == "depth_relation", "pixel_coverage"].mean()),
        },
        "competition_test_read": False,
        "training_run": False,
        "full_export_run": bool(metadata.get("full_dataset_export")),
    }
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        f"# Stage {args.stage}: ordinal Depth {'full' if args.expect_full_export else 'smoke'} export audit",
        "",
        f"- Status: **{result['status']}**",
        f"- Samples / frames / classes: {result['values']['samples']} / {result['values']['frames']:,} / {result['values']['classes']}",
        f"- Train / val frames: {result['values']['train_frames']:,} / {result['values']['val_frames']:,}",
        f"- Depth PNGs: {depth_png_count:,} (expected {expected_depth_pngs:,})",
        f"- Recomputed ordinal value mismatches: {value_mismatches:,} pixels",
        f"- Recomputed pixel-mask mismatches: {mask_mismatches:,} pixels",
        f"- Nonzero values behind invalid masks: {invalid_nonzero:,} pixels",
        f"- Raw scalar low-information flags: {result['values']['ordinal_low_information_flags_total']:,}",
        f"- Pose-valid scalar content-invalid views: {result['values']['pose_valid_ordinal_content_invalid_views']:,}",
        f"- Pose-invalid zero placeholders among those flags: {result['values']['pose_invalid_placeholder_low_information_views']:,}",
        f"- Final content-invalid views after retaining Stage 2 evidence: {result['values']['final_content_invalid_views']:,}",
        f"- Mean pixel coverage, context / relation: {result['values']['depth_context_pixel_coverage_mean']:.4f} / {result['values']['depth_relation_pixel_coverage_mean']:.4f}",
        f"- Neighboring-frame contact sheets: {preview_count}",
        "",
        "## Checks",
        "",
        *[f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in checks.items()],
        "",
        "Competition test was not read and no model was trained.",
    ]
    args.report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
