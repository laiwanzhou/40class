from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_UNION_DIR = PROJECT_DIR / "data" / "six_modality_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="以 subject 波动为参照审计 IR/Thermal 的无标签 Test 像素差异")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_UNION_DIR / "train_union_manifest.csv")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_UNION_DIR / "test_union_manifest.csv")
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "p0_six_modality_audit")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def modality_files(path: Path, modality: str) -> list[Path]:
    return sorted(path.glob("*.png" if modality == "ir" else "*.jpg"))


def frame_features(row: dict[str, str], modality: str, pixel_stride: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": row["split"],
        "sample_id": row["sample_id"],
        "class_id": int(row["class_id"]),
        "user_id": row["user_id"],
        "trial_id": row["trial_id"],
        "modality": modality,
        "directory_usable": int(row[f"{modality}_usable"]),
        "readable": 0,
        "middle_frame": "",
    }
    if not int(row[f"{modality}_usable"]):
        return result
    path = Path(row[f"{modality}_path"])
    files = modality_files(path, modality)
    if not files:
        return result
    selected = files[len(files) // 2]
    result["middle_frame"] = str(selected.resolve())
    try:
        with Image.open(selected) as image:
            rgb_full = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError):
        return result
    rgb = rgb_full[::pixel_stride, ::pixel_stride]
    luminance = (
        0.299 * rgb[..., 0].astype(np.float64)
        + 0.587 * rgb[..., 1].astype(np.float64)
        + 0.114 * rgb[..., 2].astype(np.float64)
    )
    height, width = luminance.shape
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    center = luminance[y0:y1, x0:x1]
    border_mask = np.ones_like(luminance, dtype=bool)
    border_mask[y0:y1, x0:x1] = False
    border = luminance[border_mask]
    result.update(
        {
            "readable": 1,
            "frame_count": len(files),
            "source_width": int(rgb_full.shape[1]),
            "source_height": int(rgb_full.shape[0]),
            "luminance_mean": float(luminance.mean()),
            "luminance_std": float(luminance.std()),
            "luminance_p10": float(np.quantile(luminance, 0.1)),
            "luminance_p50": float(np.quantile(luminance, 0.5)),
            "luminance_p90": float(np.quantile(luminance, 0.9)),
            "black_fraction": float(np.mean(luminance <= 1.0)),
            "saturated_fraction": float(np.mean(luminance >= 254.0)),
            "center_minus_border_luminance": float(center.mean() - border.mean()),
            "rgb_r_mean": float(rgb[..., 0].mean()),
            "rgb_g_mean": float(rgb[..., 1].mean()),
            "rgb_b_mean": float(rgb[..., 2].mean()),
        }
    )
    return result


def safe_feature(rows: list[dict[str, Any]], feature: str) -> np.ndarray:
    values = np.asarray([float(row.get(feature, math.nan)) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def compare(train: list[dict[str, Any]], test: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    train_values = safe_feature(train, feature)
    test_values = safe_feature(test, feature)
    subject_medians = []
    for user in sorted(set(str(row["user_id"]) for row in train)):
        values = safe_feature([row for row in train if row["user_id"] == user], feature)
        if len(values):
            subject_medians.append(float(np.median(values)))
    subjects = np.asarray(subject_medians, dtype=np.float64)
    train_std = float(train_values.std())
    subject_std = float(subjects.std())
    train_median = float(np.median(train_values))
    test_median = float(np.median(test_values))
    return {
        "feature": feature,
        "train_count": len(train_values),
        "test_count": len(test_values),
        "train_mean": float(train_values.mean()),
        "test_mean": float(test_values.mean()),
        "train_median": train_median,
        "test_median": test_median,
        "train_p10": float(np.quantile(train_values, 0.1)),
        "train_p90": float(np.quantile(train_values, 0.9)),
        "test_p10": float(np.quantile(test_values, 0.1)),
        "test_p90": float(np.quantile(test_values, 0.9)),
        "standardized_mean_difference_by_train_trial_std": float((test_values.mean() - train_values.mean()) / train_std) if train_std else None,
        "train_subject_median_min": float(subjects.min()),
        "train_subject_median_max": float(subjects.max()),
        "test_median_outside_train_subject_median_range": bool(test_median < subjects.min() or test_median > subjects.max()),
        "test_minus_train_median_in_subject_median_sd": float((test_median - train_median) / subject_std) if subject_std else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = read_csv(args.train_manifest.resolve())
    test_manifest = read_csv(args.test_manifest.resolve())
    feature_names = [
        "frame_count", "luminance_mean", "luminance_std", "luminance_p10", "luminance_p50",
        "luminance_p90", "black_fraction", "saturated_fraction", "center_minus_border_luminance",
        "rgb_r_mean", "rgb_g_mean", "rgb_b_mean",
    ]
    summary: dict[str, Any] = {
        "protocol": {
            "frame_sampling": "one middle frame per trial",
            "pixel_stride": int(args.pixel_stride),
            "warning": (
                "Test labels are unavailable. These unconditional differences are covariate evidence and may be confounded "
                "by unknown Test class mixture; they do not prove a causal domain shift."
            ),
        },
        "modalities": {},
    }
    all_rows = []
    all_comparisons = []
    for modality in ["ir", "thermal"]:
        print(f"提取 {modality} 中间帧统计……", flush=True)
        train_rows = [frame_features(row, modality, int(args.pixel_stride)) for row in train_manifest]
        test_rows = [frame_features(row, modality, int(args.pixel_stride)) for row in test_manifest]
        all_rows.extend(train_rows)
        all_rows.extend(test_rows)
        comparisons = [compare(train_rows, test_rows, feature) for feature in feature_names]
        for row in comparisons:
            row["modality"] = modality
        all_comparisons.extend(comparisons)
        ranked = sorted(
            comparisons,
            key=lambda row: abs(float(row["standardized_mean_difference_by_train_trial_std"] or 0.0)),
            reverse=True,
        )
        summary["modalities"][modality] = {
            "train_directory_usable": int(sum(int(row["directory_usable"]) for row in train_rows)),
            "test_directory_usable": int(sum(int(row["directory_usable"]) for row in test_rows)),
            "train_readable_middle_frames": int(sum(int(row["readable"]) for row in train_rows)),
            "test_readable_middle_frames": int(sum(int(row["readable"]) for row in test_rows)),
            "unreadable_test_sample_ids": [row["sample_id"] for row in test_rows if int(row["directory_usable"]) and not int(row["readable"])],
            "largest_five_feature_shifts": ranked[:5],
        }
    write_csv(output_dir / "ir_thermal_trial_features.csv", all_rows)
    write_csv(output_dir / "ir_thermal_train_test_comparison.csv", all_comparisons)
    (output_dir / "ir_thermal_shift.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
