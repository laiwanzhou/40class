from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from depth_encoding import decode_jet_rgb


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_UNION_DIR = PROJECT_DIR / "data" / "six_modality_audit"
TIMESTAMP_PATTERN = re.compile(r"Depth_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2}\.\d{3})_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较 Depth 的 subject 内/间波动与无标签 Test 差异")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_UNION_DIR / "train_union_manifest.csv")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_UNION_DIR / "test_union_manifest.csv")
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "p0_six_modality_audit")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_timestamp(path: Path) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(path.name)
    if not match:
        return None
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H-%M-%S.%f")


def safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def trial_features(row: dict[str, str], pixel_stride: int) -> dict[str, Any] | None:
    if int(row["depth_color_usable"]) != 1:
        return None
    trial_dir = Path(row["depth_color_path"])
    frames = sorted(trial_dir.glob("Depth_*_Color.png"))
    if not frames:
        return None
    selected = frames[len(frames) // 2]
    with Image.open(selected) as image:
        rgb_full = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = rgb_full[::pixel_stride, ::pixel_stride]
    decoded, valid, repaired = decode_jet_rgb(rgb)
    valid_values = decoded[valid].astype(np.float64)
    timestamps = [timestamp for timestamp in (parse_timestamp(path) for path in frames) if timestamp is not None]
    gaps = np.diff([timestamp.timestamp() for timestamp in timestamps]) if len(timestamps) >= 2 else np.asarray([])

    height, width = valid.shape
    yy, xx = np.mgrid[:height, :width]
    if valid.any():
        valid_x = xx[valid]
        valid_y = yy[valid]
        x_span = (valid_x.max() - valid_x.min() + 1) / width
        y_span = (valid_y.max() - valid_y.min() + 1) / height
        centroid_x = valid_x.mean() / max(width - 1, 1)
        centroid_y = valid_y.mean() / max(height - 1, 1)
    else:
        x_span = y_span = centroid_x = centroid_y = np.nan

    border = max(1, min(height, width) // 10)
    border_mask = np.zeros_like(valid, dtype=bool)
    border_mask[:border] = True
    border_mask[-border:] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    center_mask = ~border_mask

    result: dict[str, Any] = {
        "split": row["split"],
        "sample_id": row["sample_id"],
        "class_id": int(row["class_id"]),
        "user_id": row["user_id"],
        "trial_id": row["trial_id"],
        "frame_count": len(frames),
        "duration_seconds": (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) >= 2 else np.nan,
        "median_frame_gap_seconds": float(np.median(gaps)) if len(gaps) else np.nan,
        "source_width": int(rgb_full.shape[1]),
        "source_height": int(rgb_full.shape[0]),
        "middle_frame": str(selected.resolve()),
        "sampled_pixels": int(valid.size),
        "invalid_fraction": float(1.0 - valid.mean()),
        "border_invalid_fraction": float(1.0 - valid[border_mask].mean()),
        "center_invalid_fraction": float(1.0 - valid[center_mask].mean()),
        "valid_jet_mean": float(valid_values.mean()) if len(valid_values) else np.nan,
        "valid_jet_std": float(valid_values.std()) if len(valid_values) else np.nan,
        "valid_jet_p10": float(np.quantile(valid_values, 0.1)) if len(valid_values) else np.nan,
        "valid_jet_p50": float(np.quantile(valid_values, 0.5)) if len(valid_values) else np.nan,
        "valid_jet_p90": float(np.quantile(valid_values, 0.9)) if len(valid_values) else np.nan,
        "valid_bbox_width_fraction": float(x_span),
        "valid_bbox_height_fraction": float(y_span),
        "valid_centroid_x": float(centroid_x),
        "valid_centroid_y": float(centroid_y),
        "rgb_r_mean": float(rgb[..., 0].mean()),
        "rgb_g_mean": float(rgb[..., 1].mean()),
        "rgb_b_mean": float(rgb[..., 2].mean()),
        "repaired_nonblack_pixels": int(repaired),
    }
    return result


def feature_comparison(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "frame_count",
        "duration_seconds",
        "median_frame_gap_seconds",
        "invalid_fraction",
        "border_invalid_fraction",
        "center_invalid_fraction",
        "valid_jet_mean",
        "valid_jet_std",
        "valid_jet_p10",
        "valid_jet_p50",
        "valid_jet_p90",
        "valid_bbox_width_fraction",
        "valid_bbox_height_fraction",
        "valid_centroid_x",
        "valid_centroid_y",
        "rgb_r_mean",
        "rgb_g_mean",
        "rgb_b_mean",
    ]
    result = []
    users = sorted(set(str(row["user_id"]) for row in train_rows))
    for feature in features:
        train = np.asarray([float(row[feature]) for row in train_rows], dtype=np.float64)
        test = np.asarray([float(row[feature]) for row in test_rows], dtype=np.float64)
        train = train[np.isfinite(train)]
        test = test[np.isfinite(test)]
        subject_medians = []
        for user in users:
            values = np.asarray(
                [float(row[feature]) for row in train_rows if row["user_id"] == user],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if len(values):
                subject_medians.append(float(np.median(values)))
        subject_medians_array = np.asarray(subject_medians, dtype=np.float64)
        train_std = float(train.std())
        subject_std = float(subject_medians_array.std())
        test_median = float(np.median(test))
        train_median = float(np.median(train))
        result.append(
            {
                "feature": feature,
                "train_count": len(train),
                "test_count": len(test),
                "train_mean": float(train.mean()),
                "test_mean": float(test.mean()),
                "train_median": train_median,
                "test_median": test_median,
                "train_p10": float(np.quantile(train, 0.1)),
                "train_p90": float(np.quantile(train, 0.9)),
                "test_p10": float(np.quantile(test, 0.1)),
                "test_p90": float(np.quantile(test, 0.9)),
                "standardized_mean_difference_by_train_trial_std": safe_float((test.mean() - train.mean()) / train_std) if train_std else None,
                "train_subject_median_min": float(subject_medians_array.min()),
                "train_subject_median_max": float(subject_medians_array.max()),
                "test_median_outside_train_subject_median_range": bool(
                    test_median < subject_medians_array.min() or test_median > subject_medians_array.max()
                ),
                "test_minus_train_median_in_subject_median_sd": safe_float((test_median - train_median) / subject_std) if subject_std else None,
            }
        )
    return result


def subject_summary(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[str(row["user_id"])].append(row)
    result = []
    for user, rows in sorted(grouped.items()):
        result.append(
            {
                "user_id": user,
                "samples": len(rows),
                "frame_count_median": float(np.median([row["frame_count"] for row in rows])),
                "invalid_fraction_median": float(np.median([row["invalid_fraction"] for row in rows])),
                "valid_jet_mean_median": float(np.median([row["valid_jet_mean"] for row in rows])),
                "rgb_r_mean_median": float(np.median([row["rgb_r_mean"] for row in rows])),
                "rgb_g_mean_median": float(np.median([row["rgb_g_mean"] for row in rows])),
                "rgb_b_mean_median": float(np.median([row["rgb_b_mean"] for row in rows])),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = read_csv(args.train_manifest.resolve())
    test_manifest = read_csv(args.test_manifest.resolve())
    print("提取每个 trial 中间帧的 Depth 几何与 JET 统计……", flush=True)
    train_rows = [features for row in train_manifest if (features := trial_features(row, int(args.pixel_stride))) is not None]
    test_rows = [features for row in test_manifest if (features := trial_features(row, int(args.pixel_stride))) is not None]
    comparisons = feature_comparison(train_rows, test_rows)
    subjects = subject_summary(train_rows)
    write_csv(output_dir / "depth_trial_features.csv", train_rows + test_rows)
    write_csv(output_dir / "depth_train_subject_summary.csv", subjects)
    write_csv(output_dir / "depth_train_test_feature_comparison.csv", comparisons)

    ranked = sorted(
        comparisons,
        key=lambda row: abs(float(row["standardized_mean_difference_by_train_trial_std"] or 0.0)),
        reverse=True,
    )
    summary = {
        "protocol": {
            "train_trials": len(train_rows),
            "test_trials": len(test_rows),
            "frame_sampling": "one middle frame per trial for pixel statistics; all filenames for duration/gap",
            "pixel_stride": int(args.pixel_stride),
            "depth_interpretation": "OpenCV JET index plus explicit black invalid mask; not asserted to be absolute metres",
            "causal_warning": (
                "Test labels are unavailable. Unconditional Train/Test differences can be confounded by unknown class mixture; "
                "feature shifts are covariate evidence, not proof that they caused the leaderboard gap."
            ),
        },
        "ranked_feature_shifts": ranked,
        "largest_five_by_absolute_standardized_mean_difference": [
            {
                "feature": row["feature"],
                "standardized_mean_difference": row["standardized_mean_difference_by_train_trial_std"],
                "test_median_outside_train_subject_median_range": row["test_median_outside_train_subject_median_range"],
                "test_minus_train_median_in_subject_median_sd": row["test_minus_train_median_in_subject_median_sd"],
            }
            for row in ranked[:5]
        ],
        "outputs": {
            "trial_features": str((output_dir / "depth_trial_features.csv").resolve()),
            "subject_summary": str((output_dir / "depth_train_subject_summary.csv").resolve()),
            "feature_comparison": str((output_dir / "depth_train_test_feature_comparison.csv").resolve()),
        },
    }
    (output_dir / "depth_subject_test_shift.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["largest_five_by_absolute_standardized_mean_difference"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
