from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from evaluate_roi_locator_models import (
    denormalize_box,
    normalize_box,
    sanitize_box,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_V1_ROWS = (
    PROJECT_DIR / "runs" / "p13_hard_roi_locator_v1" / "all216_training_rois.csv"
)
DEFAULT_V1_FEATURES = (
    PROJECT_DIR
    / "runs"
    / "p12_roi_locator_baselines"
    / "roi_locator_features.npz"
)
DEFAULT_NEW_ROWS = (
    PROJECT_DIR
    / "data"
    / "hard_local_v1"
    / "annotation500"
    / "annotations500_final.csv"
)
DEFAULT_NEW_FEATURES = (
    PROJECT_DIR
    / "runs"
    / "p13_hard_roi_locator_v1"
    / "annotation500_locator_features.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p13_hard_roi_locator_v2"
SEED = 20260728


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hard-action ROI locator v2 on 216 + completed 500"
    )
    parser.add_argument("--v1-rows", type=Path, default=DEFAULT_V1_ROWS)
    parser.add_argument("--v1-features", type=Path, default=DEFAULT_V1_FEATURES)
    parser.add_argument("--new-rows", type=Path, default=DEFAULT_NEW_ROWS)
    parser.add_argument("--new-features", type=Path, default=DEFAULT_NEW_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trees", type=int, default=768)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a completed subset of the rolling 500 annotations.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    old_rows = read_csv(args.v1_rows.resolve())
    new_rows = read_csv(args.new_rows.resolve())
    valid_new_count = (
        1 <= len(new_rows) <= 500 if args.allow_partial else len(new_rows) == 500
    )
    if len(old_rows) != 216 or not valid_new_count:
        raise ValueError(
            "Expected 216 old rows and "
            f"{'1..500' if args.allow_partial else '500'} new rows, got "
            f"{len(old_rows)}+{len(new_rows)}"
        )
    with np.load(args.v1_features.resolve(), allow_pickle=False) as old_cache:
        old_ids = old_cache["sample_ids"].astype(str)
        old_features = old_cache["features"].astype(np.float32)
        old_fallback = old_cache["fallback"].astype(bool)
    with np.load(args.new_features.resolve(), allow_pickle=False) as new_cache:
        new_ids = new_cache["sample_ids"].astype(str)
        new_features = new_cache["features"].astype(np.float32)
    if not np.array_equal(
        old_ids, np.asarray([row["sample_id"] for row in old_rows])
    ):
        raise ValueError("Old ROI rows/features differ")
    new_lookup = {sample_id: index for index, sample_id in enumerate(new_ids)}
    requested_new_ids = [row["sample_id"] for row in new_rows]
    if len(set(requested_new_ids)) != len(requested_new_ids):
        raise ValueError("New ROI rows contain duplicate sample IDs")
    missing_new_ids = set(requested_new_ids) - set(new_lookup)
    if missing_new_ids:
        raise ValueError(
            f"New ROI feature cache is missing {sorted(missing_new_ids)[:3]}"
        )
    new_order = np.asarray(
        [new_lookup[row["sample_id"]] for row in new_rows], dtype=np.int64
    )
    new_features = new_features[new_order]
    features = np.concatenate([old_features, new_features])

    old_targets = np.stack(
        [
            normalize_box(
                np.asarray(
                    [float(row[field]) for field in ("x0", "y0", "x1", "y1")],
                    dtype=np.float32,
                )
            )
            for row in old_rows
        ]
    )
    new_targets = np.stack(
        [
            normalize_box(
                np.asarray(
                    [
                        float(row[field])
                        for field in (
                            "depth_x0",
                            "depth_y0",
                            "depth_x1",
                            "depth_y1",
                        )
                    ],
                    dtype=np.float32,
                )
            )
            for row in new_rows
        ]
    )
    targets = np.concatenate([old_targets, new_targets])
    new_fallback = np.asarray(
        [int(row["fallback"]) == 1 for row in new_rows], dtype=bool
    )
    fallback = np.concatenate([old_fallback, new_fallback])
    # Redrawn examples receive the highest correction weight; minor adjustments
    # remain more informative than direct acceptances.
    new_annotation_weights = np.asarray(
        [
            {
                "direct_accept": 1.0,
                "minor_adjustment": 1.5,
                "severe_error_redraw": 2.5,
            }[row["depth_box_assessment"]]
            for row in new_rows
        ],
        dtype=np.float64,
    )
    weights = np.concatenate(
        [
            np.where(old_fallback, 2.5, 1.0),
            new_annotation_weights * np.where(new_fallback, 1.5, 1.0),
        ]
    )
    model = ExtraTreesRegressor(
        n_estimators=int(args.trees),
        max_depth=int(args.max_depth),
        min_samples_leaf=2,
        max_features=0.7,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(features, targets, sample_weight=weights)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_samples = len(old_rows) + len(new_rows)
    model_path = output_dir / f"absolute_extra_trees_all{total_samples}.joblib"
    joblib.dump(model, model_path, compress=("gzip", 3))

    combined_rows: list[dict[str, object]] = []
    for row in old_rows:
        combined_rows.append(
            {
                "sample_id": row["sample_id"],
                "fold": int(row["fold"]),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "roi_generation": "original216",
                "supervision_source": row["supervision_source_v1"],
                "fallback": int(row["motion_fallback"]),
                "x0": float(row["x0"]),
                "y0": float(row["y0"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
            }
        )
    for row in new_rows:
        combined_rows.append(
            {
                "sample_id": row["sample_id"],
                "fold": int(row["fold"]),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "roi_generation": "rolling500",
                "supervision_source": row["depth_box_assessment"],
                "fallback": int(row["fallback"]),
                "x0": float(row["depth_x0"]),
                "y0": float(row["depth_y0"]),
                "x1": float(row["depth_x1"]),
                "y1": float(row["depth_y1"]),
            }
        )
    with (output_dir / f"all{total_samples}_training_rois.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    training_prediction = np.stack(
        [denormalize_box(sanitize_box(value)) for value in model.predict(features)]
    )
    report = {
        "status": "trained",
        "training_samples": len(combined_rows),
        "original216": len(old_rows),
        "rolling_annotations": len(new_rows),
        "rolling_target": 500,
        "partial_rolling_batch": len(new_rows) != 500,
        "fallback_samples": int(fallback.sum()),
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "trees": int(args.trees),
            "max_depth": int(args.max_depth),
        },
        "new_depth_assessment": {
            value: int(
                sum(row["depth_box_assessment"] == value for row in new_rows)
            )
            for value in (
                "direct_accept",
                "minor_adjustment",
                "severe_error_redraw",
            )
        },
        "training_fit_warning": (
            f"The {total_samples}-row fit is not an unbiased evaluation. "
            "Any classifier result using boxes from this locator must be marked "
            "exploratory/oracle-assisted until fold-pure reproduction."
        ),
        "training_prediction_shape": list(training_prediction.shape),
    }
    (output_dir / "model_card.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
