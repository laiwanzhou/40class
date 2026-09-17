from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from evaluate_roi_locator_models import (
    box_iou,
    denormalize_box,
    normalize_box,
    sanitize_box,
)


PROJECT_DIR = Path(__file__).resolve().parent
SEED = 27130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build outer-train-only, inner-subject-pure broad ROI predictions "
            "for P27 IR local-context experiments"
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p13_hard_roi_locator_v1"
            / "all216_training_rois.csv"
        ),
    )
    parser.add_argument(
        "--annotation-features",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p12_roi_locator_baselines"
            / "roi_locator_features.npz"
        ),
    )
    parser.add_argument(
        "--motion-boxes",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p12_fold_pure_locator_predictions"
            / "motion_boxes_all2914.csv"
        ),
    )
    parser.add_argument(
        "--fallback-features",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p12_fold_pure_locator_predictions"
            / "fallback_features.npz"
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "p27_strong_inner",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p27_strong_inner"
            / "inner_fold_pure_roi"
        ),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def user_from_sample_id(sample_id: str) -> str:
    parts = sample_id.split("__")
    if len(parts) < 4:
        raise ValueError(f"Unexpected sample_id: {sample_id}")
    return parts[2]


def main() -> None:
    args = parse_args()
    annotation_rows = read_csv(args.annotations.resolve())
    with np.load(args.annotation_features.resolve(), allow_pickle=False) as archive:
        annotation_ids = archive["sample_ids"].astype(str)
        annotation_features = archive["features"].astype(np.float32)
        annotation_fallback = archive["fallback"].astype(bool)
    expected_ids = np.asarray(
        [row["sample_id"] for row in annotation_rows], dtype=str
    )
    if not np.array_equal(annotation_ids, expected_ids):
        raise RuntimeError("Annotation feature cache is not aligned to annotation rows")
    annotation_targets = np.stack(
        [
            normalize_box(
                np.asarray(
                    [float(row[key]) for key in ("x0", "y0", "x1", "y1")],
                    dtype=np.float32,
                )
            )
            for row in annotation_rows
        ]
    )
    annotation_users = np.asarray(
        [
            row.get("user_id") or user_from_sample_id(row["sample_id"])
            for row in annotation_rows
        ]
    )

    motion_rows = read_csv(args.motion_boxes.resolve())
    motion = {row["sample_id"]: row for row in motion_rows}
    with np.load(args.fallback_features.resolve(), allow_pickle=False) as archive:
        fallback_ids = archive["sample_ids"].astype(str)
        fallback_vectors = archive["features"].astype(np.float32)
    fallback_features = {
        sample_id: vector
        for sample_id, vector in zip(
            fallback_ids, fallback_vectors, strict=True
        )
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_folds: dict[str, Any] = {}
    for fold in sorted(set(int(value) for value in args.folds)):
        manifest_path = args.manifest_dir.resolve() / f"fold_{fold}.csv"
        manifest = read_csv(manifest_path)
        train_users = sorted(
            {row["user_id"] for row in manifest if row["split"] == "train"}
        )
        held_users = sorted(
            {row["user_id"] for row in manifest if row["split"] == "val"}
        )
        selected_annotations = np.isin(annotation_users, train_users)
        if int(selected_annotations.sum()) < 80:
            raise RuntimeError(
                f"Fold {fold} has only {selected_annotations.sum()} ROI annotations"
            )
        sample_weight = np.where(
            annotation_fallback[selected_annotations], 2.5, 1.0
        )
        model = ExtraTreesRegressor(
            n_estimators=int(args.trees),
            max_depth=int(args.max_depth),
            min_samples_leaf=2,
            max_features=0.7,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        model.fit(
            annotation_features[selected_annotations],
            annotation_targets[selected_annotations],
            sample_weight=sample_weight,
        )
        manifest_fallback_ids = [
            row["sample_id"]
            for row in manifest
            if int(motion[row["sample_id"]]["fallback"]) == 1
        ]
        missing_fallback = [
            sample_id
            for sample_id in manifest_fallback_ids
            if sample_id not in fallback_features
        ]
        if missing_fallback:
            raise RuntimeError(
                f"Fallback cache misses {len(missing_fallback)} manifest samples"
            )
        fallback_prediction_values = model.predict(
            np.stack(
                [fallback_features[sample_id] for sample_id in manifest_fallback_ids]
            )
        )
        fallback_predictions = {
            sample_id: denormalize_box(sanitize_box(prediction))
            for sample_id, prediction in zip(
                manifest_fallback_ids, fallback_prediction_values, strict=True
            )
        }

        prediction_rows: list[dict[str, Any]] = []
        prediction_by_id: dict[str, np.ndarray] = {}
        for row in manifest:
            sample_id = row["sample_id"]
            if sample_id not in motion:
                raise RuntimeError(f"Motion cache misses {sample_id}")
            motion_row = motion[sample_id]
            is_fallback = int(motion_row["fallback"]) == 1
            if is_fallback:
                box = fallback_predictions[sample_id]
                source = "inner_pure_absolute_et_fallback"
                quality = 0.65
            else:
                box = np.asarray(
                    [
                        float(motion_row[key])
                        for key in ("x0", "y0", "x1", "y1")
                    ],
                    dtype=np.float32,
                )
                source = "label_free_motion_nonfallback"
                quality = 1.0
            prediction_by_id[sample_id] = box
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "inner_fold": fold,
                    "split": row["split"],
                    "user_id": row["user_id"],
                    "locator_training_users": json.dumps(train_users),
                    "x0": float(box[0]),
                    "y0": float(box[1]),
                    "x1": float(box[2]),
                    "y1": float(box[3]),
                    "raw_width": 640,
                    "raw_height": 480,
                    "bbox_source": source,
                    "motion_fallback": int(is_fallback),
                    "roi_quality": quality,
                    "outer_held_predictions_generated": False,
                }
            )
        write_csv(output / f"fold_{fold}_roi_predictions.csv", prediction_rows)

        held_annotation_ious: list[float] = []
        held_source_ious: dict[str, list[float]] = {}
        for annotation, user in zip(
            annotation_rows, annotation_users, strict=True
        ):
            sample_id = annotation["sample_id"]
            if user not in held_users or sample_id not in prediction_by_id:
                continue
            target = np.asarray(
                [float(annotation[key]) for key in ("x0", "y0", "x1", "y1")],
                dtype=np.float32,
            )
            value = box_iou(prediction_by_id[sample_id], target)
            held_annotation_ious.append(value)
            source = next(
                item["bbox_source"]
                for item in prediction_rows
                if item["sample_id"] == sample_id
            )
            held_source_ious.setdefault(str(source), []).append(value)
        held_audit = {
            "samples": len(held_annotation_ious),
            "mean_iou": mean(held_annotation_ious),
            "median_iou": median(held_annotation_ious),
            "iou_ge_0_5": mean(value >= 0.5 for value in held_annotation_ious),
            "by_source": {
                source: {
                    "samples": len(values),
                    "mean_iou": mean(values),
                    "median_iou": median(values),
                }
                for source, values in held_source_ious.items()
            },
        }
        summary_folds[str(fold)] = {
            "manifest": str(manifest_path),
            "train_users": train_users,
            "held_users": held_users,
            "training_annotations": int(selected_annotations.sum()),
            "prediction_samples": len(prediction_rows),
            "fallback_samples": int(
                sum(int(row["motion_fallback"]) for row in prediction_rows)
            ),
            "held_annotation_audit_only": held_audit,
            "outer_held_predictions_generated": False,
        }

    summary = {
        "protocol": "p27-inner-fold-pure-broad-roi-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "annotations_are_roi_only": True,
        "class_labels_used_as_locator_features": False,
        "sample_id_subject_trial_used_as_locator_features": False,
        "folds": summary_folds,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
