from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from evaluate_roi_locator_models import (
    annotation_box,
    box_iou,
    denormalize_box,
    normalize_box,
    sanitize_box,
)


PROJECT_DIR = Path(__file__).resolve().parent
ROI_DIR = PROJECT_DIR / "data" / "local_roi_annotation_v2"
DEFAULT_ANNOTATIONS = ROI_DIR / "roi_annotations_final.csv"
DEFAULT_FEATURES = (
    PROJECT_DIR
    / "runs"
    / "p12_roi_locator_baselines"
    / "roi_locator_features.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p13_hard_roi_locator_v1"
SEED = 20260728


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hard-action ROI locator v1 using all 216 confirmed ROIs"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trees", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def supervision_source(row: dict[str, str]) -> str:
    if row["bbox_source"] in {
        "pilot30_accepted_auto",
        "human_accepted_auto",
        "blind_review_kept_machine",
    }:
        return "accepted_machine"
    if row["annotation_mode"] == "blind":
        return "blind"
    return "correction"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    annotations = read_csv(args.annotations.resolve())
    if len(annotations) != 216:
        raise ValueError(f"Expected 216 ROI annotations, got {len(annotations)}")
    with np.load(args.features.resolve(), allow_pickle=False) as cache:
        sample_ids = cache["sample_ids"].astype(str)
        features = cache["features"].astype(np.float32)
        auto_boxes = cache["auto_boxes"].astype(np.float32)
        fallback = cache["fallback"].astype(bool)
    expected_ids = np.asarray([row["sample_id"] for row in annotations])
    if not np.array_equal(sample_ids, expected_ids):
        raise ValueError("Feature cache and final ROI rows have different ordering")
    targets = np.stack([annotation_box(row) for row in annotations])
    normalized_targets = np.stack([normalize_box(box) for box in targets])
    # Fallback is the known catastrophic failure mode. Weighting it more heavily
    # changes tree split priorities without discarding the other 173 examples.
    sample_weight = np.where(fallback, 2.5, 1.0)
    model = ExtraTreesRegressor(
        n_estimators=int(args.trees),
        max_depth=int(args.max_depth),
        min_samples_leaf=2,
        max_features=0.7,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(features, normalized_targets, sample_weight=sample_weight)
    predicted = np.stack(
        [
            denormalize_box(sanitize_box(box))
            for box in model.predict(features)
        ]
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "absolute_extra_trees_all216.joblib"
    joblib.dump(model, model_path, compress=("gzip", 3))

    training_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for index, row in enumerate(annotations):
        source = supervision_source(row)
        training_row: dict[str, object] = dict(row)
        training_row["previous_training_eligible"] = int(row["training_eligible"])
        training_row["locator_v1_training_eligible"] = 1
        training_row["supervision_source_v1"] = source
        training_row["motion_fallback"] = int(fallback[index])
        training_rows.append(training_row)
        target = targets[index]
        prediction = predicted[index]
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "fold": int(row["fold"]),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "supervision_source_v1": source,
                "motion_fallback": int(fallback[index]),
                "target_x0": float(target[0]),
                "target_y0": float(target[1]),
                "target_x1": float(target[2]),
                "target_y1": float(target[3]),
                "training_prediction_x0": float(prediction[0]),
                "training_prediction_y0": float(prediction[1]),
                "training_prediction_x1": float(prediction[2]),
                "training_prediction_y1": float(prediction[3]),
                "training_prediction_iou": box_iou(target, prediction),
                "original_motion_iou": box_iou(target, auto_boxes[index]),
            }
        )
    write_csv(output_dir / "all216_training_rois.csv", training_rows)
    write_csv(output_dir / "training_predictions_diagnostic_only.csv", prediction_rows)
    ious = np.asarray(
        [float(row["training_prediction_iou"]) for row in prediction_rows]
    )
    original_ious = np.asarray(
        [float(row["original_motion_iou"]) for row in prediction_rows]
    )
    source_counts: dict[str, int] = {}
    for row in training_rows:
        source = str(row["supervision_source_v1"])
        source_counts[source] = source_counts.get(source, 0) + 1
    report = {
        "purpose": (
            "Pre-annotation locator for the hard-action conditional Local branch. "
            "All 216 confirmed ROIs are training supervision."
        ),
        "training_samples": len(annotations),
        "supervision_sources": source_counts,
        "fallback_training_samples": int(fallback.sum()),
        "features": {
            "dimensions": int(features.shape[1]),
            "source": str(args.features.resolve()),
            "description": (
                "12 exact Depth frames; middle/motion HOG, temporal aggregates, "
                "low-resolution maps, and color histograms"
            ),
        },
        "model": {
            "type": "ExtraTreesRegressor absolute normalized bbox",
            "trees": int(args.trees),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": 2,
            "fallback_sample_weight": 2.5,
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
        },
        "deployment_v1": {
            "non_fallback": "retain motion bbox",
            "fallback": "use all216 absolute ExtraTrees bbox",
            "reason": (
                "Earlier strict fold-pure evaluation showed the hybrid is stable; "
                "global replacement and learned bad-box gating were worse."
            ),
        },
        "training_fit_diagnostic_not_generalization": {
            "mean_iou": float(ious.mean()),
            "iou_ge_0_5": float(np.mean(ious >= 0.5)),
            "original_motion_mean_iou": float(original_ious.mean()),
            "warning": (
                "These 216 rows trained the model. Generalization and acceptance "
                "rates must be measured on the newly selected 500 annotations."
            ),
        },
    }
    (output_dir / "model_card.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
