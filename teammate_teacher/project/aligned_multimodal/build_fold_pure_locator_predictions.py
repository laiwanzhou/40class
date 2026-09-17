from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np

from audit_motion_crop import analyse_trial
from evaluate_roi_locator_models import (
    denormalize_box,
    feature_vector,
    map_box,
    sanitize_box,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_MODEL_DIR = PROJECT_DIR / "runs" / "p12_roi_locator_baselines"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p12_fold_pure_locator_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all-sample fold-pure ROI predictions using the motion rule "
            "for non-fallback trials and the small absolute regressor for fallbacks."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def motion_record(row: dict[str, str]) -> dict[str, object]:
    record, _ = analyse_trial(row, 320, 240)
    mapped = map_box([float(value) for value in record["bbox"]])
    return {
        "sample_id": row["sample_id"],
        "fallback": int(bool(record["fallback"])),
        "x0": float(mapped[0]),
        "y0": float(mapped[1]),
        "x1": float(mapped[2]),
        "y1": float(mapped[3]),
    }


def fallback_feature(row: dict[str, str]) -> tuple[str, np.ndarray]:
    return row["sample_id"], feature_vector(row)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    manifest_rows = read_csv(args.manifest.resolve())
    manifest_rows.sort(key=lambda row: row["sample_id"])
    if len(manifest_rows) != 2914:
        raise ValueError(f"Expected 2914 train trials, got {len(manifest_rows)}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_cache = output_dir / "motion_boxes_all2914.csv"
    if motion_cache.exists():
        motion_rows = read_csv(motion_cache)
    else:
        motion_rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, result in enumerate(
                executor.map(motion_record, manifest_rows, chunksize=1),
                start=1,
            ):
                motion_rows.append(result)
                if index % 50 == 0 or index == len(manifest_rows):
                    print(f"Motion boxes {index}/{len(manifest_rows)}", flush=True)
        write_csv(motion_cache, motion_rows)
    motion = {row["sample_id"]: row for row in motion_rows}
    if set(motion) != {row["sample_id"] for row in manifest_rows}:
        raise ValueError("Motion-box cache does not cover the full manifest")
    fallback_rows = [
        row
        for row in manifest_rows
        if int(motion[row["sample_id"]]["fallback"]) == 1
    ]
    feature_cache = output_dir / "fallback_features.npz"
    if feature_cache.exists():
        cache = np.load(feature_cache, allow_pickle=False)
        fallback_ids = [str(value) for value in cache["sample_ids"]]
        fallback_features = cache["features"]
    else:
        fallback_ids: list[str] = []
        vectors: list[np.ndarray] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, (sample_id, vector) in enumerate(
                executor.map(fallback_feature, fallback_rows, chunksize=1),
                start=1,
            ):
                fallback_ids.append(sample_id)
                vectors.append(vector)
                if index % 25 == 0 or index == len(fallback_rows):
                    print(
                        f"Fallback features {index}/{len(fallback_rows)}",
                        flush=True,
                    )
        fallback_features = np.stack(vectors)
        np.savez_compressed(
            feature_cache,
            sample_ids=np.asarray(fallback_ids),
            features=fallback_features,
        )
    expected_fallback_ids = [row["sample_id"] for row in fallback_rows]
    if fallback_ids != expected_fallback_ids:
        raise ValueError("Fallback feature cache order mismatch")

    fallback_predictions: dict[int, dict[str, np.ndarray]] = {}
    for held_fold in range(3):
        model = joblib.load(
            args.model_dir.resolve() / f"fold_{held_fold}_absolute_et.joblib"
        )
        normalized = model.predict(fallback_features)
        fallback_predictions[held_fold] = {
            sample_id: denormalize_box(sanitize_box(prediction))
            for sample_id, prediction in zip(
                fallback_ids,
                normalized,
                strict=True,
            )
        }
    for held_fold in range(3):
        training_folds = [fold for fold in range(3) if fold != held_fold]
        rows: list[dict[str, object]] = []
        for manifest_row in manifest_rows:
            sample_id = manifest_row["sample_id"]
            original = motion[sample_id]
            is_fallback = int(original["fallback"]) == 1
            if is_fallback:
                selected = fallback_predictions[held_fold][sample_id]
                source = "absolute_et_fallback"
            else:
                selected = np.asarray(
                    [
                        float(original["x0"]),
                        float(original["y0"]),
                        float(original["x1"]),
                        float(original["y1"]),
                    ],
                    dtype=np.float32,
                )
                source = "motion_nonfallback"
            rows.append(
                {
                    "sample_id": sample_id,
                    "held_fold": held_fold,
                    "locator_training_folds": json.dumps(training_folds),
                    "x0": float(selected[0]),
                    "y0": float(selected[1]),
                    "x1": float(selected[2]),
                    "y1": float(selected[3]),
                    "raw_width": 640,
                    "raw_height": 480,
                    "bbox_source": source,
                    "motion_fallback": int(is_fallback),
                }
            )
        write_csv(output_dir / f"fold_{held_fold}_locator_predictions.csv", rows)
    summary = {
        "samples": len(manifest_rows),
        "motion_fallback": len(fallback_rows),
        "motion_fallback_fraction": len(fallback_rows) / len(manifest_rows),
        "protocol": {
            str(fold): {
                "locator_training_folds": [
                    other for other in range(3) if other != fold
                ],
                "predictions": len(manifest_rows),
            }
            for fold in range(3)
        },
        "sources": {
            "motion_nonfallback": len(manifest_rows) - len(fallback_rows),
            "absolute_et_fallback": len(fallback_rows),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
