from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from evaluate_roi_locator_models import denormalize_box, sanitize_box


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_MOTION = (
    PROJECT_DIR
    / "runs"
    / "p12_fold_pure_locator_predictions"
    / "motion_boxes_all2914.csv"
)
DEFAULT_FEATURES = (
    PROJECT_DIR
    / "runs"
    / "p12_fold_pure_locator_predictions"
    / "fallback_features.npz"
)
DEFAULT_MODEL = (
    PROJECT_DIR
    / "runs"
    / "p16_oracle_roi_locator_all286"
    / "absolute_extra_trees_all286.joblib"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "runs" / "p16_oracle_assisted_locator_predictions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate exploratory all-sample ROI predictions using all available "
            "human ROI supervision. These predictions are not fold-pure."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--motion-boxes", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--fallback-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.manifest.resolve())
    manifest_rows.sort(key=lambda row: row["sample_id"])
    if len(manifest_rows) != 2914:
        raise ValueError(f"Expected 2914 manifest rows, got {len(manifest_rows)}")
    motion_rows = read_csv(args.motion_boxes.resolve())
    motion = {row["sample_id"]: row for row in motion_rows}
    if set(motion) != {row["sample_id"] for row in manifest_rows}:
        raise ValueError("Motion boxes do not cover the complete manifest")

    with np.load(args.fallback_features.resolve(), allow_pickle=False) as cache:
        fallback_ids = cache["sample_ids"].astype(str)
        features = cache["features"].astype(np.float32)
    expected_fallback = [
        row["sample_id"]
        for row in manifest_rows
        if int(motion[row["sample_id"]]["fallback"]) == 1
    ]
    if fallback_ids.tolist() != expected_fallback:
        raise ValueError("Fallback feature order differs from the manifest")

    model = joblib.load(args.model.resolve())
    normalized = model.predict(features)
    fallback_boxes = {
        sample_id: denormalize_box(sanitize_box(prediction))
        for sample_id, prediction in zip(
            fallback_ids,
            normalized,
            strict=True,
        )
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common_rows: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        sample_id = manifest_row["sample_id"]
        original = motion[sample_id]
        is_fallback = int(original["fallback"]) == 1
        if is_fallback:
            selected = fallback_boxes[sample_id]
            source = "absolute_et_all286_oracle_fallback"
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
        common_rows.append(
            {
                "sample_id": sample_id,
                "x0": float(selected[0]),
                "y0": float(selected[1]),
                "x1": float(selected[2]),
                "y1": float(selected[3]),
                "raw_width": 640,
                "raw_height": 480,
                "bbox_source": source,
                "motion_fallback": int(is_fallback),
                "protocol": "exploratory_oracle_assisted_all286",
            }
        )
    for held_fold in range(3):
        fold_rows = [
            {
                **row,
                "held_fold": held_fold,
                "locator_training_folds": json.dumps([0, 1, 2]),
            }
            for row in common_rows
        ]
        write_csv(
            output_dir / f"fold_{held_fold}_locator_predictions.csv",
            fold_rows,
        )
    summary = {
        "status": "exploratory_oracle_assisted",
        "deployable_oof": False,
        "samples": len(common_rows),
        "fallback": len(fallback_ids),
        "nonfallback": len(common_rows) - len(fallback_ids),
        "model": str(args.model.resolve()),
        "human_roi_supervision": 286,
        "warning": (
            "All three subject folds contributed ROI supervision. Results using "
            "these boxes must not be called strict or deployable OOF."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
