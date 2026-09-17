from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np

from evaluate_roi_locator_models import (
    denormalize_box,
    feature_vector,
    sanitize_box,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SELECTION = (
    PROJECT_DIR / "data" / "hard_local_v1" / "annotation500" / "selection.csv"
)
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_MOTION = (
    PROJECT_DIR
    / "runs"
    / "p12_fold_pure_locator_predictions"
    / "motion_boxes_all2914.csv"
)
DEFAULT_MODEL = (
    PROJECT_DIR
    / "runs"
    / "p13_hard_roi_locator_v1"
    / "absolute_extra_trees_all216.joblib"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p13_hard_roi_locator_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate locator-v1 machine boxes for the selected new 500"
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--motion-boxes", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    selection = read_csv(args.selection.resolve())
    if len(selection) != 500:
        raise ValueError(f"Expected 500 selected rows, got {len(selection)}")
    manifest = {row["sample_id"]: row for row in read_csv(args.manifest.resolve())}
    motion = {
        row["sample_id"]: row for row in read_csv(args.motion_boxes.resolve())
    }

    def extract(row: dict[str, str]) -> np.ndarray:
        return feature_vector(manifest[row["sample_id"]])

    vectors: list[np.ndarray | None] = [None] * len(selection)
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {
            executor.submit(extract, row): index
            for index, row in enumerate(selection)
        }
        completed = 0
        for future, index in [(future, futures[future]) for future in futures]:
            vectors[index] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(selection):
                print(f"locator-v1 features {completed}/500", flush=True)
    if any(vector is None for vector in vectors):
        raise AssertionError("Feature extraction left an empty row")
    features = np.stack([vector for vector in vectors if vector is not None])
    model = joblib.load(args.model.resolve())
    absolute_normalized = np.stack(
        [sanitize_box(value) for value in model.predict(features)]
    )
    absolute_boxes = np.stack(
        [denormalize_box(value) for value in absolute_normalized]
    )
    # Tree disagreement is retained as a relative uncertainty feature for the
    # next rolling annotation round. It is not treated as calibrated confidence.
    tree_predictions = np.stack(
        [estimator.predict(features) for estimator in model.estimators_]
    )
    uncertainty = tree_predictions.std(axis=0).mean(axis=1)

    rows: list[dict[str, object]] = []
    for index, selected in enumerate(selection):
        sample_id = selected["sample_id"]
        motion_row = motion[sample_id]
        motion_box = np.asarray(
            [
                float(motion_row["x0"]),
                float(motion_row["y0"]),
                float(motion_row["x1"]),
                float(motion_row["y1"]),
            ],
            dtype=np.float32,
        )
        is_fallback = int(motion_row["fallback"]) == 1
        machine = absolute_boxes[index] if is_fallback else motion_box
        machine_source = (
            "locator_v1_fallback_et" if is_fallback else "motion_nonfallback"
        )
        row: dict[str, object] = dict(selected)
        row.update(
            {
                "locator_v1_machine_source": machine_source,
                "locator_v1_uncertainty": float(uncertainty[index]),
                "absolute_et_x0": float(absolute_boxes[index, 0]),
                "absolute_et_y0": float(absolute_boxes[index, 1]),
                "absolute_et_x1": float(absolute_boxes[index, 2]),
                "absolute_et_y1": float(absolute_boxes[index, 3]),
                "machine_x0": float(machine[0]),
                "machine_y0": float(machine[1]),
                "machine_x1": float(machine[2]),
                "machine_y1": float(machine[3]),
                "machine_norm_x0": float(machine[0] / 640),
                "machine_norm_y0": float(machine[1] / 480),
                "machine_norm_x1": float(machine[2] / 640),
                "machine_norm_y1": float(machine[3] / 480),
            }
        )
        rows.append(row)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "annotation500_locator_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "annotation500_locator_features.npz",
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        features=features.astype(np.float32),
        absolute_boxes=absolute_boxes.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
    )
    sources: dict[str, int] = {}
    for row in rows:
        source = str(row["locator_v1_machine_source"])
        sources[source] = sources.get(source, 0) + 1
    report = {
        "samples": len(rows),
        "features": int(features.shape[1]),
        "machine_sources": sources,
        "uncertainty": {
            "mean": float(uncertainty.mean()),
            "median": float(np.median(uncertainty)),
            "p90": float(np.quantile(uncertainty, 0.90)),
            "warning": "Relative tree disagreement, not calibrated confidence.",
        },
        "model": str(args.model.resolve()),
        "selection": str(args.selection.resolve()),
        "predictions": str(output_dir / "annotation500_locator_predictions.csv"),
    }
    (output_dir / "annotation500_prediction_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
