from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODALITY_ORDER = ("imu", "skeleton", "radar", "ir", "thermal", "depth_color")
FIELDS = (
    "modality",
    "model_name",
    "train_samples",
    "val_samples",
    "best_epoch",
    "val_accuracy",
    "val_macro_f1",
    "parameter_count",
    "checkpoint_size_mb",
    "inference_ms_per_sample",
    "config_path",
    "output_dir",
    "status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "task03")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "task03_first_round_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for order, modality in enumerate(MODALITY_ORDER):
        candidates = sorted((args.output_root / modality).glob("*/metrics.json"), key=lambda path: path.stat().st_mtime)
        candidates = [path for path in candidates if not json.loads(path.read_text(encoding="utf-8")).get("smoke_test")]
        if not candidates:
            raise FileNotFoundError(f"No completed first-round metrics for {modality}")
        metrics = json.loads(candidates[-1].read_text(encoding="utf-8"))
        row = {field: metrics[field] for field in FIELDS}
        row["modality_order"] = order
        rows.append(row)
    ranking = {row["modality"]: rank + 1 for rank, row in enumerate(sorted(rows, key=lambda item: float(item["val_accuracy"]), reverse=True))}
    for row in rows:
        row["accuracy_rank"] = ranking[row["modality"]]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.report, index=False, encoding="utf-8-sig")
    print(args.report)


if __name__ == "__main__":
    main()
