from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize an interrupted P32 fold-0 run from its saved best checkpoints"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempted-epoch", type=int, required=True)
    parser.add_argument("--reason", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    config = read_json(run / "frozen_config.json")
    history = read_json(run / "history.json")
    if not history:
        raise RuntimeError("history.json is empty")
    with (run / "history.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_epochs = list(csv.DictReader(handle))
    if len(csv_epochs) != len(history):
        raise RuntimeError("history.csv and history.json disagree")

    best_accuracy = read_json(run / "best_accuracy_metrics.json")
    best_macro = read_json(run / "best_macro_f1_metrics.json")
    best_accuracy_epoch = int(history[-1]["best_accuracy_epoch"])
    best_macro_epoch = int(history[-1]["best_macro_f1_epoch"])
    reference = config["reference_p12_fold0"]
    observed_seconds = sum(
        float(row["train_seconds"]) + float(row["val_seconds"]) for row in history
    )
    steady_rows = history[1:] if len(history) > 1 else history
    mean_steady_epoch_seconds = sum(
        float(row["train_seconds"]) + float(row["val_seconds"])
        for row in steady_rows
    ) / len(steady_rows)

    checkpoint_sizes = {}
    for name in ("best_accuracy.pt", "best_macro_f1.pt"):
        path = run / name
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"missing checkpoint: {path}")
        checkpoint_sizes[name] = {
            "bytes": path.stat().st_size,
            "mib": path.stat().st_size / 1024**2,
        }

    summary = {
        "stage": "P32_Step15_fold0_only",
        "status": "interrupted_after_saved_best",
        "fold": 0,
        "completed_epochs": len(history),
        "attempted_epoch": args.attempted_epoch,
        "stop_reason": args.reason,
        "early_stopping_reached": False,
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_accuracy_metrics": best_accuracy,
        "best_macro_f1_epoch": best_macro_epoch,
        "best_macro_f1_metrics": best_macro,
        "reference_p12_fold0": reference,
        "best_accuracy_delta_vs_p12_fold0_percentage_points": 100.0
        * (best_accuracy["overall"]["accuracy"] - reference["accuracy"]),
        "best_macro_f1_delta_vs_p12_fold0_percentage_points": 100.0
        * (best_macro["overall"]["macro_f1"] - reference["macro_f1"]),
        "beats_reference_p12_fold0_accuracy": bool(
            best_accuracy["overall"]["accuracy"] > reference["accuracy"]
        ),
        "observed_completed_epoch_seconds": observed_seconds,
        "mean_steady_epoch_seconds": mean_steady_epoch_seconds,
        "estimated_40_epoch_fold0_hours_at_observed_steady_rate": (
            mean_steady_epoch_seconds * 40.0 / 3600.0
        ),
        "checkpoint_sizes": checkpoint_sizes,
        "fold1_started": False,
        "fold2_started": False,
        "decision": "do_not_start_fold1_or_fold2_because_fold0_did_not_beat_reference",
        "config": config,
    }
    write_json_atomic(run / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
