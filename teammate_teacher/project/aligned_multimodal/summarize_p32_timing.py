from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "runs" / "p32_steps13_14_short_benchmark_80_tcn_w2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct P32 formal timing with steady repeated eval")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, nargs="+", default=(20, 30, 40))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    benchmark = json.loads((run / "benchmark.json").read_text(encoding="utf-8"))
    repeated_eval = json.loads((run / "eval_repeat.json").read_text(encoding="utf-8"))
    warm_train = benchmark["train_epochs"][1:] or benchmark["train_epochs"]
    train_seconds_per_frame = float(
        np.median(
            [row["seconds"] / row["real_frames"] for row in warm_train]
        )
    )
    steady_eval = repeated_eval["passes"][-1]
    eval_seconds_per_frame = steady_eval["seconds"] / steady_eval["real_frames"]
    per_fold = []
    for workload in benchmark["fold_workloads"]:
        seconds = (
            workload["train_frames"] * train_seconds_per_frame
            + workload["val_frames"] * eval_seconds_per_frame
        )
        per_fold.append({"fold": workload["fold"], "seconds": seconds})
    steady_three_fold_epoch = sum(row["seconds"] for row in per_fold)
    cold_train_overhead = max(
        0.0,
        benchmark["train_epochs"][0]["seconds"]
        - float(np.median([row["seconds"] for row in warm_train])),
    )
    cold_eval_overhead = max(
        0.0,
        repeated_eval["passes"][0]["seconds"]
        - repeated_eval["passes"][-1]["seconds"],
    )
    cold_overhead_three_folds = 3.0 * (cold_train_overhead + cold_eval_overhead)
    estimates = {}
    for epochs in args.epochs:
        steady_seconds = steady_three_fold_epoch * epochs
        with_cold = steady_seconds + cold_overhead_three_folds
        estimates[str(epochs)] = {
            "steady_hours": steady_seconds / 3600.0,
            "with_three_fold_cold_start_hours": with_cold / 3600.0,
            "recommended_20_percent_margin_hours": with_cold * 1.2 / 3600.0,
        }
    result = {
        "corrected": True,
        "train_seconds_per_real_frame": train_seconds_per_frame,
        "eval_seconds_per_real_frame": eval_seconds_per_frame,
        "per_fold_one_epoch": per_fold,
        "steady_three_fold_epoch_seconds": steady_three_fold_epoch,
        "steady_three_fold_epoch_minutes": steady_three_fold_epoch / 60.0,
        "cold_train_overhead_per_fold_seconds": cold_train_overhead,
        "cold_eval_overhead_per_fold_seconds": cold_eval_overhead,
        "cold_overhead_three_folds_seconds": cold_overhead_three_folds,
        "estimates": estimates,
        "note": (
            "The original benchmark.json estimate used the first cold validation pass. "
            "This corrected file uses the second pass with persistent workers."
        ),
    }
    (run / "corrected_time_estimate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
