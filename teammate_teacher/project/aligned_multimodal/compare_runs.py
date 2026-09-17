from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RUN_NAMES = (
    "ir_only",
    "depth_only",
    "skeleton_only",
    "depth_ir",
    "ir_skeleton",
    "depth_skeleton",
    "depth_ir_skeleton",
)


def main() -> None:
    rows = []
    for name in RUN_NAMES:
        path = PROJECT_DIR / "runs" / name / "metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": name,
                "modalities": "+".join(metrics["modalities"]),
                "parameters": metrics["parameters"],
                "size_mb": round(metrics["fp32_parameter_size_mb"], 2),
                "best_accuracy_epoch": metrics["best_accuracy_epoch"],
                "best_val_accuracy": metrics["best_val_accuracy"],
                "best_macro_f1_epoch": metrics["best_macro_f1_epoch"],
                "best_val_macro_f1": metrics["best_val_macro_f1"],
                "total_seconds": metrics["total_seconds"],
            }
        )
    if not rows:
        raise RuntimeError("尚无完成的 runs")
    output = PROJECT_DIR / "runs" / "comparison.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for row in rows:
        print(
            f"{row['run']:<20} acc={row['best_val_accuracy']:.4f} "
            f"F1={row['best_val_macro_f1']:.4f} size={row['size_mb']:.2f}MB"
        )


if __name__ == "__main__":
    main()
