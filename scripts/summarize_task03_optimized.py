from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task03_optimized_two_epoch"
OLD_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task03"
SUMMARY_CSV = PROJECT_ROOT / "reports" / "task03_optimized_two_epoch_summary.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "task03_optimized_two_epoch_report.md"
MODALITIES = ("imu", "skeleton", "radar", "ir", "thermal", "depth_color")
DISPLAY = {
    "imu": "IMU",
    "skeleton": "Skeleton",
    "radar": "Radar",
    "ir": "IR",
    "thermal": "Thermal",
    "depth_color": "Depth_Color",
}
FIELDS = (
    "modality",
    "model_name",
    "train_samples",
    "val_samples",
    "num_workers",
    "batch_size",
    "epochs_completed",
    "best_epoch",
    "val_accuracy",
    "val_macro_f1",
    "parameter_count",
    "checkpoint_size_mb",
    "inference_ms_per_sample",
    "epoch_1_seconds",
    "epoch_2_seconds",
    "mean_epoch_seconds",
    "train_samples_per_second",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "config_path",
    "output_dir",
    "status",
)


def latest_run(root: Path, modality: str) -> Path:
    candidates = [path.parent for path in (root / modality).glob("*/metrics.json")]
    if not candidates:
        raise FileNotFoundError(f"No completed run found for {modality} under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def validate_predictions(run_dir: Path, val_samples: int) -> None:
    with np.load(run_dir / "fold_0_val_predictions.npz") as data:
        if data["labels"].shape != (val_samples,):
            raise ValueError(f"Invalid labels shape in {run_dir}")
        if data["logits"].shape != (val_samples, 40):
            raise ValueError(f"Invalid logits shape in {run_dir}")
        if data["embeddings"].shape != (val_samples, 128):
            raise ValueError(f"Invalid embeddings shape in {run_dir}")
        if not np.array_equal(data["class_order"], np.arange(40)):
            raise ValueError(f"Invalid class order in {run_dir}")
        if not np.isfinite(data["logits"]).all() or not np.isfinite(data["embeddings"]).all():
            raise ValueError(f"Non-finite predictions in {run_dir}")


def load_new_row(modality: str) -> dict[str, Any]:
    run_dir = latest_run(OUTPUT_ROOT, modality)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    if len(history) != 2:
        raise ValueError(f"Expected two history rows in {run_dir}, got {len(history)}")
    validate_predictions(run_dir, int(metrics["val_samples"]))
    if not np.isfinite(history.select_dtypes(include=["number"]).to_numpy()).all():
        raise ValueError(f"Non-finite history value in {run_dir}")
    train_seconds = float(history["train_time_seconds"].sum())
    row = {
        "modality": metrics["modality"],
        "model_name": metrics["model_name"],
        "train_samples": int(metrics["train_samples"]),
        "val_samples": int(metrics["val_samples"]),
        "num_workers": int(metrics["num_workers"]),
        "batch_size": int(metrics["batch_size"]),
        "epochs_completed": int(metrics["epochs_completed"]),
        "best_epoch": int(metrics["best_epoch"]),
        "val_accuracy": float(metrics["val_accuracy"]),
        "val_macro_f1": float(metrics["val_macro_f1"]),
        "parameter_count": int(metrics["parameter_count"]),
        "checkpoint_size_mb": float(metrics["checkpoint_size_mb"]),
        "inference_ms_per_sample": float(metrics["inference_ms_per_sample"]),
        "epoch_1_seconds": float(history.iloc[0]["epoch_time_seconds"]),
        "epoch_2_seconds": float(history.iloc[1]["epoch_time_seconds"]),
        "mean_epoch_seconds": float(history["epoch_time_seconds"].mean()),
        "train_samples_per_second": int(metrics["train_samples"]) * len(history) / train_seconds,
        "gpu_peak_allocated_mb": float(metrics["gpu_memory_peak_mb"]),
        "gpu_peak_reserved_mb": float(metrics["gpu_memory_peak_reserved_mb"]),
        "config_path": metrics["config_path"],
        "output_dir": metrics["output_dir"],
        "status": metrics["status"],
    }
    if row["checkpoint_size_mb"] >= 95.0:
        raise ValueError(f"Checkpoint exceeds 95 MB in {run_dir}")
    return row


def load_old(modality: str) -> tuple[float, float, float]:
    run_dir = latest_run(OLD_OUTPUT_ROOT, modality)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    return (
        float(history["epoch_time_seconds"].mean()),
        float(metrics["val_accuracy"]),
        float(metrics["val_macro_f1"]),
    )


def main() -> None:
    rows = [load_new_row(modality) for modality in MODALITIES]
    pd.DataFrame(rows, columns=FIELDS).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    benchmark = pd.read_csv(PROJECT_ROOT / "reports" / "task03_input_pipeline_benchmark.csv", encoding="utf-8-sig")
    lines = [
        "# Task 03 optimized two-epoch report",
        "",
        "This is a two-epoch pipeline retest after input-pipeline optimization, not the formal 30-40 epoch training.",
        "No test data, pretrained weights, preprocessing cache, fold_1/fold_2, or multimodal fusion was used.",
        "",
        "## Old and optimized runs",
        "",
        "| Modality | Old workers | New workers | Old batch | New batch | Old mean epoch s | New mean epoch s | Speedup | Old accuracy | New accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for modality, row in zip(MODALITIES, rows, strict=True):
        old_epoch, old_accuracy, _ = load_old(modality)
        old_batch = 4 if modality in {"ir", "thermal", "depth_color"} else 16
        speedup = old_epoch / float(row["mean_epoch_seconds"])
        lines.append(
            f"| {DISPLAY[modality]} | 0 | {row['num_workers']} | {old_batch} | {row['batch_size']} | "
            f"{old_epoch:.2f} | {row['mean_epoch_seconds']:.2f} | {speedup:.2f}x | {old_accuracy:.6f} | {row['val_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Optimized results",
            "",
            "| Modality | Train/val | Accuracy | Macro-F1 | Train samples/s | GPU allocated/reserved MB | Checkpoint MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['modality']} | {row['train_samples']}/{row['val_samples']} | {row['val_accuracy']:.6f} | "
            f"{row['val_macro_f1']:.6f} | {row['train_samples_per_second']:.2f} | "
            f"{row['gpu_peak_allocated_mb']:.2f}/{row['gpu_peak_reserved_mb']:.2f} | {row['checkpoint_size_mb']:.3f} |"
        )
    depth = benchmark[
        (benchmark["modality"] == "Depth_Color")
        & (benchmark["num_workers"] == 4)
        & (benchmark["batch_size"] == 12)
    ].iloc[0]
    baseline = benchmark[
        (benchmark["modality"] == "Depth_Color")
        & (benchmark["num_workers"] == 0)
        & (benchmark["batch_size"] == 4)
    ].iloc[0]
    lines.extend(
        [
            "",
            "## Pipeline findings",
            "",
            f"Depth_Color steady-state training throughput increased from {baseline['samples_per_second']:.3f} to {depth['samples_per_second']:.3f} samples/s ({depth['samples_per_second'] / baseline['samples_per_second']:.2f}x).",
            "All tested worker configurations (0, 2, 4) exited normally, and all selected configurations completed without worker deadlock or CUDA OOM.",
            "The final visual batch is 12 because batch 16 was less than 5% faster while reserving more GPU memory. All temporal modalities retain batch 16.",
            "CPU use can remain high because image/CSV/JSON decoding is still performed online, but higher samples/s means GPU input waiting is substantially reduced.",
            "Task Manager CPU/GPU percentages and nvidia-smi process-memory observations are approximate manual snapshots and are recorded separately from the reproducible PyTorch peak metrics.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_CSV)
    print(REPORT_MD)


if __name__ == "__main__":
    main()
