from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARD_10 = (
    "Take_medicine",
    "Stir_drinks",
    "Take_a_selfie",
    "Take_body_temperature",
    "Make_a_phone_call",
    "Watch_TV",
    "Wipe_bowls",
    "Turn_pages",
    "Eat_food",
    "Drink_water",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/depth_six_patch_fold0_14train_4val.md",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "reports/depth_six_patch_fold0_14train_4val_summary.csv",
    )
    return parser.parse_args()


def duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {remainder:04.1f}s"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    per_class = pd.read_csv(run_dir / "per_class_metrics.csv", encoding="utf-8-sig")
    attention = pd.read_csv(run_dir / "patch_attention_summary.csv", encoding="utf-8-sig")
    probe = json.loads(
        (PROJECT_ROOT / "outputs/depth_six_patch_fold0_14train_4val/memory_probe.json").read_text(
            encoding="utf-8"
        )
    )
    required = {
        "best_model.pt", "last_model.pt", "config.yaml", "history.csv", "metrics.json",
        "fold_0_val_predictions.npz", "confusion_matrix.png", "per_class_metrics.csv",
        "patch_attention_summary.csv",
    }
    missing = required - {path.name for path in run_dir.iterdir()}
    if missing:
        raise FileNotFoundError(f"Missing formal artifacts: {sorted(missing)}")
    hard = per_class[per_class["action_name"].isin(HARD_10)].copy()
    if set(hard["action_name"]) != set(HARD_10):
        raise ValueError("Hard-10 class mapping is incomplete.")
    hard_macro_f1 = float(hard["f1"].mean())
    hard_zero = int((hard["f1"] == 0).sum())
    patch_columns = [f"patch_{index}_mean_weight" for index in range(1, 7)]
    patch_means = attention[patch_columns].mean()
    mean_entropy = float(attention["attention_entropy"].mean())
    max_entropy = math.log(6.0)
    total_seconds = float(history["epoch_time_seconds"].sum())
    mean_epoch_seconds = float(history["epoch_time_seconds"].mean())
    best_epoch = int(metrics["best_epoch"])
    best_history = history.loc[history["epoch"] == best_epoch].iloc[0]
    baseline_metrics_path = (
        PROJECT_ROOT
        / "outputs/task03_baseline_fold0/depth_color/baseline_fold0_20260716_depth_color/metrics.json"
    )
    comparison_note = "No ordinary Depth_Color run on the same 14/4 fold was found."
    if baseline_metrics_path.exists():
        baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        comparison_note += (
            f" The available baseline used {baseline['train_samples']}/{baseline['val_samples']} "
            f"train/validation samples, versus {metrics['train_samples']}/{metrics['val_samples']} here; "
            "its old 12/6 values are not used as evidence or deltas."
        )
    dominant_patch = int(np.argmax(patch_means.to_numpy())) + 1
    dominance_gap = float(patch_means.max() - 1.0 / 6.0)
    uniformity = mean_entropy / max_entropy
    recommendation = (
        "No. The fixed-patch result should first be compared against a newly trained ordinary Depth_Color "
        "baseline on the identical 14/4 fold; adding motion Top-2 now would confound the conclusion."
    )
    row = {
        "branch_experiment": "depth-six-patch",
        "train_samples": metrics["train_samples"],
        "val_samples": metrics["val_samples"],
        "batch_size": metrics["batch_size"],
        "probe_gpu_peak_allocated_mb": probe["gpu_peak_allocated_mb"],
        "probe_gpu_peak_reserved_mb": probe["gpu_peak_reserved_mb"],
        "formal_gpu_peak_allocated_mb": metrics["gpu_memory_peak_mb"],
        "formal_gpu_peak_reserved_mb": metrics["gpu_memory_peak_reserved_mb"],
        "epochs_completed": metrics["epochs_completed"],
        "best_epoch": best_epoch,
        "accuracy": metrics["val_accuracy"],
        "macro_f1": metrics["val_macro_f1"],
        "weighted_f1": metrics["val_weighted_f1"],
        "validation_loss": metrics["val_loss"],
        "hard_10_macro_f1": hard_macro_f1,
        "hard_10_zero_f1_classes": hard_zero,
        **{f"patch_{index}_mean_weight": patch_means.iloc[index - 1] for index in range(1, 7)},
        "mean_attention_entropy": mean_entropy,
        "max_uniform_entropy": max_entropy,
        "mean_epoch_seconds": mean_epoch_seconds,
        "total_epoch_seconds": total_seconds,
        "same_fold_baseline_available": False,
        "accuracy_delta_vs_same_fold_baseline": np.nan,
        "macro_f1_delta_vs_same_fold_baseline": np.nan,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.summary, index=False, encoding="utf-8-sig")
    hard_lines = [
        f"| {entry.action_name} | {entry.f1:.6f} | {int(entry.support)} |"
        for entry in hard.sort_values("class_id").itertuples()
    ]
    patch_text = " / ".join(f"P{i} {patch_means.iloc[i-1]:.6f}" for i in range(1, 7))
    lines = [
        "# Depth_Color global + six fixed overlapping patches (fold_0 14/4)",
        "",
        "## Result",
        "",
        f"- Formal split: {metrics['train_samples']} train / {metrics['val_samples']} validation samples; no test data read.",
        f"- Completed {metrics['epochs_completed']} of 30 requested epochs (patience 6); best Accuracy checkpoint: epoch {best_epoch}.",
        f"- Accuracy: {metrics['val_accuracy']:.6f}; Macro-F1: {metrics['val_macro_f1']:.6f}; weighted F1: {metrics['val_weighted_f1']:.6f}; validation loss: {metrics['val_loss']:.6f}.",
        f"- Best-epoch history row: Accuracy {best_history['val_accuracy']:.6f}, Macro-F1 {best_history['val_macro_f1']:.6f}, loss {best_history['val_loss']:.6f}.",
        f"- Hard-10 Macro-F1: {hard_macro_f1:.6f}; zero-F1 classes: {hard_zero}/10.",
        "",
        "## Runtime",
        "",
        f"- Actual batch size: {metrics['batch_size']} (no CUDA OOM; no gradient accumulation used).",
        f"- Real optimizer-step probe peak: {probe['gpu_peak_allocated_mb']:.2f} MB allocated / {probe['gpu_peak_reserved_mb']:.2f} MB reserved.",
        f"- Formal run peak: {metrics['gpu_memory_peak_mb']:.2f} MB allocated / {metrics['gpu_memory_peak_reserved_mb']:.2f} MB reserved.",
        f"- Mean epoch time: {mean_epoch_seconds:.2f}s; epoch-loop total: {duration(total_seconds)}.",
        "",
        "## Patch attention",
        "",
        f"- Validation mean weights: {patch_text}.",
        f"- Mean entropy: {mean_entropy:.6f}; uniform maximum ln(6): {max_entropy:.6f}; ratio: {uniformity:.2%}.",
        f"- Highest mean is patch {dominant_patch}, only {dominance_gap:+.6f} above uniform 1/6. "
        "This does not indicate a persistent single-patch collapse." if dominance_gap < 0.05 else
        f"- Patch {dominant_patch} is {dominance_gap:+.6f} above uniform 1/6, indicating a persistent spatial preference that should be inspected for background bias.",
        "- Fixed layout: P1/P2/P3 are top left/center/right; P4/P5/P6 are bottom left/center/right.",
        "",
        "## Hard-10",
        "",
        "| Class | F1 | Support |",
        "| --- | ---: | ---: |",
        *hard_lines,
        "",
        "## Baseline comparison",
        "",
        comparison_note,
        "Therefore Accuracy, Macro-F1, weighted-F1, validation-loss, per-class improvement/decline, and zero-F1 deltas versus a same-fold baseline are reported as N/A rather than mixing folds.",
        "",
        "## Decision",
        "",
        recommendation,
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary)
    print(args.report)


if __name__ == "__main__":
    main()
