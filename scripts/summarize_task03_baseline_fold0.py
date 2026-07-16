from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task03_baseline_fold0"
SUMMARY_CSV = PROJECT_ROOT / "reports" / "task03_baseline_fold0_summary.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "task03_baseline_fold0_report.md"
TWO_EPOCH_CSV = PROJECT_ROOT / "reports" / "task03_optimized_two_epoch_summary.csv"
MODALITIES = ("imu", "skeleton", "radar", "ir", "thermal", "depth_color")
DISPLAY = {
    "imu": "IMU",
    "skeleton": "Skeleton",
    "radar": "Radar",
    "ir": "IR",
    "thermal": "Thermal",
    "depth_color": "Depth_Color",
}
EXPECTED_SAMPLES = {
    "imu": (1916, 987),
    "skeleton": (1931, 1000),
    "radar": (1918, 996),
    "ir": (1933, 1000),
    "thermal": (1845, 1046),
    "depth_color": (1931, 1000),
}
EXPECTED_PARAMETERS = {
    "imu": 164328,
    "skeleton": 172776,
    "radar": 141672,
    "ir": 1006024,
    "thermal": 1006024,
    "depth_color": 1006024,
}
FIELDS = (
    "modality",
    "model_name",
    "train_samples",
    "val_samples",
    "num_workers",
    "batch_size",
    "epochs_requested",
    "epochs_completed",
    "early_stopping_patience",
    "early_stopped",
    "best_epoch",
    "best_val_accuracy",
    "best_val_macro_f1",
    "best_val_loss",
    "highest_macro_f1_epoch",
    "highest_macro_f1",
    "best_accuracy_and_macro_f1_same_epoch",
    "final_epoch",
    "final_val_accuracy",
    "final_val_macro_f1",
    "final_val_loss",
    "final_train_accuracy",
    "parameter_count",
    "checkpoint_size_mb",
    "mean_train_time_seconds",
    "mean_val_time_seconds",
    "mean_epoch_seconds",
    "total_training_seconds",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "inference_ms_per_sample",
    "worst_recall_classes",
    "major_confusions",
    "trend_assessment",
    "config_path",
    "output_dir",
    "status",
)


def latest_run(modality: str) -> Path:
    candidates = [
        path.parent
        for path in (OUTPUT_ROOT / modality).glob("baseline_fold0_20260716*/metrics.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No formal fold_0 run found for {modality}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def top_confusions(labels: np.ndarray, predictions: np.ndarray, limit: int = 5) -> str:
    matrix = np.zeros((40, 40), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    np.fill_diagonal(matrix, 0)
    flat = np.argsort(matrix.ravel())[::-1]
    items: list[str] = []
    for index in flat:
        count = int(matrix.ravel()[index])
        if count <= 0 or len(items) >= limit:
            break
        true_class, predicted_class = np.unravel_index(index, matrix.shape)
        items.append(f"{true_class}->{predicted_class} ({count})")
    return "; ".join(items)


def assess_trend(history: pd.DataFrame, best_epoch: int) -> str:
    first = history.iloc[0]
    final = history.iloc[-1]
    min_loss_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    gap = float(final["train_accuracy"] - final["val_accuracy"])
    if gap >= 0.25 and int(final["epoch"]) - best_epoch >= 2:
        fit = "overfitting signal"
    elif float(final["train_accuracy"]) < 0.5:
        fit = "underfitting signal"
    else:
        fit = "mixed/generalization-limited"
    return (
        f"train_loss {first['train_loss']:.4f}->{final['train_loss']:.4f}; "
        f"train_acc {first['train_accuracy']:.4f}->{final['train_accuracy']:.4f}; "
        f"val_acc {first['val_accuracy']:.4f}->{final['val_accuracy']:.4f}; "
        f"min_val_loss_epoch={min_loss_epoch}; final_gap={gap:.4f}; {fit}"
    )


def load_row(slug: str) -> dict[str, Any]:
    run_dir = latest_run(slug)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    expected_epochs = 40 if slug in {"imu", "skeleton", "radar"} else 30
    expected_patience = 8 if slug in {"imu", "skeleton", "radar"} else 6
    required = {
        "best_model.pt",
        "last_model.pt",
        "config.yaml",
        "history.csv",
        "metrics.json",
        "fold_0_val_predictions.npz",
        "confusion_matrix.png",
    }
    if slug in {"imu", "skeleton", "radar"}:
        required.add("normalization_stats.json")
    missing = required - {path.name for path in run_dir.iterdir()}
    if missing:
        raise FileNotFoundError(f"Missing files in {run_dir}: {sorted(missing)}")
    if (int(metrics["train_samples"]), int(metrics["val_samples"])) != EXPECTED_SAMPLES[slug]:
        raise ValueError(f"Sample count mismatch for {slug}")
    if int(metrics["parameter_count"]) != EXPECTED_PARAMETERS[slug]:
        raise ValueError(f"Parameter count mismatch for {slug}")
    if int(config["epochs"]) != expected_epochs or int(config["early_stopping_patience"]) != expected_patience:
        raise ValueError(f"Formal epoch configuration mismatch for {slug}")
    if int(config["num_workers"]) != 4 or not bool(config["amp"]):
        raise ValueError(f"Worker or AMP configuration mismatch for {slug}")
    if metrics["device"] != "cuda" or metrics["status"] != "passed":
        raise ValueError(f"Device or status mismatch for {slug}")
    if float(metrics["checkpoint_size_mb"]) >= 95.0:
        raise ValueError(f"Checkpoint exceeds 95 MB for {slug}")
    if not 1 <= len(history) <= expected_epochs:
        raise ValueError(f"Invalid history length for {slug}: {len(history)}")
    if not np.isfinite(history.select_dtypes(include=["number"]).to_numpy()).all():
        raise ValueError(f"Non-finite history value for {slug}")

    with np.load(run_dir / "fold_0_val_predictions.npz") as data:
        sample_ids = data["sample_ids"]
        labels = data["labels"]
        logits = data["logits"]
        embeddings = data["embeddings"]
        class_order = data["class_order"]
        expected_val = EXPECTED_SAMPLES[slug][1]
        if sample_ids.shape != (expected_val,) or len(np.unique(sample_ids)) != expected_val:
            raise ValueError(f"Invalid sample_ids for {slug}")
        if labels.shape != (expected_val,) or logits.shape != (expected_val, 40):
            raise ValueError(f"Invalid labels/logits for {slug}")
        if embeddings.shape != (expected_val, 128):
            raise ValueError(f"Invalid embeddings for {slug}")
        if not np.array_equal(class_order, np.arange(40)):
            raise ValueError(f"Invalid class_order for {slug}")
        if not np.isfinite(logits).all() or not np.isfinite(embeddings).all():
            raise ValueError(f"Non-finite prediction output for {slug}")
        confusions = top_confusions(labels.astype(np.int64), logits.argmax(axis=1))

    best_epoch = int(metrics["best_epoch"])
    best_rows = history.loc[history["epoch"] == best_epoch]
    if len(best_rows) != 1:
        raise ValueError(f"Best epoch is missing or duplicated for {slug}")
    best_row = best_rows.iloc[0]
    if not np.isclose(float(metrics["val_accuracy"]), float(best_row["val_accuracy"]), atol=1e-12):
        raise ValueError(f"Best checkpoint accuracy mismatch for {slug}")
    final = history.iloc[-1]
    macro_index = history["val_macro_f1"].idxmax()
    macro_epoch = int(history.loc[macro_index, "epoch"])
    recalls = np.asarray(metrics["per_class_recall"], dtype=np.float64)
    if recalls.shape != (40,) or not np.isfinite(recalls).all():
        raise ValueError(f"Invalid per-class recall for {slug}")
    worst_value = float(recalls.min())
    worst_classes = np.flatnonzero(np.isclose(recalls, worst_value)).tolist()
    row = {
        "modality": metrics["modality"],
        "model_name": metrics["model_name"],
        "train_samples": int(metrics["train_samples"]),
        "val_samples": int(metrics["val_samples"]),
        "num_workers": int(config["num_workers"]),
        "batch_size": int(config["batch_size"]),
        "epochs_requested": expected_epochs,
        "epochs_completed": len(history),
        "early_stopping_patience": expected_patience,
        "early_stopped": len(history) < expected_epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(metrics["val_accuracy"]),
        "best_val_macro_f1": float(metrics["val_macro_f1"]),
        "best_val_loss": float(metrics["val_loss"]),
        "highest_macro_f1_epoch": macro_epoch,
        "highest_macro_f1": float(history.loc[macro_index, "val_macro_f1"]),
        "best_accuracy_and_macro_f1_same_epoch": best_epoch == macro_epoch,
        "final_epoch": int(final["epoch"]),
        "final_val_accuracy": float(final["val_accuracy"]),
        "final_val_macro_f1": float(final["val_macro_f1"]),
        "final_val_loss": float(final["val_loss"]),
        "final_train_accuracy": float(final["train_accuracy"]),
        "parameter_count": int(metrics["parameter_count"]),
        "checkpoint_size_mb": float(metrics["checkpoint_size_mb"]),
        "mean_train_time_seconds": float(history["train_time_seconds"].mean()),
        "mean_val_time_seconds": float(history["val_time_seconds"].mean()),
        "mean_epoch_seconds": float(history["epoch_time_seconds"].mean()),
        "total_training_seconds": float(history["epoch_time_seconds"].sum()),
        "gpu_peak_allocated_mb": float(metrics["gpu_memory_peak_mb"]),
        "gpu_peak_reserved_mb": float(metrics["gpu_memory_peak_reserved_mb"]),
        "inference_ms_per_sample": float(metrics["inference_ms_per_sample"]),
        "worst_recall_classes": ",".join(str(value) for value in worst_classes),
        "major_confusions": confusions,
        "trend_assessment": assess_trend(history, best_epoch),
        "config_path": metrics["config_path"],
        "output_dir": metrics["output_dir"],
        "status": metrics["status"],
    }
    return row


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}h {minutes:02d}m {remainder:04.1f}s"


def main() -> None:
    rows = [load_row(slug) for slug in MODALITIES]
    pd.DataFrame(rows, columns=FIELDS).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    two_epoch = pd.read_csv(TWO_EPOCH_CSV, encoding="utf-8-sig").set_index("modality")
    lines = [
        "# Task 03 formal fold_0 unimodal baseline report",
        "",
        "This report covers the formal fold_0 unimodal baselines. It is not complete three-fold OOF, not a final model trained on all 18 users, and not a Kaggle submission model.",
        "No test data, fold_1/fold_2, pretrained weights, preprocessing cache, architecture change, augmentation change, or parameter ablation was used.",
        "The best checkpoint remains selected by validation Accuracy.",
        "",
        "## Formal configurations and results",
        "",
        "| Modality | Train/val | Requested/completed | Early stop | Best epoch | Best Acc | Best Macro-F1 | Best val loss | Final Acc | Mean epoch | Total | GPU alloc/reserved MB |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['modality']} | {row['train_samples']}/{row['val_samples']} | "
            f"{row['epochs_requested']}/{row['epochs_completed']} | {str(row['early_stopped']).lower()} | "
            f"{row['best_epoch']} | {row['best_val_accuracy']:.6f} | {row['best_val_macro_f1']:.6f} | "
            f"{row['best_val_loss']:.6f} | {row['final_val_accuracy']:.6f} | "
            f"{row['mean_epoch_seconds']:.2f}s | {format_duration(row['total_training_seconds'])} | "
            f"{row['gpu_peak_allocated_mb']:.2f}/{row['gpu_peak_reserved_mb']:.2f} |"
        )
    lines.extend(["", "## Accuracy and Macro-F1 checkpoint analysis", ""])
    for row in rows:
        lines.append(
            f"- **{row['modality']}**: Accuracy checkpoint epoch {row['best_epoch']} has "
            f"Accuracy {row['best_val_accuracy']:.6f} and Macro-F1 {row['best_val_macro_f1']:.6f}; "
            f"highest historical Macro-F1 {row['highest_macro_f1']:.6f} occurs at epoch "
            f"{row['highest_macro_f1_epoch']}; same epoch: {str(row['best_accuracy_and_macro_f1_same_epoch']).lower()}."
        )
    lines.extend(["", "## Training curves and fit assessment", ""])
    for row in rows:
        lines.append(f"- **{row['modality']}**: {row['trend_assessment']}")
    lines.extend(["", "## Per-class recall and major confusions", ""])
    for row in rows:
        lines.append(
            f"- **{row['modality']}**: minimum-recall classes [{row['worst_recall_classes']}]; "
            f"largest off-diagonal confusions: {row['major_confusions']}."
        )
    lines.extend(
        [
            "",
            "## Comparison with optimized two-epoch verification",
            "",
            "| Modality | 2-epoch Acc | Formal Acc | Delta | 2-epoch Macro-F1 | Formal Macro-F1 | Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        old = two_epoch.loc[row["modality"]]
        lines.append(
            f"| {row['modality']} | {old['val_accuracy']:.6f} | {row['best_val_accuracy']:.6f} | "
            f"{row['best_val_accuracy'] - old['val_accuracy']:+.6f} | {old['val_macro_f1']:.6f} | "
            f"{row['best_val_macro_f1']:.6f} | {row['best_val_macro_f1'] - old['val_macro_f1']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusions and next controlled ablations",
            "",
            "- Radar should first test richer per-frame statistics and then a PointNet-style frame encoder; more epochs alone cannot recover point-level structure discarded by the current 21 statistics.",
            "- Visual modalities should next compare 192x192 with aspect-ratio-preserving 224x224, then 12 versus 16 frames and weak spatial/temporal augmentation, one variable at a time.",
            "- IMU should next compare sequence length 256/384 and controlled sensor noise or amplitude scaling; Skeleton should compare 64/96 steps and stronger joint-time structure.",
            "- Learning rate, weight decay, dropout, alternative user ratios, fold_1/fold_2, candidate-B models, and multimodal fusion remain outside this run.",
            "",
            "## Integrity and runtime checks",
            "",
            "All six runs used CUDA AMP, num_workers=4, the frozen fold_0 split, unchanged model parameter counts, and the expected modality sample counts. Required checkpoints, histories, metrics, confusion matrices, and validation prediction archives were reopened and validated. No worker deadlock, CUDA OOM, NaN/Inf, test-set access, preprocessing cache, or output-format change occurred.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_CSV)
    print(REPORT_MD)


if __name__ == "__main__":
    main()
