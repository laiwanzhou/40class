from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "E0 global-only": PROJECT_ROOT / "outputs/depth_hard_global_expert_fold0/depth_color/depth_hard_global_expert_fold0",
    "E1 pose-ROI": PROJECT_ROOT / "outputs/depth_pose_roi_expert_fold0/depth_color/depth_pose_roi_expert_fold0",
}
SUMMARY_PATH = PROJECT_ROOT / "reports/depth_pose_roi_experiment_summary.csv"
REPORT_PATH = PROJECT_ROOT / "reports/depth_pose_roi_experiment.md"
TARGET_PAIRS = (
    ("Take_medicine", "Eat_food"),
    ("Take_medicine", "Drink_water"),
    ("Watch_TV", "Play_games"),
    ("Make_a_phone_call", "Take_a_selfie"),
    ("Take_body_temperature", "Make_a_phone_call"),
    ("Stir_drinks", "Wipe_bowls"),
)


def duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {remainder:04.1f}s"


def load_run(name: str, run_dir: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = {
        "best_model.pt", "last_model.pt", "config.yaml", "history.csv", "metrics.json",
        "fold_0_val_predictions.npz", "confusion_matrix.png", "per_class_metrics.csv",
    }
    if name.startswith("E1"):
        required.add("roi_attention_summary.csv")
    missing = required - {path.name for path in run_dir.iterdir()}
    if missing:
        raise FileNotFoundError(f"Missing {name} artifacts: {sorted(missing)}")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    classes = pd.read_csv(run_dir / "per_class_metrics.csv", encoding="utf-8-sig")
    with np.load(run_dir / "fold_0_val_predictions.npz") as predictions:
        details = {
            "labels": predictions["labels"].copy(),
            "predicted": predictions["logits"].argmax(axis=1),
            "action_names": predictions["action_names"].astype(str),
        }
    return metrics, history, classes, details


def confusion_rate(details: dict[str, object], source: str, target: str) -> tuple[int, int, float]:
    names = np.asarray(details["action_names"])
    source_label = int(np.flatnonzero(names == source)[0])
    target_label = int(np.flatnonzero(names == target)[0])
    labels = np.asarray(details["labels"])
    predicted = np.asarray(details["predicted"])
    total = int((labels == source_label).sum())
    count = int(((labels == source_label) & (predicted == target_label)).sum())
    return count, total, count / total if total else 0.0


def main() -> None:
    loaded = {name: load_run(name, path) for name, path in RUNS.items()}
    rows: list[dict[str, object]] = []
    for name, run_dir in RUNS.items():
        metrics, history, classes, _ = loaded[name]
        total_seconds = float(history["epoch_time_seconds"].sum())
        rows.append(
            {
                "experiment": name,
                "train_samples": metrics["train_samples"],
                "validation_samples": metrics["val_samples"],
                "best_epoch": metrics["best_epoch"],
                "epochs_completed": metrics["epochs_completed"],
                "accuracy": metrics["val_accuracy"],
                "macro_f1": metrics["val_macro_f1"],
                "weighted_f1": metrics["val_weighted_f1"],
                "validation_loss": metrics["val_loss"],
                "zero_f1_classes": int((classes["f1"] == 0).sum()),
                "gpu_peak_allocated_mb": metrics["gpu_memory_peak_mb"],
                "gpu_peak_reserved_mb": metrics["gpu_memory_peak_reserved_mb"],
                "mean_epoch_seconds": float(history["epoch_time_seconds"].mean()),
                "total_epoch_seconds": total_seconds,
                "checkpoint_size_bytes": (run_dir / "best_model.pt").stat().st_size,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    e0_metrics, e0_history, e0_classes, e0_details = loaded["E0 global-only"]
    e1_metrics, e1_history, e1_classes, e1_details = loaded["E1 pose-ROI"]
    class_comparison = e0_classes[["original_class_id", "action_name", "f1"]].rename(columns={"f1": "e0_f1"}).merge(
        e1_classes[["original_class_id", "action_name", "precision", "recall", "f1", "support"]].rename(columns={"f1": "e1_f1"}),
        on=["original_class_id", "action_name"], validate="one_to_one"
    )
    class_comparison["f1_delta"] = class_comparison["e1_f1"] - class_comparison["e0_f1"]
    attention = pd.read_csv(RUNS["E1 pose-ROI"] / "roi_attention_summary.csv", encoding="utf-8-sig")
    attention_means = attention[["upper_body_mean_weight", "left_hand_mean_weight", "right_hand_mean_weight"]].mean()
    attention_entropy = float(attention["attention_entropy"].mean())
    fallback = pd.read_csv(PROJECT_ROOT / "outputs/depth_pose_roi_probe/roi_fallback_summary.csv", encoding="utf-8-sig")
    upper_fallback = float(fallback.loc[(fallback["view"] == "upper_body") & (fallback["source"] != "keypoints"), "source_ratio"].sum())
    hand_fallback = float(fallback.loc[fallback["view"].isin(["left_hand", "right_hand"]) & ~fallback["source"].isin(["wrist", "interpolated_wrist"]), "frames"].sum() / fallback.loc[fallback["view"].isin(["left_hand", "right_hand"]), "frames"].sum())
    quality = pd.read_csv(PROJECT_ROOT / "outputs/depth_pose_roi_probe/roi_quality.csv", encoding="utf-8-sig")
    area_means = quality.groupby("view")["area_ratio"].mean()
    cache_summary = json.loads((PROJECT_ROOT / "outputs/depth_pose_roi_probe/pose_cache_summary.json").read_text(encoding="utf-8"))
    pose_bytes = int(cache_summary["weight_bytes"])
    classifier_bytes = (RUNS["E1 pose-ROI"] / "best_model.pt").stat().st_size
    combined_bytes = pose_bytes + classifier_bytes
    improved = class_comparison[class_comparison["f1_delta"] > 1e-12]
    declined = class_comparison[class_comparison["f1_delta"] < -1e-12]
    macro_delta = float(e1_metrics["val_macro_f1"] - e0_metrics["val_macro_f1"])
    zero_delta = int((e1_classes["f1"] == 0).sum() - (e0_classes["f1"] == 0).sum())
    valuable = macro_delta > 0.02 and len(improved) >= 3 and zero_delta < 0

    class_lines = [
        f"| {row.action_name} | {row.e0_f1:.6f} | {row.e1_f1:.6f} | {row.f1_delta:+.6f} | {row.precision:.6f} | {row.recall:.6f} | {int(row.support)} |"
        for row in class_comparison.sort_values("original_class_id").itertuples()
    ]
    confusion_lines = []
    for source, target in TARGET_PAIRS:
        e0_count, total, e0_rate = confusion_rate(e0_details, source, target)
        e1_count, _, e1_rate = confusion_rate(e1_details, source, target)
        confusion_lines.append(f"| {source} -> {target} | {e0_count}/{total} ({e0_rate:.2%}) | {e1_count}/{total} ({e1_rate:.2%}) | {e1_rate - e0_rate:+.2%} |")
    lines = [
        "# Pose-guided ROI hard action visual expert",
        "",
        "## Matched experiment",
        "",
        "E0 and E1 use the same 11 hard classes, fold_0 14/4 users, continuous 24-frame windows, ImageNet-pretrained MobileNetV3-Small, frame projection, single-layer GRU, optimizer, seed, image size, batch size, epoch limit, and early stopping. E1 alone receives the upper-body, left-hand, and right-hand Depth_Color crops localized from aligned IR pose coordinates.",
        "",
        "| Experiment | Train/val | Best epoch | Completed | Accuracy | Macro-F1 | Weighted F1 | Val loss | Zero-F1 | GPU MB alloc/reserved | Mean epoch | Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['train_samples']}/{row['validation_samples']} | {row['best_epoch']} | {row['epochs_completed']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['weighted_f1']:.6f} | {row['validation_loss']:.6f} | {row['zero_f1_classes']} | {row['gpu_peak_allocated_mb']:.2f}/{row['gpu_peak_reserved_mb']:.2f} | {row['mean_epoch_seconds']:.2f}s | {duration(row['total_epoch_seconds'])} |")
    lines.extend([
        "",
        "## Primary comparison",
        "",
        f"- E1 - E0 Accuracy: {e1_metrics['val_accuracy'] - e0_metrics['val_accuracy']:+.6f}.",
        f"- E1 - E0 Macro-F1: {macro_delta:+.6f}.",
        f"- E1 - E0 weighted F1: {e1_metrics['val_weighted_f1'] - e0_metrics['val_weighted_f1']:+.6f}.",
        f"- E1 - E0 zero-F1 class count: {zero_delta:+d}.",
        f"- Improved classes ({len(improved)}): {', '.join(improved['action_name']) or 'none'}.",
        f"- Declined classes ({len(declined)}): {', '.join(declined['action_name']) or 'none'}.",
        "",
        "## Per-class metrics",
        "",
        "| Action | E0 F1 | E1 F1 | Delta | E1 precision | E1 recall | Support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *class_lines,
        "",
        "## Target confusion directions",
        "",
        "| Direction | E0 | E1 | Rate delta |",
        "| --- | ---: | ---: | ---: |",
        *confusion_lines,
        "",
        "## ROI quality and gating",
        "",
        f"- Full hard-subset IR cache: person {cache_summary['person_success']:.2%}, left wrist {cache_summary['left_wrist_success']:.2%}, right wrist {cache_summary['right_wrist_success']:.2%}, at least one wrist {cache_summary['at_least_one_wrist_success']:.2%}.",
        f"- Upper-body fallback rate: {upper_fallback:.2%}; combined hand fallback rate: {hand_fallback:.2%}.",
        f"- Mean ROI area ratios: upper {area_means['upper_body']:.2%}, left hand {area_means['left_hand']:.2%}, right hand {area_means['right_hand']:.2%}.",
        f"- Validation ROI gate means: upper {attention_means['upper_body_mean_weight']:.6f}, left hand {attention_means['left_hand_mean_weight']:.6f}, right hand {attention_means['right_hand_mean_weight']:.6f}; entropy {attention_entropy:.6f} versus uniform ln(3)={math.log(3):.6f}.",
        "",
        "## Inference weights",
        "",
        f"- YOLO11n-pose locator: {pose_bytes} bytes ({pose_bytes / 1024**2:.2f} MiB).",
        f"- E1 classifier checkpoint, including MobileNet and GRU: {classifier_bytes} bytes ({classifier_bytes / 1024**2:.2f} MiB).",
        f"- Total required inference weights: {combined_bytes} bytes ({combined_bytes / 1024**2:.2f} MiB), below 100 MB.",
        "",
        "## Decision",
        "",
        ("The route meets the predefined value criteria and is a candidate for guarded integration into the existing fusion system." if valuable else "The route does not meet all predefined value criteria. Do not integrate it into the existing fusion system without a new, separately justified experiment."),
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
