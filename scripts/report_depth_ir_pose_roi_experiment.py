from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUAL_DIR = PROJECT_ROOT / "outputs/depth_ir_pose_roi_expert_fold0/depth_ir/depth_ir_pose_roi_expert_fold0_14train_4val"
COMMON_DIR = PROJECT_ROOT / "outputs/depth_ir_pose_roi_expert_fold0/depth_color/depth_pose_roi_expert_common_subset_fold0_14train_4val"
OLD_E1_DIR = PROJECT_ROOT / "outputs/depth_pose_roi_expert_fold0/depth_color/depth_pose_roi_expert_fold0"
REPORT_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_experiment.md"
SUMMARY_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_experiment_summary.csv"
PER_CLASS_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_per_class_comparison.csv"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def confusion_count(labels: np.ndarray, predictions: np.ndarray, source: int, target: int) -> tuple[int, int]:
    selected = labels == source
    return int(np.sum(selected & (predictions == target))), int(selected.sum())


def main() -> None:
    dual = load_json(DUAL_DIR / "metrics.json")
    common = load_json(COMMON_DIR / "metrics.json")
    old = load_json(OLD_E1_DIR / "metrics.json")
    ablations = load_json(DUAL_DIR / "ir_ablation_metrics.json")["modes"]
    probe = load_json(PROJECT_ROOT / "outputs/depth_ir_pose_roi_probe/initialization_and_batch_probe.json")
    dual_history = pd.read_csv(DUAL_DIR / "history.csv")
    common_history = pd.read_csv(COMMON_DIR / "history.csv")
    dual_class = pd.read_csv(DUAL_DIR / "per_class_metrics.csv")
    common_class = pd.read_csv(COMMON_DIR / "per_class_metrics.csv")
    ablation_class = pd.read_csv(DUAL_DIR / "ir_ablation_per_class.csv")
    masked = ablation_class[ablation_class["mode"] == "masked"][["expert_label", "f1", "recall"]].rename(columns={"f1": "masked_f1", "recall": "masked_recall"})
    shuffled = ablation_class[ablation_class["mode"] == "shuffled"][["expert_label", "f1", "recall"]].rename(columns={"f1": "shuffled_f1", "recall": "shuffled_recall"})
    comparison = common_class.merge(
        dual_class,
        on=["expert_label", "original_class_id", "action_name", "support"],
        suffixes=("_depth_only", "_depth_ir"),
    ).merge(masked, on="expert_label").merge(shuffled, on="expert_label")
    comparison["f1_delta_depth_ir_minus_depth_only"] = comparison["f1_depth_ir"] - comparison["f1_depth_only"]
    comparison["recall_delta_depth_ir_minus_depth_only"] = comparison["recall_depth_ir"] - comparison["recall_depth_only"]
    comparison.to_csv(PER_CLASS_PATH, index=False, encoding="utf-8-sig")

    summary_rows = [
        {"experiment": "Depth-only pose ROI common subset", "comparison_role": "strict baseline", **{key: common[key] for key in ("train_samples", "val_samples", "best_epoch", "epochs_completed", "val_accuracy", "val_macro_f1", "val_weighted_f1", "val_loss", "parameter_count", "checkpoint_size_mb")}},
        {"experiment": "Depth+IR dual-stem pose ROI", "comparison_role": "V2 normal paired", **{key: dual[key] for key in ("train_samples", "val_samples", "best_epoch", "epochs_completed", "val_accuracy", "val_macro_f1", "val_weighted_f1", "val_loss", "parameter_count", "checkpoint_size_mb")}},
        {"experiment": "Depth+IR with IR masked", "comparison_role": "same-checkpoint diagnostic", "train_samples": dual["train_samples"], "val_samples": dual["val_samples"], "best_epoch": dual["best_epoch"], "epochs_completed": dual["epochs_completed"], "val_accuracy": ablations["masked"]["accuracy"], "val_macro_f1": ablations["masked"]["macro_f1"], "val_weighted_f1": ablations["masked"]["weighted_f1"], "val_loss": ablations["masked"]["loss"], "parameter_count": dual["parameter_count"], "checkpoint_size_mb": dual["checkpoint_size_mb"]},
        {"experiment": "Depth+IR with IR sample-shuffled", "comparison_role": "same-checkpoint diagnostic", "train_samples": dual["train_samples"], "val_samples": dual["val_samples"], "best_epoch": dual["best_epoch"], "epochs_completed": dual["epochs_completed"], "val_accuracy": ablations["shuffled"]["accuracy"], "val_macro_f1": ablations["shuffled"]["macro_f1"], "val_weighted_f1": ablations["shuffled"]["weighted_f1"], "val_loss": ablations["shuffled"]["loss"], "parameter_count": dual["parameter_count"], "checkpoint_size_mb": dual["checkpoint_size_mb"]},
        {"experiment": "Old E1 pose ROI", "comparison_role": "context only; non-equivalent 596/148", **{key: old[key] for key in ("train_samples", "val_samples", "best_epoch", "epochs_completed", "val_accuracy", "val_macro_f1", "val_weighted_f1", "val_loss", "parameter_count", "checkpoint_size_mb")}},
    ]
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    with np.load(DUAL_DIR / "fold_0_val_predictions.npz") as data:
        labels = data["labels"]
        dual_predictions = data["logits"].argmax(axis=1)
        gates = data["modality_gate"].astype(np.float64)
        masks = data["temporal_mask"].astype(bool)
        roi_attention = data["roi_attention"].astype(np.float64)
    with np.load(COMMON_DIR / "fold_0_val_predictions.npz") as data:
        common_predictions = data["logits"].argmax(axis=1)
        common_labels = data["labels"]
    if not np.array_equal(labels, common_labels):
        raise ValueError("Strict baseline and V2 validation labels differ.")
    gate_weights = masks[..., None]
    view_gate = (gates * gate_weights).sum(axis=(0, 1)) / gate_weights.sum()
    gate_entropy = -gates * np.log(np.clip(gates, 1e-12, 1)) - (1 - gates) * np.log(np.clip(1 - gates, 1e-12, 1))
    mean_gate_entropy = float((gate_entropy * gate_weights).sum() / (gate_weights.sum() * gates.shape[2]))
    roi_weights = masks[..., None]
    mean_roi = (roi_attention * roi_weights).sum(axis=(0, 1)) / roi_weights.sum()
    gate_samples = pd.read_csv(DUAL_DIR / "depth_ir_modality_gate_summary.csv")
    if "upper_depth_gate" not in gate_samples and "upper_body_depth_gate" in gate_samples:
        gate_samples = gate_samples.rename(columns={"upper_body_depth_gate": "upper_depth_gate"})
        gate_samples.to_csv(DUAL_DIR / "depth_ir_modality_gate_summary.csv", index=False, encoding="utf-8-sig")
    pose_weight = PROJECT_ROOT / "yolo11n-pose.pt"
    total_weights = pose_weight.stat().st_size + (DUAL_DIR / "best_model.pt").stat().st_size
    names = comparison.sort_values("expert_label")["action_name"].tolist()
    label_by_name = {name: index for index, name in enumerate(names)}
    confusion_pairs = [
        ("Take_medicine", "Eat_food"), ("Eat_food", "Take_medicine"),
        ("Watch_TV", "Play_games"), ("Play_games", "Watch_TV"),
        ("Make_a_phone_call", "Take_a_selfie"), ("Take_a_selfie", "Make_a_phone_call"),
        ("Take_body_temperature", "Make_a_phone_call"), ("Make_a_phone_call", "Take_body_temperature"),
        ("Stir_drinks", "Wipe_bowls"), ("Wipe_bowls", "Stir_drinks"),
    ]
    lines = [
        "# Depth_Color + IR dual-stem pose-ROI hard-action expert",
        "",
        "## Pairing and comparison validity",
        "",
        "The audit parsed the absolute timestamp and frame ID from every filename; it never paired sorted array positions. Of 744 hard-subset samples, 743 are exactly aligned. `train__c10__user1__2-1-1` has 37 legacy frame names without absolute timestamps and was excluded, leaving 595 train / 148 validation samples. Therefore the old 596/148 E1 run is context only; the retrained 595/148 Depth-only model is the strict baseline.",
        "",
        "## Architecture and probes",
        "",
        "Depth and native one-channel IR are cropped with the same cached pose ROI coordinates before resizing. The pretrained RGB stem is unchanged; the IR stem is initialized by the exact RGB-kernel channel mean. A zero-initialized 1x1 gate starts at 0.5, followed by one shared MobileNetV3-Small body, the inherited ROI attention, frame projection, and GRU.",
        "",
        f"- Initial gate mean: {probe['initial_gate_mean']:.6f}.",
        f"- Real batch shapes: Depth {probe['depth_shape']}; IR {probe['ir_shape']}.",
        f"- All required gradient groups nonzero: {probe['all_required_gradients_nonzero']}.",
        f"- Real batch=4 peak allocated GPU memory: {probe['peak_gpu_memory_mb']:.2f} MiB; no batch reduction was needed.",
        "- Fourteen representative samples (11 action representatives plus far, near, and off-center geometry cases) were rendered at three temporal positions each and inspected.",
        "",
        "## Main results",
        "",
        "| Experiment | Train/val | Best/completed | Accuracy | Macro-F1 | Weighted F1 | Val loss | Zero-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics, zero in (
        ("Depth-only common subset", common, sum(value == 0 for value in common["per_class_f1"])),
        ("Depth+IR paired", dual, sum(value == 0 for value in dual["per_class_f1"])),
        ("Old E1 (context only)", old, sum(value == 0 for value in old["per_class_f1"])),
    ):
        lines.append(f"| {label} | {metrics['train_samples']}/{metrics['val_samples']} | {metrics['best_epoch']}/{metrics['epochs_completed']} | {metrics['val_accuracy']:.6f} | {metrics['val_macro_f1']:.6f} | {metrics['val_weighted_f1']:.6f} | {metrics['val_loss']:.6f} | {zero} |")
    lines.extend([
        "",
        f"Strict V2 - Depth-only delta: Accuracy {dual['val_accuracy'] - common['val_accuracy']:+.6f}, Macro-F1 {dual['val_macro_f1'] - common['val_macro_f1']:+.6f}, weighted F1 {dual['val_weighted_f1'] - common['val_weighted_f1']:+.6f}, validation loss {dual['val_loss'] - common['val_loss']:+.6f}. Zero-F1 classes decrease from {sum(value == 0 for value in common['per_class_f1'])} to {sum(value == 0 for value in dual['per_class_f1'])}.",
        f"Training time: Depth-only {common_history['epoch_time_seconds'].sum():.1f}s total / {common_history['epoch_time_seconds'].mean():.1f}s mean epoch; Depth+IR {dual_history['epoch_time_seconds'].sum():.1f}s total / {dual_history['epoch_time_seconds'].mean():.1f}s mean epoch.",
        "",
        "## IR contribution diagnostics",
        "",
        "| Validation input | Accuracy | Macro-F1 | Weighted F1 | Changed predictions vs paired |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for mode, label in (("normal", "Normal paired IR"), ("masked", "IR normalized-zero masked"), ("shuffled", "IR sample-shuffled, no self-pair")):
        item = ablations[mode]
        lines.append(f"| {label} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['weighted_f1']:.6f} | {item['prediction_change_count_vs_normal']}/148 ({item['prediction_change_rate_vs_normal']:.2%}) |")
    lines.extend([
        "",
        "Both destructive diagnostics substantially reduce performance, so V2 uses aligned IR content rather than benefiting only from added parameters.",
        "",
        "## Per-class strict comparison",
        "",
        "| Action | Support | Depth F1 | Depth+IR F1 | Delta | Masked F1 | Shuffled F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comparison.sort_values("expert_label").itertuples():
        lines.append(f"| {row.action_name} | {row.support} | {row.f1_depth_only:.6f} | {row.f1_depth_ir:.6f} | {row.f1_delta_depth_ir_minus_depth_only:+.6f} | {row.masked_f1:.6f} | {row.shuffled_f1:.6f} |")
    lines.extend(["", "## Target confusion directions", "", "| Direction | Depth-only | Depth+IR |", "| --- | ---: | ---: |"])
    for source_name, target_name in confusion_pairs:
        source = label_by_name[source_name]
        target = label_by_name[target_name]
        common_count, support = confusion_count(labels, common_predictions, source, target)
        dual_count, _ = confusion_count(labels, dual_predictions, source, target)
        lines.append(f"| {source_name} -> {target_name} | {common_count}/{support} ({common_count / support:.2%}) | {dual_count}/{support} ({dual_count / support:.2%}) |")
    lines.extend(["", "## Difficult-class prediction distributions", ""])
    for source_name in ("Watch_TV", "Play_games", "Take_medicine", "Wipe_bowls"):
        source = label_by_name[source_name]
        selected = dual_predictions[labels == source]
        values, counts = np.unique(selected, return_counts=True)
        distribution = "; ".join(f"{names[int(value)]} {int(count)}" for value, count in sorted(zip(values, counts, strict=True), key=lambda item: item[1], reverse=True))
        lines.append(f"- {source_name}: {distribution}.")
    lines.extend([
        "",
        "## Learned gates and weight budget",
        "",
        f"Mean Depth gate by view (IR weight is `1-g`): global {view_gate[0]:.6f}, upper {view_gate[1]:.6f}, left hand {view_gate[2]:.6f}, right hand {view_gate[3]:.6f}. Mean binary gate entropy is {mean_gate_entropy:.6f} versus maximum ln(2)=0.693147. The scalar summaries stay near balanced fusion rather than collapsing to one modality; destructive IR tests show that this balance carries useful IR information.",
        f"ROI attention means: upper {mean_roi[0]:.6f}, left hand {mean_roi[1]:.6f}, right hand {mean_roi[2]:.6f}.",
        "",
        "### Per-class modality gates",
        "",
        "| Action | Global Depth | Upper Depth | Left-hand Depth | Right-hand Depth |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    grouped_gates = gate_samples.groupby("true_label", as_index=False)[["global_depth_gate", "upper_depth_gate", "left_hand_depth_gate", "right_hand_depth_gate"]].mean()
    for row in grouped_gates.itertuples():
        lines.append(f"| {names[int(row.true_label)]} | {row.global_depth_gate:.6f} | {row.upper_depth_gate:.6f} | {row.left_hand_depth_gate:.6f} | {row.right_hand_depth_gate:.6f} |")
    lines.extend([
        "",
        "Watch_TV and Play_games do not show a meaningful shift toward global IR, and both remain zero-F1. Drink_water and Eat_food also keep nearly balanced hand gates; the learned scalar means are not strongly class-specific.",
        f"YOLO pose weights plus V2 classifier checkpoint total {total_weights} bytes ({total_weights / 1024**2:.2f} MiB), below 100 MB.",
        "",
        "## Decision",
        "",
        "V2 is a positive exploratory result on the exact common subset: aggregate metrics improve, zero-F1 classes decrease, and both IR masking and sample shuffling cause large degradations. Six of the eight continuation criteria are met. TV/gaming remain unresolved and Drink_water declines, so extending this exact expert unchanged to all 40 classes is not yet recommended as a replacement mainline. It is reasonable to retain it as an auxiliary visual branch for a later 40-class fusion experiment.",
        "",
        "The next targeted experiment should prioritize a long-duration screen-activity branch for Watch_TV / Play_games. Hand-head or joint-two-hand ROIs are secondary candidates for medicine/eating/drinking and stirring/wiping, but the present IR result already improves Wipe_bowls and phone-related local interactions, while screen activities remain the clear gap.",
        "",
        "The experiment remains isolated on its branch; no existing task03 fusion or training framework was replaced.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)
    print(SUMMARY_PATH)
    print(PER_CLASS_PATH)


if __name__ == "__main__":
    main()
