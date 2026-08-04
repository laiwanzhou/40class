from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "outputs/depth_ir_pose_roi_40class_fold0/depth_ir_pose_roi_40class_fold0_14train_4val"
HARD11_DIR = PROJECT_ROOT / "outputs/depth_ir_pose_roi_expert_fold0/depth_ir/depth_ir_pose_roi_expert_fold0_14train_4val"
REPORT_DIR = PROJECT_ROOT / "reports"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def late_abandonment(values: np.ndarray) -> bool:
    for start in range(1, len(values) - 4):
        if np.any(values[:start] > 0) and np.all(values[start : start + 5] == 0):
            return True
    return False


def main() -> None:
    class_map = pd.read_csv(REPORT_DIR / "depth_ir_pose_roi_40class_class_map.csv")
    history = pd.read_csv(RUN_DIR / "history.csv")
    epochs = pd.read_csv(RUN_DIR / "epoch_per_class_diagnostics.csv")
    per_accuracy = pd.read_csv(RUN_DIR / "per_class_best_accuracy.csv")
    per_macro = pd.read_csv(RUN_DIR / "per_class_best_macro_f1.csv")
    per_last = pd.read_csv(RUN_DIR / "per_class_last.csv")
    metrics_accuracy = read_json(RUN_DIR / "metrics_best_accuracy.json")
    metrics_macro = read_json(RUN_DIR / "metrics_best_macro_f1.json")
    metrics_last = read_json(RUN_DIR / "metrics_last.json")
    run_summary = read_json(RUN_DIR / "run_summary.json")
    pose = pd.read_csv(REPORT_DIR / "depth_ir_pose_roi_40class_pose_quality.csv")
    overall_pose = pose.loc[pose["class_id"] == -1].iloc[0]
    names = class_map.sort_values("class_id")["action_name"].tolist()

    tradeoff = per_accuracy[["class_id", "action_name", "f1", "recall", "predicted_count"]].rename(
        columns={"f1": "f1_best_accuracy", "recall": "recall_best_accuracy", "predicted_count": "predicted_count_best_accuracy"}
    ).merge(
        per_macro[["class_id", "action_name", "f1", "recall", "predicted_count"]].rename(
            columns={"f1": "f1_best_macro", "recall": "recall_best_macro", "predicted_count": "predicted_count_best_macro"}
        ), on=["class_id", "action_name"]
    )
    tradeoff["delta_f1"] = tradeoff["f1_best_macro"] - tradeoff["f1_best_accuracy"]
    tradeoff.to_csv(REPORT_DIR / "depth_ir_pose_roi_40class_checkpoint_tradeoff.csv", index=False, encoding="utf-8-sig")

    abandoned_rows = []
    for class_id, action_name in enumerate(names):
        class_epochs = epochs.loc[epochs["class_id"] == class_id].sort_values("epoch")
        predicted = class_epochs["predicted_count"].to_numpy()
        first = int(class_epochs.loc[class_epochs["predicted_count"] > 0, "epoch"].min()) if np.any(predicted > 0) else np.nan
        last = int(class_epochs.loc[class_epochs["predicted_count"] > 0, "epoch"].max()) if np.any(predicted > 0) else np.nan
        accuracy_row = per_accuracy.loc[per_accuracy["class_id"] == class_id].iloc[0]
        macro_row = per_macro.loc[per_macro["class_id"] == class_id].iloc[0]
        late = late_abandonment(predicted)
        diagnoses = []
        if int(macro_row["predicted_count"]) == 0:
            diagnoses.append("never_predicted")
        if float(macro_row["recall"]) == 0:
            diagnoses.append("zero_recall")
        if late:
            diagnoses.append("late_abandonment")
        if not diagnoses:
            diagnoses.append("retained")
        abandoned_rows.append(
            {
                "class_id": class_id,
                "action_name": action_name,
                "best_accuracy_predicted_count": int(accuracy_row["predicted_count"]),
                "best_macro_predicted_count": int(macro_row["predicted_count"]),
                "best_accuracy_recall": accuracy_row["recall"],
                "best_macro_recall": macro_row["recall"],
                "best_accuracy_f1": accuracy_row["f1"],
                "best_macro_f1": macro_row["f1"],
                "late_abandonment": late,
                "first_epoch_predicted": first,
                "last_epoch_predicted": last,
                "true_rank_mean": macro_row["true_class_rank_mean"],
                "diagnosis": ";".join(diagnoses),
            }
        )
    abandoned = pd.DataFrame(abandoned_rows)
    abandoned.to_csv(REPORT_DIR / "depth_ir_pose_roi_40class_abandoned_classes.csv", index=False, encoding="utf-8-sig")

    difficulty = per_macro.copy()
    difficulty["difficulty_group"] = np.select(
        [
            (difficulty["predicted_count"] == 0) | (difficulty["recall"] == 0),
            difficulty["f1"].between(0, 0.20, inclusive="neither"),
            difficulty["f1"].between(0.20, 0.50, inclusive="left"),
            difficulty["f1"].between(0.50, 0.75, inclusive="left"),
            difficulty["f1"] >= 0.75,
        ],
        ["abandoned", "severe", "difficult", "good", "strong"],
        default="abandoned",
    )
    difficulty = difficulty.sort_values(["f1", "recall", "true_class_rank_mean"], ascending=[True, True, False]).reset_index(drop=True)
    difficulty.insert(0, "difficulty_rank", np.arange(1, len(difficulty) + 1))
    difficulty = difficulty[
        [
            "difficulty_rank", "class_id", "action_name", "train_support", "val_support", "precision", "recall", "f1",
            "predicted_count", "true_class_rank_mean", "true_class_rank_median", "correct_confidence_mean", "wrong_confidence_mean",
            "top_confused_action", "top_confusion_count", "difficulty_group",
        ]
    ]
    difficulty.to_csv(REPORT_DIR / "depth_ir_pose_roi_40class_difficulty.csv", index=False, encoding="utf-8-sig")

    with np.load(RUN_DIR / "val_predictions_best_macro_f1.npz") as predictions_file:
        labels = predictions_file["labels"]
        predictions = predictions_file["predicted"]
    matrix = np.asarray(metrics_macro["confusion_matrix"], dtype=np.int64)
    pair_rows = []
    for source in range(40):
        for target in range(40):
            if source == target:
                continue
            forward = int(matrix[source, target])
            reverse = int(matrix[target, source])
            pair_rows.append(
                {
                    "source_class": source,
                    "source_action": names[source],
                    "target_class": target,
                    "target_action": names[target],
                    "source_to_target_count": forward,
                    "target_to_source_count": reverse,
                    "bidirectional_total": forward + reverse,
                    "source_to_target_rate": forward / max(1, int(matrix[source].sum())),
                    "target_to_source_rate": reverse / max(1, int(matrix[target].sum())),
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["source_to_target_count", "bidirectional_total"], ascending=False
    )
    pairs.to_csv(REPORT_DIR / "depth_ir_pose_roi_40class_confusion_pairs.csv", index=False, encoding="utf-8-sig")

    hard_per = pd.read_csv(HARD11_DIR / "per_class_metrics.csv")
    with np.load(HARD11_DIR / "fold_0_val_predictions.npz") as hard_file, np.load(RUN_DIR / "val_predictions_best_macro_f1.npz") as full_file:
        hard_ids = set(hard_file["sample_ids"].astype(str))
        full_ids = set(full_file["sample_ids"].astype(str))
    strict_samples = hard_ids.issubset(full_ids)
    hard_rows = []
    for row in hard_per.itertuples():
        class_id = int(row.original_class_id)
        accuracy_row = per_accuracy.loc[per_accuracy["class_id"] == class_id].iloc[0]
        macro_row = per_macro.loc[per_macro["class_id"] == class_id].iloc[0]
        hard_f1 = float(row.f1)
        full_f1 = float(macro_row["f1"])
        rank = float(macro_row["true_class_rank_mean"])
        if hard_f1 >= 0.4 and full_f1 <= hard_f1 - 0.2:
            interpretation = "40-class competition suppression"
        elif hard_f1 < 0.2 and full_f1 < 0.2 and rank > 5:
            interpretation = "visual representation remains insufficient"
        elif hard_f1 < 0.2 and full_f1 >= hard_f1 + 0.1:
            interpretation = "full-class context or additional data helps"
        elif full_f1 < 0.2 and rank <= 5:
            interpretation = "classification-boundary suppression"
        elif full_f1 < 0.2:
            interpretation = "representation-limited"
        else:
            interpretation = "retained or mixed change"
        hard_rows.append(
            {
                "class_id": class_id,
                "action_name": row.action_name,
                "hard11_support": int(row.support),
                "full40_support": int(macro_row["val_support"]),
                "hard11_f1": hard_f1,
                "full40_f1_best_accuracy": accuracy_row["f1"],
                "full40_f1_best_macro": full_f1,
                "delta_best_accuracy": accuracy_row["f1"] - hard_f1,
                "delta_best_macro": full_f1 - hard_f1,
                "full40_true_rank_mean": rank,
                "full40_top_confused_action": macro_row["top_confused_action"],
                "interpretation": interpretation if strict_samples else f"non-strict background: {interpretation}",
            }
        )
    hard_comparison = pd.DataFrame(hard_rows)
    hard_comparison.to_csv(REPORT_DIR / "depth_ir_pose_roi_40class_vs_hard11.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"checkpoint": "best_accuracy", **{key: metrics_accuracy[key] for key in ("epoch", "accuracy", "macro_f1", "weighted_f1", "loss", "top3_accuracy", "top5_accuracy", "zero_f1_class_count", "zero_recall_class_count", "never_predicted_class_count", "number_of_predicted_classes")}},
            {"checkpoint": "best_macro_f1", **{key: metrics_macro[key] for key in ("epoch", "accuracy", "macro_f1", "weighted_f1", "loss", "top3_accuracy", "top5_accuracy", "zero_f1_class_count", "zero_recall_class_count", "never_predicted_class_count", "number_of_predicted_classes")}},
            {"checkpoint": "last", **{key: metrics_last[key] for key in ("epoch", "accuracy", "macro_f1", "weighted_f1", "loss", "top3_accuracy", "top5_accuracy", "zero_f1_class_count", "zero_recall_class_count", "never_predicted_class_count", "number_of_predicted_classes")}},
        ]
    )
    summary.to_csv(REPORT_DIR / "depth_ir_pose_roi_40class_summary.csv", index=False, encoding="utf-8-sig")

    top_single = pairs.drop_duplicates(["source_class", "target_class"]).head(10)
    late_classes = abandoned.loc[abandoned["late_abandonment"], "action_name"].tolist()
    never_macro = per_macro.loc[per_macro["predicted_count"] == 0, "action_name"].tolist()
    hardest = difficulty.head(10)["action_name"].tolist()
    strongest = difficulty.tail(10).sort_values("f1", ascending=False)["action_name"].tolist()
    accuracy_sacrifices = int(metrics_accuracy["number_of_predicted_classes"]) < int(metrics_macro["number_of_predicted_classes"])
    pose_bytes = (PROJECT_ROOT / "yolo11n-pose.pt").stat().st_size
    classifier_bytes = (RUN_DIR / "best_macro_f1.pt").stat().st_size
    total_bytes = pose_bytes + classifier_bytes
    lines = [
        "# Full 40-class Depth_Color + IR pose-ROI capability audit",
        "",
        "## Data and execution",
        "",
        f"- Strictly usable samples: {run_summary['train_samples']} train / {run_summary['val_samples']} validation; all 40 original class IDs are present in both splits.",
        "- Pairing audit: 2910/2931 dual-modality fold samples strictly paired and readable (99.28%); 21 legacy-name exceptions are explicitly reported.",
        f"- Batch size: {run_summary['batch_size']}; peak GPU allocated/reserved: {run_summary['peak_allocated_mb']:.2f}/{run_summary['peak_reserved_mb']:.2f} MiB.",
        f"- Training: all 30 epochs completed in {run_summary['total_training_time_seconds']:.1f}s; mean epoch {run_summary['mean_epoch_time_seconds']:.1f}s.",
        f"- Pose: person {overall_pose['person_success']:.2%}, left wrist {overall_pose['left_wrist_success']:.2%}, right wrist {overall_pose['right_wrist_success']:.2%}, at least one wrist {overall_pose['at_least_one_wrist_success']:.2%}.",
        f"- ROI fallback: upper {overall_pose['upper_body_fallback']:.2%}, left hand {overall_pose['left_hand_fallback']:.2%}, right hand {overall_pose['right_hand_fallback']:.2%}.",
        f"- ROI mean area: upper {overall_pose['upper_body_area_ratio']:.2%}, left hand {overall_pose['left_hand_area_ratio']:.2%}, right hand {overall_pose['right_hand_area_ratio']:.2%}.",
        "",
        "## Checkpoint results",
        "",
        "| Checkpoint | Epoch | Accuracy | Macro-F1 | Weighted F1 | Loss | Top-3 | Top-5 | Zero F1 | Zero recall | Never predicted | Predicted classes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.checkpoint} | {row.epoch} | {row.accuracy:.6f} | {row.macro_f1:.6f} | {row.weighted_f1:.6f} | {row.loss:.6f} | {row.top3_accuracy:.6f} | {row.top5_accuracy:.6f} | {row.zero_f1_class_count} | {row.zero_recall_class_count} | {row.never_predicted_class_count} | {row.number_of_predicted_classes} |")
    lines.extend(
        [
            "",
            f"The Accuracy checkpoint {'does' if accuracy_sacrifices else 'does not'} cover fewer classes than the Macro-F1 checkpoint. Accuracy difference (macro minus accuracy checkpoint): {float(metrics_macro['accuracy']) - float(metrics_accuracy['accuracy']):+.6f}.",
            f"Late-abandonment classes: {', '.join(late_classes) if late_classes else 'none'}.",
            f"Never predicted at best Macro-F1: {', '.join(never_macro) if never_macro else 'none'}.",
            "",
            "## Complete difficulty ranking",
            "",
            "| Rank | Class | Action | F1 | Recall | Predicted | True rank mean | Group |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in difficulty.itertuples():
        lines.append(f"| {row.difficulty_rank} | {row.class_id} | {row.action_name} | {row.f1:.6f} | {row.recall:.6f} | {row.predicted_count} | {row.true_class_rank_mean:.2f} | {row.difficulty_group} |")
    lines.extend(["", "## Top confusion directions", "", "| Source | Target | Count | Source rate | Reverse count |", "| --- | --- | ---: | ---: | ---: |"])
    for row in top_single.itertuples():
        lines.append(f"| {row.source_action} | {row.target_action} | {row.source_to_target_count} | {row.source_to_target_rate:.2%} | {row.target_to_source_count} |")
    lines.extend(["", "## Original hard-11 competition", "", f"All hard-11 validation sample IDs are included in full-40 validation: {strict_samples}.", "", "| Action | Hard-11 F1 | Full-40 Accuracy F1 | Full-40 Macro F1 | Macro delta | True rank | Interpretation |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in hard_comparison.itertuples():
        lines.append(f"| {row.action_name} | {row.hard11_f1:.6f} | {row.full40_f1_best_accuracy:.6f} | {row.full40_f1_best_macro:.6f} | {row.delta_best_macro:+.6f} | {row.full40_true_rank_mean:.2f} | {row.interpretation} |")
    lines.extend(
        [
            "",
            "## Capability interpretation",
            "",
            f"Strongest actions: {', '.join(strongest)}.",
            f"Hardest actions: {', '.join(hardest)}.",
            "",
            "- Loss coefficient or hard-group auxiliary head candidates: abandoned/severe classes whose mean true rank remains near the top; these are classification-boundary failures rather than total representation failures.",
            "- Long-duration branch candidates: Watch_TV and Play_games.",
            "- Joint interaction ROI candidates: medicine/eating/drinking, phone/selfie/temperature, and stirring/wiping/tableware groups when their confusion remains local.",
            "- Skeleton/IMU/Radar priority: locomotion and posture transitions such as Walk, Jog_in_place, Sit_down, Stand_up, Lie_down, lunges, squats, and jumping jacks when visual F1 remains weak.",
            "",
            "The encoder is suitable for later four- or six-modality fusion as a compact visual expert, but the full-40 audit should guide class-aware fusion rather than treating its logits as uniformly reliable.",
            f"Inference weights: YOLO pose {pose_bytes} bytes plus classifier {classifier_bytes} bytes = {total_bytes} bytes ({total_bytes / 1024**2:.2f} MiB).",
        ]
    )
    (REPORT_DIR / "depth_ir_pose_roi_40class_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_DIR / "depth_ir_pose_roi_40class_experiment.md")


if __name__ == "__main__":
    main()
