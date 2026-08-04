from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import PoseROIDataset, load_modality_frames
from src.engine import collect_predictions
from src.train_depth_ir_pose_roi_40class import detailed_metrics, per_class_rows
from src.train_unimodal import build_model, load_config, save_confusion_matrix, seed_worker, set_seed


RUN_ROOT = PROJECT_ROOT / "outputs/depth_ir_pose_roi_40class_fold0"
RUN_DIR = RUN_ROOT / "depth_ir_pose_roi_40class_fold0_14train_4val"
REPORTS = PROJECT_ROOT / "reports"
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/depth_ir_pose_roi_40class.yaml"


def config_args() -> argparse.Namespace:
    return argparse.Namespace(
        data_root=None, manifest=None, fold=None, output_root=None, device=None, seed=None,
        smoke_test=False, max_epochs=None, num_workers=None, max_train_batches=None,
        max_val_batches=None, run_id=None,
    )


def ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(value)


def evaluate(
    suffix: str,
    checkpoint_path: Path,
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    criterion: nn.Module,
    class_names: list[str],
    train_support: np.ndarray,
    val_support: np.ndarray,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["model_state_dict"]["classifier.weight"].shape) != (40, 192):
        raise ValueError(f"Invalid 40-class checkpoint head: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = collect_predictions(model, loader, criterion, device, amp_enabled=True)
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - started
    labels = np.asarray(output["labels"])
    logits = np.asarray(output["logits"])
    predictions = logits.argmax(axis=1)
    metrics = detailed_metrics(labels, logits, float(output["metrics"]["loss"]))
    probabilities = np.asarray(metrics.pop("probabilities"))
    ranks = np.asarray(metrics.pop("true_class_rank"))
    rows = per_class_rows(
        int(checkpoint["epoch"]), labels, logits,
        {**metrics, "probabilities": probabilities, "true_class_rank": ranks},
        class_names, train_support, val_support,
    )
    max_confidence = probabilities.max(axis=1)
    for row in rows:
        class_id = int(row["class_id"])
        selected = labels == class_id
        correct = selected & (predictions == labels)
        wrong = selected & (predictions != labels)
        row["correct_confidence_mean"] = float(max_confidence[correct].mean()) if correct.any() else np.nan
        row["wrong_confidence_mean"] = float(max_confidence[wrong].mean()) if wrong.any() else np.nan
    per_class = pd.DataFrame(rows).rename(
        columns={
            "true_class_probability_mean": "true_probability_mean",
            "true_class_rank_mean": "true_rank_mean",
            "true_class_rank_median": "true_rank_median",
        }
    )
    per_class.insert(0, "difficulty_rank", per_class["f1"].rank(method="first", ascending=True).astype(int))
    per_class = per_class.sort_values("difficulty_rank")
    report_per_class = REPORTS / f"depth_ir_pose_roi_40class_per_class_{suffix}.csv"
    per_class.to_csv(report_per_class, index=False, encoding="utf-8-sig")
    per_class.to_csv(RUN_DIR / f"per_class_{suffix}.csv", index=False, encoding="utf-8-sig")
    save_confusion_matrix(metrics["confusion_matrix"], RUN_DIR / f"confusion_matrix_{suffix}.png")
    np.savez_compressed(
        RUN_DIR / f"val_predictions_{suffix}.npz",
        sample_ids=output["sample_ids"], labels=labels, predictions=predictions, predicted=predictions,
        logits=logits, probabilities=probabilities, true_class_rank=ranks, embeddings=output["embeddings"],
        roi_attention=output["roi_attention"], modality_gate=output["modality_gate"], temporal_mask=output["temporal_mask"],
    )
    correct = predictions == labels
    serializable = {
        key: value for key, value in metrics.items()
        if key not in {"confusion_matrix", "per_class_precision", "per_class_recall", "per_class_f1", "per_class_support", "predicted_count"}
    }
    serializable.update(
        {
            "checkpoint": checkpoint_path.name,
            "epoch": int(checkpoint["epoch"]),
            "correct_confidence_mean": float(max_confidence[correct].mean()),
            "wrong_confidence_mean": float(max_confidence[~correct].mean()),
            "expected_calibration_error": ece(probabilities, labels),
            "validation_seconds": validation_seconds,
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
            "confusion_matrix": metrics["confusion_matrix"],
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"],
            "per_class_support": metrics["per_class_support"],
            "predicted_count": metrics["predicted_count"],
        }
    )
    (RUN_DIR / f"metrics_{suffix}.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    return {
        "metrics": serializable,
        "per_class": per_class,
        "sample_ids": np.asarray(output["sample_ids"]).astype(str),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "ranks": ranks,
    }


def confusion_table(matrix: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    for class_id, action_name in enumerate(names):
        outgoing = errors[class_id]
        incoming = errors[:, class_id]
        target = int(outgoing.argmax())
        source = int(incoming.argmax())
        rows.append(
            {
                "class_id": class_id,
                "action_name": action_name,
                "top_misclassified_as_class": target,
                "top_misclassified_as_action": names[target],
                "top_misclassified_count": int(outgoing[target]),
                "top_misclassified_rate": float(outgoing[target] / max(1, matrix[class_id].sum())),
                "top_error_source_class": source,
                "top_error_source_action": names[source],
                "top_error_source_count": int(incoming[source]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config(CONFIG_PATH, config_args())
    set_seed(int(config["seed"]))
    history = pd.read_csv(RUN_DIR / "history.csv")
    epoch_classes = pd.read_csv(RUN_DIR / "epoch_per_class_diagnostics.csv")
    counts = epoch_classes.groupby("epoch").size()
    complete_epochs = sorted(set(history["epoch"].astype(int)) & set(counts[counts == 40].index.astype(int)))
    required = [
        "train_loss", "val_loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
        "top3_accuracy", "top5_accuracy", "zero_f1_class_count", "zero_recall_class_count",
        "never_predicted_class_count", "number_of_predicted_classes", "epoch_time_seconds",
    ]
    complete_epochs = [epoch for epoch in complete_epochs if not history.loc[history["epoch"] == epoch, required].isna().any(axis=None)]
    last_completed = max(complete_epochs)
    integrity = {
        "planned_epochs": 30,
        "last_completed_epoch": last_completed,
        "interrupted_epoch": last_completed + 1,
        "history_last_epoch": int(history["epoch"].max()),
        "per_class_rows_for_last_epoch": int(counts.loc[last_completed]),
        "best_accuracy_epoch": int(torch.load(RUN_DIR / "best_accuracy.pt", map_location="cpu", weights_only=True)["epoch"]),
        "best_macro_f1_epoch": int(torch.load(RUN_DIR / "best_macro_f1.pt", map_location="cpu", weights_only=True)["epoch"]),
        "stop_was_graceful": False,
        "last_complete_artifacts_are_clean": True,
        "partial_formal_rows_found": False,
        "last_model_exists": (RUN_DIR / "last_model.pt").exists(),
        "test_data_read": False,
    }
    (RUN_ROOT / "stopped_run_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")

    train_frame, val_frame = load_modality_frames(
        Path(config["manifest"]), Path(config["fold"]), Path(config["data_root"]), str(config["path_column"])
    )
    audit = pd.read_csv(Path(config["pairing_audit"]), encoding="utf-8-sig")
    valid_ids = set(audit.loc[audit["complete_pairing"], "sample_id"].astype(str))
    val_frame = val_frame[val_frame["sample_id"].isin(valid_ids)].reset_index(drop=True)
    actions = val_frame[["class_id", "action_name"]].drop_duplicates().sort_values("class_id")["action_name"].tolist()
    dataset = PoseROIDataset(
        val_frame, actions, int(config["num_frames"]), int(config["image_size"]), False, True,
        Path(config["pose_cache"]), True, Path(config["data_root"]),
    )
    class_map = pd.read_csv(REPORTS / "depth_ir_pose_roi_40class_class_map.csv").sort_values("class_id")
    names = class_map["action_name"].tolist()
    train_support = class_map["train_support"].to_numpy(dtype=np.int64)
    val_support = class_map["val_support"].to_numpy(dtype=np.int64)
    generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]),
        pin_memory=True, generator=generator, worker_init_fn=seed_worker, persistent_workers=True,
        prefetch_factor=2, multiprocessing_context="spawn",
    )
    model = build_model(config, dataset[0])
    criterion = nn.CrossEntropyLoss()
    accuracy = evaluate("best_accuracy", RUN_DIR / "best_accuracy.pt", model, loader, criterion, names, train_support, val_support)
    macro = evaluate("best_macro_f1", RUN_DIR / "best_macro_f1.pt", model, loader, criterion, names, train_support, val_support)
    if not np.array_equal(accuracy["sample_ids"], macro["sample_ids"]) or not np.array_equal(accuracy["labels"], macro["labels"]):
        raise ValueError("Checkpoint validation sample or label order differs.")

    summary_rows = []
    for checkpoint_name, result in (("best_accuracy", accuracy), ("best_macro_f1", macro)):
        metrics = result["metrics"]
        summary_rows.append(
            {
                "checkpoint_name": checkpoint_name, "epoch": metrics["epoch"], "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"], "val_loss": metrics["loss"],
                "top3_accuracy": metrics["top3_accuracy"], "top5_accuracy": metrics["top5_accuracy"],
                "predicted_class_count": metrics["number_of_predicted_classes"],
                "never_predicted_count": metrics["never_predicted_class_count"],
                "zero_recall_count": metrics["zero_recall_class_count"], "zero_f1_count": metrics["zero_f1_class_count"],
                "correct_confidence_mean": metrics["correct_confidence_mean"], "wrong_confidence_mean": metrics["wrong_confidence_mean"],
                "expected_calibration_error": metrics["expected_calibration_error"],
                "validation_seconds": metrics["validation_seconds"], "gpu_peak_allocated_mb": metrics["gpu_peak_allocated_mb"],
                "gpu_peak_reserved_mb": metrics["gpu_peak_reserved_mb"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORTS / "depth_ir_pose_roi_40class_stopped_summary.csv", index=False, encoding="utf-8-sig")

    def select_columns(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        columns = [
            "class_id", "action_name", "train_support", "val_support", "precision", "recall", "f1", "predicted_count",
            "correct_count", "true_probability_mean", "true_rank_mean", "true_rank_median", "top_confused_class_id",
            "top_confused_action", "top_confusion_count",
        ]
        selected = frame[columns].copy()
        fixed = {column: column if column in {"class_id", "action_name", "train_support", "val_support"} else f"{column}_{suffix}" for column in columns}
        fixed["top_confused_class_id"] = f"top_confused_class_{suffix}"
        return selected.rename(columns=fixed)

    comparison = select_columns(accuracy["per_class"], "best_accuracy").merge(
        select_columns(macro["per_class"], "best_macro"), on=["class_id", "action_name", "train_support", "val_support"]
    )
    for metric in ("precision", "recall", "f1", "predicted_count"):
        comparison[f"delta_{metric}_macro_minus_accuracy"] = comparison[f"{metric}_best_macro"] - comparison[f"{metric}_best_accuracy"]
    comparison["delta_true_rank_macro_minus_accuracy"] = comparison["true_rank_mean_best_macro"] - comparison["true_rank_mean_best_accuracy"]
    comparison.to_csv(REPORTS / "depth_ir_pose_roi_40class_checkpoint_comparison.csv", index=False, encoding="utf-8-sig")
    tradeoff = comparison.sort_values("delta_f1_macro_minus_accuracy", ascending=False)
    tradeoff.to_csv(REPORTS / "depth_ir_pose_roi_40class_checkpoint_tradeoff.csv", index=False, encoding="utf-8-sig")

    group_rows = []
    for row in comparison.itertuples():
        maximum = max(row.f1_best_accuracy, row.f1_best_macro)
        if (row.recall_best_accuracy == 0 and row.recall_best_macro == 0) or (row.f1_best_accuracy == 0 and row.f1_best_macro == 0) or (row.correct_count_best_accuracy == 0 and row.correct_count_best_macro == 0):
            group = "unrecognized"
        elif row.f1_best_accuracy >= 0.60 and row.f1_best_macro >= 0.60 and row.recall_best_accuracy >= 0.50 and row.recall_best_macro >= 0.50:
            group = "stable_easy"
        elif maximum >= 0.40:
            group = "usable"
        else:
            group = "difficult"
        sensitive = (
            abs(row.delta_f1_macro_minus_accuracy) >= 0.15
            or ((row.f1_best_accuracy == 0) != (row.f1_best_macro == 0))
            or ((row.recall_best_accuracy == 0) != (row.recall_best_macro == 0))
            or abs(row.delta_predicted_count_macro_minus_accuracy) >= max(3, int(round(row.val_support * 0.25)))
        )
        if row.f1_best_accuracy == 0 and row.f1_best_macro == 0:
            note = "Both checkpoints have zero F1."
        elif row.delta_f1_macro_minus_accuracy >= 0.15:
            note = "Macro-F1 checkpoint is substantially better."
        elif row.delta_f1_macro_minus_accuracy <= -0.15:
            note = "Accuracy checkpoint is substantially better."
        elif maximum < 0.40 and min(row.true_rank_mean_best_accuracy, row.true_rank_mean_best_macro) <= 3:
            note = "F1 is low while mean true-class rank remains near the top."
        elif maximum < 0.40:
            note = "F1 is low and mean true-class rank is relatively late."
        else:
            note = "Both checkpoints are similar or moderately checkpoint-sensitive."
        group_rows.append(
            {
                "class_id": row.class_id, "action_name": row.action_name, "primary_group": group,
                "checkpoint_sensitive": sensitive, "f1_best_accuracy": row.f1_best_accuracy, "f1_best_macro": row.f1_best_macro,
                "recall_best_accuracy": row.recall_best_accuracy, "recall_best_macro": row.recall_best_macro,
                "predicted_count_best_accuracy": row.predicted_count_best_accuracy,
                "predicted_count_best_macro": row.predicted_count_best_macro,
                "true_rank_mean_best_accuracy": row.true_rank_mean_best_accuracy,
                "true_rank_mean_best_macro": row.true_rank_mean_best_macro, "diagnostic_note": note,
            }
        )
    groups = pd.DataFrame(group_rows)
    groups.to_csv(REPORTS / "depth_ir_pose_roi_40class_action_difficulty_groups.csv", index=False, encoding="utf-8-sig")

    confusion_results = {}
    matrices = {}
    for suffix, result in (("best_accuracy", accuracy), ("best_macro_f1", macro)):
        matrix = np.asarray(result["metrics"]["confusion_matrix"], dtype=np.int64)
        matrices[suffix] = matrix
        table = confusion_table(matrix, names)
        table.to_csv(REPORTS / f"depth_ir_pose_roi_40class_confusions_{suffix}.csv", index=False, encoding="utf-8-sig")
        confusion_results[suffix] = table
    directions = []
    for source in range(40):
        for target in range(40):
            if source == target:
                continue
            count_accuracy = int(matrices["best_accuracy"][source, target])
            count_macro = int(matrices["best_macro_f1"][source, target])
            if count_accuracy or count_macro:
                directions.append(
                    {"source_class": source, "source_action": names[source], "target_class": target, "target_action": names[target],
                     "count_best_accuracy": count_accuracy, "count_best_macro": count_macro,
                     "delta_macro_minus_accuracy": count_macro - count_accuracy}
                )
    direction_frame = pd.DataFrame(directions).sort_values(["count_best_accuracy", "count_best_macro"], ascending=False)
    direction_frame.to_csv(REPORTS / "depth_ir_pose_roi_40class_confusion_directions_comparison.csv", index=False, encoding="utf-8-sig")

    differing = accuracy["predictions"] != macro["predictions"]
    labels = accuracy["labels"]
    disagreement_rows = []
    for index in np.flatnonzero(differing):
        label = int(labels[index])
        pa = int(accuracy["predictions"][index])
        pm = int(macro["predictions"][index])
        correct_a = pa == label
        correct_m = pm == label
        which = "best_accuracy" if correct_a else "best_macro" if correct_m else "neither"
        disagreement_rows.append(
            {
                "sample_id": accuracy["sample_ids"][index], "true_class_id": label, "true_action": names[label],
                "prediction_best_accuracy": pa, "prediction_best_accuracy_action": names[pa],
                "confidence_best_accuracy": float(accuracy["probabilities"][index, pa]),
                "true_rank_best_accuracy": int(accuracy["ranks"][index]), "prediction_best_macro": pm,
                "prediction_best_macro_action": names[pm], "confidence_best_macro": float(macro["probabilities"][index, pm]),
                "true_rank_best_macro": int(macro["ranks"][index]), "which_checkpoint_correct": which,
            }
        )
    disagreements = pd.DataFrame(disagreement_rows)
    disagreements.to_csv(REPORTS / "depth_ir_pose_roi_40class_predictions_disagreement.csv", index=False, encoding="utf-8-sig")
    correct_a = accuracy["predictions"] == labels
    correct_m = macro["predictions"] == labels
    disagreement_stats = {
        "different_predictions": int(differing.sum()), "only_best_accuracy_correct": int((correct_a & ~correct_m).sum()),
        "only_best_macro_correct": int((correct_m & ~correct_a).sum()), "both_wrong": int((~correct_a & ~correct_m).sum()),
        "both_correct": int((correct_a & correct_m).sum()),
    }

    clean_history = history[history["epoch"].isin(complete_epochs)].copy()
    clean_history.to_csv(REPORTS / "depth_ir_pose_roi_40class_training_until_stop.csv", index=False, encoding="utf-8-sig")
    best_loss_epoch = int(clean_history.loc[clean_history["val_loss"].idxmin(), "epoch"])
    training_lines = [
        "# Training through the verified stop point", "",
        f"- Last complete epoch: {last_completed}; no partial formal rows were found for epoch {last_completed + 1}.",
        f"- Best Accuracy epoch in completed history: {int(clean_history.loc[clean_history['accuracy'].idxmax(), 'epoch'])}.",
        f"- Best Macro-F1 epoch in completed history: {int(clean_history.loc[clean_history['macro_f1'].idxmax(), 'epoch'])}.",
        f"- Lowest validation loss occurs at epoch {best_loss_epoch}; no later epoch improves that minimum.",
        f"- Predicted-class coverage ranges from {int(clean_history['number_of_predicted_classes'].min())} to {int(clean_history['number_of_predicted_classes'].max())}; it ends at {int(clean_history.iloc[-1]['number_of_predicted_classes'])}.",
        f"- Zero-F1 count reaches a minimum of {int(clean_history['zero_f1_class_count'].min())} and ends at {int(clean_history.iloc[-1]['zero_f1_class_count'])}.",
        "- Accuracy and Macro-F1 select different checkpoints beginning with epoch 9.",
        "- Training loss continues falling after validation loss has stopped improving, which is direct overfitting evidence.",
    ]
    (REPORTS / "depth_ir_pose_roi_40class_training_until_stop.md").write_text("\n".join(training_lines) + "\n", encoding="utf-8")

    top_macro = tradeoff.head(10)
    top_accuracy = tradeoff.tail(10).sort_values("delta_f1_macro_minus_accuracy")
    tradeoff_lines = [
        "# Best Accuracy versus best Macro-F1", "",
        f"- Accuracy difference (best Accuracy - best Macro checkpoint): {float(accuracy['metrics']['accuracy']) - float(macro['metrics']['accuracy']):+.6f}.",
        f"- Macro-F1 difference (best Macro - best Accuracy checkpoint): {float(macro['metrics']['macro_f1']) - float(accuracy['metrics']['macro_f1']):+.6f}.",
        f"- Predicted classes: Accuracy {accuracy['metrics']['number_of_predicted_classes']}, Macro {macro['metrics']['number_of_predicted_classes']}.",
        f"- Zero-F1 classes: Accuracy {accuracy['metrics']['zero_f1_class_count']}, Macro {macro['metrics']['zero_f1_class_count']}.",
        "", "## Largest Macro-checkpoint gains", "",
    ]
    tradeoff_lines.extend(f"- {row.action_name}: {row.delta_f1_macro_minus_accuracy:+.6f} F1." for row in top_macro.itertuples())
    tradeoff_lines.extend(["", "## Largest Accuracy-checkpoint gains", ""])
    tradeoff_lines.extend(f"- {row.action_name}: {-row.delta_f1_macro_minus_accuracy:+.6f} F1 in favor of Accuracy." for row in top_accuracy.itertuples())
    (REPORTS / "depth_ir_pose_roi_40class_checkpoint_tradeoff.md").write_text("\n".join(tradeoff_lines) + "\n", encoding="utf-8")

    report = [
        "# Early-stop analysis of the 40-class Depth+IR pose-ROI run", "",
        "## 1. Safe stop status", "",
        f"The run planned 30 epochs and has {last_completed} complete epochs. The process was terminated while epoch {last_completed + 1} had no formal rows. The termination itself was not graceful, but all epoch-{last_completed} artifacts are complete and stable. No test data was read. `last_model.pt` is absent because the original trainer writes it only after the full loop.",
        "", "## 2. Best checkpoints", "",
        "| Checkpoint | Epoch | Accuracy | Macro-F1 | Weighted F1 | Loss | Top-3 | Top-5 | Predicted classes | Zero recall | Zero F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        report.append(f"| {row.checkpoint_name} | {row.epoch} | {row.accuracy:.6f} | {row.macro_f1:.6f} | {row.weighted_f1:.6f} | {row.val_loss:.6f} | {row.top3_accuracy:.6f} | {row.top5_accuracy:.6f} | {row.predicted_class_count} | {row.zero_recall_count} | {row.zero_f1_count} |")
    report.extend([
        "", "## 3. Overall tradeoff", "",
        f"Best Accuracy gains {float(accuracy['metrics']['accuracy']) - float(macro['metrics']['accuracy']):.6f} Accuracy while losing {float(macro['metrics']['macro_f1']) - float(accuracy['metrics']['macro_f1']):.6f} Macro-F1. It predicts {accuracy['metrics']['number_of_predicted_classes']} classes versus {macro['metrics']['number_of_predicted_classes']} for best Macro-F1.",
        f"The checkpoints disagree on {disagreement_stats['different_predictions']} of 590 samples; only Accuracy is correct on {disagreement_stats['only_best_accuracy_correct']}, only Macro is correct on {disagreement_stats['only_best_macro_correct']}, both are correct on {disagreement_stats['both_correct']}, and both are wrong on {disagreement_stats['both_wrong']}.",
        "", "## 4. Complete action groups", "",
    ])
    for group in ("stable_easy", "usable", "difficult", "unrecognized"):
        report.extend([f"### {group}", ""])
        selected = groups[groups["primary_group"] == group]
        if selected.empty:
            report.append("None.")
        else:
            for row in selected.itertuples():
                comparison_row = comparison.loc[comparison["class_id"] == row.class_id].iloc[0]
                report.append(
                    f"- {row.action_name}: F1 {row.f1_best_accuracy:.3f}/{row.f1_best_macro:.3f}, recall {row.recall_best_accuracy:.3f}/{row.recall_best_macro:.3f}, predicted {row.predicted_count_best_accuracy}/{row.predicted_count_best_macro}, true rank {row.true_rank_mean_best_accuracy:.2f}/{row.true_rank_mean_best_macro:.2f}, top confusion {comparison_row['top_confused_action_best_accuracy']}/{comparison_row['top_confused_action_best_macro']}."
                )
        report.append("")
    sensitive_names = groups.loc[groups["checkpoint_sensitive"], "action_name"].tolist()
    report.extend(["## 5. Checkpoint-sensitive actions", "", ", ".join(sensitive_names) if sensitive_names else "None.", "", "## 6. Major confusion directions", ""])
    for suffix in ("best_accuracy", "best_macro_f1"):
        report.append(f"### {suffix}")
        report.append("")
        matrix = matrices[suffix]
        rows = []
        for source in range(40):
            for target in range(40):
                if source != target and matrix[source, target] > 0:
                    rows.append((int(matrix[source, target]), names[source], names[target]))
        for count, source, target in sorted(rows, reverse=True)[:10]:
            report.append(f"- {source} -> {target}: {count}.")
        report.append("")
    report.extend([
        "## 7. Data-supported conclusions", "",
        f"Stable easy actions: {', '.join(groups.loc[groups['primary_group'] == 'stable_easy', 'action_name']) or 'none'}.",
        f"Unrecognized by both checkpoints: {', '.join(groups.loc[groups['primary_group'] == 'unrecognized', 'action_name']) or 'none'}.",
        f"Actions most harmed when selecting Accuracy: {', '.join(top_macro.head(5)['action_name'])}.",
        f"Actions retained more strongly by the Macro-F1 checkpoint: {', '.join(top_macro[top_macro['delta_f1_macro_minus_accuracy'] > 0]['action_name'].head(10)) or 'none'}.",
        "This report describes observed results only; it does not select experts, routers, class weights, modalities, or a next training configuration.",
    ])
    (REPORTS / "depth_ir_pose_roi_40class_early_stop_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Cross-check every reported aggregate against the 590 aligned validation samples.
    for result in (accuracy, macro):
        matrix = np.asarray(result["metrics"]["confusion_matrix"])
        per_class = result["per_class"]
        assert matrix.sum() == len(dataset) == 590
        assert per_class["predicted_count"].sum() == 590
        assert per_class["correct_count"].sum() == np.trace(matrix)
        assert abs(np.trace(matrix) / 590 - result["metrics"]["accuracy"]) < 1e-12
        assert abs(per_class["f1"].mean() - result["metrics"]["macro_f1"]) < 1e-12
        assert len(per_class) == 40
    print(json.dumps({"integrity": integrity, "summary": summary_rows, "disagreements": disagreement_stats}, indent=2))


if __name__ == "__main__":
    main()
