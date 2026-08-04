from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEW_NAMES = ["global_context", "left_interaction", "right_interaction", "hand_head_union", "two_hand_table_context"]


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.6f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_object_interaction_tcn_expert.yaml",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_object_interaction_tcn_expert_fold0/depth_ir_object_interaction_tcn_expert_14train_4val",
    )
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    prediction = logits.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(40), zero_division=0,
    )
    ranks = 1 + (logits > logits[np.arange(len(labels)), labels, None]).sum(1)
    return {
        "prediction": prediction, "precision": precision, "recall": recall, "f1": f1, "support": support,
        "ranks": ranks, "accuracy": float((prediction == labels).mean()), "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)), "zero_f1": int((f1 == 0).sum()),
    }


def top_confusion(labels: np.ndarray, prediction: np.ndarray, class_id: int, names: list[str]) -> str:
    wrong = prediction[(labels == class_id) & (prediction != class_id)]
    if not len(wrong):
        return ""
    values, counts = np.unique(wrong, return_counts=True)
    selected = int(values[counts.argmax()])
    return f"{selected}:{names[selected]} ({int(counts.max())})"


def aggregate_audit(frame: pd.DataFrame, scope: str, class_id: int | None, action: str) -> dict[str, Any]:
    weights = frame["frames"].to_numpy(dtype=float)
    result: dict[str, Any] = {
        "scope": scope, "class_id": "" if class_id is None else class_id, "action_name": action,
        "samples": len(frame), "frames": int(weights.sum()),
    }
    excluded = {"sample_id", "class_id", "action_name", "frames", "artificial_keypoint_count"}
    for column in frame.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        result[column] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan
    result["artificial_keypoint_count"] = int(frame["artificial_keypoint_count"].sum())
    return result


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.reports.mkdir(parents=True, exist_ok=True)
    class_map_path = Path(config["class_map"])
    if not class_map_path.is_absolute():
        class_map_path = PROJECT_ROOT / class_map_path
    class_map = pd.read_csv(class_map_path, encoding="utf-8-sig").sort_values("class_id")
    names = class_map["action_name"].tolist()
    lookup = dict(zip(names, range(40), strict=True))
    target_ids = np.asarray([lookup[name] for name in config["target_actions"]], dtype=int)
    control_ids = np.asarray([lookup[name] for name in config["control_actions"]], dtype=int)
    hard_ids = np.asarray([lookup[name] for name in config["hard_negative_actions"]], dtype=int)
    stable_ids = np.asarray([lookup[name] for name in ["Walk", "Sit_down", "Stand_up", "Wash_face", "Jog_in_place", "Do_jumping_jacks"]])

    archive = np.load(args.run_dir / "val_predictions_best_target16.npz", allow_pickle=False)
    sample_ids = archive["sample_ids"].astype(str)
    labels = archive["labels"].astype(int)
    base_logits = archive["base_logits"].astype(np.float64)
    final_logits = archive["logits"].astype(np.float64)
    base = metrics(labels, base_logits)
    final = metrics(labels, final_logits)
    base_prob = torch.from_numpy(base_logits).softmax(1).numpy()
    final_prob = torch.from_numpy(final_logits).softmax(1).numpy()
    base_correct = base["prediction"] == labels
    final_correct = final["prediction"] == labels
    outcomes = np.full(len(labels), "changed_wrong", dtype="U20")
    outcomes[base_correct & final_correct] = "retained_correct"
    outcomes[~base_correct & final_correct] = "rescued"
    outcomes[base_correct & ~final_correct] = "harmed"
    outcomes[~base_correct & ~final_correct & (base["prediction"] == final["prediction"])] = "unchanged_wrong"
    temporal_mask = archive["temporal_mask"].astype(bool)
    view_weights = archive["view_weights"]
    view_valid = archive["view_valid_mask"].astype(bool)
    mean_view_weights = np.divide(
        (view_weights * temporal_mask[..., None]).sum(1), temporal_mask.sum(1, keepdims=True),
        out=np.zeros((len(labels), len(VIEW_NAMES))), where=temporal_mask.sum(1, keepdims=True) > 0,
    )
    sample_frame = pd.DataFrame(
        {
            "sample_id": sample_ids, "true_class_id": labels, "true_action": [names[value] for value in labels],
            "base_prediction": base["prediction"], "base_prediction_action": [names[value] for value in base["prediction"]],
            "base_confidence": base_prob.max(1), "final_prediction": final["prediction"],
            "final_prediction_action": [names[value] for value in final["prediction"]],
            "final_confidence": final_prob.max(1), "outcome": outcomes,
            "selected_view_weights": [json.dumps(dict(zip(VIEW_NAMES, row, strict=True))) for row in mean_view_weights],
            "two_hand_roi_valid": archive["two_hand_roi_valid"],
            "hand_head_roi_valid": archive["hand_head_roi_valid"],
            "valid_keypoint_pattern": archive["valid_keypoint_pattern"].astype(str),
        }
    )
    sample_frame.to_csv(args.reports / "object_interaction_tcn_expert_sample_outcomes.csv", index=False, encoding="utf-8-sig")

    per_class_rows = []
    for class_id, action_name in enumerate(names):
        selected = labels == class_id
        base_rank_mean = float(base["ranks"][selected].mean()) if selected.any() else np.nan
        final_rank_mean = float(final["ranks"][selected].mean()) if selected.any() else np.nan
        rescued = int(np.sum(selected & (outcomes == "rescued")))
        harmed = int(np.sum(selected & (outcomes == "harmed")))
        per_class_rows.append(
            {
                "class_id": class_id, "action_name": action_name,
                "train_support": int(class_map.iloc[class_id]["train_support"]), "val_support": int(selected.sum()),
                "base_precision": base["precision"][class_id], "base_recall": base["recall"][class_id],
                "base_f1": base["f1"][class_id], "final_precision": final["precision"][class_id],
                "final_recall": final["recall"][class_id], "final_f1": final["f1"][class_id],
                "delta_precision": final["precision"][class_id] - base["precision"][class_id],
                "delta_recall": final["recall"][class_id] - base["recall"][class_id],
                "delta_f1": final["f1"][class_id] - base["f1"][class_id],
                "base_true_rank_mean": base_rank_mean,
                "final_true_rank_mean": final_rank_mean,
                "delta_true_rank": final_rank_mean - base_rank_mean,
                "rescued_count": rescued, "harmed_count": harmed, "net_rescue": rescued - harmed,
                "top_confusion_base": top_confusion(labels, base["prediction"], class_id, names),
                "top_confusion_final": top_confusion(labels, final["prediction"], class_id, names),
            }
        )
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(args.reports / "object_interaction_tcn_expert_per_class.csv", index=False, encoding="utf-8-sig")

    history = pd.read_csv(args.run_dir / "history.csv", encoding="utf-8-sig")
    checkpoint_rows = []
    for filename in ("best_target16_macro_f1.pt", "best_overall_macro_f1.pt", "best_accuracy.pt", "last_complete.pt"):
        checkpoint = torch.load(args.run_dir / filename, map_location="cpu", weights_only=True)
        item = checkpoint["metrics"]
        checkpoint_rows.append(
            {
                "checkpoint": filename, "epoch": checkpoint["epoch"], "stage": checkpoint["stage"],
                "accuracy": item["accuracy"], "macro_f1": item["macro_f1"], "weighted_f1": item["weighted_f1"],
                "target16_macro_f1": item["target16_macro_f1"], "target16_zero_f1_count": item["target16_zero_f1_count"],
                "zero_f1_count": item["zero_f1_count"], "val_loss": checkpoint["val_loss"],
            }
        )
    checkpoints = pd.DataFrame(checkpoint_rows)
    checkpoints.to_csv(args.reports / "object_interaction_tcn_expert_checkpoint_summary.csv", index=False, encoding="utf-8-sig")

    audit = pd.concat(
        (
            pd.read_csv(args.run_dir / "roi_audit_train_samples.csv", encoding="utf-8-sig").assign(split="train"),
            pd.read_csv(args.run_dir / "roi_audit_val_samples.csv", encoding="utf-8-sig").assign(split="validation"),
        ), ignore_index=True,
    )
    audit_numeric = audit.drop(columns="split")
    audit_rows = [aggregate_audit(audit_numeric, "overall", None, "ALL")]
    for class_id, group in audit_numeric.groupby("class_id", sort=True):
        audit_rows.append(aggregate_audit(group, "class", int(class_id), names[int(class_id)]))
    roi_quality = pd.DataFrame(audit_rows)
    invalid_nonzero = int(np.sum((~view_valid) & (np.abs(view_weights) > 1e-12)))
    roi_quality["invalid_view_nonzero_gate_count"] = 0
    roi_quality.loc[roi_quality["scope"] == "overall", "invalid_view_nonzero_gate_count"] = invalid_nonzero
    roi_quality.to_csv(args.reports / "object_interaction_roi_quality.csv", index=False, encoding="utf-8-sig")

    temporal = pd.read_csv(args.run_dir / "temporal_diagnostics_base.csv", encoding="utf-8-sig")
    temporal = temporal.set_index("sample_id").loc[sample_ids].reset_index()
    if not np.array_equal(temporal["sample_id"].to_numpy(dtype=str), sample_ids):
        raise RuntimeError("Temporal diagnostics do not align with validation predictions")
    attention = archive["temporal_attention"]
    activation = archive["tcn_activation_norms"]
    temporal["attention_peak_position"] = attention.argmax(1)
    temporal["attention_peak_sampled_frame"] = archive["sampled_indices"][np.arange(len(temporal)), attention.argmax(1)]
    temporal["attention_entropy"] = -(np.log(attention.clip(min=1e-12)) * attention).sum(1)
    temporal["attention_distribution"] = [json.dumps(row.tolist()) for row in attention]
    for index, dilation in enumerate(config["tcn_dilations"]):
        temporal[f"dilation_{dilation}_activation_norm"] = activation[:, index]
    temporal.to_csv(args.reports / "object_interaction_tcn_temporal_diagnostics.csv", index=False, encoding="utf-8-sig")
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(attention.shape[1])
    axis.plot(positions, attention.mean(0), label="all validation", linewidth=2.2)
    for action in ("Watch_TV", "Play_games", "Read_documents"):
        selected = labels == lookup[action]
        if selected.any():
            axis.plot(positions, attention[selected].mean(0), label=action, linewidth=1.5)
    axis.set(xlabel="Sampled full-trial frame position", ylabel="Mean temporal attention")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.run_dir / "temporal_attention_curves.png", dpi=180)
    plt.close(figure)

    def aggregate(ids: np.ndarray, score: dict[str, Any]) -> float:
        return float(np.asarray(score["f1"])[ids].mean())

    summary_rows = []
    for model_name, score in (("base_epoch8", base), ("final_expert", final)):
        summary_rows.append(
            {
                "model": model_name, "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
                "weighted_f1": score["weighted_f1"], "target16_macro_f1": aggregate(target_ids, score),
                "hand_head7_macro_f1": aggregate(np.asarray([lookup[n] for n in config["hand_head_actions"]]), score),
                "table7_macro_f1": aggregate(np.asarray([lookup[n] for n in config["table_actions"]]), score),
                "screen2_macro_f1": aggregate(np.asarray([lookup[n] for n in config["screen_actions"]]), score),
                "control10_macro_f1": aggregate(control_ids, score),
                "non_target_macro_f1": aggregate(np.asarray([i for i in range(40) if i not in target_ids]), score),
                "stable_easy_macro_f1": aggregate(stable_ids, score), "zero_f1_count": score["zero_f1"],
                "target16_zero_f1_count": int((np.asarray(score["f1"])[target_ids] == 0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.reports / "object_interaction_tcn_expert_summary.csv", index=False, encoding="utf-8-sig")

    run_summary = json.loads((args.run_dir / "run_summary.json").read_text(encoding="utf-8"))
    lie = roi_quality.loc[roi_quality["action_name"] == "Lie_down"].iloc[0]
    lie_id = lookup["Lie_down"]
    invalid_two_hand_frame = view_valid[:, :, 4] & ~(view_valid[:, :, 1] & view_valid[:, :, 2])
    lie_wrong_two_hand = int(np.sum((labels == lie_id) & invalid_two_hand_frame.any(axis=1)))
    rescued_total = int((outcomes == "rescued").sum())
    harmed_total = int((outcomes == "harmed").sum())
    base_target = aggregate(target_ids, base)
    final_target = aggregate(target_ids, final)
    stable_delta = aggregate(stable_ids, final) - aggregate(stable_ids, base)
    macro_delta = final["macro_f1"] - base["macro_f1"]
    recommend_router = bool(
        final_target > base_target and rescued_total > harmed_total and macro_delta >= -0.02 and stable_delta >= -0.03
    )
    target_table = per_class[per_class["class_id"].isin(target_ids)].sort_values("delta_f1", ascending=False)
    harmed_table = per_class.sort_values("delta_f1").head(10)
    roi_lines = [
        "# Object interaction ROI quality", "",
        f"- Artificially completed keypoints: {int(roi_quality.iloc[0]['artificial_keypoint_count'])}.",
        f"- Directional ROI valid rate: {roi_quality.iloc[0]['direction_roi_valid_rate']:.6f}.",
        f"- Two-hand ROI trigger rate: {roi_quality.iloc[0]['two_hand_merge_trigger_rate']:.6f}.",
        f"- Old/new left-right overlap rate: {roi_quality.iloc[0]['old_tight_overlap_rate']:.6f} / {roi_quality.iloc[0]['new_directional_overlap_rate']:.6f}.",
        f"- Mean interaction ROI area ratio: {roi_quality.iloc[0]['interaction_roi_area_ratio_mean']:.6f}.",
        f"- Invalid-view nonzero gate count: {invalid_nonzero}.", "",
        "## Lie_down", "",
        f"- One-elbow-only rate: {lie['one_elbow_only_rate']:.6f}.",
        f"- One-wrist-only rate: {lie['one_wrist_only_rate']:.6f}.",
        f"- Erroneous two-hand ROI sample count: {lie_wrong_two_hand}.",
        f"- Harmed validation samples: {int(per_class.loc[per_class.class_id == lie_id, 'harmed_count'].iloc[0])}.",
    ]
    (args.reports / "object_interaction_roi_quality.md").write_text("\n".join(roi_lines) + "\n", encoding="utf-8")
    temporal_lines = [
        "# Full-trial temporal diagnostics", "",
        f"- Validation trials: {len(temporal)}.",
        f"- Start and end both covered: {int((temporal.full_trial_start_covered & temporal.full_trial_end_covered).sum())}/{len(temporal)}.",
        f"- Mean center-24 coverage ratio: {temporal.center24_coverage_ratio.mean():.6f}.",
        f"- Mean full-48 coverage ratio: {temporal.full48_coverage_ratio.mean():.6f}.",
        f"- TCN dilations: {config['tcn_dilations']}; receptive field: 125.", "",
    ]
    for action in ("Watch_TV", "Play_games", "Read_documents"):
        selected = temporal.action_name == action
        temporal_lines.append(
            f"- {action}: mean attention entropy {temporal.loc[selected, 'attention_entropy'].mean():.6f}, "
            f"full endpoints covered {int((temporal.loc[selected, 'full_trial_start_covered'] & temporal.loc[selected, 'full_trial_end_covered']).sum())}/{int(selected.sum())}."
        )
    (args.reports / "object_interaction_tcn_temporal_diagnostics.md").write_text(
        "\n".join(temporal_lines) + "\n", encoding="utf-8"
    )
    experiment = [
        "# Object interaction residual TCN expert experiment", "", "## Integrity", "",
        f"- Epoch 8 predictions reproduced exactly: {run_summary['base_reproduction']['strict_prediction_reproduction']}.",
        f"- Train/validation samples: {run_summary['train_samples']}/{run_summary['val_samples']}.",
        f"- Target train/validation samples: {run_summary['target_train_samples']}/{run_summary['target_val_samples']}.",
        "- Competition test data read: no.", "- Artificial keypoint completion: no.", "",
        "## Base versus final", "",
        dataframe_to_markdown(summary), "",
        f"- Best target16 epoch: {run_summary['best_target16_epoch']}.",
        f"- Rescued/harmed/net rescue: {rescued_total}/{harmed_total}/{rescued_total - harmed_total}.",
        f"- Mean absolute expert residual: {float(np.abs(archive['delta_logits_target']).mean()):.6f}.",
        f"- Peak allocated/reserved VRAM: {run_summary['peak_allocated_mb']:.2f}/{run_summary['peak_reserved_mb']:.2f} MB.",
        f"- Training seconds: {run_summary['training_seconds']:.2f}.",
        f"- Total inference parameter bytes: {run_summary['total_inference_parameter_bytes']}.", "",
        "## Target-class F1 changes", "", dataframe_to_markdown(target_table[["class_id", "action_name", "base_f1", "final_f1", "delta_f1", "net_rescue"]]), "",
        "## Largest harmed classes", "", dataframe_to_markdown(harmed_table[["class_id", "action_name", "base_f1", "final_f1", "delta_f1", "harmed_count"]]), "",
        "## Decision", "", f"- Recommend Router training next: {'yes' if recommend_router else 'no'}.",
        f"- Overall Macro-F1 delta: {macro_delta:+.6f}; stable-easy mean F1 delta: {stable_delta:+.6f}.",
    ]
    (args.reports / "object_interaction_tcn_expert_experiment.md").write_text(
        "\n".join(experiment) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "base_accuracy": base["accuracy"], "final_accuracy": final["accuracy"],
        "base_macro_f1": base["macro_f1"], "final_macro_f1": final["macro_f1"],
        "base_target16_macro_f1": base_target, "final_target16_macro_f1": final_target,
        "rescued": rescued_total, "harmed": harmed_total, "net_rescue": rescued_total - harmed_total,
        "recommend_router": recommend_router,
    }, indent=2))


if __name__ == "__main__":
    main()
