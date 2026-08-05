from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEW_NAMES = ["global_context", "left_interaction", "right_interaction", "hand_head_union", "two_hand_table_context"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_object_interaction_resnet18_expert.yaml",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_object_interaction_resnet18_expert_fold0/depth_ir_object_interaction_resnet18_expert_14train_4val",
    )
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def score(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    prediction = logits.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(40), zero_division=0,
    )
    probability = torch.from_numpy(logits.astype(np.float64)).softmax(1).numpy()
    confidence = probability.max(1)
    correct = prediction == labels
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return {
        "prediction": prediction, "precision": precision, "recall": recall, "f1": f1, "support": support,
        "rank": 1 + (logits > logits[np.arange(len(labels)), labels, None]).sum(1),
        "confidence": confidence, "accuracy": float(correct.mean()), "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)), "ece": ece,
    }


def outcomes(labels: np.ndarray, base_prediction: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    base_correct = base_prediction == labels
    final_correct = prediction == labels
    values = np.full(len(labels), "changed_wrong", dtype="U20")
    values[base_correct & final_correct] = "retained_correct"
    values[~base_correct & final_correct] = "rescued"
    values[base_correct & ~final_correct] = "harmed"
    values[~base_correct & ~final_correct & (base_prediction == prediction)] = "unchanged_wrong"
    return values


def mean_view_weights(archive: Any) -> np.ndarray:
    weights = archive["view_weights"]
    mask = archive["temporal_mask"].astype(bool)
    return np.divide(
        (weights * mask[..., None]).sum(1), mask.sum(1, keepdims=True),
        out=np.zeros((len(mask), len(VIEW_NAMES))), where=mask.sum(1, keepdims=True) > 0,
    )


def markdown(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.6f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mobile_dir = PROJECT_ROOT / config["mobilenet_run_dir"]
    mobile = np.load(mobile_dir / "val_predictions_best_target16.npz", allow_pickle=False)
    resnet = np.load(args.run_dir / "val_predictions_best_target16.npz", allow_pickle=False)
    sample_ids = resnet["sample_ids"].astype(str)
    labels = resnet["labels"].astype(int)
    if not np.array_equal(sample_ids, mobile["sample_ids"].astype(str)):
        raise RuntimeError("MobileNet and ResNet sample_id order differs")
    if not np.array_equal(labels, mobile["labels"].astype(int)):
        raise RuntimeError("MobileNet and ResNet labels differ")
    if not np.allclose(resnet["base_logits"], mobile["base_logits"], rtol=0, atol=0):
        raise RuntimeError("Frozen base logits differ between experiments")

    class_map = pd.read_csv(PROJECT_ROOT / config["class_map"], encoding="utf-8-sig").sort_values("class_id")
    names = class_map["action_name"].tolist()
    lookup = dict(zip(names, range(40), strict=True))
    target_ids = np.asarray([lookup[name] for name in config["target_actions"]])
    group_ids = {
        "target16_macro_f1": target_ids,
        "hand_head7_macro_f1": np.asarray([lookup[name] for name in config["hand_head_actions"]]),
        "table7_macro_f1": np.asarray([lookup[name] for name in config["table_actions"]]),
        "screen2_macro_f1": np.asarray([lookup[name] for name in config["screen_actions"]]),
    }
    scores = {
        "B0_epoch8": score(labels, resnet["base_logits"]),
        "M1_mobilenet": score(labels, mobile["logits"]),
        "R1_resnet18": score(labels, resnet["logits"]),
    }
    base_prediction = scores["B0_epoch8"]["prediction"]
    mobile_outcomes = outcomes(labels, base_prediction, scores["M1_mobilenet"]["prediction"])
    resnet_outcomes = outcomes(labels, base_prediction, scores["R1_resnet18"]["prediction"])
    mobile_checkpoint = torch.load(mobile_dir / "best_target16_macro_f1.pt", map_location="cpu", weights_only=True)
    resnet_checkpoint = torch.load(args.run_dir / "best_target16_macro_f1.pt", map_location="cpu", weights_only=True)
    mobile_summary = json.loads((mobile_dir / "run_summary.json").read_text(encoding="utf-8"))
    resnet_summary = json.loads((args.run_dir / "run_summary.json").read_text(encoding="utf-8"))
    mobile_benchmark = json.loads((args.run_dir / "mobilenet_validation_benchmark.json").read_text(encoding="utf-8"))
    base_benchmark = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_stopped_summary.csv", encoding="utf-8-sig")
    base_validation_seconds = float(base_benchmark.loc[base_benchmark.checkpoint_name == "best_macro_f1", "validation_seconds"].iloc[0])
    if not np.array_equal(
        np.asarray(mobile_summary["target_class_weights"]), np.asarray(resnet_summary["target_class_weights"]),
    ):
        raise RuntimeError("Target16 class weights differ between experiments")
    for split in ("train", "val"):
        mobile_audit = pd.read_csv(mobile_dir / f"roi_audit_{split}_samples.csv", encoding="utf-8-sig")
        resnet_audit = pd.read_csv(args.run_dir / f"roi_audit_{split}_samples.csv", encoding="utf-8-sig")
        pd.testing.assert_frame_equal(mobile_audit, resnet_audit, check_dtype=False, check_exact=True)
    budget = json.loads((args.run_dir / "inference_weight_budget.json").read_text(encoding="utf-8"))

    summary_rows = []
    for model, result in scores.items():
        outcome_values = outcomes(labels, base_prediction, result["prediction"])
        checkpoint = None if model == "B0_epoch8" else (mobile_checkpoint if model == "M1_mobilenet" else resnet_checkpoint)
        run_summary = None if model == "B0_epoch8" else (mobile_summary if model == "M1_mobilenet" else resnet_summary)
        row = {
            "model": model, "checkpoint_epoch": 8 if checkpoint is None else int(checkpoint["epoch"]),
            "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "weighted_f1": result["weighted_f1"],
            **{name: float(result["f1"][ids].mean()) for name, ids in group_ids.items()},
            "zero_f1_count": int((result["f1"] == 0).sum()),
            "target16_zero_f1_count": int((result["f1"][target_ids] == 0).sum()),
            "rescued": int((outcome_values == "rescued").sum()), "harmed": int((outcome_values == "harmed").sum()),
            "net_rescue": int((outcome_values == "rescued").sum() - (outcome_values == "harmed").sum()),
            "val_loss": float(torch.nn.functional.cross_entropy(torch.from_numpy((resnet["base_logits"] if checkpoint is None else (mobile["logits"] if model == "M1_mobilenet" else resnet["logits"])).astype(np.float64)), torch.from_numpy(labels))),
            "ece": result["ece"],
            "validation_seconds": (
                base_validation_seconds if checkpoint is None else
                mobile_benchmark["validation_seconds"] if model == "M1_mobilenet" else checkpoint["validation_seconds"]
            ),
            "peak_allocated_mb": 0.0 if run_summary is None else run_summary["peak_allocated_mb"],
            "model_parameter_bytes": resnet_summary["base_parameter_bytes"] if run_summary is None else run_summary["total_inference_parameter_bytes"],
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.reports / "object_interaction_resnet18_expert_summary.csv", index=False, encoding="utf-8-sig")

    checkpoint_rows = []
    for filename in ("best_target16_macro_f1.pt", "best_overall_macro_f1.pt", "best_accuracy.pt", "last_complete.pt"):
        checkpoint = torch.load(args.run_dir / filename, map_location="cpu", weights_only=True)
        metrics = checkpoint["metrics"]
        checkpoint_rows.append({
            "checkpoint": filename, "epoch": checkpoint["epoch"], "stage": checkpoint["stage"],
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"], "target16_macro_f1": metrics["target16_macro_f1"],
            "target16_zero_f1_count": metrics["target16_zero_f1_count"], "zero_f1_count": metrics["zero_f1_count"],
            "rescued": metrics["rescued_count"], "harmed": metrics["harmed_count"], "net_rescue": metrics["net_rescue"],
            "val_loss": checkpoint["val_loss"], "validation_seconds": checkpoint["validation_seconds"],
            "checkpoint_bytes": (args.run_dir / filename).stat().st_size,
        })
    pd.DataFrame(checkpoint_rows).to_csv(
        args.reports / "object_interaction_resnet18_expert_checkpoint_summary.csv", index=False, encoding="utf-8-sig"
    )

    per_rows = []
    for class_id, action in enumerate(names):
        selected = labels == class_id
        row: dict[str, Any] = {"class_id": class_id, "action_name": action, "val_support": int(selected.sum())}
        for prefix, model, outcome_values in (
            ("base", "B0_epoch8", outcomes(labels, base_prediction, base_prediction)),
            ("mobilenet", "M1_mobilenet", mobile_outcomes),
            ("resnet18", "R1_resnet18", resnet_outcomes),
        ):
            result = scores[model]
            row[f"{prefix}_f1"] = result["f1"][class_id]
            row[f"{prefix}_recall"] = result["recall"][class_id]
            row[f"{prefix}_true_rank_mean"] = float(result["rank"][selected].mean())
            if prefix != "base":
                row[f"{prefix}_rescued"] = int(np.sum(selected & (outcome_values == "rescued")))
                row[f"{prefix}_harmed"] = int(np.sum(selected & (outcome_values == "harmed")))
                row[f"{prefix}_net_rescue"] = row[f"{prefix}_rescued"] - row[f"{prefix}_harmed"]
        row["delta_resnet_vs_base"] = row["resnet18_f1"] - row["base_f1"]
        row["delta_resnet_vs_mobilenet"] = row["resnet18_f1"] - row["mobilenet_f1"]
        per_rows.append(row)
    per_class = pd.DataFrame(per_rows)
    per_class.to_csv(args.reports / "object_interaction_resnet18_expert_per_class.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(args.reports / "object_interaction_resnet18_vs_mobilenet.csv", index=False, encoding="utf-8-sig")

    mobile_views = mean_view_weights(mobile)
    resnet_views = mean_view_weights(resnet)
    samples = pd.DataFrame({
        "sample_id": sample_ids, "true_action": [names[value] for value in labels],
        "base_prediction": [names[value] for value in base_prediction],
        "mobilenet_prediction": [names[value] for value in scores["M1_mobilenet"]["prediction"]],
        "resnet18_prediction": [names[value] for value in scores["R1_resnet18"]["prediction"]],
        "base_confidence": scores["B0_epoch8"]["confidence"],
        "mobilenet_confidence": scores["M1_mobilenet"]["confidence"],
        "resnet18_confidence": scores["R1_resnet18"]["confidence"],
        "mobilenet_outcome": mobile_outcomes, "resnet18_outcome": resnet_outcomes,
        "mobilenet_true_rank": scores["M1_mobilenet"]["rank"], "resnet18_true_rank": scores["R1_resnet18"]["rank"],
        "mobilenet_selected_view_weights": [json.dumps(dict(zip(VIEW_NAMES, row, strict=True))) for row in mobile_views],
        "resnet18_selected_view_weights": [json.dumps(dict(zip(VIEW_NAMES, row, strict=True))) for row in resnet_views],
        "view_valid_count": resnet["view_valid_mask"].sum(axis=(1, 2)),
        "hand_head_roi_valid": resnet["hand_head_roi_valid"], "two_hand_roi_valid": resnet["two_hand_roi_valid"],
    })
    samples.to_csv(args.reports / "object_interaction_resnet18_vs_mobilenet_samples.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(args.reports / "object_interaction_resnet18_expert_sample_outcomes.csv", index=False, encoding="utf-8-sig")

    view_rows = []
    for scope, selected in [("overall", np.ones(len(labels), dtype=bool))] + [(name, labels == lookup[name]) for name in config["target_actions"]]:
        for index, view in enumerate(VIEW_NAMES):
            view_rows.append({
                "scope": scope, "view": view, "mobilenet_mean_weight": mobile_views[selected, index].mean(),
                "resnet18_mean_weight": resnet_views[selected, index].mean(),
                "delta_resnet_minus_mobilenet": resnet_views[selected, index].mean() - mobile_views[selected, index].mean(),
            })
    pd.DataFrame(view_rows).to_csv(args.reports / "object_interaction_resnet18_view_diagnostics.csv", index=False, encoding="utf-8-sig")

    def entropy(values: np.ndarray) -> np.ndarray:
        return -(values.clip(1e-12) * np.log(values.clip(1e-12))).sum(1)
    temporal = pd.DataFrame({
        "sample_id": sample_ids, "action_name": [names[value] for value in labels],
        "mobilenet_attention_entropy": entropy(mobile["temporal_attention"]),
        "resnet18_attention_entropy": entropy(resnet["temporal_attention"]),
        "mobilenet_depth_gate_mean": mobile["modality_gate"].mean(axis=(1, 2)),
        "resnet18_depth_gate_mean": resnet["modality_gate"].mean(axis=(1, 2)),
        "mobilenet_embedding_norm": np.linalg.norm(mobile["embeddings"], axis=1),
        "resnet18_embedding_norm": np.linalg.norm(resnet["embeddings"], axis=1),
        "mobilenet_residual_abs_mean": np.abs(mobile["delta_logits_target"]).mean(axis=1),
        "resnet18_residual_abs_mean": np.abs(resnet["delta_logits_target"]).mean(axis=1),
    })
    temporal.to_csv(args.reports / "object_interaction_resnet18_temporal_diagnostics.csv", index=False, encoding="utf-8-sig")

    weight_rows = [
        {"component": "frozen_base", "bytes": budget["frozen_base_fp32_bytes"]},
        {"component": "resnet18_expert", "bytes": budget["resnet18_expert_fp32_bytes"]},
        {"component": "yolo11n_pose", "bytes": budget["yolo11n_pose_file_bytes"]},
        {"component": "actual_inference_bundle", "bytes": budget["actual_bundle_bytes"]},
    ]
    pd.DataFrame(weight_rows).to_csv(args.reports / "object_interaction_resnet18_weight_budget.csv", index=False, encoding="utf-8-sig")

    target_table = per_class[per_class.class_id.isin(target_ids)][
        ["action_name", "base_f1", "mobilenet_f1", "resnet18_f1", "delta_resnet_vs_mobilenet"]
    ]
    direct = {
        "resnet_only_correct": int(np.sum((scores["R1_resnet18"]["prediction"] == labels) & (scores["M1_mobilenet"]["prediction"] != labels))),
        "mobilenet_only_correct": int(np.sum((scores["M1_mobilenet"]["prediction"] == labels) & (scores["R1_resnet18"]["prediction"] != labels))),
        "both_correct": int(np.sum((scores["R1_resnet18"]["prediction"] == labels) & (scores["M1_mobilenet"]["prediction"] == labels))),
        "both_wrong": int(np.sum((scores["R1_resnet18"]["prediction"] != labels) & (scores["M1_mobilenet"]["prediction"] != labels))),
        "different_but_both_wrong": int(np.sum(
            (scores["R1_resnet18"]["prediction"] != labels)
            & (scores["M1_mobilenet"]["prediction"] != labels)
            & (scores["R1_resnet18"]["prediction"] != scores["M1_mobilenet"]["prediction"])
        )),
    }
    stable_ids = np.asarray([lookup[name] for name in ["Walk", "Sit_down", "Stand_up", "Wash_face", "Jog_in_place", "Do_jumping_jacks"]])
    recovery_names = ["Wipe_bowls", "Take_medicine"]
    recovery_pass = all(
        float(per_class.loc[per_class.action_name == name, "resnet18_f1"].iloc[0])
        > float(per_class.loc[per_class.action_name == name, "mobilenet_f1"].iloc[0])
        for name in recovery_names
    )
    signals = {name: bool(value) for name, value in {
        "target16_macro_f1_higher": summary.loc[2, "target16_macro_f1"] > summary.loc[1, "target16_macro_f1"],
        "overall_macro_f1_not_lower": summary.loc[2, "macro_f1"] >= summary.loc[1, "macro_f1"],
        "net_rescue_higher": summary.loc[2, "net_rescue"] > summary.loc[1, "net_rescue"],
        "harmed_not_higher": summary.loc[2, "harmed"] <= summary.loc[1, "harmed"],
        "target16_zero_f1_not_higher": summary.loc[2, "target16_zero_f1_count"] <= summary.loc[1, "target16_zero_f1_count"],
        "damaged_targets_recovered": recovery_pass,
        "under_100_mib": bool(budget["under_100_mib"]),
        "stable_easy_not_materially_lower": float(scores["R1_resnet18"]["f1"][stable_ids].mean() - scores["M1_mobilenet"]["f1"][stable_ids].mean()) >= -0.01,
    }.items()}
    keep = sum(bool(value) for value in signals.values()) >= 6
    experiment = [
        "# ResNet18 object interaction expert experiment", "", "## Integrity", "",
        f"- ImageNet weights: ResNet18_Weights.IMAGENET1K_V1; loaded: {resnet_summary['expert_metadata']['pretrained_weights_loaded']}.",
        "- Depth stem: native RGB 3-channel conv1/bn1; IR stem: native 1-channel conv1 initialized by RGB-channel mean with independent bn1.",
        "- Shared ResNet body count: 1; layer4 BN running statistics: frozen.",
        f"- Batch/accumulation/chunk: {config['batch_size']}/{config['gradient_accumulation_steps']}/{config['encoder_chunk_size']}.",
        f"- Epoch 8 strict reproduction: {resnet_summary['base_reproduction']['strict_prediction_reproduction']}; test read: no.", "",
        "- ROI train/validation audit tables match MobileNet exactly; Target16 class weights match exactly.", "",
        "## B0 / M1 / R1", "", markdown(summary), "",
        "## Target16", "", markdown(target_table), "",
        "## Sample comparison", "", f"- {json.dumps(direct, ensure_ascii=False)}",
        f"- R1 rescued/harmed/net: {summary.loc[2, 'rescued']}/{summary.loc[2, 'harmed']}/{summary.loc[2, 'net_rescue']}.", "",
        f"- Retention signals passed: {sum(bool(value) for value in signals.values())}/{len(signals)}; {json.dumps(signals)}.", "",
        "## Resource budget", "",
        f"- Peak allocated/reserved VRAM: {resnet_summary['peak_allocated_mb']:.2f}/{resnet_summary['peak_reserved_mb']:.2f} MB.",
        f"- Training seconds: {resnet_summary['training_seconds']:.2f}.",
        f"- Actual inference bundle: {budget['actual_bundle_bytes']} bytes ({budget['actual_bundle_mib']:.2f} MiB); under 100 MiB: {budget['under_100_mib']}.", "",
        "## Decision", "", f"- Retain ResNet18 expert: {'yes' if keep else 'no'}.",
        "- Capacity is not the sole bottleneck: the stronger backbone improves aggregate metrics and recovers Wipe_bowls/Take_medicine, but Write, Make_a_phone_call, and Watch_TV remain zero-F1 and several interaction classes trade places.",
        "- Evidence points more strongly to small-object visibility/ROI evidence, limited examples, and class-boundary ambiguity than to backbone capacity alone. Do not replace the frozen 40-class base model in this experiment.",
    ]
    (args.reports / "object_interaction_resnet18_expert_experiment.md").write_text("\n".join(experiment) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary.to_dict(orient="records"), "direct": direct, "signals": signals, "retain_resnet18": keep}, indent=2))


if __name__ == "__main__":
    main()
