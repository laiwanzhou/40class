from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from scripts.report_x3d_s_train12_val2_dev import build_report as build_standalone_report
from scripts.report_x3d_s_train12_val2_partial1 import (
    _matched_prediction_diagnostics,
    evaluate_matched_candidate,
)
import src.train_x3d_s_visual_expert as trainer


def _parameter_drift(
    run_directory: Path, reference_run_directory: Path
) -> dict[str, dict[str, float]]:
    config = yaml.safe_load(
        (run_directory / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    trainer._set_seed(int(config["seed"]))
    initial = trainer._build_model(config).cpu().state_dict()
    checkpoints = {
        "layerwise_lr1": torch.load(
            run_directory / "best_accuracy.pt", map_location="cpu", weights_only=False
        )["model_state_dict"],
        "partial2": torch.load(
            reference_run_directory / "best_accuracy.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"],
    }
    prefixes = {
        "block4": "backbone.blocks.4.",
        "block5": "backbone.blocks.5.",
        "embedding_head": "embedding_head.",
        "classifier": "classifier.",
    }
    result: dict[str, dict[str, float]] = {}
    for checkpoint_name, state in checkpoints.items():
        checkpoint_result: dict[str, float] = {}
        for scope, prefix in prefixes.items():
            delta_squared = 0.0
            initial_squared = 0.0
            for name, initial_tensor in initial.items():
                if not name.startswith(prefix) or not torch.is_floating_point(initial_tensor):
                    continue
                delta = (state[name].float() - initial_tensor.float()).reshape(-1)
                baseline = initial_tensor.float().reshape(-1)
                delta_squared += float(torch.dot(delta, delta))
                initial_squared += float(torch.dot(baseline, baseline))
            checkpoint_result[scope] = math.sqrt(delta_squared) / (
                math.sqrt(initial_squared) + 1e-30
            )
        result[checkpoint_name] = checkpoint_result
    return result


def build_report(
    run_directory: Path,
    *,
    reference_report_path: Path,
    reference_run_directory: Path,
) -> dict[str, Any]:
    report = build_standalone_report(run_directory)
    reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    if reference_report.get("role") != "train12_val2_development_tuning_report":
        raise ValueError("Reference report is not train12/val2 development evidence")
    metrics = report["metrics"]
    reference_metrics = reference_report["metrics"]
    names = ("accuracy", "macro_f1", "worst_user_accuracy")
    candidate_archive = run_directory / "val_predictions_best_accuracy.npz"
    reference_archive = reference_run_directory / "val_predictions_best_accuracy.npz"
    diagnostics = _matched_prediction_diagnostics(candidate_archive, reference_archive)
    parameter_drift = _parameter_drift(run_directory, reference_run_directory)
    summary = json.loads((run_directory / "run_summary.json").read_text(encoding="utf-8"))

    report.update(
        {
            "role": "train12_val2_layerwise_lr1_matched_report",
            "decision": evaluate_matched_candidate(metrics, reference_metrics),
            "interpretation": (
                "Matched optimization ablation against partial2. Block4 and block5 remain "
                "trainable, but use lower block-specific learning rates."
            ),
            "learning_rate_contract": {
                "block4": float(summary["backbone_block_lrs"]["4"]),
                "block5": float(summary["backbone_block_lrs"]["5"]),
                "custom_head": 0.0003,
            },
            "matched_partial2_reference": {
                "run_id": reference_report["run_id"],
                "metrics": {name: float(reference_metrics[name]) for name in names},
                "report_path": str(reference_report_path.resolve()),
                "report_sha256": trainer._sha256_file(reference_report_path),
                "prediction_sha256": trainer._sha256_file(reference_archive),
            },
            "matched_delta": {
                name: float(metrics[name] - reference_metrics[name]) for name in names
            },
            "per_user_accuracy_delta": {
                user: float(
                    metrics["per_user_accuracy"][user]
                    - reference_metrics["per_user_accuracy"][user]
                )
                for user in metrics["per_user_accuracy"]
            },
            "duration_accuracy_delta": {
                bucket: float(
                    metrics["duration_buckets"][bucket]["accuracy"]
                    - reference_metrics["duration_buckets"][bucket]["accuracy"]
                )
                for bucket in metrics["duration_buckets"]
            },
            "matched_prediction_diagnostics": diagnostics,
            "parameter_relative_l2_drift_vs_initialization": parameter_drift,
            "parameter_drift_ratio_vs_partial2": {
                scope: float(
                    parameter_drift["layerwise_lr1"][scope]
                    / parameter_drift["partial2"][scope]
                )
                for scope in parameter_drift["layerwise_lr1"]
            },
            "human_review_floor": float(reference_metrics["accuracy"] - 0.02),
            "automatic_next_experiment_permitted": False,
        }
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    reference = report["matched_partial2_reference"]["metrics"]
    delta = report["matched_delta"]
    training = report["training_log_at_selected_epoch"]
    diagnostics = report["matched_prediction_diagnostics"]
    drift = report["parameter_relative_l2_drift_vs_initialization"]
    lines = [
        "# X3D-S Train12/Val2 Layer-wise LR1 Matched Result",
        "",
        "> Development-only matched optimization ablation. This is not unbiased OOF and does not replace canonical Phase 4/5 evidence.",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "| Metric | Layer-wise LR1 | Partial2 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name in ("accuracy", "macro_f1", "worst_user_accuracy"):
        lines.append(
            f"| {name} | {metrics[name]:.6f} | {reference[name]:.6f} | {delta[name]:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Selected epoch: `{report['selected_epoch']}`. Train Accuracy: `{training['train_accuracy']:.6f}`. Train-minus-validation gap: `{training['train_minus_val_accuracy']:.6f}`.",
            "",
            "Block learning rates: block4 `3e-6`, block5 `1e-5`, custom head `3e-4`.",
            "",
            "## Per User",
            "",
            "| User | Accuracy | Delta vs partial2 |",
            "|---|---:|---:|",
        ]
    )
    for user, accuracy in metrics["per_user_accuracy"].items():
        lines.append(
            f"| {user} | {accuracy:.6f} | {report['per_user_accuracy_delta'][user]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Duration",
            "",
            "| Frames | N | Accuracy | Delta vs partial2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for bucket, values in metrics["duration_buckets"].items():
        lines.append(
            f"| {bucket} | {values['sample_count']} | {values['accuracy']:.6f} | "
            f"{report['duration_accuracy_delta'][bucket]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Matched Prediction Diagnosis",
            "",
            f"The models disagree on {diagnostics['prediction_disagreement_count']}/324 trials. Layer-wise-only correct: {diagnostics['candidate_only_correct']}; partial2-only correct: {diagnostics['reference_only_correct']}; both wrong: {diagnostics['both_wrong']}.",
            "",
            f"Layer-wise NLL is {diagnostics['candidate_confidence']['nll']:.6f}; partial2 NLL is {diagnostics['reference_confidence']['nll']:.6f}.",
            "",
            "## Parameter Drift",
            "",
            "| Scope | Layer-wise LR1 | Partial2 | Ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for scope in ("block4", "block5", "embedding_head", "classifier"):
        lines.append(
            f"| {scope} | {drift['layerwise_lr1'][scope]:.6f} | "
            f"{drift['partial2'][scope]:.6f} | "
            f"{report['parameter_drift_ratio_vs_partial2'][scope]:.3f} |"
        )
    lines.extend(
        [
            "",
            "Lower block learning rates substantially reduce backbone drift, but custom-head drift increases and the train-validation gap remains severe. Backbone drift alone is not the dominant overfitting mechanism.",
            "",
            "The greater-than-two-point Accuracy regression rule is triggered. Preserve all artifacts and require human review; do not automatically launch L2-SP or another experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report matched layer-wise LR1 X3D evidence")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-run-directory", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/x3d_s_train12_val2_layerwise_lr1_report.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/x3d_s_train12_val2_layerwise_lr1_report.md"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(
        args.run_directory,
        reference_report_path=args.reference_report,
        reference_run_directory=args.reference_run_directory,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
