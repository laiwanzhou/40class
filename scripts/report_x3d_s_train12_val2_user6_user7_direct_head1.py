from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from scripts.report_x3d_s_fold0_dev import _duration_bucket
from scripts.report_x3d_s_train12_val2_dev import build_report as build_standalone_report
from scripts.report_x3d_s_train12_val2_partial1 import _matched_prediction_diagnostics
from src.engine.metrics import classification_metrics
import src.train_x3d_s_visual_expert as trainer


METRIC_NAMES = ("accuracy", "macro_f1", "worst_user_accuracy")


def evaluate_direct_head_candidate(
    metrics: Mapping[str, float], reference: Mapping[str, float]
) -> str:
    if float(metrics["accuracy"]) < float(reference["accuracy"]) - 0.02 - 1e-12:
        return "human_review_regression"
    no_worse = all(float(metrics[name]) >= float(reference[name]) for name in METRIC_NAMES)
    improves = any(float(metrics[name]) > float(reference[name]) for name in METRIC_NAMES)
    return "preferred" if no_worse and improves else "non_winning_ablation"


def _relative_l2_drift(
    initial: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    *,
    prefix: str,
) -> float:
    delta_squared = 0.0
    initial_squared = 0.0
    matched = 0
    for name, initial_tensor in initial.items():
        if not name.startswith(prefix) or not torch.is_floating_point(initial_tensor):
            continue
        if name not in trained or trained[name].shape != initial_tensor.shape:
            raise ValueError(f"Checkpoint differs from seeded initialization at {name}")
        delta = (trained[name].float() - initial_tensor.float()).reshape(-1)
        baseline = initial_tensor.float().reshape(-1)
        delta_squared += float(torch.dot(delta, delta))
        initial_squared += float(torch.dot(baseline, baseline))
        matched += 1
    if matched == 0:
        raise ValueError(f"No floating-point parameters matched prefix {prefix}")
    return math.sqrt(delta_squared) / (math.sqrt(initial_squared) + 1e-30)


def _direct_classifier_drift(run_directory: Path) -> float:
    config = yaml.safe_load(
        (run_directory / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    if trainer._resolved_head_type(config) != "direct":
        raise ValueError("Direct-Head report requires head_type=direct")
    trainer._set_seed(int(config["seed"]))
    initial = trainer._build_model(config).cpu().state_dict()
    checkpoint = torch.load(
        run_directory / "best_accuracy.pt", map_location="cpu", weights_only=False
    )
    trained = checkpoint["model_state_dict"]
    return _relative_l2_drift(initial, trained, prefix="classifier.")


def _group_metrics(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        predictions = archive["logits"].argmax(axis=1)
        users = archive["user_ids"].astype(str)
        frames = archive["num_frames"].astype(np.int64)

    def metrics(mask: np.ndarray) -> dict[str, float]:
        values = classification_metrics(labels[mask], predictions[mask], num_classes=40)
        return {
            "accuracy": float(values["accuracy"]),
            "macro_f1": float(values["macro_f1"]),
        }

    per_user = {user: metrics(users == user) for user in sorted(np.unique(users))}
    buckets = np.asarray([_duration_bucket(int(value)) for value in frames])
    duration = {
        bucket: metrics(buckets == bucket)
        for bucket in ("<=13", "14-32", "33-64", ">64")
    }
    return {"per_user": per_user, "duration": duration}


def _delta_groups(
    candidate: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if set(candidate) != set(reference):
        raise ValueError("Matched group populations differ")
    return {
        group: {
            name: float(candidate[group][name] - reference[group][name])
            for name in ("accuracy", "macro_f1")
        }
        for group in candidate
    }


def build_report(
    run_directory: Path,
    *,
    reference_report_path: Path,
    reference_run_directory: Path,
) -> dict[str, Any]:
    report = build_standalone_report(run_directory)
    reference = json.loads(reference_report_path.read_text(encoding="utf-8"))
    if reference.get("role") != "train12_val2_matched_reference_report":
        raise ValueError("Reference is not the frozen user6/user7 matched report")
    if reference.get("split", {}).get("validation_user_ids") not in (None, ["user6", "user7"]):
        raise ValueError("Historical user21/user22 evidence cannot be a matched reference")

    metrics = report["metrics"]
    reference_metrics = reference["metrics"]
    candidate_archive = run_directory / "val_predictions_best_accuracy.npz"
    reference_archive = reference_run_directory / "val_predictions_best_accuracy.npz"
    diagnostics = _matched_prediction_diagnostics(candidate_archive, reference_archive)
    candidate_groups = _group_metrics(candidate_archive)
    reference_groups = _group_metrics(reference_archive)
    summary = json.loads((run_directory / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("head_type") != "direct" or int(summary.get("embedding_dim", -1)) != 2048:
        raise ValueError("Run summary is not the frozen Direct-Head architecture")
    if int(summary.get("custom_head_parameter_count", -1)) != 81960:
        raise ValueError("Direct classifier parameter count differs from preregistration")

    candidate_training = report["training_log_at_selected_epoch"]
    reference_training = reference["training_log_at_selected_epoch"]
    candidate_gap = {
        "accuracy": float(candidate_training["train_accuracy"] - metrics["accuracy"]),
        "macro_f1": float(candidate_training["train_macro_f1"] - metrics["macro_f1"]),
    }
    reference_gap = {
        "accuracy": float(reference_training["train_accuracy"] - reference_metrics["accuracy"]),
        "macro_f1": float(reference_training["train_macro_f1"] - reference_metrics["macro_f1"]),
    }
    report["resources"].update(
        {
            "checkpoint_bytes": int(summary["checkpoint_bytes"]["best_accuracy"]),
            "prediction_archive_bytes": int(
                summary["prediction_archive_bytes"]["best_accuracy"]
            ),
            "peak_cuda_memory_bytes": int(summary["peak_cuda_memory_bytes"]),
            "ir_route_serialized_weight_subtotal": int(
                summary["ir_route_serialized_weight_subtotal"]
            ),
        }
    )
    report.update(
        {
            "role": "train12_val2_direct_head1_matched_report",
            "decision": evaluate_direct_head_candidate(metrics, reference_metrics),
            "interpretation": (
                "Matched composite-head replacement against the frozen user6/user7 "
                "partial2 reference. This development result cannot replace canonical OOF."
            ),
            "head_replacement": "composite_projector_and_classifier_to_direct_classifier",
            "model_contract": {
                "head_type": "direct",
                "embedding_dim": 2048,
                "custom_head_parameter_count": 81960,
            },
            "matched_partial2_reference": {
                "run_id": reference["run_id"],
                "metrics": {name: float(reference_metrics[name]) for name in METRIC_NAMES},
                "report_path": str(reference_report_path.resolve()),
                "report_sha256": trainer._sha256_file(reference_report_path),
                "prediction_sha256": trainer._sha256_file(reference_archive),
            },
            "matched_delta": {
                name: float(metrics[name] - reference_metrics[name]) for name in METRIC_NAMES
            },
            "train_to_validation_gap": candidate_gap,
            "reference_train_to_validation_gap": reference_gap,
            "train_to_validation_gap_delta": {
                name: float(candidate_gap[name] - reference_gap[name])
                for name in ("accuracy", "macro_f1")
            },
            "per_user_delta": _delta_groups(
                candidate_groups["per_user"], reference_groups["per_user"]
            ),
            "duration_delta": _delta_groups(
                candidate_groups["duration"], reference_groups["duration"]
            ),
            "matched_prediction_diagnostics": diagnostics,
            "direct_classifier_relative_l2_drift_vs_initialization": (
                _direct_classifier_drift(run_directory)
            ),
            "human_review_floor": float(reference_metrics["accuracy"] - 0.02),
            "automatic_next_experiment_permitted": False,
        }
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    reference = report["matched_partial2_reference"]["metrics"]
    delta = report["matched_delta"]
    lines = [
        "# X3D-S User6/User7 Direct-Head1 Matched Result",
        "",
        "> Development-only composite-head replacement. This is not unbiased OOF and does not replace canonical Phase 4/5 evidence.",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "| Metric | Direct-Head1 | Partial2 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name in METRIC_NAMES:
        lines.append(
            f"| {name} | {metrics[name]:.6f} | {reference[name]:.6f} | {delta[name]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "The intervention replaces the learned 2048-to-256 projector plus classifier "
            "with dropout and a direct 2048-to-40 classifier. All other frozen recipe fields "
            "remain matched.",
            "",
            f"Accuracy train-to-validation gap: `{report['train_to_validation_gap']['accuracy']:.6f}`. "
            f"Macro-F1 gap: `{report['train_to_validation_gap']['macro_f1']:.6f}`.",
            "",
            f"Direct classifier relative L2 drift from its seeded initialization: "
            f"`{report['direct_classifier_relative_l2_drift_vs_initialization']:.6f}`.",
            "",
            "No follow-up IR experiment is authorized automatically by this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report matched Direct-Head1 X3D evidence")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-run-directory", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/x3d_s_train12_val2_user6_user7_direct_head1_report.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/x3d_s_train12_val2_user6_user7_direct_head1_report.md"),
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
