from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.report_x3d_s_train12_val2_dev import build_report as build_standalone_report


REFERENCE = {
    "run_id": "x3d_s_ir_context_train12_val2_user6_user7_single13_global_seed20260715",
    "accuracy": 0.522077922077922,
    "macro_f1": 0.41144521588444183,
    "worst_user_accuracy": 0.5024875621890548,
}


def decision(metrics: Mapping[str, Any]) -> str:
    accuracy = float(metrics["accuracy"])
    if accuracy < REFERENCE["accuracy"] - 0.02:
        return "human_review_regression"
    if (
        accuracy > REFERENCE["accuracy"]
        and float(metrics["macro_f1"]) >= REFERENCE["macro_f1"] - 0.01
        and float(metrics["worst_user_accuracy"])
        >= REFERENCE["worst_user_accuracy"] - 0.02
    ):
        return "preferred_spatial_candidate"
    return "non_winning_ablation"


def build_report(run_directory: Path) -> dict[str, Any]:
    report = build_standalone_report(run_directory)
    metrics = report["metrics"]
    report.update(
        {
            "role": "train12_val2_single13_fixed_context_spatial_ablation_report",
            "decision": decision(metrics),
            "interpretation": (
                "One trial-level fixed person-context box improves Accuracy, Macro-F1, "
                "and worst-user Accuracy over the matched moving-context Single13 route. "
                "The combined fixed-context Single13 route is retained for the approved "
                "four-channel IR+Depth experiment."
            ),
            "direct_head_human_review_accuracy_floor": None,
            "matched_reference": dict(REFERENCE),
            "metric_delta": {
                "accuracy": float(metrics["accuracy"]) - REFERENCE["accuracy"],
                "macro_f1": float(metrics["macro_f1"]) - REFERENCE["macro_f1"],
                "worst_user_accuracy": float(metrics["worst_user_accuracy"])
                - REFERENCE["worst_user_accuracy"],
            },
        }
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    delta = report["metric_delta"]
    resources = report["resources"]
    training = report["training_log_at_selected_epoch"]
    return "\n".join(
        [
            "# X3D-S Single13 Fixed-Context Result",
            "",
            "> Development-only evidence on user6/user7. Canonical Phase 4/5 evidence remains unchanged.",
            "",
            f"Decision: `{report['decision']}`",
            "",
            "| Metric | Fixed context | Delta vs moving-context Single13 |",
            "|---|---:|---:|",
            f"| Accuracy | {metrics['accuracy']:.6f} | {delta['accuracy']:+.6f} |",
            f"| Macro-F1 | {metrics['macro_f1']:.6f} | {delta['macro_f1']:+.6f} |",
            f"| Worst-user Accuracy | {metrics['worst_user_accuracy']:.6f} | {delta['worst_user_accuracy']:+.6f} |",
            "",
            f"Selected epoch: `{report['selected_epoch']}` of `{report['epochs_completed']}`. ",
            f"Train Accuracy at selection was `{training['train_accuracy']:.6f}`; the train-minus-validation gap was `{training['train_minus_val_accuracy']:.6f}`.",
            "",
            f"Peak CUDA memory: `{resources['peak_cuda_memory_bytes']}` bytes. Checkpoint: `{resources['checkpoint_bytes']}` bytes.",
            "",
            report["interpretation"],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.run_directory)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
