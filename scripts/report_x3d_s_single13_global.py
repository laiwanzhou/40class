from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.report_x3d_s_train12_val2_dev import build_report as build_standalone_report


REFERENCE = {
    "run_id": "x3d_s_ir_context_train12_val2_user6_user7_partial2_seed20260715",
    "accuracy": 0.5324675324675324,
    "macro_f1": 0.42157462519220035,
    "worst_user_accuracy": 0.5323383084577115,
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
        return "preferred_temporal_candidate"
    return "non_winning_ablation"


def build_report(run_directory: Path) -> dict[str, Any]:
    report = build_standalone_report(run_directory)
    metrics = report["metrics"]
    report.update(
        {
            "role": "train12_val2_single13_global_temporal_ablation_report",
            "decision": decision(metrics),
            "interpretation": (
                "Replacing adaptive Kx13 local windows with one globally stratified "
                "13-frame clip reduces compute and the selected-epoch train/validation "
                "gap, but does not improve the matched user6/user7 development metrics."
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
            "# X3D-S Single13-Global Result",
            "",
            "> Development-only evidence on user6/user7. Canonical Phase 4/5 evidence remains unchanged.",
            "",
            f"Decision: `{report['decision']}`",
            "",
            "| Metric | Single13 | Delta vs adaptive Partial2 |",
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
