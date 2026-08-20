from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.report_x3d_s_train12_val2_dev import build_report as build_standalone_report


REFERENCE = {
    "run_id": "x3d_s_ir_context_train12_val2_user6_user7_single13_fixed_context_seed20260715",
    "accuracy": 0.5376623376623376,
    "macro_f1": 0.4275141328945328,
    "worst_user_accuracy": 0.5174129353233831,
}
STABILITY_REVIEW_ACCURACY_GATE = 0.63
LOADER_AMENDMENT = {
    "manual_user_authorization": True,
    "num_workers": 4,
    "persistent_workers": False,
    "memory_smoke_decision": "pass_workers4_nonpersistent",
    "peak_root_and_descendant_working_set_gib": 5.683,
    "minimum_system_available_gib": 13.342,
    "fallback_to_workers2_required": False,
}


def decision(metrics: Mapping[str, Any]) -> str:
    accuracy = float(metrics["accuracy"])
    if accuracy < REFERENCE["accuracy"] - 0.02:
        return "human_review_regression"
    if accuracy >= STABILITY_REVIEW_ACCURACY_GATE:
        return "eligible_for_manual_stability_review"
    if (
        accuracy > REFERENCE["accuracy"]
        and float(metrics["macro_f1"]) >= REFERENCE["macro_f1"] - 0.01
        and float(metrics["worst_user_accuracy"])
        >= REFERENCE["worst_user_accuracy"] - 0.02
    ):
        return "improved_but_below_stability_gate"
    return "non_winning_ablation"


def build_report(run_directory: Path) -> dict[str, Any]:
    report = build_standalone_report(run_directory)
    metrics = report["metrics"]
    report.update(
        {
            "role": "train12_val2_ir_depth4_early_fusion_report",
            "decision": decision(metrics),
            "interpretation": (
                "Aligned Depth_Color RGB and IR gray use one fixed trial context and "
                "one globally stratified 13-frame clip. Stability experiments remain "
                "locked unless this single-seed development Accuracy reaches 0.63."
            ),
            "direct_head_human_review_accuracy_floor": None,
            "matched_reference": dict(REFERENCE),
            "stability_review_accuracy_gate": STABILITY_REVIEW_ACCURACY_GATE,
            "stability_work_automatically_authorized": False,
            "loader_protocol_amendment": dict(LOADER_AMENDMENT),
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
    training = report["training_log_at_selected_epoch"]
    return "\n".join(
        [
            "# X3D-S Single13 Fixed-Context IR+Depth4 Result",
            "",
            "> Development-only evidence on user6/user7. No three-fold or multi-seed stability work is authorized by this report.",
            "",
            f"Decision: `{report['decision']}`",
            "",
            "| Metric | IR+Depth4 | Delta vs IR fixed context |",
            "|---|---:|---:|",
            f"| Accuracy | {metrics['accuracy']:.6f} | {delta['accuracy']:+.6f} |",
            f"| Macro-F1 | {metrics['macro_f1']:.6f} | {delta['macro_f1']:+.6f} |",
            f"| Worst-user Accuracy | {metrics['worst_user_accuracy']:.6f} | {delta['worst_user_accuracy']:+.6f} |",
            "",
            f"Selected epoch: `{report['selected_epoch']}` of `{report['epochs_completed']}`. Train-minus-validation Accuracy gap: `{training['train_minus_val_accuracy']:.6f}`.",
            "",
            f"The frozen eligibility gate for a separately approved stability review is Accuracy >= `{report['stability_review_accuracy_gate']:.2f}`.",
            "",
            "The four-worker non-persistent loader was manually authorized after the original workers=0 run completed one CPU-limited epoch. A CUDA memory smoke passed with 13.342 GiB minimum system memory available; this operational amendment did not change the scientific intervention.",
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
