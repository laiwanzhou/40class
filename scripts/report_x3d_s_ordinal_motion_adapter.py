from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.report_x3d_s_train12_val2_dev import build_report as build_base_report
import src.train_x3d_s_visual_expert as trainer


EXPECTED_RUN_ID = (
    "x3d_s_ir_ordinal_motion_adapter_train12_val2_user6_user7_"
    "single13_fixed_context_workers4_seed20260715"
)
ANCHOR_REFERENCE = {
    "run_id": (
        "x3d_s_ir_anchored_depth_adapter_train12_val2_user6_user7_"
        "single13_fixed_context_workers4_seed20260715"
    ),
    "accuracy": 0.548051948051948,
    "macro_f1": 0.4330774727821246,
    "worst_user_accuracy": 0.5434782608695652,
}
IR_REFERENCE = {
    "run_id": "x3d_s_ir_context_train12_val2_user6_user7_single13_fixed_context_seed20260715",
    "accuracy": 0.5376623376623376,
    "macro_f1": 0.4275141328945328,
    "worst_user_accuracy": 0.5174129353233831,
}
FAILED_EXPANDED_STEM = {
    "run_id": "x3d_s_ir_depth4_train12_val2_user6_user7_single13_fixed_context_workers4_seed20260715",
    "accuracy": 0.45714285714285713,
    "macro_f1": 0.35535251791744926,
    "worst_user_accuracy": 0.42786069651741293,
}
STABILITY_REVIEW_ACCURACY_GATE = 0.63
HUMAN_REVIEW_ACCURACY_FLOOR = ANCHOR_REFERENCE["accuracy"] - 0.02
CHANNEL_ORDER = [
    "depth_relative_displacement",
    "depth_time_normalized_velocity",
    "depth_motion_magnitude",
    "ir_gray",
]
DEPTH_MOTION_CONTRACT = {
    "source": "inverse_opencv_jet_ordinal",
    "reference": "per_pixel_temporal_median",
    "displacement_scale": "selected_clip_valid_ordinal_iqr_min_8",
    "velocity_scale": "8_ordinal_levels_per_source_frame",
    "velocity_timebase": "source_frame_index",
    "clip_range": [-1.0, 1.0],
    "invalid_policy": "strict_temporal_intersection_zero_fill",
}


def decision(metrics: Mapping[str, Any]) -> str:
    accuracy = float(metrics["accuracy"])
    if accuracy < HUMAN_REVIEW_ACCURACY_FLOOR:
        return "human_review_regression"
    if accuracy >= STABILITY_REVIEW_ACCURACY_GATE:
        return "eligible_for_manual_stability_review"
    if accuracy > ANCHOR_REFERENCE["accuracy"]:
        return "improved_but_below_stability_gate"
    return "non_winning_ablation"


def validate_candidate_identity(
    *,
    run_id: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    computed_config_sha256: str,
) -> None:
    if run_id != EXPECTED_RUN_ID:
        raise ValueError("Ordinal-motion adapter report received the wrong run ID")
    if config.get("input_view") != "depth_ordinal_motion_plus_ir_gray":
        raise ValueError("Ordinal-motion adapter requires its registered input view")
    if int(config.get("input_channels", 0)) != 4:
        raise ValueError("Ordinal-motion adapter requires four input channels")
    if config.get("fusion_strategy") != "ir_anchored_depth_residual":
        raise ValueError("Ordinal-motion adapter fusion strategy changed")
    early_fusion = config.get("early_fusion", {})
    if not isinstance(early_fusion, Mapping):
        raise ValueError("early_fusion must be a mapping")
    if list(early_fusion.get("channel_order", ())) != CHANNEL_ORDER:
        raise ValueError("Ordinal-motion channel order changed")
    if early_fusion.get("stem_initialization") != (
        "standard_k400_rgb_after_ir_anchor_zero_depth_residual"
    ):
        raise ValueError("IR-anchor and zero-Depth initialization changed")
    if dict(config.get("depth_motion", {})) != DEPTH_MOTION_CONTRACT:
        raise ValueError("Ordinal Depth motion contract changed")
    optimizer = config.get("optimizer", {})
    if not isinstance(optimizer, Mapping) or float(
        optimizer.get("input_adapter_lr", 0.0)
    ) != 3e-4:
        raise ValueError("Input adapter learning rate changed")
    temporal = config.get("temporal", {})
    spatial = config.get("spatial_input", {})
    loader = config.get("loader", {})
    if not isinstance(temporal, Mapping) or (
        temporal.get("sampling_mode") != "global_single_clip"
        or int(temporal.get("local_frames", 0)) != 13
    ):
        raise ValueError("Single13 temporal contract changed")
    if not isinstance(spatial, Mapping) or spatial.get("crop_mode") != (
        "fixed_trial_person_context"
    ):
        raise ValueError("Fixed trial context contract changed")
    if not isinstance(loader, Mapping) or (
        int(loader.get("num_workers", -1)) != 4
        or loader.get("persistent_workers") is not False
    ):
        raise ValueError("Four-worker non-persistent loader contract changed")
    if provenance.get("development_split_name") != "train12_val2_user6_user7":
        raise ValueError("Development split changed")
    if int(provenance.get("seed", -1)) != 20260715:
        raise ValueError("Canonical development seed changed")
    if provenance.get("smoke_test") is not False:
        raise ValueError("Smoke evidence cannot be reported as the formal result")
    expected_hashes = {
        str(summary.get("resolved_config_sha256", "")),
        str(provenance.get("resolved_config_sha256", "")),
        str(computed_config_sha256),
    }
    if len(expected_hashes) != 1 or len(next(iter(expected_hashes))) != 64:
        raise ValueError("Resolved config SHA256 identity check failed")


def build_report(run_directory: Path) -> dict[str, Any]:
    config = yaml.safe_load(
        (run_directory / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (run_directory / "run_summary.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (run_directory / "development_provenance.json").read_text(encoding="utf-8")
    )
    if not isinstance(config, Mapping):
        raise ValueError("Resolved config must be a mapping")
    validate_candidate_identity(
        run_id=run_directory.name,
        config=config,
        summary=summary,
        provenance=provenance,
        computed_config_sha256=trainer.resolved_config_sha256(config),
    )
    report = build_base_report(run_directory)
    metrics = report["metrics"]
    report.update(
        {
            "role": "train12_val2_ordinal_motion_adapter_report",
            "decision": decision(metrics),
            "interpretation": (
                "Depth is represented only by fixed-context aligned ordinal relative "
                "displacement, source-time-normalized velocity, and motion magnitude. "
                "The unchanged repeated-IR path remains exact at initialization and "
                "Depth enters through the zero-initialized nine-parameter residual adapter."
            ),
            "matched_anchor_reference": dict(ANCHOR_REFERENCE),
            "matched_ir_reference": dict(IR_REFERENCE),
            "failed_expanded_stem_reference": dict(FAILED_EXPANDED_STEM),
            "stability_review_accuracy_gate": STABILITY_REVIEW_ACCURACY_GATE,
            "stability_work_automatically_authorized": False,
            "human_review_accuracy_floor": HUMAN_REVIEW_ACCURACY_FLOOR,
            "metric_delta_vs_anchor": {
                "accuracy": float(metrics["accuracy"]) - ANCHOR_REFERENCE["accuracy"],
                "macro_f1": float(metrics["macro_f1"]) - ANCHOR_REFERENCE["macro_f1"],
                "worst_user_accuracy": float(metrics["worst_user_accuracy"])
                - ANCHOR_REFERENCE["worst_user_accuracy"],
            },
            "metric_delta_vs_ir": {
                "accuracy": float(metrics["accuracy"]) - IR_REFERENCE["accuracy"],
                "macro_f1": float(metrics["macro_f1"]) - IR_REFERENCE["macro_f1"],
                "worst_user_accuracy": float(metrics["worst_user_accuracy"])
                - IR_REFERENCE["worst_user_accuracy"],
            },
        }
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    anchor = report["metric_delta_vs_anchor"]
    ir = report["metric_delta_vs_ir"]
    rows = [
        ("Accuracy", "accuracy"),
        ("Macro-F1", "macro_f1"),
        ("Worst-user Accuracy", "worst_user_accuracy"),
    ]
    lines = [
        "# X3D-S Ordinal Depth Motion Adapter Result",
        "",
        "> Development-only evidence on user6/user7. No folds, extra seeds, heldout evaluation, or stability work are authorized.",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "| Metric | Candidate | Delta vs absolute-Depth anchor | Delta vs IR |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {label} | {metrics[key]:.6f} | {anchor[key]:+.6f} | {ir[key]:+.6f} |"
        for label, key in rows
    )
    lines.extend(
        [
            "",
            f"Selected epoch: `{report['selected_epoch']}` of `{report['epochs_completed']}`.",
            "",
            f"Human review is mandatory below Accuracy `{report['human_review_accuracy_floor']:.6f}`; artifacts remain preserved.",
            "",
            f"A separately approved stability review requires Accuracy >= `{report['stability_review_accuracy_gate']:.2f}`.",
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


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
