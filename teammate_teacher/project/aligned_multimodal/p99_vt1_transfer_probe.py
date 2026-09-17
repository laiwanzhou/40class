"""Fixed-budget Student transfer probe for the P99-VT1 Teacher."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from p99_transfer_probe import (
    branch_command,
    read_base_rows,
    rescue_harm,
    resolve,
    sha256,
    summarize_branch,
    write_target,
)
from p99_visual_transfer_probe import focus_group_transfer_audit, paired_exact_pvalue


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/p99_vt1_transfer_h1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99-VT1 fixed-budget Student probe")
    parser.add_argument("--mode", choices=("build", "run", "summarize", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_target(config: dict[str, Any], output: Path) -> dict[str, Any]:
    teacher_summary_path = resolve(config["teacher_summary"])
    teacher_summary = json.loads(teacher_summary_path.read_text(encoding="utf-8"))
    if teacher_summary.get("stage") != "P99_VT1_H1":
        raise ValueError("teacher summary is not a P99-VT1 H1 result")
    if not teacher_summary.get("student_gate", {}).get("passed"):
        raise ValueError("P99-VT1 did not pass the pre-registered Student gate")

    source_path = resolve(config["teacher_predictions"])
    probability_key = str(config["teacher_probability_key"])
    with np.load(source_path, allow_pickle=False) as source:
        source_ids = source["sample_ids"].astype(str)
        source_users = source["users"].astype(str)
        probability = np.asarray(source[probability_key], dtype=np.float32)
        source_contains_evaluation_labels = "labels" in source.files
    anchor_path = resolve(config["anchor_control_target"])
    with np.load(anchor_path, allow_pickle=False) as anchor:
        anchor_ids = anchor["sample_ids"].astype(str)
        anchor_users = anchor["users"].astype(str)
        anchor_mask = anchor["target_mask"].astype(bool)
    selected_ids = anchor_ids[anchor_mask]
    selected_users = anchor_users[anchor_mask]
    lookup = {value: index for index, value in enumerate(source_ids)}
    missing = [value for value in selected_ids if value not in lookup]
    if missing:
        raise KeyError(f"VT1 predictions miss {len(missing)} anchor-control rows")
    order = np.asarray([lookup[value] for value in selected_ids], dtype=np.int64)
    if not np.array_equal(source_users[order], selected_users):
        raise RuntimeError("VT1 target users disagree with anchor control")

    target = write_target(
        output / "targets/anchor_internvideo2_vt1_targets.npz",
        selected_ids,
        selected_users,
        probability[order],
        "P99_VT1_anchor_InternVideo2_inner_OOF_pool",
    )
    manifest = {
        "stage": "P99_VT1_H1_transfer_target_build",
        "status": "complete",
        "teacher_summary": str(teacher_summary_path),
        "teacher_summary_sha256": sha256(teacher_summary_path),
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "source_contains_evaluation_labels": source_contains_evaluation_labels,
        "labels_read_or_written_to_target": False,
        "rows": int(len(selected_ids)),
        "users": sorted(set(selected_users.tolist())),
        "target": target,
        "anchor_control_target": str(anchor_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "target_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run_branch(config: dict[str, Any], output: Path) -> None:
    manifest_path = output / "target_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("build the VT1 transfer target before training")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = Path(manifest["target"]["path"])
    branch_output = output / "student_anchor_internvideo2_vt1"
    summary_path = branch_output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "formal"
            and Path(previous["structured_targets"]).resolve() == target.resolve()
            and int(previous["config"]["epochs"]) == int(config["epochs"])
        ):
            print(f"reuse complete VT1 branch: {branch_output}", flush=True)
            return
        raise RuntimeError(f"incompatible existing VT1 branch: {branch_output}")
    command = branch_command(config, target, branch_output)
    print(json.dumps({"branch": "anchor_internvideo2_vt1", "command": command}), flush=True)
    subprocess.run(command, cwd=resolve("."), check=True)


def summarize(config: dict[str, Any], output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "target_manifest.json").read_text(encoding="utf-8"))
    base_dir = resolve(config["base_checkpoint"]).parent
    ids, labels, users = read_base_rows(base_dir / "subject_holdout_predictions.csv")
    anchor_result, anchor_prediction = summarize_branch(
        "anchor_control",
        resolve(config["anchor_control_run"]),
        resolve(config["anchor_control_target"]),
        ids,
        labels,
        users,
        None,
    )
    vt1_result, vt1_prediction = summarize_branch(
        "anchor_internvideo2_vt1",
        output / "student_anchor_internvideo2_vt1",
        Path(manifest["target"]["path"]),
        ids,
        labels,
        users,
        anchor_prediction,
    )
    comparison = vt1_result["student_vs_anchor_control"]
    comparison["mcnemar_exact_pvalue"] = paired_exact_pvalue(
        labels, anchor_prediction, vt1_prediction
    )
    vt1_result["focus_group_transfer"] = focus_group_transfer_audit(
        labels, anchor_prediction, vt1_prediction, config["focus_groups"]
    )

    reference_dir = resolve(config["reference_structured_run"])
    reference_target = resolve(config["reference_structured_targets"])
    reference_result, reference_prediction = summarize_branch(
        "p87_structured_4epoch",
        reference_dir,
        reference_target,
        ids,
        labels,
        users,
        anchor_prediction,
        target_key="structured_distillation_probability",
    )
    reference_result["student_vs_anchor_control"] = rescue_harm(
        labels, anchor_prediction, reference_prediction
    )
    report = {
        "stage": "P99_VT1_H1_fixed_budget_transfer_probe",
        "status": "complete",
        "protocol": (
            "VT1 and anchor controls start from the identical P87-S C0 checkpoint and use "
            "the same four-epoch optimizer scope; only the label-free teacher probability differs."
        ),
        "branches": {
            "anchor_control": anchor_result,
            "anchor_internvideo2_vt1": vt1_result,
            "p87_structured_4epoch_reference": reference_result,
        },
        "selection_rule": config["student_selection"],
        "h2_h3_accessed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if bool(config.get("h2_h3_access")):
        raise ValueError("P99-VT1 H1 transfer probe cannot access H2/H3")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"build", "all"}:
        print(json.dumps(build_target(config, output), ensure_ascii=False, indent=2), flush=True)
    if args.mode in {"run", "all"}:
        run_branch(config, output)
    if args.mode in {"summarize", "all"}:
        report = summarize(config, output)
        print(
            json.dumps(
                {
                    name: {
                        "teacher_correct": value["teacher_metrics"]["correct"],
                        "student_correct": value["student_metrics"]["correct"],
                        "student_top5": value["student_metrics"]["top5"],
                        "student_macro_f1": value["student_metrics"]["macro_f1"],
                        "vs_anchor": value.get("student_vs_anchor_control"),
                        "checkpoint_bytes": value["checkpoint_bytes"],
                        "ground_truth_seen": value["training_ground_truth_fields_seen"],
                    }
                    for name, value in report["branches"].items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
