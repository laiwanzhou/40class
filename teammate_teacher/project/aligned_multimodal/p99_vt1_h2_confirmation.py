"""One-shot joint H2 confirmation for the frozen P99-VT1 Teacher and Student."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
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
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_vt1_joint_h2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99-VT1 frozen joint H2 confirmation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_ids_users_without_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([row["sample_id"] for row in rows]),
        np.asarray([row["user_id"] for row in rows]),
    )


def run_hidden(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise RuntimeError(f"frozen H2 subprocess failed ({result.returncode}):\n{tail}")


def ensure_teacher(config: dict[str, Any]) -> tuple[Path, Path]:
    output = resolve(config["teacher_h2_output"])
    summary_path = output / "summary.json"
    predictions_path = output / "h2_predictions.npz"
    if summary_path.exists() and predictions_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("stage") != "P99_VT1_H2_confirmation":
            raise RuntimeError("existing H2 Teacher output has the wrong stage")
        return summary_path, predictions_path
    command = [
        sys.executable,
        str(HERE / "p99_visual_anchor_teacher.py"),
        "--stage", "h2_confirmation",
        "--config", str(resolve(config["teacher_config"])),
        "--visual-config", str(resolve(config["visual_config"])),
        "--v0-h1-summary", str(resolve(config["v0_h1_summary"])),
        "--h1-summary", str(resolve(config["vt1_h1_summary"])),
        "--output-dir", str(output),
    ]
    run_hidden(command, PROJECT)
    if not summary_path.exists() or not predictions_path.exists():
        raise RuntimeError("H2 Teacher completed without required artifacts")
    return summary_path, predictions_path


def build_targets(
    config: dict[str, Any], output: Path, summary_path: Path, predictions_path: Path
) -> dict[str, Any]:
    teacher_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if teacher_summary.get("stage") != "P99_VT1_H2_confirmation":
        raise ValueError("Teacher artifact is not the frozen VT1 H2 confirmation")
    with np.load(predictions_path, allow_pickle=False) as source:
        source_ids = source["sample_ids"].astype(str)
        source_users = source["users"].astype(str)
        anchor_probability = np.asarray(source["anchor_probability"], dtype=np.float32)
        vt1_probability = np.asarray(source["direct_probability"], dtype=np.float32)
        source_contains_evaluation_labels = "labels" in source.files

    base_dir = resolve(config["base_checkpoint"]).parent
    target_ids, target_users = read_ids_users_without_labels(
        base_dir / "subject_holdout_predictions.csv"
    )
    lookup = {value: index for index, value in enumerate(source_ids)}
    missing = [value for value in target_ids if value not in lookup]
    if missing:
        raise KeyError(f"H2 Teacher predictions miss {len(missing)} Student rows")
    order = np.asarray([lookup[value] for value in target_ids], dtype=np.int64)
    if not np.array_equal(source_users[order], target_users):
        raise RuntimeError("H2 Teacher/Student user alignment differs")

    target_dir = output / "targets"
    targets = {
        "anchor_control": write_target(
            target_dir / "anchor_control_targets.npz",
            target_ids,
            target_users,
            anchor_probability[order],
            "P91_source_safe_H2_anchor_fixed_0.94",
        ),
        "anchor_internvideo2_vt1": write_target(
            target_dir / "anchor_internvideo2_vt1_targets.npz",
            target_ids,
            target_users,
            vt1_probability[order],
            "P99_VT1_frozen_H2_anchor_InternVideo2_pool",
        ),
    }
    manifest = {
        "stage": "P99_VT1_H2_joint_target_build",
        "status": "complete",
        "teacher_summary": str(summary_path),
        "teacher_summary_sha256": sha256(summary_path),
        "teacher_predictions": str(predictions_path),
        "teacher_predictions_sha256": sha256(predictions_path),
        "source_contains_evaluation_labels": source_contains_evaluation_labels,
        "labels_read_or_written_to_targets": False,
        "rows": int(len(target_ids)),
        "users": sorted(set(target_users.tolist())),
        "targets": targets,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "target_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def ensure_students(config: dict[str, Any], output: Path, manifest: dict[str, Any]) -> None:
    for name in ("anchor_control", "anchor_internvideo2_vt1"):
        target = Path(manifest["targets"][name]["path"])
        branch_output = output / f"student_{name}"
        summary_path = branch_output / "summary.json"
        if summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "formal"
                and Path(previous["structured_targets"]).resolve() == target.resolve()
                and int(previous["config"]["epochs"]) == int(config["epochs"])
            ):
                continue
            raise RuntimeError(f"incompatible existing H2 Student branch: {branch_output}")
        run_hidden(branch_command(config, target, branch_output), PROJECT)


def gate_side(
    comparison: dict[str, Any],
    per_user_delta: dict[str, int],
    baseline_worst: float,
    candidate_worst: float,
    gate: dict[str, Any],
) -> dict[str, Any]:
    positive = sum(value > 0 for value in per_user_delta.values())
    nonnegative = sum(value >= 0 for value in per_user_delta.values())
    worst_regression_pp = 100.0 * (baseline_worst - candidate_worst)
    checks = {
        "net": comparison["net"] >= int(gate["minimum_net"]),
        "positive_users": positive >= int(gate["minimum_positive_users"]),
        "nonnegative_users": nonnegative >= int(gate["minimum_nonnegative_users"]),
        "worst_user": worst_regression_pp <= float(gate["maximum_worst_user_regression_pp"]),
        "paired_exact": comparison["mcnemar_exact_pvalue"]
        <= float(gate["maximum_paired_exact_pvalue"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "positive_users": int(positive),
        "nonnegative_users": int(nonnegative),
        "worst_user_regression_pp": float(worst_regression_pp),
    }


def summarize(
    config: dict[str, Any], output: Path, manifest: dict[str, Any], teacher_summary_path: Path
) -> dict[str, Any]:
    ids, labels, users = read_base_rows(
        resolve(config["base_checkpoint"]).parent / "subject_holdout_predictions.csv"
    )
    anchor_result, anchor_prediction = summarize_branch(
        "anchor_control",
        output / "student_anchor_control",
        Path(manifest["targets"]["anchor_control"]["path"]),
        ids,
        labels,
        users,
        None,
    )
    vt1_result, vt1_prediction = summarize_branch(
        "anchor_internvideo2_vt1",
        output / "student_anchor_internvideo2_vt1",
        Path(manifest["targets"]["anchor_internvideo2_vt1"]["path"]),
        ids,
        labels,
        users,
        anchor_prediction,
    )
    student_comparison = vt1_result["student_vs_anchor_control"]
    student_comparison["mcnemar_exact_pvalue"] = paired_exact_pvalue(
        labels, anchor_prediction, vt1_prediction
    )
    vt1_result["focus_group_transfer"] = focus_group_transfer_audit(
        labels, anchor_prediction, vt1_prediction, config["focus_groups"]
    )

    reference_result, _ = summarize_branch(
        "p87_structured_4epoch",
        resolve(config["reference_structured_run"]),
        resolve(config["reference_structured_targets"]),
        ids,
        labels,
        users,
        anchor_prediction,
        target_key="structured_distillation_probability",
    )

    teacher = json.loads(teacher_summary_path.read_text(encoding="utf-8"))
    teacher_comparison = dict(teacher["vs_anchor"])
    teacher_delta = {user: int(value["delta"]) for user, value in teacher["per_user"].items()}
    teacher_anchor_rates = {
        user: value["anchor_correct"] / value["rows"] for user, value in teacher["per_user"].items()
    }
    teacher_candidate_rates = {
        user: value["teacher_correct"] / value["rows"] for user, value in teacher["per_user"].items()
    }
    student_delta = {
        user: int(value["student_vs_anchor_control"])
        for user, value in vt1_result["per_user"].items()
    }
    anchor_student_rates = {
        user: value["student_correct"] / value["rows"]
        for user, value in anchor_result["per_user"].items()
    }
    vt1_student_rates = {
        user: value["student_correct"] / value["rows"]
        for user, value in vt1_result["per_user"].items()
    }
    gate = config["joint_confirmation_gate"]
    teacher_gate = gate_side(
        teacher_comparison,
        teacher_delta,
        min(teacher_anchor_rates.values()),
        min(teacher_candidate_rates.values()),
        gate,
    )
    student_gate = gate_side(
        student_comparison,
        student_delta,
        min(anchor_student_rates.values()),
        min(vt1_student_rates.values()),
        gate,
    )
    joint_passed = bool(teacher_gate["passed"] and student_gate["passed"])
    report = {
        "stage": "P99_VT1_frozen_joint_H2_confirmation",
        "status": "complete",
        "protocol": (
            "Teacher and both fixed-budget Student branches were frozen before H2; "
            "all subprocess output was withheld until the full joint comparison completed."
        ),
        "teacher": teacher,
        "students": {
            "anchor_control": anchor_result,
            "anchor_internvideo2_vt1": vt1_result,
            "p87_structured_4epoch_reference": reference_result,
        },
        "joint_confirmation": {
            "passed": joint_passed,
            "teacher_gate": teacher_gate,
            "student_gate": student_gate,
            "failure_policy": config["failure_policy"],
            "next_action": (
                "retain VT1 as the confirmed Teacher-Student shortlist"
                if joint_passed
                else "close VT1 without H2 tuning; H3 remains inaccessible"
            ),
        },
        "h3_accessed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if bool(config.get("h3_access")):
        raise ValueError("P99-VT1 joint H2 confirmation cannot access H3")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher_summary, teacher_predictions = ensure_teacher(config)
    manifest = build_targets(config, output, teacher_summary, teacher_predictions)
    ensure_students(config, output, manifest)
    report = summarize(config, output, manifest, teacher_summary)
    teacher = report["teacher"]
    students = report["students"]
    compact = {
        "teacher": {
            "anchor_correct": teacher["anchor_metrics"]["correct"],
            "candidate_correct": teacher["metrics"]["correct"],
            "top5": teacher["metrics"]["top5"],
            "vs_anchor": teacher["vs_anchor"],
            "per_user": teacher["per_user"],
        },
        "students": {
            name: {
                "teacher_correct": value["teacher_metrics"]["correct"],
                "student_correct": value["student_metrics"]["correct"],
                "student_top5": value["student_metrics"]["top5"],
                "student_macro_f1": value["student_metrics"]["macro_f1"],
                "vs_anchor": value.get("student_vs_anchor_control"),
                "per_user": value["per_user"],
                "checkpoint_bytes": value["checkpoint_bytes"],
                "ground_truth_seen": value["training_ground_truth_fields_seen"],
            }
            for name, value in students.items()
        },
        "joint_confirmation": report["joint_confirmation"],
        "h3_accessed": False,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
