"""Equal-budget P99 Student transfer probe for the three V0 Visual experts."""

from __future__ import annotations

import argparse
import json
import math
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


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/p99_visual_transfer_h1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99 Visual Student transfer probe")
    parser.add_argument("--mode", choices=("build", "run", "summarize", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def probability_key(expert_name: str) -> str:
    return f"{expert_name}_direct_probability"


def paired_exact_pvalue(labels: np.ndarray, control: np.ndarray, candidate: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for paired correctness."""
    control_correct = control == labels
    candidate_correct = candidate == labels
    rescue = int(np.sum(~control_correct & candidate_correct))
    harm = int(np.sum(control_correct & ~candidate_correct))
    discordant = rescue + harm
    if discordant == 0:
        return 1.0
    lower = min(rescue, harm)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def focus_group_transfer_audit(
    labels: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    focus_groups: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    control_correct = control == labels
    candidate_correct = candidate == labels
    result: dict[str, dict[str, int]] = {}
    for name, classes in focus_groups.items():
        mask = np.isin(labels, np.asarray(classes, dtype=np.int64))
        result[name] = {
            "rows": int(np.sum(mask)),
            "anchor_correct": int(np.sum(mask & control_correct)),
            "student_correct": int(np.sum(mask & candidate_correct)),
            "rescue": int(np.sum(mask & ~control_correct & candidate_correct)),
            "harm": int(np.sum(mask & control_correct & ~candidate_correct)),
        }
    return result


def build_targets(config: dict[str, Any], output: Path) -> dict[str, Any]:
    source_path = resolve(config["visual_predictions"])
    anchor_path = resolve(config["anchor_control_target"])
    with np.load(source_path, allow_pickle=False) as source:
        source_ids = source["sample_ids"].astype(str)
        source_users = source["users"].astype(str)
        probabilities = {
            name: np.asarray(source[probability_key(name)], dtype=np.float32)
            for name in config["targets"]
        }
        # The source result legitimately contains evaluation labels, but target
        # construction neither reads nor copies them.
        source_contains_evaluation_labels = "labels" in source.files
    with np.load(anchor_path, allow_pickle=False) as anchor:
        anchor_ids = anchor["sample_ids"].astype(str)
        anchor_users = anchor["users"].astype(str)
        anchor_mask = anchor["target_mask"].astype(bool)
    selected_ids = anchor_ids[anchor_mask]
    selected_users = anchor_users[anchor_mask]
    lookup = {value: index for index, value in enumerate(source_ids)}
    missing = [value for value in selected_ids if value not in lookup]
    if missing:
        raise KeyError(f"Visual V0 predictions miss {len(missing)} anchor-control rows")
    order = np.asarray([lookup[value] for value in selected_ids], dtype=np.int64)
    if not np.array_equal(source_users[order], selected_users):
        raise RuntimeError("Visual target users disagree with anchor control")

    target_dir = output / "targets"
    targets = {
        name: write_target(
            target_dir / f"{name}_targets.npz",
            selected_ids,
            selected_users,
            probabilities[name][order],
            f"P99_V0_{name}_source_OOF",
        )
        for name in config["targets"]
    }
    manifest = {
        "stage": "P99_V0_H1_visual_transfer_target_build",
        "status": "complete",
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "source_contains_evaluation_labels": source_contains_evaluation_labels,
        "labels_read_or_written_to_targets": False,
        "rows": int(len(selected_ids)),
        "users": sorted(set(selected_users.tolist())),
        "targets": targets,
        "anchor_control_target": str(anchor_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "target_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run_branches(config: dict[str, Any], output: Path) -> None:
    manifest_path = output / "target_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("build Visual transfer targets before training")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in config["targets"]:
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
                print(f"reuse complete branch {name}: {branch_output}", flush=True)
                continue
            raise RuntimeError(f"incompatible existing Visual branch: {branch_output}")
        command = branch_command(config, target, branch_output)
        print(json.dumps({"branch": name, "command": command}, ensure_ascii=False), flush=True)
        subprocess.run(command, cwd=resolve("."), check=True)


def summarize(config: dict[str, Any], output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "target_manifest.json").read_text(encoding="utf-8"))
    base_dir = resolve(config["base_checkpoint"]).parent
    ids, labels, users = read_base_rows(base_dir / "subject_holdout_predictions.csv")
    anchor_run = resolve(config["anchor_control_run"])
    anchor_target = resolve(config["anchor_control_target"])
    anchor_result, anchor_prediction = summarize_branch(
        "anchor_control", anchor_run, anchor_target, ids, labels, users, None
    )
    branches: dict[str, Any] = {"anchor_control": anchor_result}
    for name in config["targets"]:
        result, prediction = summarize_branch(
            name,
            output / f"student_{name}",
            Path(manifest["targets"][name]["path"]),
            ids,
            labels,
            users,
            anchor_prediction,
        )
        result["student_vs_anchor_control"]["mcnemar_exact_pvalue"] = paired_exact_pvalue(
            labels, anchor_prediction, prediction
        )
        result["focus_group_transfer"] = focus_group_transfer_audit(
            labels, anchor_prediction, prediction, config["focus_groups"]
        )
        branches[name] = result

    reference_dir = resolve(config["reference_structured_run"])
    reference_target = resolve(config["reference_structured_targets"])
    if reference_dir.joinpath("summary.json").exists():
        reference, reference_prediction = summarize_branch(
            "p87_structured_4epoch",
            reference_dir,
            reference_target,
            ids,
            labels,
            users,
            anchor_prediction,
            target_key="structured_distillation_probability",
        )
        reference["student_vs_anchor_control"] = rescue_harm(
            labels, anchor_prediction, reference_prediction
        )
        branches["p87_structured_4epoch_reference"] = reference

    report = {
        "stage": "P99_V0_H1_fixed_budget_visual_transfer_probe",
        "status": "complete",
        "protocol": (
            "All new Visual branches start from the identical P87-S C0 checkpoint, "
            "train four epochs with the same optimizer scope, and differ only in the "
            "label-free source-OOF teacher probability. The completed anchor branch is reused."
        ),
        "branches": branches,
        "selection_rule": config["selection"],
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
        raise ValueError("P99 Visual H1 transfer probe cannot access H2/H3")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"build", "all"}:
        print(json.dumps(build_targets(config, output), ensure_ascii=False, indent=2), flush=True)
    if args.mode in {"run", "all"}:
        run_branches(config, output)
    if args.mode in {"summarize", "all"}:
        report = summarize(config, output)
        compact = {
            name: {
                "teacher_correct": value["teacher_metrics"]["correct"],
                "teacher_top5": value["teacher_metrics"]["top5"],
                "student_correct": value["student_metrics"]["correct"],
                "student_top5": value["student_metrics"]["top5"],
                "student_macro_f1": value["student_metrics"]["macro_f1"],
                "gap": value["teacher_to_student_correct_gap"],
                "vs_anchor": value.get("student_vs_anchor_control"),
                "bytes": value["checkpoint_bytes"],
                "ground_truth_seen": value["training_ground_truth_fields_seen"],
            }
            for name, value in report["branches"].items()
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
