from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p86_mc3_visual_model import P86MC3VisualStudent
from train_p86_visual_student_oof import metric_dict


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_p86_mobind_fusion_proxy.py"
VISUAL_HEAD_TRAINER = HERE / "train_p93_visual_head_from_sequence.py"
SEQUENCE_BUILDER = HERE / "build_p86_mc3_sequence_cache.py"
MOTION_PRETRAINER = HERE / "train_p86_mobind_pretrain.py"
IMU_TEACHER = HERE / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"

SOURCE_USERS = ("user1", "user2", "user6", "user8", "user17", "user21", "user23")
H2_USERS = ("user5", "user7", "user16", "user18", "user19")
H3_USERS = ("user3", "user4", "user9", "user20", "user22", "user24")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired P86/P93-v2 source LOUO and, only after the frozen H1 "
            "gate passes, one H2 confirmation. This script has no H3 path."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "runs/p93_temporal_mobind_v2_h1h2_v1",
    )
    parser.add_argument("--stage-a-epochs", type=int, default=4)
    parser.add_argument("--stage-b-epochs", type=int, default=20)
    parser.add_argument("--visual-head-epochs", type=int, default=16)
    parser.add_argument("--motion-pretrain-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument("--temporal-residual-budget", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def completed(run: Path, train_users: tuple[str, ...], holdout_users: tuple[str, ...]) -> bool:
    summary_path = run / "summary.json"
    if not summary_path.exists():
        return False
    summary = read_json(summary_path)
    return (
        summary.get("status") == "formal_subject_holdout"
        and summary.get("training_subjects") == sorted(train_users)
        and summary.get("holdout_subjects") == sorted(holdout_users)
        and summary.get("excluded_subjects") == sorted(
            set(SOURCE_USERS + H2_USERS + H3_USERS)
            - set(train_users)
            - set(holdout_users)
        )
    )


def completed_visual(
    run: Path, train_users: tuple[str, ...], holdout_users: tuple[str, ...]
) -> bool:
    summary_path = run / "summary.json"
    if not summary_path.exists() or not (run / "visual_student.pt").exists():
        return False
    summary = read_json(summary_path)
    return (
        summary.get("status") == "formal"
        and summary.get("training_subjects") == sorted(train_users)
        and summary.get("holdout_subjects") == sorted(holdout_users)
    )


def completed_motion(
    run: Path, train_users: tuple[str, ...], holdout_users: tuple[str, ...]
) -> bool:
    summary_path = run / "summary.json"
    if not summary_path.exists() or not (run / "mobind_lite.pt").exists():
        return False
    summary = read_json(summary_path)
    return (
        summary.get("status") == "formal_subject_holdout"
        and summary.get("training_subjects") == sorted(train_users)
        and summary.get("holdout_subjects") == sorted(holdout_users)
    )


def prepare_shared_backbone(args: argparse.Namespace) -> tuple[Path, Path]:
    shared = args.output_dir.resolve() / "_shared_label_free_backbone"
    shared.mkdir(parents=True, exist_ok=True)
    checkpoint_path = shared / "kinetics_visual_init.pt"
    if not checkpoint_path.exists():
        model = P86MC3VisualStudent(
            classes=40,
            width=512,
            dropout=0.18,
            fusion_mode="gated",
            enable_distillation_projection=False,
            frames=16,
            kinetics_pretrained=True,
            temporal_modeling=True,
        )
        model.freeze_low_level("layer2")
        checkpoint = {
            "stage": "P93_label_free_kinetics_visual_init",
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_config": {
                "classes": 40,
                "width": 512,
                "dropout": 0.18,
                "fusion_mode": "gated",
                "enable_distillation_projection": False,
                "freeze_through": "layer2",
                "frames": 16,
                "input_resolution": 160,
                "backbone": "mc3_18_temporal",
                "temporal_modeling": True,
            },
            "labels_used": False,
        }
        torch.save(checkpoint, checkpoint_path)
    sequence_cache = shared / "kinetics_sequence_cache"
    cache_summary = sequence_cache / "summary.json"
    cache_complete = False
    if cache_summary.exists():
        summary = read_json(cache_summary)
        cache_complete = summary.get("completed") == summary.get("total") == 2914
    if not cache_complete:
        run_checked(
            [
                sys.executable,
                str(SEQUENCE_BUILDER),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(sequence_cache),
                "--batch-size",
                str(args.cache_batch_size),
                "--workers",
                str(args.workers),
            ]
        )
    return checkpoint_path, sequence_cache


def common_command(
    args: argparse.Namespace,
    output: Path,
    train_users: tuple[str, ...],
    holdout_users: tuple[str, ...],
    visual_checkpoint: Path,
    sequence_cache: Path,
    pretrain_checkpoint: Path,
) -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        "--visual-checkpoint",
        str(visual_checkpoint),
        "--sequence-cache",
        str(sequence_cache),
        "--pretrain-checkpoint",
        str(pretrain_checkpoint),
        "--output-dir",
        str(output),
        "--modality",
        "separate",
        "--stage-a-epochs",
        str(args.stage_a_epochs),
        "--stage-b-epochs",
        str(args.stage_b_epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--joint-freeze-pretrained-encoders",
        "--distillation-weight",
        "1.0",
        "--selective-anchor-weight",
        "0.3",
        "--reliability-weight",
        "0.0",
        "--visual-corruption-probability",
        "0.75",
        "--visual-feature-dropout",
        "0.3",
        "--visual-view-dropout",
        "0.4",
        "--live-visual-anchor",
        "--train-users",
        *train_users,
        "--subject-holdout-users",
        *holdout_users,
    ]


def run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=HERE, check=True)


def run_pair(
    args: argparse.Namespace,
    name: str,
    train_users: tuple[str, ...],
    holdout_users: tuple[str, ...],
    initial_visual_checkpoint: Path,
    sequence_cache: Path,
) -> tuple[Path, Path]:
    pair_dir = args.output_dir.resolve() / name
    visual = pair_dir / "visual_anchor"
    motion = pair_dir / "motion_pretrain"
    anchor = pair_dir / "p86_anchor"
    candidate = pair_dir / "p93_v2"
    pair_dir.mkdir(parents=True, exist_ok=True)
    if args.force or not completed_visual(visual, train_users, holdout_users):
        run_checked(
            [
                sys.executable,
                str(VISUAL_HEAD_TRAINER),
                "--initial-checkpoint",
                str(initial_visual_checkpoint),
                "--sequence-cache",
                str(sequence_cache),
                "--output-dir",
                str(visual),
                "--epochs",
                str(args.visual_head_epochs),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--train-users",
                *train_users,
                "--holdout-users",
                *holdout_users,
            ]
        )
    if args.force or not completed_motion(motion, train_users, holdout_users):
        run_checked(
            [
                sys.executable,
                str(MOTION_PRETRAINER),
                "--output-dir",
                str(motion),
                "--epochs",
                str(args.motion_pretrain_epochs),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--learning-rate",
                "0.0002",
                "--semantic-weight",
                "2.0",
                "--token-weight",
                "0.1",
                "--local-weight",
                "0.05",
                "--global-weight",
                "0.15",
                "--reconstruction-weight",
                "0.1",
                "--distillation-weight",
                "0.75",
                "--teacher-feature-weight",
                "0.25",
                "--imu-teacher-logits",
                str(IMU_TEACHER),
                "--imu-teacher-weight",
                "1.0",
                "--train-users",
                *train_users,
                "--subject-holdout-users",
                *holdout_users,
            ]
        )
    if args.force or not completed(anchor, train_users, holdout_users):
        run_checked(
            common_command(
                args,
                anchor,
                train_users,
                holdout_users,
                visual / "visual_student.pt",
                sequence_cache,
                motion / "mobind_lite.pt",
            )
            + ["--fusion-position", "clip"]
        )
    if args.force or not completed(candidate, train_users, holdout_users):
        run_checked(
            common_command(
                args,
                candidate,
                train_users,
                holdout_users,
                visual / "visual_student.pt",
                sequence_cache,
                motion / "mobind_lite.pt",
            )
            + [
                "--fusion-position",
                "temporal_v2",
                "--temporal-anchor-checkpoint",
                str(anchor / "unified_student.pt"),
                "--temporal-radius",
                str(args.temporal_radius),
                "--temporal-residual-budget",
                str(args.temporal_residual_budget),
            ]
        )
    return anchor, candidate


def align_rows(
    anchor_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    old = {row["sample_id"]: row for row in anchor_rows}
    new = {row["sample_id"]: row for row in candidate_rows}
    if set(old) != set(new):
        raise RuntimeError("paired P86/P93 sample universes differ")
    output = []
    for sample_id in sorted(old):
        anchor = old[sample_id]
        candidate = new[sample_id]
        if (anchor["label"], anchor["user_id"]) != (
            candidate["label"],
            candidate["user_id"],
        ):
            raise RuntimeError(f"paired metadata mismatch for {sample_id}")
        label = int(anchor["label"])
        old_prediction = int(anchor["prediction"])
        new_prediction = int(candidate["prediction"])
        if old_prediction != label and new_prediction == label:
            transition = "rescue"
        elif old_prediction == label and new_prediction != label:
            transition = "harm"
        elif old_prediction == label:
            transition = "stable_correct"
        else:
            transition = "stable_wrong"
        output.append(
            {
                "sample_id": sample_id,
                "user_id": anchor["user_id"],
                "label": label,
                "p86_prediction": old_prediction,
                "p93_prediction": new_prediction,
                "transition": transition,
                "p86_confidence": float(anchor["confidence"]),
                "p93_confidence": float(candidate["confidence"]),
                "temporal_residual_rms_ratio": float(
                    candidate["mean_temporal_residual_rms_ratio"]
                ),
                "temporal_skeleton_part_attention_entropy": float(
                    candidate["temporal_skeleton_part_attention_entropy"]
                ),
                "temporal_imu_part_attention_entropy": float(
                    candidate["temporal_imu_part_attention_entropy"]
                ),
            }
        )
    return output


def paired_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    users = [str(row["user_id"]) for row in rows]
    p86 = np.asarray([row["p86_prediction"] for row in rows], dtype=np.int64)
    p93 = np.asarray([row["p93_prediction"] for row in rows], dtype=np.int64)
    p86_metrics = metric_dict(labels, p86, users)
    p93_metrics = metric_dict(labels, p93, users)
    user_delta = {
        user: sum(
            int(row["p93_prediction"] == row["label"])
            - int(row["p86_prediction"] == row["label"])
            for row in rows
            if row["user_id"] == user
        )
        for user in sorted(set(users))
    }
    rescue = sum(row["transition"] == "rescue" for row in rows)
    harm = sum(row["transition"] == "harm" for row in rows)
    return {
        "p86": p86_metrics,
        "p93": p93_metrics,
        "delta_correct": int(p93_metrics["correct"] - p86_metrics["correct"]),
        "delta_accuracy_pp": 100.0
        * (p93_metrics["accuracy"] - p86_metrics["accuracy"]),
        "rescue": rescue,
        "harm": harm,
        "worst_user_delta_correct": min(user_delta.values()),
        "user_delta_correct": user_delta,
        "mean_temporal_residual_rms_ratio": float(
            np.mean([row["temporal_residual_rms_ratio"] for row in rows])
        ),
        "mean_temporal_skeleton_part_attention_entropy": float(
            np.mean(
                [
                    row["temporal_skeleton_part_attention_entropy"]
                    for row in rows
                ]
            )
        ),
        "mean_temporal_imu_part_attention_entropy": float(
            np.mean(
                [row["temporal_imu_part_attention_entropy"] for row in rows]
            )
        ),
    }


def counterfactual_totals(candidate_dirs: list[Path]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for ablation in ("zero", "reverse_time", "sample_roll"):
        rows = []
        maximum_error = 0.0
        for candidate in candidate_dirs:
            rows.extend(
                read_rows(candidate / f"counterfactual_{ablation}_predictions.csv")
            )
            audit = read_json(candidate / "summary.json")["counterfactual_audit"][
                ablation
            ]
            error = audit["maximum_logit_error_vs_p86_anchor"]
            if error is not None:
                maximum_error = max(maximum_error, float(error))
        correct = sum(int(row["prediction"]) == int(row["label"]) for row in rows)
        totals[ablation] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / max(len(rows), 1),
            "maximum_logit_error_vs_p86_anchor": (
                maximum_error if ablation == "zero" else None
            ),
        }
    return totals


def collect_pairs(pairs: list[tuple[Path, Path]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = []
    for anchor, candidate in pairs:
        rows.extend(
            align_rows(
                read_rows(anchor / "subject_holdout_predictions.csv"),
                read_rows(candidate / "subject_holdout_predictions.csv"),
            )
        )
        candidates.append(candidate)
    return rows, counterfactual_totals(candidates)


def mechanism_gate(
    paired: dict[str, Any],
    counterfactual: dict[str, Any],
    minimum_gain: int,
) -> dict[str, Any]:
    exact_zero = counterfactual["zero"]["maximum_logit_error_vs_p86_anchor"] <= 1e-4
    aligned_not_worse = paired["p93"]["correct"] >= max(
        counterfactual["reverse_time"]["correct"],
        counterfactual["sample_roll"]["correct"],
    )
    passed = (
        paired["delta_correct"] >= minimum_gain
        and paired["worst_user_delta_correct"] >= -2
        and exact_zero
        and aligned_not_worse
    )
    return {
        "passed": passed,
        "criteria": {
            "minimum_delta_correct": minimum_gain,
            "minimum_worst_user_delta_correct": -2,
            "maximum_zero_logit_error": 1e-4,
            "aligned_must_not_underperform_reverse_or_sample_roll": True,
        },
        "observed": {
            "delta_correct": paired["delta_correct"],
            "worst_user_delta_correct": paired["worst_user_delta_correct"],
            "zero_logit_error": counterfactual["zero"][
                "maximum_logit_error_vs_p86_anchor"
            ],
            "aligned_correct": paired["p93"]["correct"],
            "reverse_time_correct": counterfactual["reverse_time"]["correct"],
            "sample_roll_correct": counterfactual["sample_roll"]["correct"],
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_visual_checkpoint, sequence_cache = prepare_shared_backbone(args)
    h1_pairs = []
    for held_user in SOURCE_USERS:
        train_users = tuple(user for user in SOURCE_USERS if user != held_user)
        h1_pairs.append(
            run_pair(
                args,
                f"h1_louo_{held_user}",
                train_users,
                (held_user,),
                initial_visual_checkpoint,
                sequence_cache,
            )
        )
    h1_rows, h1_counterfactual = collect_pairs(h1_pairs)
    h1_metrics = paired_metrics(h1_rows)
    h1_gate = mechanism_gate(h1_metrics, h1_counterfactual, minimum_gain=5)
    write_rows(output / "h1_paired_predictions.csv", h1_rows)

    h2_result = None
    decision = "CLOSED_NO_EVIDENCE"
    if h1_gate["passed"]:
        h2_pair = run_pair(
            args,
            "h2_confirmation",
            SOURCE_USERS,
            H2_USERS,
            initial_visual_checkpoint,
            sequence_cache,
        )
        h2_rows, h2_counterfactual = collect_pairs([h2_pair])
        h2_metrics = paired_metrics(h2_rows)
        h2_gate = mechanism_gate(h2_metrics, h2_counterfactual, minimum_gain=1)
        write_rows(output / "h2_paired_predictions.csv", h2_rows)
        h2_result = {
            "paired": h2_metrics,
            "counterfactual": h2_counterfactual,
            "gate": h2_gate,
        }
        decision = "CLOSED_SUCCESS" if h2_gate["passed"] else "CLOSED_NO_EVIDENCE"

    summary = {
        "stage": "P93_temporal_mobind_v2_H1_LOUO_H2_frozen",
        "status": "complete",
        "decision": decision,
        "protocol": {
            "source_users": list(SOURCE_USERS),
            "h1": "paired seven-fold leave-one-user-out P86 anchor vs P93-v2",
            "h2": "one refit on all source users and frozen confirmation only if H1 passes",
            "h3_users_excluded": list(H3_USERS),
            "h3_code_path_exists": False,
            "teacher_pool_changed": False,
            "temporal_radius": args.temporal_radius,
            "temporal_residual_budget": args.temporal_residual_budget,
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.stage_b_epochs,
            "visual_head_epochs": args.visual_head_epochs,
            "motion_pretrain_epochs": args.motion_pretrain_epochs,
            "visual_backbone_initialization": "label-free Kinetics MC3-18",
            "visual_backbone_frozen_across_all_user_folds": True,
            "motion_pretrain_refit_from_scratch_per_user_fold": True,
        },
        "h1": {
            "paired": h1_metrics,
            "counterfactual": h1_counterfactual,
            "gate": h1_gate,
        },
        "h2": h2_result,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# P93-v2 temporal MoBind H1/H2 report",
        "",
        "P86's complete clip-local/global path is frozen. P93-v2 adds only a "
        "zero-initialized, fixed-budget, modality-private temporal residual.",
        "",
        f"- H1 P86: `{h1_metrics['p86']['correct']}/{h1_metrics['p86']['total']}`",
        f"- H1 P93-v2: `{h1_metrics['p93']['correct']}/{h1_metrics['p93']['total']}`",
        f"- H1 transition: `{h1_metrics['rescue']}` rescue / `{h1_metrics['harm']}` harm / net `{h1_metrics['delta_correct']}`.",
        f"- H1 worst user delta: `{h1_metrics['worst_user_delta_correct']}`.",
        f"- H1 zero residual: `{h1_counterfactual['zero']['correct']}/{h1_counterfactual['zero']['total']}`; maximum logit error vs P86 `{h1_counterfactual['zero']['maximum_logit_error_vs_p86_anchor']}`.",
        f"- H1 reverse-time: `{h1_counterfactual['reverse_time']['correct']}/{h1_counterfactual['reverse_time']['total']}`.",
        f"- H1 sample-roll: `{h1_counterfactual['sample_roll']['correct']}/{h1_counterfactual['sample_roll']['total']}`.",
        f"- H1 gate passed: `{h1_gate['passed']}`.",
        f"- Decision: `{decision}`.",
        "",
        "H3 was excluded and this orchestrator has no H3 evaluation path.",
    ]
    if h2_result is not None:
        h2 = h2_result["paired"]
        report[13:13] = [
            f"- H2 P86: `{h2['p86']['correct']}/{h2['p86']['total']}`",
            f"- H2 P93-v2: `{h2['p93']['correct']}/{h2['p93']['total']}`",
            f"- H2 transition: `{h2['rescue']}` rescue / `{h2['harm']}` harm / net `{h2['delta_correct']}`.",
            f"- H2 gate passed: `{h2_result['gate']['passed']}`.",
        ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
