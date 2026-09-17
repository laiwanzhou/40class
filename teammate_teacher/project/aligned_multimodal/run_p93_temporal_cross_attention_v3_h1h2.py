from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import run_p93_temporal_mobind_v2_h1h2 as v2
from train_p86_visual_student_oof import metric_dict


HERE = Path(__file__).resolve().parent
SOURCE_USERS = v2.SOURCE_USERS
H2_USERS = v2.H2_USERS
H3_USERS = v2.H3_USERS


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(HERE)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled P93-v3 local cross-attention pooling test on the "
            "exact paired P86 anchors from P93-v2. H2 runs only after the frozen "
            "H1 gate passes; this script has no H3 path."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "runs/p93_temporal_cross_attention_v3_h1h2_v1",
    )
    parser.add_argument(
        "--paired-anchor-root",
        type=Path,
        default=HERE / "runs/p93_temporal_mobind_v2_h1h2_v1",
        help="Root containing the leakage-safe P86 anchors frozen for P93-v2.",
    )
    parser.add_argument("--stage-a-epochs", type=int, default=4)
    parser.add_argument("--stage-b-epochs", type=int, default=20)
    parser.add_argument("--visual-head-epochs", type=int, default=16)
    parser.add_argument("--motion-pretrain-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument("--temporal-attention-logit-limit", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def completed_candidate(
    run: Path,
    train_users: tuple[str, ...],
    holdout_users: tuple[str, ...],
) -> bool:
    if not v2.completed(run, train_users, holdout_users):
        return False
    summary = v2.read_json(run / "summary.json")
    return summary.get("fusion_position") == "temporal_v3"


def shared_backbone(args: argparse.Namespace) -> tuple[Path, Path]:
    shared_args = argparse.Namespace(
        **{**vars(args), "output_dir": args.paired_anchor_root.resolve()}
    )
    return v2.prepare_shared_backbone(shared_args)


def prepare_anchor(
    args: argparse.Namespace,
    name: str,
    train_users: tuple[str, ...],
    holdout_users: tuple[str, ...],
    initial_visual_checkpoint: Path,
    sequence_cache: Path,
) -> tuple[Path, Path, Path]:
    frozen_pair = args.paired_anchor_root.resolve() / name
    frozen_visual = frozen_pair / "visual_anchor"
    frozen_motion = frozen_pair / "motion_pretrain"
    frozen_anchor = frozen_pair / "p86_anchor"
    if (
        v2.completed_visual(frozen_visual, train_users, holdout_users)
        and v2.completed_motion(frozen_motion, train_users, holdout_users)
        and v2.completed(frozen_anchor, train_users, holdout_users)
    ):
        return frozen_visual, frozen_motion, frozen_anchor

    # H1 should normally reuse the exact P93-v2 assets. If H1 passes and no H2
    # anchor exists yet, build H2 with the identical frozen subject-safe recipe.
    pair_dir = args.output_dir.resolve() / name / "paired_anchor_assets"
    visual = pair_dir / "visual_anchor"
    motion = pair_dir / "motion_pretrain"
    anchor = pair_dir / "p86_anchor"
    pair_dir.mkdir(parents=True, exist_ok=True)
    if args.force or not v2.completed_visual(visual, train_users, holdout_users):
        v2.run_checked(
            [
                sys.executable,
                str(v2.VISUAL_HEAD_TRAINER),
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
    if args.force or not v2.completed_motion(motion, train_users, holdout_users):
        v2.run_checked(
            [
                sys.executable,
                str(v2.MOTION_PRETRAINER),
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
                str(v2.IMU_TEACHER),
                "--imu-teacher-weight",
                "1.0",
                "--train-users",
                *train_users,
                "--subject-holdout-users",
                *holdout_users,
            ]
        )
    if args.force or not v2.completed(anchor, train_users, holdout_users):
        v2.run_checked(
            v2.common_command(
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
    return visual, motion, anchor


def run_pair(
    args: argparse.Namespace,
    name: str,
    train_users: tuple[str, ...],
    holdout_users: tuple[str, ...],
    initial_visual_checkpoint: Path,
    sequence_cache: Path,
) -> tuple[Path, Path]:
    visual, motion, anchor = prepare_anchor(
        args,
        name,
        train_users,
        holdout_users,
        initial_visual_checkpoint,
        sequence_cache,
    )
    candidate = args.output_dir.resolve() / name / "p93_v3"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not completed_candidate(candidate, train_users, holdout_users):
        v2.run_checked(
            v2.common_command(
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
                "temporal_v3",
                "--temporal-anchor-checkpoint",
                str(anchor / "unified_student.pt"),
                "--temporal-radius",
                str(args.temporal_radius),
                "--temporal-attention-logit-limit",
                str(args.temporal_attention_logit_limit),
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
        raise RuntimeError("paired P86/P93-v3 sample universes differ")
    output: list[dict[str, Any]] = []
    audit_fields = (
        "mean_temporal_effect_rms_ratio",
        "mean_temporal_skeleton_weight",
        "mean_temporal_imu_weight",
        "temporal_skeleton_availability",
        "temporal_imu_availability",
        "temporal_skeleton_cross_attention_entropy",
        "temporal_imu_cross_attention_entropy",
        "temporal_skeleton_mean_abs_offset",
        "temporal_imu_mean_abs_offset",
        "temporal_pool_attention_entropy",
        "temporal_pool_weight_l1_from_uniform",
        "mean_temporal_pool_logit_abs",
    )
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
                **{field: float(candidate[field]) for field in audit_fields},
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
    audit_fields = (
        "mean_temporal_effect_rms_ratio",
        "mean_temporal_skeleton_weight",
        "mean_temporal_imu_weight",
        "temporal_skeleton_availability",
        "temporal_imu_availability",
        "temporal_skeleton_cross_attention_entropy",
        "temporal_imu_cross_attention_entropy",
        "temporal_skeleton_mean_abs_offset",
        "temporal_imu_mean_abs_offset",
        "temporal_pool_attention_entropy",
        "temporal_pool_weight_l1_from_uniform",
        "mean_temporal_pool_logit_abs",
    )
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
        **{
            (field if field.startswith("mean_") else f"mean_{field}"): float(
                np.mean([row[field] for row in rows])
            )
            for field in audit_fields
        },
    }


def collect_pairs(
    pairs: list[tuple[Path, Path]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = []
    for anchor, candidate in pairs:
        rows.extend(
            align_rows(
                v2.read_rows(anchor / "subject_holdout_predictions.csv"),
                v2.read_rows(candidate / "subject_holdout_predictions.csv"),
            )
        )
        candidates.append(candidate)
    return rows, v2.counterfactual_totals(candidates)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_visual_checkpoint, sequence_cache = shared_backbone(args)
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
    h1_gate = v2.mechanism_gate(h1_metrics, h1_counterfactual, minimum_gain=5)
    v2.write_rows(output / "h1_paired_predictions.csv", h1_rows)

    h2_result = None
    decision = "CLOSED_TEMPORAL_FUSION_NO_EVIDENCE"
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
        h2_gate = v2.mechanism_gate(
            h2_metrics, h2_counterfactual, minimum_gain=1
        )
        h2_result = {
            "paired": h2_metrics,
            "counterfactual": h2_counterfactual,
            "gate": h2_gate,
        }
        v2.write_rows(output / "h2_paired_predictions.csv", h2_rows)
        decision = (
            "TEMPORAL_INTERACTION_SUPPORTED"
            if h2_gate["passed"]
            else "CLOSED_TEMPORAL_FUSION_NO_EVIDENCE"
        )

    summary = {
        "stage": "P93_temporal_cross_attention_pool_v3_H1_LOUO_H2_frozen",
        "status": "complete",
        "decision": decision,
        "protocol": {
            "only_changed_variable": (
                "replace v2 motion-vector residual injection with modality-private "
                "local cross-attention conditioned visual temporal pooling"
            ),
            "source_users": list(SOURCE_USERS),
            "h1": "paired seven-fold leave-one-user-out frozen P86 vs P93-v3",
            "h2": "one frozen confirmation only if H1 passes",
            "h3_users_excluded": list(H3_USERS),
            "h3_code_path_exists": False,
            "paired_anchor_root": display_path(args.paired_anchor_root),
            "p86_anchor_reused_without_retraining_for_h1": True,
            "teacher_pool_changed": False,
            "backbone_changed": False,
            "loss_changed": False,
            "training_protocol_changed": False,
            "temporal_radius": args.temporal_radius,
            "maximum_pooling_logit": args.temporal_attention_logit_limit,
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.stage_b_epochs,
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
        "# P93-v3 local cross-attention temporal-pooling report",
        "",
        "The exact P86 anchor is frozen. Local Visual-to-Skeleton/IMU cross-attention "
        "changes only the weights of existing visual temporal tokens; motion vectors "
        "are not injected into the visual representation.",
        "",
        f"- H1 P86: `{h1_metrics['p86']['correct']}/{h1_metrics['p86']['total']}`",
        f"- H1 P93-v3: `{h1_metrics['p93']['correct']}/{h1_metrics['p93']['total']}`",
        f"- H1 transition: `{h1_metrics['rescue']}` rescue / `{h1_metrics['harm']}` harm / net `{h1_metrics['delta_correct']}`.",
        f"- H1 worst user delta: `{h1_metrics['worst_user_delta_correct']}`.",
        f"- H1 zero interaction: `{h1_counterfactual['zero']['correct']}/{h1_counterfactual['zero']['total']}`; maximum logit error vs P86 `{h1_counterfactual['zero']['maximum_logit_error_vs_p86_anchor']}`.",
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
            f"- H2 P93-v3: `{h2['p93']['correct']}/{h2['p93']['total']}`",
            f"- H2 transition: `{h2['rescue']}` rescue / `{h2['harm']}` harm / net `{h2['delta_correct']}`.",
            f"- H2 gate passed: `{h2_result['gate']['passed']}`.",
        ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
