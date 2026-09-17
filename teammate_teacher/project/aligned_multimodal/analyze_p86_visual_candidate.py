from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_p86_visual_student_fold0 import metrics, write_csv


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Outer confirmation and error audit for a P86 visual candidate."
    )
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument(
        "--rollback-run",
        type=Path,
        default=PROJECT_DIR / "runs/p86_visual_pixel_fold0_hybrid_v2",
    )
    parser.add_argument(
        "--p12-oof",
        type=Path,
        default=PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz",
    )
    parser.add_argument(
        "--teacher-oof",
        type=Path,
        default=PROJECT_DIR
        / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def align_rows(rows: list[dict[str, str]], sample_ids: list[str]) -> np.ndarray:
    lookup = {row["sample_id"]: int(row["prediction"]) for row in rows}
    missing = set(sample_ids) - set(lookup)
    if missing:
        raise RuntimeError(f"prediction file is missing samples: {sorted(missing)[:3]}")
    return np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)


def align_npz(
    path: Path,
    sample_ids: list[str],
    *,
    prediction_key: str | None = None,
    logits_key: str | None = None,
) -> np.ndarray:
    with np.load(path.resolve(), allow_pickle=False) as data:
        lookup = {str(sample_id): index for index, sample_id in enumerate(data["sample_ids"])}
        indices = np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
        if prediction_key is not None:
            return np.asarray(data[prediction_key][indices], dtype=np.int64)
        if logits_key is None:
            raise ValueError("prediction_key or logits_key required")
        return np.asarray(data[logits_key][indices]).argmax(1).astype(np.int64)


def metrics_with_users(
    labels: np.ndarray, prediction: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    result = dict(metrics(labels, prediction))
    per_user = {
        str(user): float((prediction[users == user] == labels[users == user]).mean())
        for user in sorted(set(users))
    }
    result["per_user_accuracy"] = per_user
    result["worst_subject_accuracy"] = min(per_user.values())
    return result


def main() -> None:
    args = parse_args()
    candidate_run = args.candidate_run.resolve()
    run_summary = json.loads(
        (candidate_run / "summary.json").read_text(encoding="utf-8")
    )
    if run_summary.get("status") == "rejected_before_outer_held":
        raise RuntimeError("candidate failed the inner gate and has no legal outer predictions")
    candidate_rows = read_predictions(candidate_run / "outer_predictions.csv")
    sample_ids = [row["sample_id"] for row in candidate_rows]
    labels = np.asarray([int(row["label"]) for row in candidate_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in candidate_rows])
    predictions = {
        "candidate": np.asarray(
            [int(row["prediction"]) for row in candidate_rows], dtype=np.int64
        ),
        "v2_rollback": align_rows(
            read_predictions(args.rollback_run.resolve() / "outer_predictions.csv"), sample_ids
        ),
        "p12_depth": align_npz(args.p12_oof, sample_ids, logits_key="depth_logits"),
        "p12_final": align_npz(
            args.p12_oof, sample_ids, prediction_key="final_predictions"
        ),
        "large_teacher": align_npz(
            args.teacher_oof, sample_ids, logits_key="early_late_logits"
        ),
    }
    overall = {
        name: metrics_with_users(labels, prediction, users)
        for name, prediction in predictions.items()
    }

    pairwise: list[dict[str, Any]] = []
    for baseline in ("v2_rollback", "p12_depth", "p12_final", "large_teacher"):
        candidate_correct = predictions["candidate"] == labels
        baseline_correct = predictions[baseline] == labels
        rescued = candidate_correct & ~baseline_correct
        harmed = ~candidate_correct & baseline_correct
        pairwise.append(
            {
                "candidate": "candidate",
                "baseline": baseline,
                "rescued": int(rescued.sum()),
                "harmed": int(harmed.sum()),
                "net_correct": int(rescued.sum() - harmed.sum()),
                "both_correct": int((candidate_correct & baseline_correct).sum()),
                "both_wrong": int((~candidate_correct & ~baseline_correct).sum()),
            }
        )

    per_user: list[dict[str, Any]] = []
    improved_users = 0
    for user in sorted(set(users)):
        mask = users == user
        row: dict[str, Any] = {"user_id": str(user), "samples": int(mask.sum())}
        for name, prediction in predictions.items():
            row[f"{name}_accuracy"] = float((prediction[mask] == labels[mask]).mean())
        row["candidate_vs_v2_delta"] = (
            row["candidate_accuracy"] - row["v2_rollback_accuracy"]
        )
        improved_users += int(row["candidate_vs_v2_delta"] > 0.0)
        per_user.append(row)

    per_class: list[dict[str, Any]] = []
    for class_id in range(40):
        mask = labels == class_id
        row: dict[str, Any] = {"class_id": class_id, "samples": int(mask.sum())}
        for name, prediction in predictions.items():
            row[f"{name}_correct"] = int((prediction[mask] == labels[mask]).sum())
            row[f"{name}_accuracy"] = float((prediction[mask] == labels[mask]).mean())
        row["candidate_vs_v2_correct_delta"] = (
            row["candidate_correct"] - row["v2_rollback_correct"]
        )
        per_class.append(row)

    candidate_metrics = overall["candidate"]
    rollback_metrics = overall["v2_rollback"]
    depth_metrics = overall["p12_depth"]
    outer_confirmed = (
        candidate_metrics["accuracy"] > max(
            rollback_metrics["accuracy"], depth_metrics["accuracy"]
        )
        and candidate_metrics["macro_f1"] >= rollback_metrics["macro_f1"] - 0.005
        and candidate_metrics["worst_subject_accuracy"]
        >= rollback_metrics["worst_subject_accuracy"] - 0.02
        and improved_users >= 4
    )
    decision = {
        "outer_confirmed": bool(outer_confirmed),
        "improved_subjects_vs_v2": improved_users,
        "total_outer_subjects": len(set(users)),
        "criteria": (
            "Candidate must exceed both V2 and P12 depth accuracy, preserve V2 macro-F1 within "
            "0.5pp and worst-subject within 2pp, and improve at least four of six subjects."
        ),
        "next_action": (
            "freeze_candidate_as_visual_anchor_and_begin_modality_increment_audit"
            if outer_confirmed
            else "retain_v2_rollback_and_continue_visual_stage_error_driven_iteration"
        ),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "pairwise.csv", pairwise)
    write_csv(output / "per_user.csv", per_user)
    write_csv(output / "per_class.csv", per_class)
    summary = {
        "protocol": "p86-visual-candidate-outer-confirmation-v1",
        "candidate_run": str(candidate_run),
        "overall": overall,
        "pairwise": pairwise,
        "decision": decision,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
