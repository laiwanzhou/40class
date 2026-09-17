from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_p86_visual_candidate import metrics_with_users
from analyze_p86_visual_student_fold0 import write_csv


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble and audit the frozen P86 V9 three-fold visual OOF."
    )
    parser.add_argument(
        "--fold0-run", type=Path, default=PROJECT_DIR / "runs/p86_visual_mc3_fold0_v9"
    )
    parser.add_argument(
        "--fold1-run", type=Path, default=PROJECT_DIR / "runs/p86_visual_mc3_fold1_v9"
    )
    parser.add_argument(
        "--fold2-run", type=Path, default=PROJECT_DIR / "runs/p86_visual_mc3_fold2_v9"
    )
    parser.add_argument(
        "--p12-oof", type=Path, default=PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
    )
    parser.add_argument(
        "--teacher-oof",
        type=Path,
        default=PROJECT_DIR
        / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "runs/p86_visual_mc3_oof_v9_audit"
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def main() -> None:
    args = parse_args()
    runs = [args.fold0_run.resolve(), args.fold1_run.resolve(), args.fold2_run.resolve()]
    all_rows: list[dict[str, Any]] = []
    checkpoint_bytes: dict[str, int] = {}
    for fold, run in enumerate(runs):
        rows = read_rows(run / "outer_predictions.csv")
        for row in rows:
            row["outer_fold"] = fold
        all_rows.extend(rows)
        checkpoint_bytes[f"fold{fold}"] = (run / "visual_student.pt").stat().st_size
    sample_ids = [str(row["sample_id"]) for row in all_rows]
    if len(sample_ids) != 2914 or len(set(sample_ids)) != 2914:
        raise RuntimeError(
            f"expected one OOF prediction for 2914 samples, got {len(sample_ids)} rows and "
            f"{len(set(sample_ids))} unique ids"
        )
    labels = np.asarray([int(row["label"]) for row in all_rows], dtype=np.int64)
    users = np.asarray([str(row["user_id"]) for row in all_rows])
    folds = np.asarray([int(row["outer_fold"]) for row in all_rows], dtype=np.int64)
    predictions = {
        "v9_visual": np.asarray(
            [int(row["prediction"]) for row in all_rows], dtype=np.int64
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
    per_fold: list[dict[str, Any]] = []
    improved_folds_vs_depth = 0
    for fold in range(3):
        mask = folds == fold
        row: dict[str, Any] = {"fold": fold, "samples": int(mask.sum())}
        for name, prediction in predictions.items():
            result = metrics_with_users(labels[mask], prediction[mask], users[mask])
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "worst_subject_accuracy"):
                row[f"{name}_{key}"] = result[key]
        row["v9_vs_p12_depth_accuracy_delta"] = (
            row["v9_visual_accuracy"] - row["p12_depth_accuracy"]
        )
        improved_folds_vs_depth += int(row["v9_vs_p12_depth_accuracy_delta"] > 0.0)
        per_fold.append(row)

    pairwise: list[dict[str, Any]] = []
    for baseline in ("p12_depth", "p12_final", "large_teacher"):
        candidate_correct = predictions["v9_visual"] == labels
        baseline_correct = predictions[baseline] == labels
        pairwise.append(
            {
                "candidate": "v9_visual",
                "baseline": baseline,
                "rescued": int((candidate_correct & ~baseline_correct).sum()),
                "harmed": int((~candidate_correct & baseline_correct).sum()),
                "net_correct": int(candidate_correct.sum() - baseline_correct.sum()),
                "both_correct": int((candidate_correct & baseline_correct).sum()),
                "both_wrong": int((~candidate_correct & ~baseline_correct).sum()),
            }
        )

    per_class: list[dict[str, Any]] = []
    for class_id in range(40):
        mask = labels == class_id
        row: dict[str, Any] = {"class_id": class_id, "samples": int(mask.sum())}
        for name, prediction in predictions.items():
            row[f"{name}_correct"] = int((prediction[mask] == labels[mask]).sum())
            row[f"{name}_accuracy"] = float((prediction[mask] == labels[mask]).mean())
        per_class.append(row)

    deployment_checkpoint_ok = all(size < 100_000_000 for size in checkpoint_bytes.values())
    passed = (
        improved_folds_vs_depth >= 2
        and overall["v9_visual"]["accuracy"] > overall["p12_depth"]["accuracy"]
        and overall["v9_visual"]["worst_subject_accuracy"]
        >= overall["p12_depth"]["worst_subject_accuracy"] - 0.02
        and deployment_checkpoint_ok
    )
    decision = {
        "passed_visual_threefold_gate": bool(passed),
        "folds_improved_vs_p12_depth": improved_folds_vs_depth,
        "required_improved_folds": 2,
        "all_single_fold_checkpoints_under_100000000_bytes": deployment_checkpoint_ok,
        "deployment_note": (
            "Three fold models are used only to estimate OOF. Final Test deployment must refit "
            "one fixed V9 model on all training subjects and must not ensemble the three folds."
        ),
        "next_action": (
            "freeze_v9_visual_anchor_and_start_skeleton_imu_increment_loop"
            if passed
            else "stay_in_visual_stage_and_audit_cross_fold_failure"
        ),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_fold.csv", per_fold)
    write_csv(output / "pairwise.csv", pairwise)
    write_csv(output / "per_class.csv", per_class)
    summary = {
        "protocol": "p86-v9-visual-threefold-subject-disjoint-oof-v1",
        "samples": len(sample_ids),
        "overall": overall,
        "per_fold": per_fold,
        "pairwise": pairwise,
        "checkpoint_bytes": checkpoint_bytes,
        "decision": decision,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
