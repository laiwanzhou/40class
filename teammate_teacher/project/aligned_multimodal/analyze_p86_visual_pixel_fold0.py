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
    parser = argparse.ArgumentParser(description="Audit P86 V2 fold0 visual representation.")
    parser.add_argument(
        "--mechanism-run",
        type=Path,
        default=PROJECT_DIR / "runs/p86_visual_pixel_fold0_mechanism_v2",
    )
    parser.add_argument(
        "--hybrid-run",
        type=Path,
        default=PROJECT_DIR / "runs/p86_visual_pixel_fold0_hybrid_v2",
    )
    parser.add_argument(
        "--v1-hybrid-run",
        type=Path,
        default=PROJECT_DIR / "runs/p86_visual_student_fold0_hybrid_v1",
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
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs/p86_visual_pixel_fold0_audit_v2",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def align_npz(
    path: Path, sample_ids: list[str], *, prediction_key: str | None = None, logits_key: str | None = None
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
    run_files = {
        "v2_mechanism": args.mechanism_run.resolve() / "outer_predictions.csv",
        "v2_hybrid": args.hybrid_run.resolve() / "outer_predictions.csv",
        "v1_hybrid": args.v1_hybrid_run.resolve() / "outer_predictions.csv",
    }
    loaded = {name: read_csv(path) for name, path in run_files.items()}
    reference = loaded["v2_hybrid"]
    sample_ids = [row["sample_id"] for row in reference]
    labels = np.asarray([int(row["label"]) for row in reference], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in reference])
    predictions: dict[str, np.ndarray] = {}
    for name, rows in loaded.items():
        if [row["sample_id"] for row in rows] != sample_ids:
            raise RuntimeError(f"sample order mismatch: {name}")
        predictions[name] = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    predictions["p12_depth"] = align_npz(
        args.p12_oof, sample_ids, logits_key="depth_logits"
    )
    predictions["p12_final"] = align_npz(
        args.p12_oof, sample_ids, prediction_key="final_predictions"
    )
    predictions["teacher"] = align_npz(
        args.teacher_oof, sample_ids, logits_key="early_late_logits"
    )
    overall = {name: metrics(labels, prediction) for name, prediction in predictions.items()}

    pairs = (
        ("v2_hybrid", "v2_mechanism"),
        ("v2_hybrid", "v1_hybrid"),
        ("v2_hybrid", "p12_depth"),
        ("v2_hybrid", "p12_final"),
        ("v2_hybrid", "teacher"),
        ("p12_depth", "teacher"),
    )
    pairwise: list[dict[str, Any]] = []
    for candidate, baseline in pairs:
        candidate_correct = predictions[candidate] == labels
        baseline_correct = predictions[baseline] == labels
        rescued = candidate_correct & ~baseline_correct
        harmed = ~candidate_correct & baseline_correct
        pairwise.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "rescued": int(rescued.sum()),
                "harmed": int(harmed.sum()),
                "net_correct": int(rescued.sum() - harmed.sum()),
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
        row["v2_hybrid_vs_v1_correct_delta"] = row["v2_hybrid_correct"] - row["v1_hybrid_correct"]
        row["v2_hybrid_vs_p12_depth_correct_delta"] = (
            row["v2_hybrid_correct"] - row["p12_depth_correct"]
        )
        per_class.append(row)

    per_user: list[dict[str, Any]] = []
    for user in sorted(set(users)):
        mask = users == user
        row = {"user_id": str(user), "samples": int(mask.sum())}
        for name, prediction in predictions.items():
            row[f"{name}_accuracy"] = float((prediction[mask] == labels[mask]).mean())
        per_user.append(row)

    sample_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "user_id": str(users[index]),
            "label": int(labels[index]),
        }
        for name, prediction in predictions.items():
            row[f"{name}_prediction"] = int(prediction[index])
            row[f"{name}_correct"] = int(prediction[index] == labels[index])
        sample_rows.append(row)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "pairwise.csv", pairwise)
    write_csv(output / "per_class.csv", per_class)
    write_csv(output / "per_user.csv", per_user)
    write_csv(output / "sample_comparison.csv", sample_rows)
    decision = {
        "passed_visual_gate": False,
        "status": "iterate_visual_distillation_or_representation",
        "reason": (
            "The trainable multi-view representation improves over V1 but only reaches the P12 "
            "depth branch. Keep V2 as rollback and test direct aligned clip-feature distillation "
            "before increasing model size or moving to Skeleton/IMU."
        ),
    }
    summary = {
        "protocol": "p86-visual-pixel-fold0-audit-v2",
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
