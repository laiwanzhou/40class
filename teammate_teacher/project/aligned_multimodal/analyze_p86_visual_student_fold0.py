from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = {
    "mechanism": PROJECT_DIR / "runs/p86_visual_student_fold0_mechanism_v1",
    "kd": PROJECT_DIR / "runs/p86_visual_student_fold0_kd_v1",
    "hybrid": PROJECT_DIR / "runs/p86_visual_student_fold0_hybrid_v1",
}
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_TEACHER = (
    PROJECT_DIR
    / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86_visual_student_fold0_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P86 fold0 visual-student failures.")
    parser.add_argument("--mechanism-run", type=Path, default=DEFAULT_RUNS["mechanism"])
    parser.add_argument("--kd-run", type=Path, default=DEFAULT_RUNS["kd"])
    parser.add_argument("--hybrid-run", type=Path, default=DEFAULT_RUNS["hybrid"])
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--teacher-oof", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(labels.size),
        "correct": int((labels == predictions).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def align_npz_predictions(
    path: Path,
    sample_ids: list[str],
    *,
    prediction_key: str | None = None,
    logits_key: str | None = None,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        index = {str(sample_id): row for row, sample_id in enumerate(data["sample_ids"])}
        missing = [sample_id for sample_id in sample_ids if sample_id not in index]
        if missing:
            raise RuntimeError(f"{path} is missing {missing[:3]}")
        rows = np.asarray([index[sample_id] for sample_id in sample_ids], dtype=np.int64)
        if prediction_key is not None:
            return np.asarray(data[prediction_key][rows], dtype=np.int64)
        if logits_key is None:
            raise ValueError("prediction_key or logits_key is required")
        return np.asarray(data[logits_key][rows]).argmax(axis=1).astype(np.int64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_paths = {
        "mechanism": args.mechanism_run.resolve(),
        "kd": args.kd_run.resolve(),
        "hybrid": args.hybrid_run.resolve(),
    }
    loaded = {
        name: read_predictions(path / "outer_predictions.csv") for name, path in run_paths.items()
    }
    reference = loaded["mechanism"]
    sample_ids = [row["sample_id"] for row in reference]
    users = np.asarray([row["user_id"] for row in reference])
    labels = np.asarray([int(row["label"]) for row in reference], dtype=np.int64)
    predictions: dict[str, np.ndarray] = {}
    for name, rows in loaded.items():
        if [row["sample_id"] for row in rows] != sample_ids:
            raise RuntimeError(f"sample order mismatch for {name}")
        if not np.array_equal(labels, np.asarray([int(row["label"]) for row in rows])):
            raise RuntimeError(f"label mismatch for {name}")
        predictions[name] = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)

    predictions["p12"] = align_npz_predictions(
        args.p12_oof.resolve(), sample_ids, prediction_key="final_predictions"
    )
    predictions["teacher"] = align_npz_predictions(
        args.teacher_oof.resolve(), sample_ids, logits_key="early_late_logits"
    )

    overall = {name: metrics(labels, pred) for name, pred in predictions.items()}
    per_user: list[dict[str, Any]] = []
    for user in sorted(set(users)):
        mask = users == user
        row: dict[str, Any] = {"user_id": str(user), "samples": int(mask.sum())}
        for name, pred in predictions.items():
            row[f"{name}_accuracy"] = float(accuracy_score(labels[mask], pred[mask]))
            row[f"{name}_correct"] = int((labels[mask] == pred[mask]).sum())
        per_user.append(row)

    per_class: list[dict[str, Any]] = []
    for class_id in range(40):
        mask = labels == class_id
        row = {"class_id": class_id, "samples": int(mask.sum())}
        for name, pred in predictions.items():
            row[f"{name}_accuracy"] = float(accuracy_score(labels[mask], pred[mask])) if mask.any() else 0.0
            row[f"{name}_correct"] = int((labels[mask] == pred[mask]).sum())
        row["hybrid_vs_mechanism_correct_delta"] = (
            row["hybrid_correct"] - row["mechanism_correct"]
        )
        row["hybrid_vs_p12_correct_delta"] = row["hybrid_correct"] - row["p12_correct"]
        row["hybrid_vs_teacher_correct_delta"] = row["hybrid_correct"] - row["teacher_correct"]
        per_class.append(row)

    pairwise: list[dict[str, Any]] = []
    pairs = [
        ("kd", "mechanism"),
        ("hybrid", "mechanism"),
        ("hybrid", "kd"),
        ("hybrid", "p12"),
        ("hybrid", "teacher"),
        ("p12", "teacher"),
    ]
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

    sample_rows: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(sample_ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "user_id": str(users[row_index]),
            "label": int(labels[row_index]),
        }
        for name, pred in predictions.items():
            row[f"{name}_prediction"] = int(pred[row_index])
            row[f"{name}_correct"] = int(pred[row_index] == labels[row_index])
        sample_rows.append(row)

    # The audit gate is deliberately conservative: V1 cannot advance merely because
    # one distillation objective beats another weak student.
    best_student = max(("mechanism", "kd", "hybrid"), key=lambda name: overall[name]["accuracy"])
    decision = {
        "status": "iterate_visual_representation",
        "best_student": best_student,
        "best_student_accuracy": overall[best_student]["accuracy"],
        "p12_accuracy": overall["p12"]["accuracy"],
        "teacher_accuracy": overall["teacher"]["accuracy"],
        "passed_visual_gate": False,
        "reason": (
            "Training-only teacher signals help the student, but the frozen globally pooled "
            "P30 representation remains far below both P12 and the teacher. Do not advance "
            "to additional outer folds or Skeleton/IMU; test a stronger spatial or trainable "
            "small visual representation under the same fold0 protocol."
        ),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_user.csv", per_user)
    write_csv(output / "per_class.csv", per_class)
    write_csv(output / "pairwise.csv", pairwise)
    write_csv(output / "sample_comparison.csv", sample_rows)
    summary = {
        "protocol": "p86-visual-student-fold0-error-audit-v1",
        "runs": {name: str(path) for name, path in run_paths.items()},
        "p12_oof": str(args.p12_oof.resolve()),
        "teacher_oof": str(args.teacher_oof.resolve()),
        "overall": overall,
        "pairwise": pairwise,
        "decision": decision,
        "artifacts": {
            "per_user": str(output / "per_user.csv"),
            "per_class": str(output / "per_class.csv"),
            "pairwise": str(output / "pairwise.csv"),
            "sample_comparison": str(output / "sample_comparison.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
