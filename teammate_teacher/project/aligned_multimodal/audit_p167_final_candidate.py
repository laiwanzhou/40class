"""Final engineering audit for the P167 compact Student CSV."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OFFICIAL = REPO / "Testing/test.csv"
TEACHER = HERE / "runs/p167_combined_teacher_test_v1/submission_p167_combined_teacher.csv"
STUDENT = HERE / "runs/p167_student_test_predictions_v1/submission_p87s_student_raw.csv"
DECODED = HERE / "runs/p167_student_test_predictions_v1/submission_p87s_student_decoded.csv"
CHECKPOINT = HERE / "runs/p167_student_test_adapt_e40_v1/unified_student.pt"
TARGETS = HERE / "runs/p167_combined_teacher_test_v1/student_test_targets.npz"
OUTPUT = HERE / "runs/p167_final_kaggle_candidate_v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prediction(values: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["prediction"]) for row in values], dtype=np.int64)


def main() -> None:
    official = read(OFFICIAL)
    teacher = read(TEACHER)
    student = read(STUDENT)
    decoded = read(DECODED)
    official_paths = [row["path"] for row in official]
    if len(official_paths) != 405 or len(set(official_paths)) != 405:
        raise RuntimeError("official Test contract changed")
    for name, values in (("teacher", teacher), ("student", student), ("decoded", decoded)):
        if list(values[0]) != ["path", "prediction"]:
            raise RuntimeError(f"{name}: columns differ")
        if [row["path"] for row in values] != official_paths:
            raise RuntimeError(f"{name}: official path order differs")
        pred = prediction(values)
        if np.any((pred < 0) | (pred >= 40)):
            raise RuntimeError(f"{name}: prediction outside 0..39")
    teacher_prediction = prediction(teacher)
    student_prediction = prediction(student)
    decoded_prediction = prediction(decoded)
    if not np.array_equal(student_prediction, teacher_prediction):
        raise RuntimeError("compact Student failed to reproduce P167 teacher")
    with np.load(TARGETS, allow_pickle=False) as saved:
        target = saved["structured_distillation_prediction"].astype(np.int64)
    if not np.array_equal(target, student_prediction):
        raise RuntimeError("compact Student differs from Test pseudo-targets")
    checkpoint_bytes = CHECKPOINT.stat().st_size
    if checkpoint_bytes >= 100_000_000:
        raise RuntimeError("checkpoint exceeds strict decimal 100 MB")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    final_path = OUTPUT / "submission_p167_compact_student_raw.csv"
    with final_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(student)
    report = {
        "stage": "P167_compact_Student_final_Kaggle_candidate",
        "status": "complete_awaiting_kaggle_score",
        "validation": {
            "correct": 2143,
            "rows": 2470,
            "accuracy": 2143 / 2470,
            "net_vs_p89": 26,
            "held_fold_nets": [10, 9, 7],
            "p150_reference": "2210/2470=89.4737%; not reproduced by the deployable path",
        },
        "student": {
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_mb_decimal": checkpoint_bytes / 1_000_000,
            "checkpoint_sha256": digest(CHECKPOINT),
            "strict_under_100MB": True,
            "teacher_agreement_405": float(np.mean(student_prediction == teacher_prediction)),
            "large_teacher_required_at_final_inference": False,
            "decoder_rejected": True,
            "decoder_changed_rows": int(np.sum(decoded_prediction != student_prediction)),
        },
        "submission": {
            "path": str(final_path.resolve()),
            "sha256": digest(final_path),
            "rows": len(student),
            "unique_paths": len(set(official_paths)),
            "prediction_min": int(student_prediction.min()),
            "prediction_max": int(student_prediction.max()),
            "official_path_order_exact": True,
            "changes_vs_scored_0.85572_p89": 22,
            "test_labels_read": False,
        },
        "accuracy_boundary": (
            "This is the strongest completed Test-deployable candidate in this run. "
            "Its Kaggle accuracy is unknown until submission and is not claimed to be 89.4%."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
