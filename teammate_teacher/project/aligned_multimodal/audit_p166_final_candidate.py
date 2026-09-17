"""Audit and freeze the P166 compact Student Kaggle CSV."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OFFICIAL = REPO / "Testing/test.csv"
TEACHER = HERE / "runs/p166_a18_p89_confidence_gap_test_v1/submission_p166_a18_p89_confidence_gap.csv"
STUDENT = HERE / "runs/p166_student_test_predictions_v1/submission_p87s_student_raw.csv"
DECODED = HERE / "runs/p166_student_test_predictions_v1/submission_p87s_student_decoded.csv"
CHECKPOINT = HERE / "runs/p166_student_test_adapt_e40_v1/unified_student.pt"
TARGETS = HERE / "runs/p166_a18_p89_confidence_gap_test_v1/student_test_targets.npz"
OUTPUT = HERE / "runs/p166_final_kaggle_candidate_v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def predictions(values: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["prediction"]) for row in values], dtype=np.int64)


def main() -> None:
    official = rows(OFFICIAL)
    teacher_rows = rows(TEACHER)
    student_rows = rows(STUDENT)
    decoded_rows = rows(DECODED)
    expected_paths = [row["path"] for row in official]
    if len(official) != 405:
        raise RuntimeError("official Test row count changed")
    for name, values in (
        ("teacher", teacher_rows),
        ("student", student_rows),
        ("decoded", decoded_rows),
    ):
        if list(values[0]) != ["path", "prediction"]:
            raise RuntimeError(f"{name}: columns changed")
        if [row["path"] for row in values] != expected_paths:
            raise RuntimeError(f"{name}: official path order differs")
        value = predictions(values)
        if np.any((value < 0) | (value >= 40)):
            raise RuntimeError(f"{name}: prediction outside 0..39")
    teacher_prediction = predictions(teacher_rows)
    student_prediction = predictions(student_rows)
    decoded_prediction = predictions(decoded_rows)
    if not np.array_equal(student_prediction, teacher_prediction):
        raise RuntimeError("compact Student raw prediction does not reproduce P166 teacher")
    with np.load(TARGETS, allow_pickle=False) as saved:
        target_prediction = saved["structured_distillation_prediction"].astype(np.int64)
    if len(target_prediction) != 405 or not np.array_equal(target_prediction, student_prediction):
        raise RuntimeError("Student CSV differs from the 405 pseudo-targets")
    checkpoint_bytes = CHECKPOINT.stat().st_size
    if checkpoint_bytes >= 100_000_000:
        raise RuntimeError("Student checkpoint violates decimal 100 MB limit")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    final_path = OUTPUT / "submission_p166_compact_student_raw.csv"
    with final_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(student_rows)
    report = {
        "stage": "P166_compact_Student_final_Kaggle_candidate",
        "status": "complete_awaiting_kaggle_score",
        "selection": {
            "teacher_rule": "A18 confidence - P89 max confidence >= 0.47835912972024364",
            "oof": "2137/2470 = 86.5182%; H1/H2/H3 all positive",
            "test_replacements_vs_scored_0.85572_P89": 21,
            "decoder_rejected": True,
            "decoder_changed_rows": int(np.sum(decoded_prediction != student_prediction)),
        },
        "student": {
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_mb_decimal": checkpoint_bytes / 1_000_000,
            "checkpoint_sha256": digest(CHECKPOINT),
            "strict_under_100MB": True,
            "raw_teacher_agreement": float(np.mean(student_prediction == teacher_prediction)),
            "large_teacher_required_at_final_inference": False,
        },
        "submission": {
            "path": str(final_path.resolve()),
            "sha256": digest(final_path),
            "rows": len(student_rows),
            "unique_paths": len(set(expected_paths)),
            "prediction_min": int(student_prediction.min()),
            "prediction_max": int(student_prediction.max()),
            "official_order_exact": True,
            "test_labels_read": False,
        },
        "accuracy_boundary": (
            "No Test labels were available. OOF evidence supports this candidate over P89, "
            "but only the Kaggle score can establish its Test accuracy."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
