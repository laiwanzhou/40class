"""Final audit and packaging for the P199 current 88.54% OOF candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OFFICIAL = REPO / "Testing/test.csv"
TEACHER = HERE / "runs/p191_source_truth_calibrated_p150_v1/submission_p191_calibrated_meta.csv"
RAW = HERE / "runs/p199_student_test_predictions_v1/submission_p87s_student_raw.csv"
DECODED = HERE / "runs/p199_student_test_predictions_v1/submission_p87s_student_decoded.csv"
CHECKPOINT = HERE / "runs/p199_student_test_adapt_e40_v1/unified_student.pt"
TARGETS = HERE / "runs/p199_current_champion_teacher_v1/student_test_targets.npz"
OUTPUT = HERE / "runs/p199_final_kaggle_candidate_v1"


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pred(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    official = read(OFFICIAL)
    teacher = read(TEACHER)
    raw = read(RAW)
    decoded = read(DECODED)
    paths = [row["path"] for row in official]
    if len(paths) != 405 or len(set(paths)) != 405:
        raise RuntimeError("P199 official Test contract changed")
    for name, rows in (("teacher", teacher), ("raw", raw), ("decoded", decoded)):
        if list(rows[0]) != ["path", "prediction"]:
            raise RuntimeError(f"P199 {name} columns differ")
        if [row["path"] for row in rows] != paths:
            raise RuntimeError(f"P199 {name} path order differs")
        values = pred(rows)
        if np.any((values < 0) | (values >= 40)):
            raise RuntimeError(f"P199 {name} label outside 0..39")
    teacher_pred = pred(teacher)
    raw_pred = pred(raw)
    decoded_pred = pred(decoded)
    if not np.array_equal(raw_pred, teacher_pred):
        raise RuntimeError("P199 compact Student raw differs from teacher")
    with np.load(TARGETS, allow_pickle=False) as saved:
        target = saved["structured_distillation_prediction"].astype(np.int64)
    if not np.array_equal(target, raw_pred):
        raise RuntimeError("P199 raw CSV differs from pseudo-targets")
    checkpoint_bytes = CHECKPOINT.stat().st_size
    if checkpoint_bytes >= 100_000_000:
        raise RuntimeError("P199 checkpoint exceeds decimal 100 MB")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    final_path = OUTPUT / "submission_p199_compact_student_raw.csv"
    with final_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(raw)
    report = {
        "stage": "P199_compact_Student_final_Kaggle_candidate",
        "status": "complete_awaiting_kaggle_score",
        "validation": {
            "correct": 2187,
            "rows": 2470,
            "accuracy": 2187 / 2470,
            "fold_nets_vs_P180": [1, 1, 0],
            "P180_correct": 2185,
            "P150_reference_correct": 2210,
            "target_0.91_correct": 2248,
        },
        "student": {
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_mb_decimal": checkpoint_bytes / 1_000_000,
            "checkpoint_sha256": digest(CHECKPOINT),
            "strict_under_100MB": True,
            "teacher_agreement_405": float(np.mean(raw_pred == teacher_pred)),
            "large_teacher_required_at_final_inference": False,
            "decoder_rejected": True,
            "decoder_changed_rows": int(np.sum(decoded_pred != raw_pred)),
        },
        "submission": {
            "path": str(final_path.resolve()),
            "sha256": digest(final_path),
            "rows": len(raw),
            "unique_paths": len(set(paths)),
            "prediction_min": int(raw_pred.min()),
            "prediction_max": int(raw_pred.max()),
            "official_path_order_exact": True,
            "test_labels_read": False,
        },
        "accuracy_boundary": (
            "Strict OOF is 88.5425%. Kaggle Test accuracy remains unknown until upload; "
            "this artifact is not claimed to be 91%."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
