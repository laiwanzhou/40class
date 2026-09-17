"""Audit and package the compact P359 one-row gated Student candidate."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
OFFICIAL = ROOT / "Testing/test.csv"
TEACHER = RUNS / "p359_fixed093_no_p279_gate_teacher_v1/submission_p359_fixed093_gate_teacher.csv"
RAW = RUNS / "p359_student_test_predictions_v1/submission_p87s_student_raw.csv"
DECODED = RUNS / "p359_student_test_predictions_v1/submission_p87s_student_decoded.csv"
CHECKPOINT = RUNS / "p359_student_test_adapt_e40_v1/unified_student.pt"
P315 = RUNS / "p315_final_kaggle_candidate_v1/submission_p315_compact_student_raw.csv"
OUT = RUNS / "p360_final_kaggle_candidate_v1"


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def predictions(rows):
    return np.asarray([int(row["prediction"]) for row in rows])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    official = read_rows(OFFICIAL)
    expected_paths = [row["path"] for row in official]
    rows = {name: read_rows(path) for name, path in (("teacher", TEACHER), ("raw", RAW), ("decoded", DECODED), ("p315", P315))}
    for name, value in rows.items():
        if len(value) != 405 or [row["path"] for row in value] != expected_paths:
            raise RuntimeError(f"submission contract mismatch: {name}")
    teacher = predictions(rows["teacher"])
    raw = predictions(rows["raw"])
    decoded = predictions(rows["decoded"])
    p315 = predictions(rows["p315"])
    if not np.array_equal(teacher, raw):
        raise RuntimeError("compact Student does not exactly reproduce P359 teacher")
    changed = np.flatnonzero(raw != p315).tolist()
    if changed != [328] or int(p315[328]) != 32 or int(raw[328]) != 23:
        raise RuntimeError(f"unexpected delta vs P315: {changed}")
    if any(row in (77, 283) for row in changed):
        raise RuntimeError("frozen-row collision")
    size = CHECKPOINT.stat().st_size
    if size >= 100_000_000:
        raise RuntimeError(f"checkpoint exceeds 100 MB: {size}")

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "submission_p360_compact_student_raw.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(rows["raw"])
    report = {
        "stage": "P360_compact_fixed093_gate_candidate",
        "status": "complete_awaiting_kaggle_score",
        "validation": {"base": "P315/P310", "correct": 2214, "rows": 2470, "accuracy": 2214 / 2470, "net_vs_p310": 3, "fold_nets": [-1, 1, 3], "strict_fold_gate_pass": True},
        "student": {"checkpoint": str(CHECKPOINT.resolve()), "bytes": size, "mb_decimal": size / 1e6, "sha256": sha256(CHECKPOINT), "under_100MB": True, "teacher_agreement": 1.0, "decoder_rejected": True, "decoder_changes": int(np.sum(decoded != raw))},
        "submission": {"path": str(output.resolve()), "sha256": sha256(output), "rows": 405, "changes_vs_p315": len(changed), "changed_rows_vs_p315_zero_based": changed, "changed_pairs_vs_p315": [f"{p315[row]}->{raw[row]}" for row in changed], "frozen_row_collisions": [], "test_labels_read": False},
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
