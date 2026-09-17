"""Audit and package the compact P400 raw-detail Student candidate."""
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
TEACHER = RUNS / "p400_candidate_conditioned_raw_detail_test_v1/submission_p400_raw_detail_teacher.csv"
RAW = RUNS / "p400_student_test_predictions_v1/submission_p87s_student_raw.csv"
DECODED = RUNS / "p400_student_test_predictions_v1/submission_p87s_student_decoded.csv"
CHECKPOINT = RUNS / "p400_student_test_adapt_e40_v1/unified_student.pt"
P315 = RUNS / "p315_final_kaggle_candidate_v1/submission_p315_compact_student_raw.csv"
P400_SUMMARY = RUNS / "p400_candidate_conditioned_raw_detail_test_v1/summary.json"
OUT = RUNS / "p401_final_kaggle_candidate_v1"


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prediction(rows):
    return np.asarray([int(row["prediction"]) for row in rows], dtype=int)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    official = read_rows(OFFICIAL)
    expected_paths = [row["path"] for row in official]
    rows = {
        "teacher": read_rows(TEACHER),
        "raw": read_rows(RAW),
        "decoded": read_rows(DECODED),
        "p315": read_rows(P315),
    }
    for name, value in rows.items():
        if len(value) != 405 or [row["path"] for row in value] != expected_paths:
            raise RuntimeError(f"submission contract mismatch: {name}")
    teacher = prediction(rows["teacher"])
    raw = prediction(rows["raw"])
    decoded = prediction(rows["decoded"])
    p315 = prediction(rows["p315"])
    if not np.array_equal(raw, teacher):
        raise RuntimeError("compact Student does not exactly reproduce P400 teacher")
    changed = np.flatnonzero(raw != p315).tolist()
    if changed != [227, 369]:
        raise RuntimeError(f"unexpected P401 delta vs P315: {changed}")
    if any(row in (77, 283, 328) for row in changed):
        raise RuntimeError("P401 collides with frozen Kaggle-feedback rows")
    decoder_changed = np.flatnonzero(decoded != raw).tolist()
    if decoder_changed != [369]:
        raise RuntimeError(f"unexpected old decoder delta: {decoder_changed}")
    size = CHECKPOINT.stat().st_size
    if size >= 100_000_000:
        raise RuntimeError(f"checkpoint exceeds 100 MB: {size}")
    source = json.loads(P400_SUMMARY.read_text(encoding="utf-8"))

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "submission_p401_compact_student_raw.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(rows["raw"])
    report = {
        "stage": "P401_compact_candidate_conditioned_raw_detail_student",
        "status": "complete_awaiting_kaggle_score",
        "validation": {
            "base": "P310/P315 label-identical teacher",
            "correct": 2215,
            "rows": 2470,
            "accuracy": 2215 / 2470,
            "net_vs_p310": 4,
            "fold_nets": [0, 2, 2],
            "strict_gate_pass": True,
            "full_source_loso_nets": [1, 1, 5],
        },
        "student": {
            "checkpoint": str(CHECKPOINT.resolve()),
            "bytes": size,
            "mb_decimal": size / 1e6,
            "sha256": sha256(CHECKPOINT),
            "under_100MB": True,
            "teacher_agreement": 1.0,
            "old_decoder_rejected": True,
            "old_decoder_changed_rows": decoder_changed,
        },
        "submission": {
            "path": str(output.resolve()),
            "sha256": sha256(output),
            "rows": len(raw),
            "changes_vs_p315": len(changed),
            "changed_rows_zero_based": changed,
            "changed_sample_ids": source["test"]["changed_sample_ids"],
            "changed_pairs": [f"{p315[row]}->{raw[row]}" for row in changed],
            "candidate_margins": source["test"]["candidate_margins"],
            "frozen_row_collisions": [],
            "test_labels_read": False,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: compact P400 raw-detail Student package; old decoder rejected because it reverses row 369.\n"
        + json.dumps(report["submission"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
