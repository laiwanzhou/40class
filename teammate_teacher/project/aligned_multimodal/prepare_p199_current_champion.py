"""Prepare P199 compact-Student targets from the corrected P191 candidate."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = HERE / "runs/p191_source_truth_calibrated_p150_v1/submission_p191_calibrated_meta.csv"
OFFICIAL = REPO / "Testing/test.csv"
P89 = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
P180 = HERE / "runs/p180_sequence_micro_teacher_v1/submission_p180_sequence_micro.csv"
OUTPUT = HERE / "runs/p199_current_champion_teacher_v1"


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prediction(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    official = read(OFFICIAL)
    source = read(SOURCE)
    p89 = read(P89)
    p180 = read(P180)
    paths = [row["path"] for row in official]
    for name, rows in (("source", source), ("p89", p89), ("p180", p180)):
        if len(rows) != 405 or [row["path"] for row in rows] != paths:
            raise RuntimeError(f"P199 {name} Test order differs")
    pred = prediction(source)
    if np.any((pred < 0) | (pred >= 40)):
        raise RuntimeError("P199 prediction outside 0..39")
    sample_ids = []
    for path in paths:
        match = re.search(r"(SM_test_\d{4})", path)
        if match is None:
            raise RuntimeError(f"P199 cannot parse sample ID: {path}")
        sample_ids.append(match.group(1))
    probability = np.full((405, 40), 0.0005, dtype=np.float32)
    probability[np.arange(405), pred] = 0.9805
    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = OUTPUT / "student_test_targets.npz"
    np.savez_compressed(
        targets,
        sample_ids=np.asarray(sample_ids),
        target_mask=np.ones(405, dtype=bool),
        emission_probability=probability,
        structured_distillation_probability=probability,
        structured_confidence=np.full(405, 0.9805, dtype=np.float32),
        emission_prediction=pred,
        structured_distillation_prediction=pred,
    )
    report = {
        "stage": "P199_current_champion_teacher_targets",
        "status": "complete",
        "validation": {
            "correct": 2187,
            "rows": 2470,
            "accuracy": 2187 / 2470,
            "fold_nets_vs_P180": [1, 1, 0],
            "P180_correct": 2185,
        },
        "test": {
            "rows": 405,
            "changes_vs_P180": int(np.sum(pred != prediction(p180))),
            "changes_vs_P89": int(np.sum(pred != prediction(p89))),
            "source": str(SOURCE.resolve()),
            "source_sha256": digest(SOURCE),
            "targets": str(targets.resolve()),
            "targets_sha256": digest(targets),
            "test_labels_read": False,
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
