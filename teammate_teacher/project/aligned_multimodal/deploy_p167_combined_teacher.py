"""Freeze P167: P166 confidence gap plus non-overlapping P165 group routes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "runs/p167_combined_teacher_test_v1"
P89 = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
P166 = HERE / "runs/p166_a18_p89_confidence_gap_test_v1/submission_p166_a18_p89_confidence_gap.csv"
P165 = HERE / "runs/p165_deployable_group_teacher_v1/submission_p165_deployable_group_teacher.csv"
OOF_A18 = REPO / "runs/a18_p89_selective_replacement_v1/crossfit_predictions.npz"
OOF_P165 = HERE / "runs/p165_deployable_group_teacher_v1/predictions.npz"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
SIZES = (663, 834, 973)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def read_prediction(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray([int(row["prediction"]) for row in csv.DictReader(handle)])


def main() -> None:
    with np.load(OOF_A18, allow_pickle=False) as saved:
        labels = saved["labels"].astype(np.int64)
        base = saved["p89_safe_prediction"].astype(np.int64)
        a18 = saved["confidence_gap_only_prediction"].astype(np.int64)
        a18_route = saved["confidence_gap_only_replacement_mask"].astype(bool)
    with np.load(OOF_P165, allow_pickle=False) as saved:
        group = np.concatenate(
            [saved[f"{name}_held_prediction"].astype(np.int64) for name in SPLITS]
        )
    group_route = group != base
    combined = np.where(a18_route, a18, np.where(group_route, group, base))
    if int(np.sum(combined == labels)) != 2143:
        raise RuntimeError("frozen P167 OOF count changed")
    fold_rows = []
    low = 0
    for name, size in zip(SPLITS, SIZES, strict=True):
        high = low + size
        fold_rows.append(
            {
                "cohort": name,
                "rows": size,
                "base_correct": int(np.sum(base[low:high] == labels[low:high])),
                "correct": int(np.sum(combined[low:high] == labels[low:high])),
            }
        )
        fold_rows[-1]["net"] = fold_rows[-1]["correct"] - fold_rows[-1]["base_correct"]
        low = high
    if any(int(row["net"]) <= 0 for row in fold_rows):
        raise RuntimeError("P167 must remain positive in every held cohort")

    base_test = read_prediction(P89)
    p166_test = read_prediction(P166)
    p165_test = read_prediction(P165)
    a18_test_route = p166_test != base_test
    group_test_route = p165_test != base_test
    prediction = np.where(
        a18_test_route,
        p166_test,
        np.where(group_test_route, p165_test, base_test),
    ).astype(np.int64)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p167_combined_teacher.csv"
    submission_io.write_submission(submission, submission_io.read_rows(P89), prediction)
    probability = np.full((len(prediction), 40), 0.0005, dtype=np.float32)
    probability[np.arange(len(prediction)), prediction] = 0.9805
    targets = OUTPUT / "student_test_targets.npz"
    np.savez_compressed(
        targets,
        sample_ids=np.asarray(
            [f"SM_test_{row:04d}" for row in range(1, len(prediction) + 1)]
        ),
        target_mask=np.ones(len(prediction), dtype=bool),
        emission_probability=probability,
        structured_distillation_probability=probability,
        structured_confidence=np.full(len(prediction), 0.9805, dtype=np.float32),
        emission_prediction=prediction,
        structured_distillation_prediction=prediction,
    )
    report = {
        "stage": "P167_combined_deployable_teacher",
        "status": "complete_frozen_no_test_label",
        "oof": {
            "rows": len(labels),
            "base_correct": int(np.sum(base == labels)),
            "correct": int(np.sum(combined == labels)),
            "accuracy": float(np.mean(combined == labels)),
            "net_vs_p89": int(np.sum(combined == labels) - np.sum(base == labels)),
            "changed": int(np.sum(combined != base)),
            "folds": fold_rows,
            "combination_rule": "P166 confidence-gap priority, then P165 group route, else P89",
        },
        "test": {
            "rows": len(prediction),
            "p166_routes": int(a18_test_route.sum()),
            "p165_routes": int(group_test_route.sum()),
            "route_overlap": int(np.sum(a18_test_route & group_test_route)),
            "changes_vs_0.85572_p89": int(np.sum(prediction != base_test)),
            "submission": str(submission.resolve()),
            "submission_sha256": digest(submission),
            "targets": str(targets.resolve()),
            "targets_sha256": digest(targets),
            "test_labels_read": False,
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
