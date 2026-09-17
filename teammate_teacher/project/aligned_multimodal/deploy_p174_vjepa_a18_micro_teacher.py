"""Freeze P174: A18 confidence-gap, P173 V-JEPA group, then micro-union."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "runs/p174_vjepa_a18_micro_teacher_v1"
P89 = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
A18 = HERE / "runs/p166_a18_p89_confidence_gap_test_v1/submission_p166_a18_p89_confidence_gap.csv"
GROUP = HERE / "runs/p173_vjepa_augmented_group_teacher_v1/submission_p173_vjepa_augmented_group.csv"
P168 = HERE / "runs/p168_historical_micro_union_test_v1/submission_p168_historical_micro_union.csv"
P167 = HERE / "runs/p167_combined_teacher_test_v1/submission_p167_combined_teacher.csv"
OOF_A18 = REPO / "runs/a18_p89_selective_replacement_v1/crossfit_predictions.npz"
OOF_GROUP = HERE / "runs/p173_vjepa_augmented_group_teacher_v1/predictions.npz"
OOF_MICRO = HERE / "runs/p89_verified_micro_union_audit_v1/validation_predictions.npz"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")


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
        sample_ids = saved["sample_ids"].astype(str)
        labels = saved["labels"].astype(np.int64)
        base = saved["p89_safe_prediction"].astype(np.int64)
        a18 = saved["confidence_gap_only_prediction"].astype(np.int64)
        a18_route = saved["confidence_gap_only_replacement_mask"].astype(bool)
    with np.load(OOF_GROUP, allow_pickle=False) as saved:
        group = np.concatenate(
            [saved[f"{name}_held_prediction"].astype(np.int64) for name in SPLITS]
        )
    with np.load(OOF_MICRO, allow_pickle=False) as saved:
        micro = np.concatenate(
            [saved["h1_union"], saved["h2_union"], saved["h3_union"]]
        ).astype(np.int64)
    prediction = np.where(
        a18_route,
        a18,
        np.where(group != base, group, np.where(micro != base, micro, base)),
    )
    correct = int(np.sum(prediction == labels))
    if correct != 2162:
        raise RuntimeError(f"P174 frozen OOF count changed: {correct}")
    fold_nets = []
    for low, high in ((0, 663), (663, 1497), (1497, 2470)):
        fold_nets.append(
            int(np.sum(prediction[low:high] == labels[low:high]))
            - int(np.sum(base[low:high] == labels[low:high]))
        )
    if fold_nets != [15, 13, 17]:
        raise RuntimeError(f"P174 fold nets changed: {fold_nets}")

    base_test = read_prediction(P89)
    a18_test = read_prediction(A18)
    group_test = read_prediction(GROUP)
    p168_test = read_prediction(P168)
    p167_test = read_prediction(P167)
    micro_test = np.where(p168_test != p167_test, p168_test, base_test)
    test_prediction = np.where(
        a18_test != base_test,
        a18_test,
        np.where(
            group_test != base_test,
            group_test,
            np.where(micro_test != base_test, micro_test, base_test),
        ),
    ).astype(np.int64)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p174_vjepa_a18_micro.csv"
    submission_io.write_submission(
        submission, submission_io.read_rows(P89), test_prediction
    )
    probability = np.full((len(test_prediction), 40), 0.0005, dtype=np.float32)
    probability[np.arange(len(test_prediction)), test_prediction] = 0.9805
    targets = OUTPUT / "student_test_targets.npz"
    np.savez_compressed(
        targets,
        sample_ids=np.asarray(
            [f"SM_test_{row:04d}" for row in range(1, len(test_prediction) + 1)]
        ),
        target_mask=np.ones(len(test_prediction), dtype=bool),
        emission_probability=probability,
        structured_distillation_probability=probability,
        structured_confidence=np.full(len(test_prediction), 0.9805, dtype=np.float32),
        emission_prediction=test_prediction,
        structured_distillation_prediction=test_prediction,
    )
    report = {
        "stage": "P174_VJEPA_A18_micro_deployable_teacher",
        "status": "complete_no_test_label",
        "oof": {
            "rows": len(labels),
            "base_correct": int(np.sum(base == labels)),
            "correct": correct,
            "accuracy": correct / len(labels),
            "net_vs_p89": correct - int(np.sum(base == labels)),
            "held_fold_nets": fold_nets,
            "changed": int(np.sum(prediction != base)),
            "gap_to_0.91_correct": int(np.ceil(0.91 * len(labels))) - correct,
        },
        "test": {
            "a18_routes": int(np.sum(a18_test != base_test)),
            "group_routes": int(np.sum(group_test != base_test)),
            "micro_only_routes": int(np.sum(micro_test != base_test)),
            "changes_vs_p89": int(np.sum(test_prediction != base_test)),
            "changes_vs_p168": int(np.sum(test_prediction != p168_test)),
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
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        sample_ids=sample_ids,
        labels=labels,
        base_prediction=base,
        prediction=prediction,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
