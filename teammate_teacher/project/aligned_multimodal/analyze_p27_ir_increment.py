from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from probe_p27r3_incremental_information import metric_bundle, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
SKELETON_BASE = (
    PROJECT_DIR
    / "runs"
    / "p27_strong_inner"
    / "skeleton_imu"
    / "calibration"
    / "cross_fitted_logits.npz"
)
IR_BASE = (
    PROJECT_DIR
    / "runs"
    / "p27_strong_inner"
    / "ir_skeleton_imu"
    / "calibration"
    / "cross_fitted_logits.npz"
)
OUTPUT = PROJECT_DIR / "runs" / "p27_strong_inner" / "ir_increment_audit"


def read_class_names() -> dict[int, str]:
    with (PROJECT_DIR / "data" / "manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = csv.DictReader(handle)
        return {int(row["class_id"]): row["class_name"] for row in rows}


def align(
    source: np.lib.npyio.NpzFile, sample_ids: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    if bool(source["outer_held_predictions_generated"]):
        raise RuntimeError("Outer-held predictions are forbidden")
    positions = {
        str(sample_id): index
        for index, sample_id in enumerate(source["sample_ids"].astype(str))
    }
    indices = np.asarray([positions[str(sample_id)] for sample_id in sample_ids])
    if not np.array_equal(source["labels"][indices], labels):
        raise RuntimeError("Labels do not align")
    return source["logits"][indices].astype(np.float32)


def flatten(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def main() -> None:
    skeleton = np.load(SKELETON_BASE, allow_pickle=False)
    ir = np.load(IR_BASE, allow_pickle=False)
    sample_ids = skeleton["sample_ids"].astype(str)
    labels = skeleton["labels"].astype(np.int64)
    subjects = skeleton["subjects"].astype(str)
    inner_folds = skeleton["inner_folds"].astype(np.int64)
    skeleton_logits = skeleton["logits"].astype(np.float32)
    ir_logits = align(ir, sample_ids, labels)
    skeleton_predictions = skeleton_logits.argmax(axis=1)
    ir_predictions = ir_logits.argmax(axis=1)
    class_names = read_class_names()

    overall_rows: list[dict[str, Any]] = []
    for method, predictions in (
        ("skeleton_rfimu", skeleton_predictions),
        ("ir_skeleton_rfimu", ir_predictions),
    ):
        overall_rows.append(
            {"method": method, **flatten(metric_bundle(labels, predictions))}
        )

    subject_rows: list[dict[str, Any]] = []
    for subject in sorted(set(subjects.tolist())):
        selected = subjects == subject
        base_accuracy = float(
            np.mean(skeleton_predictions[selected] == labels[selected])
        )
        ir_accuracy = float(np.mean(ir_predictions[selected] == labels[selected]))
        subject_rows.append(
            {
                "subject": subject,
                "samples": int(selected.sum()),
                "skeleton_rfimu_accuracy": base_accuracy,
                "ir_skeleton_rfimu_accuracy": ir_accuracy,
                "delta_pp": 100.0 * (ir_accuracy - base_accuracy),
            }
        )

    class_rows: list[dict[str, Any]] = []
    for class_id in range(40):
        selected = labels == class_id
        base_recall = float(
            np.mean(skeleton_predictions[selected] == class_id)
        )
        ir_recall = float(np.mean(ir_predictions[selected] == class_id))
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "samples": int(selected.sum()),
                "skeleton_rfimu_recall": base_recall,
                "ir_skeleton_rfimu_recall": ir_recall,
                "delta_pp": 100.0 * (ir_recall - base_recall),
            }
        )

    base_correct = skeleton_predictions == labels
    ir_correct = ir_predictions == labels
    rescues = (~base_correct) & ir_correct
    new_errors = base_correct & (~ir_correct)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(3):
        selected = inner_folds == fold
        fold_rows.append(
            {
                "inner_fold": fold,
                "samples": int(selected.sum()),
                "skeleton_rfimu_accuracy": float(base_correct[selected].mean()),
                "ir_skeleton_rfimu_accuracy": float(ir_correct[selected].mean()),
                "rescues": int(rescues[selected].sum()),
                "new_errors": int(new_errors[selected].sum()),
                "net_rescues": int(
                    rescues[selected].sum() - new_errors[selected].sum()
                ),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "metrics.csv", overall_rows)
    write_csv(OUTPUT / "per_subject.csv", subject_rows)
    write_csv(OUTPUT / "per_class.csv", class_rows)
    write_csv(OUTPUT / "per_fold_rescues.csv", fold_rows)
    summary = {
        "protocol": "outer-fold-0 train subjects only; three fixed subject-disjoint inner folds; both systems use leave-one-inner-fold-out NLL calibration",
        "outer_held_predictions_generated": False,
        "samples": int(len(labels)),
        "rescues": int(rescues.sum()),
        "new_errors": int(new_errors.sum()),
        "net_rescues": int(rescues.sum() - new_errors.sum()),
        "subjects_improved": int(
            sum(float(row["delta_pp"]) > 0 for row in subject_rows)
        ),
        "subjects_tied": int(
            sum(float(row["delta_pp"]) == 0 for row in subject_rows)
        ),
        "subjects_harmed": int(
            sum(float(row["delta_pp"]) < 0 for row in subject_rows)
        ),
        "metrics": {row["method"]: row for row in overall_rows},
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
