"""Deploy the frozen outer-cross-fit A18/P89 confidence-gap rule to Test.

The rule and its threshold are read from the completed OOF audit.  No Test
label, class-specific correction, or Test-side threshold search is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "runs/p166_a18_p89_confidence_gap_test_v1"
THRESHOLDS = REPO / "runs/a18_p89_selective_replacement_v1/selected_thresholds.json"
A18 = HERE / "runs/a18_full_teacher_test_v1/test_predictions.npz"
P89_PROBABILITY = HERE / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
IMU = HERE / "runs/p3_sd_imu_rf_full18/test_logits.npz"
P89_SUBMISSION = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def normalise(values: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    return probability / probability.sum(axis=1, keepdims=True)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def read_prediction(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray(
            [int(row["prediction"]) for row in csv.DictReader(handle)], dtype=np.int64
        )


def align(values, source_ids, target_ids):
    lookup = {value: row for row, value in enumerate(source_ids.astype(str))}
    rows = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(values)[rows]


def main() -> None:
    selected = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["confidence_gap_only"]
    outer_thresholds = [
        float(selected[name]["thresholds"]["confidence_gap"])
        for name in ("H1_selection", "H2_confirmation", "H3_independent_fold0")
    ]
    threshold = float(np.median(np.asarray(outer_thresholds)))

    with np.load(A18, allow_pickle=False) as saved:
        sample_ids = saved["sample_ids"].astype(str)
        visual_available = saved["visual_available"].astype(bool)
        a18_probability = normalise(saved["selected_probability"])
    with np.load(P89_PROBABILITY, allow_pickle=False) as saved:
        p89_ids = saved["sample_ids"].astype(str)
        base_probability = normalise(saved["base_probability"])
    if not np.array_equal(sample_ids, p89_ids):
        raise RuntimeError("A18/P89 Test sample order differs")
    with np.load(IMU, allow_pickle=False) as saved:
        imu_probability = softmax(saved["imu_logits"], 3.0)
        imu_ids = saved["sample_ids"].astype(str)
    p89_probability = normalise(
        0.95 * base_probability + 0.05 * align(imu_probability, imu_ids, sample_ids)
    )
    p89_prediction = read_prediction(P89_SUBMISSION)
    a18_prediction = a18_probability.argmax(axis=1).astype(np.int64)
    confidence_gap = a18_probability.max(axis=1) - p89_probability.max(axis=1)
    route = (
        visual_available
        & (a18_prediction != p89_prediction)
        & (confidence_gap >= threshold)
    )
    prediction = p89_prediction.copy()
    prediction[route] = a18_prediction[route]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p166_a18_p89_confidence_gap.csv"
    submission_io.write_submission(
        submission, submission_io.read_rows(P89_SUBMISSION), prediction
    )
    hard_probability = np.full((len(sample_ids), 40), 0.0005, dtype=np.float32)
    hard_probability[np.arange(len(sample_ids)), prediction] = 0.9805
    # P89 supplies the frozen fallback target for the four unreadable-IR rows,
    # so the compact Student can remain the only inference checkpoint.
    target_rows = np.ones(len(sample_ids), dtype=bool)
    targets = OUTPUT / "student_test_targets.npz"
    np.savez_compressed(
        targets,
        sample_ids=sample_ids[target_rows],
        target_mask=np.ones(int(target_rows.sum()), dtype=bool),
        emission_probability=hard_probability[target_rows],
        structured_distillation_probability=hard_probability[target_rows],
        structured_confidence=np.full(int(target_rows.sum()), 0.9805, dtype=np.float32),
        teacher_prediction=prediction[target_rows],
        emission_prediction=prediction[target_rows],
        structured_distillation_prediction=prediction[target_rows],
    )
    audit_rows = [
        {
            "sample_id": sample_id,
            "p89": int(p89_prediction[row]),
            "a18": int(a18_prediction[row]),
            "confidence_gap": float(confidence_gap[row]),
            "replaced": int(route[row]),
            "prediction": int(prediction[row]),
        }
        for row, sample_id in enumerate(sample_ids)
    ]
    with (OUTPUT / "prediction_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    report = {
        "stage": "P166_A18_P89_confidence_gap_Test_deployment",
        "status": "complete_frozen_rule_no_test_label",
        "protocol": {
            "default": "P89 0.85572 safe submission",
            "replacement": "A18 selected/session prediction",
            "rule": "A18 confidence - P89 max confidence >= outer-threshold median",
            "outer_thresholds": outer_thresholds,
            "threshold": threshold,
            "oof_evidence": "2137/2470=86.5182%; all three held cohorts positive",
            "test_labels_read": False,
            "test_threshold_or_class_sweep": False,
        },
        "test": {
            "rows": len(sample_ids),
            "visual_available": int(visual_available.sum()),
            "p89_a18_disagreements": int(np.sum(p89_prediction != a18_prediction)),
            "replacements": int(route.sum()),
            "submission": str(submission.resolve()),
            "submission_sha256": digest(submission),
            "student_targets": str(targets.resolve()),
            "student_targets_sha256": digest(targets),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
