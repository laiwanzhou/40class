"""Deploy the validated count/template micro-union and add it behind P167."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
from audit_p87_sequence_decoder import align_metadata, build_sessions
from p89_deterministic_triple_repeat import test_protocol
from p89_peer_supported_triple_repair import test_probability
from p89_supported_template_gate import (
    MAXIMUM_LENGTH,
    apply_gate,
    fit_supported_templates,
    session_candidates,
)
from p89_verified_micro_union_audit import label_free_union


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "runs/p168_historical_micro_union_test_v1"
P89 = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
P167 = HERE / "runs/p167_combined_teacher_test_v1/submission_p167_combined_teacher.csv"
COUNT = HERE / "runs/p89_count_preserving_imu_v2/predictions.npz"
TEMPLATE_VALIDATION = HERE / "runs/p89_supported_template_gate_v1/validation_predictions.npz"
TEMPLATE_SUMMARY = HERE / "runs/p89_supported_template_gate_v1/summary.json"
MICRO_VALIDATION = HERE / "runs/p89_verified_micro_union_audit_v1/validation_predictions.npz"
TEACHER = HERE / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
TRAIN_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
OOF_A18 = REPO / "runs/a18_p89_selective_replacement_v1/crossfit_predictions.npz"
OOF_P165 = HERE / "runs/p165_deployable_group_teacher_v1/predictions.npz"
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


def aligned(values, source_ids, target_ids):
    lookup = {value: row for row, value in enumerate(source_ids.astype(str))}
    rows = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(values)[rows]


def frozen_oof():
    with np.load(OOF_A18, allow_pickle=False) as saved:
        labels = saved["labels"].astype(np.int64)
        base = saved["p89_safe_prediction"].astype(np.int64)
        a18 = saved["confidence_gap_only_prediction"].astype(np.int64)
        a18_route = saved["confidence_gap_only_replacement_mask"].astype(bool)
    with np.load(OOF_P165, allow_pickle=False) as saved:
        group = np.concatenate(
            [saved[f"{name}_held_prediction"].astype(np.int64) for name in SPLITS]
        )
    p167 = np.where(a18_route, a18, np.where(group != base, group, base))
    with np.load(MICRO_VALIDATION, allow_pickle=False) as saved:
        micro = np.concatenate(
            [saved["h1_union"], saved["h2_union"], saved["h3_union"]]
        ).astype(np.int64)
    combined = np.where(p167 != base, p167, np.where(micro != base, micro, base))
    if int(np.sum(combined == labels)) != 2146:
        raise RuntimeError("P168 frozen OOF correct count changed")
    gains = []
    for low, high in ((0, 663), (663, 1497), (1497, 2470)):
        gains.append(
            int(np.sum(combined[low:high] == labels[low:high]))
            - int(np.sum(base[low:high] == labels[low:high]))
        )
    if gains != [10, 12, 7]:
        raise RuntimeError(f"P168 fold gains changed: {gains}")
    return labels, base, combined, gains


def deploy_micro():
    sample_ids, probability, safe = test_probability()
    protocol = test_protocol(sample_ids, probability, safe)
    with np.load(TEACHER, allow_pickle=False) as saved:
        train_ids = saved["oof_sample_ids"].astype(str)
        train_labels = saved["oof_labels"].astype(np.int64)
    metadata = align_metadata(TRAIN_METADATA, train_ids)
    sessions = build_sessions(
        np.arange(len(train_ids), dtype=np.int64),
        metadata,
        protocol[8].gap_seconds,
        "known_user",
    )
    templates = fit_supported_templates(
        train_labels, sessions, metadata, MAXIMUM_LENGTH
    )
    records = session_candidates(
        probability, protocol[6], templates, protocol[7], protocol[8]
    )
    configuration = json.loads(TEMPLATE_SUMMARY.read_text(encoding="utf-8"))["H1"][
        "selected"
    ]["configuration"]
    template_prediction, template_audit = apply_gate(safe, records, configuration)
    with np.load(COUNT, allow_pickle=False) as saved:
        count_prediction = aligned(
            saved["test_prediction"], saved["test_sample_ids"], sample_ids
        ).astype(np.int64)
    micro, overlap = label_free_union(safe, count_prediction, template_prediction)
    return sample_ids, safe, count_prediction, template_prediction, micro, {
        "configuration": configuration,
        "template": template_audit,
        "union": overlap,
        "template_count_by_length": {
            str(length): int(len(value[0])) for length, value in templates.items()
        },
    }


def main() -> None:
    labels, oof_base, oof_prediction, gains = frozen_oof()
    sample_ids, safe, count, template, micro, micro_audit = deploy_micro()
    p167 = read_prediction(P167)
    if len(p167) != len(sample_ids):
        raise RuntimeError("P167/Test micro row count differs")
    prediction = np.where(
        p167 != safe,
        p167,
        np.where(micro != safe, micro, safe),
    ).astype(np.int64)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p168_historical_micro_union.csv"
    submission_io.write_submission(submission, submission_io.read_rows(P89), prediction)
    probability = np.full((len(prediction), 40), 0.0005, dtype=np.float32)
    probability[np.arange(len(prediction)), prediction] = 0.9805
    targets = OUTPUT / "student_test_targets.npz"
    np.savez_compressed(
        targets,
        sample_ids=sample_ids,
        target_mask=np.ones(len(sample_ids), dtype=bool),
        emission_probability=probability,
        structured_distillation_probability=probability,
        structured_confidence=np.full(len(sample_ids), 0.9805, dtype=np.float32),
        emission_prediction=prediction,
        structured_distillation_prediction=prediction,
    )
    report = {
        "stage": "P168_P167_plus_historical_verified_micro_union",
        "status": "complete_frozen_no_test_label",
        "oof": {
            "rows": len(labels),
            "base_correct": int(np.sum(oof_base == labels)),
            "correct": int(np.sum(oof_prediction == labels)),
            "accuracy": float(np.mean(oof_prediction == labels)),
            "net_vs_p89": int(np.sum(oof_prediction == labels) - np.sum(oof_base == labels)),
            "held_fold_nets": gains,
        },
        "test": {
            "count_changes_vs_p89": int(np.sum(count != safe)),
            "template_changes_vs_p89": int(np.sum(template != safe)),
            "micro_changes_vs_p89": int(np.sum(micro != safe)),
            "p167_changes_vs_p89": int(np.sum(p167 != safe)),
            "final_changes_vs_p89": int(np.sum(prediction != safe)),
            "p167_micro_overlap": int(np.sum((p167 != safe) & (micro != safe))),
            "p167_micro_conflict": int(
                np.sum((p167 != safe) & (micro != safe) & (p167 != micro))
            ),
            "micro_audit": micro_audit,
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
