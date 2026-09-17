from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import align_metadata, build_sessions, decode_sessions
from p89_cohort_class_prior import class_biases, date_number
from p89_deploy_supervised_router import load_decoder
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_cohort_class_prior_test_v1"


def test_groups(metadata) -> np.ndarray:
    dates = date_number(metadata.dates)
    valid = sorted(set(dates[np.isfinite(dates)].tolist()))
    cluster_by_date = {}
    cluster = 0
    previous = None
    for value in valid:
        if previous is not None and value - previous > 5.0:
            cluster += 1
        cluster_by_date[value] = cluster
        previous = value
    result = np.asarray(["unknown"] * len(dates), dtype=object)
    for index, value in enumerate(dates):
        if np.isfinite(value):
            result[index] = f"test_cluster_{cluster_by_date[value]}"
    return result.astype(str)


def main() -> None:
    source = json.loads(
        (PROJECT_DIR / "runs/p89_cohort_class_prior_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    configuration = source["H1_selected"]["configuration"]
    grouping_source = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping_config = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = full40.align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    test = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    sample_ids = test["sample_ids"].astype(str)
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        sample_ids,
    )
    groups = test_groups(metadata)
    biases, neighbors = class_biases(
        all_labels,
        all_metadata,
        np.ones(len(all_labels), dtype=bool),
        metadata,
        groups,
        int(configuration["neighbors"]),
    )
    logp = np.log(np.maximum(base_probability, 1e-12)) + float(
        configuration["weight"]
    ) * biases
    adjusted = np.exp(logp - np.logaddexp.reduce(logp, axis=1, keepdims=True))
    transition, decoder = load_decoder()
    indices = np.arange(len(sample_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    reproduced = decode_sessions(
        np.log(np.maximum(base_probability, 1e-12)), sessions, transition, decoder
    )
    if not np.array_equal(reproduced, p87):
        raise RuntimeError("failed to reproduce immutable P87 Test prediction")
    protocol_value = (
        sample_ids,
        None,
        adjusted,
        p87,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        None,
    )
    prediction, grouping = joint_decode(
        adjusted,
        p87,
        protocol_value,
        grouping_config,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_cohort_prior_joint.csv"
    io.write_submission(submission, source_rows, prediction)
    np.savez_compressed(
        OUTPUT / "test_predictions.npz",
        sample_ids=sample_ids,
        groups=groups,
        p87_prediction=p87,
        prediction=prediction,
    )
    report = {
        "stage": "P89_recording_cohort_class_prior_Test_v1",
        "protocol": (
            "Split Test into date-contiguous recording cohorts without labels, "
            "fit the frozen nearest-subject class prior on all labeled users, "
            "then apply the confirmed joint repeated-take decoder."
        ),
        "configuration": configuration,
        "test_group_counts": {
            group: int(np.sum(groups == group)) for group in sorted(set(groups.tolist()))
        },
        "nearest_training_users": neighbors,
        "grouping": grouping,
        "changes_vs_p87": int(np.sum(prediction != p87)),
        "submission": {
            "path": str(submission.resolve()),
            "sha256": io.digest(submission),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
