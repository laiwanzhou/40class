from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
)
from p88_train_depth_residual import rescue_harm
from p89_cohort_class_prior import class_biases
from p89_deploy_cohort_prior import test_groups
from p89_deploy_supervised_router import load_decoder
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import aligned_imu, softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_imu_cohort_probability_blend_v1"
IMU_WEIGHT = 0.05
IMU_TEMPERATURE = 3.0


def evaluate(
    protocol_value,
    imu_probability: np.ndarray,
    imu_present: np.ndarray,
    all_labels: np.ndarray,
    all_metadata,
    fit_mask: np.ndarray,
    target_groups: np.ndarray,
    grouping: GlobalRepeatConfig,
    neighbors: int,
    cohort_weight: float,
):
    adjusted = protocol_value[2].copy()
    adjusted[imu_present] = (
        (1.0 - IMU_WEIGHT) * adjusted[imu_present]
        + IMU_WEIGHT * imu_probability[imu_present]
    )
    biases, nearest = class_biases(
        all_labels,
        all_metadata,
        fit_mask,
        protocol_value[4],
        target_groups,
        neighbors,
    )
    logp = np.log(np.maximum(adjusted, 1e-12)) + cohort_weight * biases
    adjusted = np.exp(logp - np.logaddexp.reduce(logp, axis=1, keepdims=True))
    adjusted_protocol = list(protocol_value)
    adjusted_protocol[2] = adjusted
    prediction, grouping_audit = joint_decode(
        adjusted,
        protocol_value[3],
        tuple(adjusted_protocol),
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    users = protocol_value[4].users.astype(str)
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        per_user[user] = int(
            np.sum(prediction[rows] == protocol_value[1][rows])
            - np.sum(protocol_value[3][rows] == protocol_value[1][rows])
        )
    return {
        "configuration": {
            "imu_weight": IMU_WEIGHT,
            "imu_temperature": IMU_TEMPERATURE,
            "neighbors": neighbors,
            "cohort_weight": cohort_weight,
        },
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "per_user_gain": per_user,
        "minimum_user_gain": min(per_user.values()),
        "positive_users": int(sum(value > 0 for value in per_user.values())),
        "nearest_fit_users": nearest,
        "grouping": grouping_audit,
    }, prediction


def main() -> None:
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    imu = np.load(
        PROJECT_DIR
        / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    )
    imu_ids = imu["sample_ids"].astype(str)
    imu_logits = np.asarray(imu["imu_logits"], dtype=np.float64)
    grouping_source = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_imu, h1_present = aligned_imu(
        imu_ids, imu_logits, h1[0], IMU_TEMPERATURE
    )
    h2_imu, h2_present = aligned_imu(
        imu_ids, imu_logits, h2[0], IMU_TEMPERATURE
    )
    fit_h1 = ~np.isin(all_metadata.users.astype(str), full40.H1_USERS)
    fit_h2 = ~np.isin(all_metadata.users.astype(str), full40.H2_USERS)
    candidates = []
    predictions = []
    for neighbors in (1, 2, 3, 4, 5, 8):
        for cohort_weight in (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
            item, prediction = evaluate(
                h1,
                h1_imu,
                h1_present,
                all_labels,
                all_metadata,
                fit_h1,
                h1[4].users.astype(str),
                grouping,
                neighbors,
                cohort_weight,
            )
            candidates.append(item)
            predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["positive_users"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_p87"]["net"],
            -candidates[index]["rescue_harm_vs_p87"]["harm"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    config = selected["configuration"]
    confirmation, h2_prediction = evaluate(
        h2,
        h2_imu,
        h2_present,
        all_labels,
        all_metadata,
        fit_h2,
        h2[4].users.astype(str),
        grouping,
        int(config["neighbors"]),
        float(config["cohort_weight"]),
    )

    test = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    test_ids = test["sample_ids"].astype(str)
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    test_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        test_ids,
    )
    groups = test_groups(test_metadata)
    biases, nearest = class_biases(
        all_labels,
        all_metadata,
        np.ones(len(all_labels), dtype=bool),
        test_metadata,
        groups,
        int(config["neighbors"]),
    )
    test_imu = np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz")
    test_imu_ids = test_imu["sample_ids"].astype(str)
    lookup = {value: index for index, value in enumerate(test_imu_ids)}
    test_imu_probability = softmax(
        np.asarray(test_imu["imu_logits"], dtype=np.float64)[
            np.asarray([lookup[value] for value in test_ids], dtype=np.int64)
        ],
        IMU_TEMPERATURE,
    )
    adjusted = (1.0 - IMU_WEIGHT) * base_probability + IMU_WEIGHT * test_imu_probability
    logp = np.log(np.maximum(adjusted, 1e-12)) + float(config["cohort_weight"]) * biases
    adjusted = np.exp(logp - np.logaddexp.reduce(logp, axis=1, keepdims=True))
    transition, decoder = load_decoder()
    indices = np.arange(len(test_ids), dtype=np.int64)
    sessions = build_sessions(indices, test_metadata, decoder.gap_seconds, "anonymous_date")
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    reproduced = decode_sessions(
        np.log(np.maximum(base_probability, 1e-12)),
        sessions,
        transition,
        decoder,
    )
    if not np.array_equal(reproduced, p87):
        raise RuntimeError("failed to reproduce immutable P87")
    protocol_value = (
        test_ids,
        None,
        adjusted,
        p87,
        test_metadata,
        indices,
        sessions,
        transition,
        decoder,
        None,
    )
    test_prediction, test_grouping = joint_decode(
        adjusted,
        p87,
        protocol_value,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    known_best = io.read_prediction(
        PROJECT_DIR
        / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_imu_cohort_probability_blend.csv"
    io.write_submission(submission, source_rows, test_prediction)
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
        test_prediction=test_prediction,
    )
    report = {
        "stage": "P89_IMU_plus_recording_cohort_probability_blend_v1",
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "Test": {
            "groups": {group: int(np.sum(groups == group)) for group in sorted(set(groups))},
            "nearest_training_users": nearest,
            "grouping": test_grouping,
            "changes_vs_p87": int(np.sum(test_prediction != p87)),
            "changes_vs_known_0.85572": int(np.sum(test_prediction != known_best)),
            "submission": str(submission.resolve()),
            "sha256": io.digest(submission),
        },
        "all_H1_candidates": [candidates[index] for index in order],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "all_H1_candidates"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
