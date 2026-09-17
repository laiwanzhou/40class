from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
)
from p88_train_depth_residual import rescue_harm
from p89_deploy_cohort_prior import test_groups
from p89_deploy_supervised_router import load_decoder
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_probability_blend import evaluate as imu_evaluate
from p89_imu_rescue_gate import aligned_imu, softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_count_preserving_imu_v2"
PRIOR_WEIGHT = 0.0
PRIOR_SIGMA = 1.0
COUNT_DEVIATION_WEIGHT = 0.25
STAY_WEIGHT = 2.5
IMU_WEIGHT = 0.05
IMU_TEMPERATURE = 3.0


def count_quota(
    predicted_count: np.ndarray,
    total: int,
    observed_counts: np.ndarray,
) -> np.ndarray:
    maximum = min(total, max(60, int(np.max(predicted_count) + 15)))
    values = np.arange(maximum + 1, dtype=np.float64)
    costs = np.zeros((40, maximum + 1), dtype=np.float64)
    for class_id in range(40):
        kernel = np.mean(
            np.exp(
                -0.5
                * (
                    (values[:, None] - observed_counts[None, :, class_id])
                    / PRIOR_SIGMA
                )
                ** 2
            ),
            axis=1,
        )
        costs[class_id] = (
            COUNT_DEVIATION_WEIGHT
            * (values - float(predicted_count[class_id])) ** 2
            - PRIOR_WEIGHT * np.log(kernel + 1e-6)
        )

    dynamic = np.full((41, total + 1), np.inf, dtype=np.float64)
    back = np.full((40, total + 1), -1, dtype=np.int64)
    dynamic[0, 0] = 0.0
    for class_id in range(40):
        for subtotal in range(total + 1):
            maximum_count = min(maximum, subtotal)
            candidate_counts = np.arange(maximum_count + 1, dtype=np.int64)
            candidate_costs = (
                dynamic[class_id, subtotal - candidate_counts]
                + costs[class_id, : maximum_count + 1]
            )
            selected = int(np.argmin(candidate_costs))
            dynamic[class_id + 1, subtotal] = candidate_costs[selected]
            back[class_id, subtotal] = selected
    quota = np.zeros(40, dtype=np.int64)
    subtotal = total
    for class_id in range(39, -1, -1):
        quota[class_id] = int(back[class_id, subtotal])
        subtotal -= int(quota[class_id])
    if subtotal != 0 or int(quota.sum()) != total:
        raise RuntimeError("count-prior dynamic program failed")
    return quota


def transport(
    adjusted_probability: np.ndarray,
    safe_prediction: np.ndarray,
    groups: np.ndarray,
    observed_counts: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    output = safe_prediction.copy()
    audit = {}
    for group in sorted(set(groups.astype(str).tolist())):
        indices = np.flatnonzero(groups.astype(str) == group)
        if group == "unknown" or len(indices) < 2:
            audit[group] = {"rows": int(len(indices)), "skipped": True}
            continue
        predicted_count = np.bincount(safe_prediction[indices], minlength=40)
        quota = count_quota(predicted_count, len(indices), observed_counts)
        slots = np.repeat(np.arange(40, dtype=np.int64), quota)
        cost = -np.log(np.maximum(adjusted_probability[indices][:, slots], 1e-12))
        cost -= STAY_WEIGHT * (
            safe_prediction[indices, None] == slots[None, :]
        )
        row_assignment, slot_assignment = linear_sum_assignment(cost)
        output[indices[row_assignment]] = slots[slot_assignment]
        audit[group] = {
            "rows": int(len(indices)),
            "changes": int(np.sum(output[indices] != safe_prediction[indices])),
            "count_changes": {
                str(class_id): {
                    "before": int(predicted_count[class_id]),
                    "after": int(quota[class_id]),
                }
                for class_id in range(40)
                if int(predicted_count[class_id]) != int(quota[class_id])
            },
        }
    return output, audit


def prepare_validation(protocol_value, imu_ids, imu_logits, grouping):
    imu_probability, present = aligned_imu(
        imu_ids, imu_logits, protocol_value[0], IMU_TEMPERATURE
    )
    adjusted = protocol_value[2].copy()
    adjusted[present] = (
        (1.0 - IMU_WEIGHT) * adjusted[present]
        + IMU_WEIGHT * imu_probability[present]
    )
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    safe_prediction = imu_evaluate(
        protocol_value,
        imu_probability,
        present,
        IMU_WEIGHT,
        "joint",
        grouping,
    )[1]
    return adjusted, safe_prediction


def validation_result(protocol_value, prediction, safe_prediction, audit):
    users = protocol_value[4].users.astype(str)
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        per_user[user] = {
            "vs_p87": int(
                np.sum(prediction[rows] == protocol_value[1][rows])
                - np.sum(protocol_value[3][rows] == protocol_value[1][rows])
            ),
            "vs_safe_0.85572_recipe": int(
                np.sum(prediction[rows] == protocol_value[1][rows])
                - np.sum(safe_prediction[rows] == protocol_value[1][rows])
            ),
        }
    return {
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "rescue_harm_vs_safe": rescue_harm(
            protocol_value[1], safe_prediction, prediction
        ),
        "per_user_gain": per_user,
        "transport": audit,
    }


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
    validation = []
    validation_predictions = []
    for protocol_value, excluded_users, split in (
        (h1, full40.H1_USERS, "H1_selection"),
        (h2, full40.H2_USERS, "H2_confirmation"),
    ):
        adjusted, safe_prediction = prepare_validation(
            protocol_value, imu_ids, imu_logits, grouping
        )
        fit_users = sorted(
            set(all_metadata.users.astype(str).tolist()) - set(excluded_users)
        )
        observed_counts = np.stack(
            [
                np.bincount(
                    all_labels[all_metadata.users.astype(str) == user], minlength=40
                )
                for user in fit_users
            ]
        )
        prediction, audit = transport(
            adjusted,
            safe_prediction,
            protocol_value[4].users.astype(str),
            observed_counts,
        )
        validation.append(
            {
                "split": split,
                **validation_result(
                    protocol_value, prediction, safe_prediction, audit
                ),
            }
        )
        validation_predictions.append(prediction)

    test = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    test_ids = test["sample_ids"].astype(str)
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
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
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        test_ids,
    )
    groups = test_groups(metadata)
    transition, decoder = load_decoder()
    indices = np.arange(len(test_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
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
    safe_prediction = io.read_prediction(
        PROJECT_DIR
        / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
    )
    fit_users = sorted(set(all_metadata.users.astype(str).tolist()))
    observed_counts = np.stack(
        [
            np.bincount(
                all_labels[all_metadata.users.astype(str) == user], minlength=40
            )
            for user in fit_users
        ]
    )
    test_prediction, test_audit = transport(
        adjusted, safe_prediction, groups, observed_counts
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_count_preserving_imu_v2.csv"
    io.write_submission(submission, source_rows, test_prediction)
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=validation_predictions[0],
        h2_prediction=validation_predictions[1],
        test_sample_ids=test_ids,
        test_prediction=test_prediction,
    )
    report = {
        "stage": "P89_count_preserving_IMU_transport_v2",
        "protocol": (
            "Start from the LB-validated 0.85572 IMU+joint recipe. Use only the "
            "candidate's own per-cohort class count vector as an invariant, then "
            "solve a global minimum-cost label reassignment with a strong stay "
            "penalty. No external Test count or inferred class quota is used."
        ),
        "configuration": {
            "prior_weight": PRIOR_WEIGHT,
            "prior_sigma": PRIOR_SIGMA,
            "count_deviation_weight": COUNT_DEVIATION_WEIGHT,
            "stay_weight": STAY_WEIGHT,
            "imu_weight": IMU_WEIGHT,
            "imu_temperature": IMU_TEMPERATURE,
        },
        "validation": validation,
        "Test": {
            "group_counts": {
                group: int(np.sum(groups == group)) for group in sorted(set(groups))
            },
            "changes_vs_p87": int(np.sum(test_prediction != p87)),
            "changes_vs_known_0.85572": int(
                np.sum(test_prediction != safe_prediction)
            ),
            "transport": test_audit,
            "submission": str(submission.resolve()),
            "sha256": io.digest(submission),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
