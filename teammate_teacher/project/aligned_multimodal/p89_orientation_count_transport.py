from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
import p89_count_regularized_imu_submission as count_transport
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import align_metadata, build_sessions, classification_metrics, decode_sessions
from p88_train_depth_residual import rescue_harm
from p89_deploy_cohort_prior import test_groups
from p89_deploy_supervised_router import load_decoder
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_probability_blend import evaluate
from p89_imu_rescue_gate import aligned_imu, softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_orientation_count_transport_v1"
TEMPERATURE = 0.75
OUTER_WEIGHT = 0.06
STAY_WEIGHT = 2.0


def orientation_candidate(protocol_value, ids, logits, grouping):
    probability, present = aligned_imu(
        ids, logits, protocol_value[0], TEMPERATURE
    )
    adjusted = protocol_value[2].copy()
    adjusted[present] = (
        (1.0 - OUTER_WEIGHT) * adjusted[present]
        + OUTER_WEIGHT * probability[present]
    )
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    candidate = evaluate(
        protocol_value,
        probability,
        present,
        OUTER_WEIGHT,
        "joint",
        grouping,
    )[1]
    return adjusted, candidate


def preserve_own_counts(adjusted, candidate, groups):
    # PRIOR_WEIGHT=0 means observed_counts is mathematically unused.  Pass an
    # explicit neutral matrix and keep the candidate's own count vector exactly.
    original_stay = count_transport.STAY_WEIGHT
    count_transport.STAY_WEIGHT = STAY_WEIGHT
    try:
        return count_transport.transport(
            adjusted,
            candidate,
            groups,
            np.zeros((1, 40), dtype=np.float64),
        )
    finally:
        count_transport.STAY_WEIGHT = original_stay


def per_user(protocol_value, prediction, references):
    users = protocol_value[4].users.astype(str)
    result = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        correct = int(np.sum(prediction[rows] == protocol_value[1][rows]))
        item = {"correct": correct, "rows": int(np.sum(rows))}
        for name, reference in references.items():
            item[f"gain_vs_{name}"] = correct - int(
                np.sum(reference[rows] == protocol_value[1][rows])
            )
        result[user] = item
    return result


def main() -> None:
    orientation = np.load(
        PROJECT_DIR / "runs/p89_imu_orientation_expert_v1/oof_logits.npz"
    )
    orientation_ids = orientation["sample_ids"].astype(str)
    orientation_logits = np.asarray(orientation["imu_logits"], dtype=np.float64)
    p3 = np.load(
        PROJECT_DIR
        / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    )
    grouping_source = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
    count_v2 = np.load(
        PROJECT_DIR / "runs/p89_count_preserving_imu_v2/predictions.npz"
    )
    validation = []
    saved = {}
    for split, protocol_value, count_key in (
        (
            "H1_selection",
            full40.protocol(full40.H1_RUN, full40.H1_USERS),
            "h1_prediction",
        ),
        (
            "H2_confirmation",
            full40.protocol(full40.H2_RUN, full40.H2_USERS),
            "h2_prediction",
        ),
    ):
        adjusted, orientation_prediction = orientation_candidate(
            protocol_value, orientation_ids, orientation_logits, grouping
        )
        prediction, audit = preserve_own_counts(
            adjusted,
            orientation_prediction,
            protocol_value[4].users.astype(str),
        )
        p3_probability, p3_present = aligned_imu(
            p3["sample_ids"].astype(str),
            np.asarray(p3["imu_logits"], dtype=np.float64),
            protocol_value[0],
            3.0,
        )
        known_best_recipe = evaluate(
            protocol_value,
            p3_probability,
            p3_present,
            0.05,
            "joint",
            grouping,
        )[1]
        references = {
            "p87": protocol_value[3],
            "known_0.85572_recipe": known_best_recipe,
            "count_v2": count_v2[count_key],
        }
        validation.append(
            {
                "split": split,
                "metrics": classification_metrics(protocol_value[1], prediction),
                "orientation_before_transport": classification_metrics(
                    protocol_value[1], orientation_prediction
                ),
                "rescue_harm_vs_p87": rescue_harm(
                    protocol_value[1], protocol_value[3], prediction
                ),
                "rescue_harm_vs_known_0.85572_recipe": rescue_harm(
                    protocol_value[1], known_best_recipe, prediction
                ),
                "rescue_harm_vs_count_v2": rescue_harm(
                    protocol_value[1], count_v2[count_key], prediction
                ),
                "per_user": per_user(protocol_value, prediction, references),
                "transport": audit,
            }
        )
        saved[split] = (protocol_value[0], prediction)

    test = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    test_ids = test["sample_ids"].astype(str)
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    test_orientation = np.load(
        PROJECT_DIR / "runs/p89_imu_orientation_expert_v1/test_logits.npz"
    )
    lookup = {
        value: index
        for index, value in enumerate(test_orientation["sample_ids"].astype(str))
    }
    orientation_probability = softmax(
        np.asarray(test_orientation["imu_logits"], dtype=np.float64)[
            np.asarray([lookup[value] for value in test_ids], dtype=np.int64)
        ],
        TEMPERATURE,
    )
    adjusted = (
        (1.0 - OUTER_WEIGHT) * base_probability
        + OUTER_WEIGHT * orientation_probability
    )
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    transition, decoder = load_decoder()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        test_ids,
    )
    indices = np.arange(len(test_ids), dtype=np.int64)
    sessions = build_sessions(
        indices, metadata, decoder.gap_seconds, "anonymous_date"
    )
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
    test_protocol = (
        test_ids,
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
    orientation_prediction, grouping_audit = joint_decode(
        adjusted,
        p87,
        test_protocol,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    groups = test_groups(metadata)
    prediction, transport_audit = preserve_own_counts(
        adjusted, orientation_prediction, groups
    )
    known_best = io.read_prediction(
        PROJECT_DIR
        / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
    )
    count_v2_test = io.read_prediction(
        PROJECT_DIR
        / "runs/p89_count_preserving_imu_v2/submission_p89_count_preserving_imu_v2.csv"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_orientation_count_transport.csv"
    io.write_submission(submission, source_rows, prediction)
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        h1_sample_ids=saved["H1_selection"][0],
        h1_prediction=saved["H1_selection"][1],
        h2_sample_ids=saved["H2_confirmation"][0],
        h2_prediction=saved["H2_confirmation"][1],
        test_sample_ids=test_ids,
        test_prediction=prediction,
        test_probability=adjusted,
    )
    report = {
        "stage": "P89_orientation_IMU_plus_count_preserving_transport_v1",
        "protocol": (
            "H1-selected orientation-only IMU temperature/outer weight, H2 frozen "
            "confirmation, followed by an exact own-count-preserving assignment. "
            "No external Test quota and no Test label feedback."
        ),
        "configuration": {
            "temperature": TEMPERATURE,
            "outer_weight": OUTER_WEIGHT,
            "stay_weight": STAY_WEIGHT,
        },
        "validation": validation,
        "Test": {
            "grouping": grouping_audit,
            "transport": transport_audit,
            "orientation_changes_vs_p87": int(
                np.sum(orientation_prediction != p87)
            ),
            "final_changes_vs_p87": int(np.sum(prediction != p87)),
            "changes_vs_known_0.85572": int(np.sum(prediction != known_best)),
            "changes_vs_count_v2": int(np.sum(prediction != count_v2_test)),
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
