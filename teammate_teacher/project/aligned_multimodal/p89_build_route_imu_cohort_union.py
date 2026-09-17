from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
import p89_build_quad_consensus_submission as quad
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_route_imu_cohort_union_v1"


def evaluate(
    protocol_value,
    first_ids: np.ndarray,
    first_prediction: np.ndarray,
    cohort_ids: np.ndarray,
    cohort_prediction: np.ndarray,
) -> dict[str, object]:
    first = quad.align_prediction(first_ids, first_prediction, protocol_value[0])
    cohort = quad.align_prediction(cohort_ids, cohort_prediction, protocol_value[0])
    prediction, merge = io.merge(protocol_value[3], first, cohort)
    users = protocol_value[4].users.astype(str)
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        per_user[user] = int(
            np.sum(prediction[rows] == protocol_value[1][rows])
            - np.sum(protocol_value[3][rows] == protocol_value[1][rows])
        )
    return {
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "merge": merge,
        "per_user_gain": per_user,
    }


def main() -> None:
    route = np.load(
        PROJECT_DIR / "runs/p89_route_vote_gate_v1/validation_predictions.npz"
    )
    imu = np.load(
        PROJECT_DIR / "runs/p89_imu_probability_blend_v1/validation_predictions.npz"
    )
    cohort = np.load(
        PROJECT_DIR / "runs/p89_cohort_class_prior_v1/validation_predictions.npz"
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_first, _ = io.merge(h1[3], route["h1_prediction"], imu["h1_prediction"])
    h2_first, _ = io.merge(h2[3], route["h2_prediction"], imu["h2_prediction"])
    h1_result = evaluate(
        h1, h1[0], h1_first, cohort["h1_sample_ids"], cohort["h1_prediction"]
    )
    h2_result = evaluate(
        h2, h2[0], h2_first, cohort["h2_sample_ids"], cohort["h2_prediction"]
    )
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    first_test = io.read_prediction(
        PROJECT_DIR / "runs/p89_route_imu_union_v1/submission_p89_route_imu_union.csv"
    )
    cohort_test = io.read_prediction(
        PROJECT_DIR
        / "runs/p89_cohort_class_prior_test_v1/submission_p89_cohort_prior_joint.csv"
    )
    prediction, test_merge = io.merge(p87, first_test, cohort_test)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_route_imu_cohort_union.csv"
    io.write_submission(submission, rows, prediction)
    report = {
        "stage": "P89_route_IMU_cohort_union_v1",
        "protocol": (
            "Union of route-vote plus low-weight IMU consensus with the stable "
            "date-cohort prior route. Conflicts revert to P87."
        ),
        "H1": h1_result,
        "H2": h2_result,
        "Test": {
            "merge": test_merge,
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
