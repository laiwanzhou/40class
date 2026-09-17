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
OUTPUT = PROJECT_DIR / "runs/p89_route_imu_union_v1"


def evaluate(
    protocol_value,
    route_ids: np.ndarray,
    route_prediction: np.ndarray,
    imu_ids: np.ndarray,
    imu_prediction: np.ndarray,
) -> dict[str, object]:
    route = quad.align_prediction(route_ids, route_prediction, protocol_value[0])
    imu = quad.align_prediction(imu_ids, imu_prediction, protocol_value[0])
    prediction, merge = io.merge(protocol_value[3], route, imu)
    users = protocol_value[4].users.astype(str)
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        base_correct = int(np.sum(protocol_value[3][rows] == protocol_value[1][rows]))
        candidate_correct = int(np.sum(prediction[rows] == protocol_value[1][rows]))
        per_user[user] = candidate_correct - base_correct
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
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_result = evaluate(
        h1,
        route["h1_sample_ids"],
        route["h1_prediction"],
        imu["h1_sample_ids"],
        imu["h1_prediction"],
    )
    h2_result = evaluate(
        h2,
        route["h2_sample_ids"],
        route["h2_prediction"],
        imu["h2_sample_ids"],
        imu["h2_prediction"],
    )
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    route_test = io.read_prediction(
        PROJECT_DIR / "runs/p89_route_vote_gate_v1/submission_p89_route_vote_gate.csv"
    )
    imu_test = io.read_prediction(
        PROJECT_DIR
        / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
    )
    prediction, test_merge = io.merge(p87, route_test, imu_test)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_route_imu_union.csv"
    io.write_submission(submission, rows, prediction)
    report = {
        "stage": "P89_route_gate_plus_low_weight_IMU_union_v1",
        "protocol": (
            "Union of the validated route-vote gate and independently validated "
            "low-weight IMU shared-path decoder. Conflicts revert to P87."
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
