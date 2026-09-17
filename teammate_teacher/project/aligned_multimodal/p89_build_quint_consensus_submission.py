from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as dual
import p89_build_evidence_rollback as detail
import p89_build_final_test_submissions_nometa  # noqa: F401
import p89_build_final_test_submissions as detail_deploy
import p89_build_quad_consensus_submission as quad
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_quint_consensus_v1"


def validate(
    protocol_value,
    split: str,
    holdout_users: list[str],
    all_ids: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    detail_ids: np.ndarray,
    detail_probability: np.ndarray,
    full_probability: np.ndarray,
    full_features: np.ndarray,
    configuration: dict[str, object],
    stored_detail: np.lib.npyio.NpzFile,
    router_ids: np.ndarray,
    router_prediction: np.ndarray,
    joint_ids: np.ndarray,
    joint_prediction: np.ndarray,
) -> dict[str, object]:
    triple, component_audit = quad.triple_prediction(
        protocol_value,
        split,
        holdout_users,
        all_ids,
        labels,
        users,
        detail_ids,
        detail_probability,
        full_probability,
        full_features,
        configuration,
        stored_detail,
    )
    routed = quad.align_prediction(router_ids, router_prediction, protocol_value[0])
    quad_prediction, quad_audit = dual.merge(protocol_value[3], triple, routed)
    joint = quad.align_prediction(joint_ids, joint_prediction, protocol_value[0])
    prediction, quint_audit = dual.merge(
        protocol_value[3], quad_prediction, joint
    )
    per_user = {}
    for user in holdout_users:
        rows = protocol_value[4].users == user
        per_user[user] = {
            "p87_correct": int(
                np.sum(protocol_value[3][rows] == protocol_value[1][rows])
            ),
            "candidate_correct": int(
                np.sum(prediction[rows] == protocol_value[1][rows])
            ),
            "gain": int(
                np.sum(prediction[rows] == protocol_value[1][rows])
                - np.sum(protocol_value[3][rows] == protocol_value[1][rows])
            ),
            "changes": int(np.sum(prediction[rows] != protocol_value[3][rows])),
        }
    return {
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "component_merge": component_audit,
        "quad_merge": quad_audit,
        "quad_plus_joint_merge": quint_audit,
        "per_user": per_user,
    }


def main() -> None:
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    labels = teacher["oof_labels"].astype(np.int64)
    metadata = full40.align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    detail_reference = detail_deploy.load(
        PROJECT_DIR / "runs/p46_validation70_final_v1/crossfit_logits.npz"
    )
    detail_ids = detail_reference["sample_ids"].astype(str)
    detail_features, _ = detail_deploy.training_features(detail_ids)
    detail_probability = detail.expert_probability(detail_features)
    full_probability, expert_names = full40.train_probabilities(all_ids)
    source_summary = json.loads(
        (PROJECT_DIR / "runs/p89_full40_scale_invariant_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    configuration = source_summary["selected"]
    full_features = full40.feature_matrix(
        full_probability, str(configuration["feature_kind"])
    )
    stored_detail = np.load(
        PROJECT_DIR / "runs/p89_final_nometa_validation_v1/predictions.npz"
    )
    routed = np.load(
        PROJECT_DIR
        / "runs/p89_supervised_router_h1_to_h2_v3/confirmation_predictions.npz"
    )
    joint = np.load(
        PROJECT_DIR / "runs/p89_global_joint_repeat_v1/validation_predictions.npz"
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_result = validate(
        h1,
        "h1",
        full40.H1_USERS,
        all_ids,
        labels,
        metadata.users,
        detail_ids,
        detail_probability,
        full_probability,
        full_features,
        configuration,
        stored_detail,
        routed["selection_sample_ids"],
        routed["selection_routed_prediction"],
        joint["h1_sample_ids"],
        joint["h1_prediction"],
    )
    h2_result = validate(
        h2,
        "h2",
        full40.H2_USERS,
        all_ids,
        labels,
        metadata.users,
        detail_ids,
        detail_probability,
        full_probability,
        full_features,
        configuration,
        stored_detail,
        routed["sample_ids"],
        routed["routed_prediction"],
        joint["h2_sample_ids"],
        joint["h2_prediction"],
    )

    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = dual.read_rows(p87_path)
    p87 = dual.read_prediction(p87_path)
    quad_test = dual.read_prediction(
        PROJECT_DIR / "runs/p89_quad_consensus_v1/submission_p89_quad_consensus.csv"
    )
    joint_test = dual.read_prediction(
        PROJECT_DIR
        / "runs/p89_global_joint_repeat_test_v1/submission_p89_global_joint_repeat.csv"
    )
    prediction, test_merge = dual.merge(p87, quad_test, joint_test)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_quint_consensus.csv"
    dual.write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_quint_consensus_v1",
        "protocol": (
            "Union of the four independently validated correction routes and "
            "the shared-path repeated-take decoder. Any changed-label conflict "
            "reverts to immutable P87."
        ),
        "H1": h1_result,
        "H2": h2_result,
        "Test": {
            "merge": test_merge,
            "path": str(submission.resolve()),
            "sha256": dual.digest(submission),
        },
        "full40_expert_names": expert_names,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
