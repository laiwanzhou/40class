from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as dual
import p89_build_evidence_rollback as detail
import p89_build_final_test_submissions_nometa  # noqa: F401
import p89_build_final_test_submissions as detail_deploy
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_triple_consensus_v1"


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
) -> dict[str, object]:
    detail_prediction, full_prediction = dual.component_predictions(
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
    dual_prediction, dual_audit = dual.merge(
        protocol_value[3], detail_prediction, full_prediction
    )
    repeat_prediction, repeat_grouping = decode_aligned_repeat(
        np.log(np.maximum(protocol_value[2], 1e-12)),
        protocol_value[5],
        protocol_value[4],
        protocol_value[7],
        protocol_value[8],
        protocol_value[9],
    )
    prediction, final_audit = dual.merge(
        protocol_value[3], dual_prediction, repeat_prediction
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
            "changes": int(np.sum(prediction[rows] != protocol_value[3][rows])),
        }
    return {
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "dual_merge": dual_audit,
        "dual_plus_repeat_merge": final_audit,
        "repeat_grouping": repeat_grouping,
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
    )

    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = dual.read_rows(p87_path)
    p87 = dual.read_prediction(p87_path)
    dual_test = dual.read_prediction(
        PROJECT_DIR / "runs/p89_dual_consensus_v1/submission_p89_dual_consensus.csv"
    )
    repeat_test = dual.read_prediction(
        PROJECT_DIR / "runs/p88_final_test_predictions_v1/submission_p88_repeat.csv"
    )
    prediction, test_merge = dual.merge(p87, dual_test, repeat_test)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_triple_consensus.csv"
    dual.write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_triple_consensus_Detail21_full40_repeat_v1",
        "protocol": (
            "Union of the deployable Detail21 evidence gate, full40 expert-vote "
            "gate, and H1-selected/H2-confirmed repeated-take consensus. Any "
            "cross-component label conflict reverts to immutable P87."
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
