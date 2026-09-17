from __future__ import annotations

import itertools
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
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_route_vote_gate_v1"
ROUTE_NAMES = ("detail", "full40", "repeat", "router", "joint")


def components(
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
) -> tuple[np.ndarray, np.ndarray]:
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
    repeat_prediction, _ = decode_aligned_repeat(
        np.log(np.maximum(protocol_value[2], 1e-12)),
        protocol_value[5],
        protocol_value[4],
        protocol_value[7],
        protocol_value[8],
        protocol_value[9],
    )
    routed = quad.align_prediction(router_ids, router_prediction, protocol_value[0])
    joint = quad.align_prediction(joint_ids, joint_prediction, protocol_value[0])
    routes = np.stack(
        (detail_prediction, full_prediction, repeat_prediction, routed, joint),
        axis=1,
    )
    candidate, _ = dual.merge(protocol_value[3], detail_prediction, full_prediction)
    candidate, _ = dual.merge(protocol_value[3], candidate, repeat_prediction)
    candidate, _ = dual.merge(protocol_value[3], candidate, routed)
    candidate, _ = dual.merge(protocol_value[3], candidate, joint)
    return routes, candidate


def gated_prediction(
    base: np.ndarray,
    routes: np.ndarray,
    candidate: np.ndarray,
    configuration: dict[str, object],
) -> np.ndarray:
    support = routes == candidate[:, None]
    votes = np.sum(support, axis=1)
    allowed = votes >= int(configuration["minimum_route_votes"])
    for index, name in enumerate(ROUTE_NAMES):
        if bool(configuration[f"allow_{name}_alone"]):
            allowed |= support[:, index]
    accepted = (candidate != base) & allowed
    output = base.copy()
    output[accepted] = candidate[accepted]
    return output


def evaluate(
    protocol_value,
    routes: np.ndarray,
    candidate: np.ndarray,
    configuration: dict[str, object],
) -> dict[str, object]:
    prediction = gated_prediction(protocol_value[3], routes, candidate, configuration)
    users = protocol_value[4].users.astype(str)
    gains = []
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        base_correct = int(np.sum(protocol_value[3][rows] == protocol_value[1][rows]))
        candidate_correct = int(np.sum(prediction[rows] == protocol_value[1][rows]))
        gains.append(candidate_correct - base_correct)
        per_user[user] = candidate_correct - base_correct
    return {
        "configuration": configuration,
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "minimum_user_gain": int(min(gains)),
        "positive_users": int(np.sum(np.asarray(gains) > 0)),
        "per_user_gain": per_user,
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
        PROJECT_DIR
        / "runs/p89_global_joint_grouping_tuned_v1/validation_predictions.npz"
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    common = (
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
    h1_routes, h1_candidate = components(
        h1,
        "h1",
        full40.H1_USERS,
        *common,
        routed["selection_sample_ids"],
        routed["selection_routed_prediction"],
        joint["h1_sample_ids"],
        joint["h1_prediction"],
    )
    h2_routes, h2_candidate = components(
        h2,
        "h2",
        full40.H2_USERS,
        *common,
        routed["sample_ids"],
        routed["routed_prediction"],
        joint["h2_sample_ids"],
        joint["h2_prediction"],
    )
    candidates = []
    for minimum_votes in (2, 3, 4, 5, 6):
        for allowed in itertools.product((False, True), repeat=len(ROUTE_NAMES)):
            gate_configuration = {"minimum_route_votes": minimum_votes}
            gate_configuration.update(
                {
                    f"allow_{name}_alone": value
                    for name, value in zip(ROUTE_NAMES, allowed, strict=True)
                }
            )
            candidates.append(
                evaluate(h1, h1_routes, h1_candidate, gate_configuration)
            )
    candidates.sort(
        key=lambda item: (
            item["minimum_user_gain"] >= 0,
            item["metrics"]["correct"],
            item["positive_users"],
            item["metrics"]["balanced_accuracy"],
            item["rescue_harm_vs_p87"]["net"],
            -item["rescue_harm_vs_p87"]["harm"],
            -item["rescue_harm_vs_p87"]["changed"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    confirmation = evaluate(
        h2, h2_routes, h2_candidate, selected["configuration"]
    )

    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = dual.read_rows(p87_path)
    p87 = dual.read_prediction(p87_path)
    test_routes = np.stack(
        (
            dual.read_prediction(PROJECT_DIR / "runs/p89_evidence_rollback_v1/submission_p89_evidence_rollback.csv"),
            dual.read_prediction(PROJECT_DIR / "runs/p89_full40_consensus_gate_v1/submission_p89_full40_consensus.csv"),
            dual.read_prediction(PROJECT_DIR / "runs/p88_final_test_predictions_v1/submission_p88_repeat.csv"),
            dual.read_prediction(PROJECT_DIR / "runs/p89_supervised_router_test_v1/submission_p89_supervised_router.csv"),
            dual.read_prediction(PROJECT_DIR / "runs/p89_global_joint_repeat_test_v1/submission_p89_global_joint_repeat.csv"),
        ),
        axis=1,
    )
    test_candidate = dual.read_prediction(
        PROJECT_DIR / "runs/p89_quint_consensus_v1/submission_p89_quint_consensus.csv"
    )
    prediction = gated_prediction(
        p87, test_routes, test_candidate, selected["configuration"]
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=gated_prediction(
            h1[3], h1_routes, h1_candidate, selected["configuration"]
        ),
        h2_prediction=gated_prediction(
            h2[3], h2_routes, h2_candidate, selected["configuration"]
        ),
    )
    submission = OUTPUT / "submission_p89_route_vote_gate.csv"
    dual.write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_validated_route_vote_gate_v1",
        "protocol": (
            "Treat five deployable correction routes as votes. Select the minimum "
            "agreement and which individually validated routes may act alone on "
            "H1 under no-user-regression, then transfer once to H2 and Test."
        ),
        "route_names": ROUTE_NAMES,
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "Test": {
            "changes_vs_p87": int(np.sum(prediction != p87)),
            "path": str(submission.resolve()),
            "sha256": dual.digest(submission),
        },
        "full40_expert_names": expert_names,
        "all_H1_candidates": candidates,
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
