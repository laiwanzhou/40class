from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_classwise_weighted_vote_v1"


def reliability(
    probability: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    alpha: float,
) -> np.ndarray:
    top = np.argmax(probability, axis=2)
    output = np.empty((probability.shape[1], 40), dtype=np.float64)
    for expert in range(probability.shape[1]):
        global_accuracy = float(np.mean(top[fit, expert] == labels[fit]))
        for class_id in range(40):
            selected = fit & (top[:, expert] == class_id)
            count = int(np.sum(selected))
            correct = int(np.sum(labels[selected] == class_id))
            output[expert, class_id] = (
                correct + alpha * global_accuracy
            ) / (count + alpha)
    return output


def evidence(
    probability: np.ndarray,
    reliability_value: np.ndarray,
    current_prediction: np.ndarray,
    mode: str,
) -> dict[str, np.ndarray]:
    top = np.argmax(probability, axis=2)
    confidence = np.max(probability, axis=2)
    if mode.startswith("logit"):
        base_weight = np.log(
            np.clip(reliability_value, 1e-3, 1.0 - 1e-3)
            / np.clip(1.0 - reliability_value, 1e-3, 1.0)
        )
        base_weight = np.maximum(base_weight, 0.0)
    else:
        base_weight = reliability_value
    scores = np.zeros((len(probability), 40), dtype=np.float64)
    counts = np.zeros((len(probability), 40), dtype=np.int64)
    reliability_sum = np.zeros((len(probability), 40), dtype=np.float64)
    for expert in range(probability.shape[1]):
        labels = top[:, expert]
        rows = np.arange(len(probability))
        weights = base_weight[expert, labels]
        if mode.endswith("confidence"):
            weights = weights * confidence[:, expert]
        scores[rows, labels] += weights
        counts[rows, labels] += 1
        reliability_sum[rows, labels] += reliability_value[expert, labels]
    winner = np.argmax(scores, axis=1)
    rows = np.arange(len(probability))
    winner_score = scores[rows, winner]
    current_score = scores[rows, current_prediction]
    winner_count = counts[rows, winner]
    winner_mean_reliability = reliability_sum[rows, winner] / np.maximum(
        winner_count, 1
    )
    return {
        "winner": winner,
        "winner_score": winner_score,
        "score_margin": winner_score - current_score,
        "winner_count": winner_count,
        "winner_mean_reliability": winner_mean_reliability,
    }


def predict(
    current: np.ndarray,
    values: dict[str, np.ndarray],
    configuration: dict[str, object],
) -> np.ndarray:
    accepted = (
        (values["winner"] != current)
        & (values["winner_count"] >= int(configuration["minimum_votes"]))
        & (values["score_margin"] >= float(configuration["minimum_score_margin"]))
        & (
            values["winner_mean_reliability"]
            >= float(configuration["minimum_mean_reliability"])
        )
    )
    output = current.copy()
    output[accepted] = values["winner"][accepted]
    return output


def evaluate(
    protocol_value,
    current: np.ndarray,
    values: dict[str, np.ndarray],
    configuration: dict[str, object],
) -> dict[str, object]:
    prediction = predict(current, values, configuration)
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
        "rescue_harm_vs_route_gate": rescue_harm(
            protocol_value[1], current, prediction
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
    train_probability, expert_names = full40.train_probabilities(all_ids)
    lookup = {value: index for index, value in enumerate(all_ids)}
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_rows = np.asarray([lookup[value] for value in h1[0].astype(str)], dtype=np.int64)
    h2_rows = np.asarray([lookup[value] for value in h2[0].astype(str)], dtype=np.int64)
    route = np.load(
        PROJECT_DIR / "runs/p89_route_vote_gate_v1/validation_predictions.npz"
    )
    h1_current = route["h1_prediction"]
    h2_current = route["h2_prediction"]
    fit_h1 = ~np.isin(metadata.users.astype(str), full40.H1_USERS)
    fit_h2 = ~np.isin(metadata.users.astype(str), full40.H2_USERS)
    candidate_results = []
    cached_h1 = {}
    for alpha in (3.0, 10.0, 30.0):
        reliability_h1 = reliability(train_probability, labels, fit_h1, alpha)
        for mode in ("precision", "precision_confidence", "logit", "logit_confidence"):
            key = (alpha, mode)
            cached_h1[key] = evidence(
                train_probability[h1_rows], reliability_h1, h1_current, mode
            )
            for votes in (2, 3, 4, 5, 7, 10, 14):
                for margin in (0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 3.0):
                    for minimum_reliability in (0.40, 0.50, 0.60, 0.70, 0.80):
                        configuration = {
                            "alpha": alpha,
                            "mode": mode,
                            "minimum_votes": votes,
                            "minimum_score_margin": margin,
                            "minimum_mean_reliability": minimum_reliability,
                        }
                        candidate_results.append(
                            evaluate(
                                h1,
                                h1_current,
                                cached_h1[key],
                                configuration,
                            )
                        )
    candidate_results.sort(
        key=lambda item: (
            item["minimum_user_gain"] >= 0,
            item["metrics"]["correct"],
            item["positive_users"],
            item["metrics"]["balanced_accuracy"],
            item["rescue_harm_vs_p87"]["net"],
            -item["rescue_harm_vs_route_gate"]["harm"],
            -item["rescue_harm_vs_route_gate"]["changed"],
        ),
        reverse=True,
    )
    selected = candidate_results[0]
    configuration = selected["configuration"]
    reliability_h2 = reliability(
        train_probability, labels, fit_h2, float(configuration["alpha"])
    )
    h2_values = evidence(
        train_probability[h2_rows],
        reliability_h2,
        h2_current,
        str(configuration["mode"]),
    )
    confirmation = evaluate(h2, h2_current, h2_values, configuration)

    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    test_current = io.read_prediction(
        PROJECT_DIR / "runs/p89_route_vote_gate_v1/submission_p89_route_vote_gate.csv"
    )
    test_base = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    test_ids = test_base["sample_ids"].astype(str)
    routed_ids = test_base["detail_sample_ids"].astype(str)
    routed_probability, test_names = full40.test_probabilities(routed_ids)
    if test_names != expert_names:
        raise RuntimeError("Test expert order differs")
    test_lookup = {value: index for index, value in enumerate(test_ids)}
    positions = np.asarray([test_lookup[value] for value in routed_ids], dtype=np.int64)
    final_reliability = reliability(
        train_probability,
        labels,
        np.ones(len(labels), dtype=bool),
        float(configuration["alpha"]),
    )
    test_values = evidence(
        routed_probability,
        final_reliability,
        test_current[positions],
        str(configuration["mode"]),
    )
    prediction = test_current.copy()
    prediction[positions] = predict(
        test_current[positions], test_values, configuration
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_classwise_weighted_vote.csv"
    io.write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_classwise_expert_reliability_vote_v1",
        "protocol": (
            "Estimate expert-by-predicted-class precision only on non-holdout "
            "subjects, select a small weighted-vote gate on H1, transfer once to "
            "H2, and refit the frozen reliability estimator on all labels for Test."
        ),
        "expert_names": expert_names,
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidate_results),
        "Test": {
            "changes_vs_route_gate": int(np.sum(prediction != test_current)),
            "changes_vs_p87": int(np.sum(prediction != p87)),
            "path": str(submission.resolve()),
            "sha256": io.digest(submission),
        },
        "all_H1_candidates": candidate_results,
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
