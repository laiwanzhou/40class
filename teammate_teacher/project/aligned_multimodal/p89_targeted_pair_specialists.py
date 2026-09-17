from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_targeted_pair_specialists_v1"


def align_rows(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    return np.asarray(
        [lookup[value] for value in target_ids.astype(str)], dtype=np.int64
    )


def fit_scores(
    train_features: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    eval_features: np.ndarray,
    current: np.ndarray,
    pairs: list[tuple[int, int]],
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    best_target = current.copy()
    best_score = np.zeros(len(current), dtype=np.float64)
    for source, target in pairs:
        train_rows = fit & np.isin(labels, (source, target))
        if np.sum(labels[train_rows] == source) < 5 or np.sum(labels[train_rows] == target) < 5:
            continue
        eval_rows = current == source
        if not np.any(eval_rows):
            continue
        scaler = StandardScaler()
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            solver="liblinear",
            max_iter=500,
        )
        model.fit(scaler.fit_transform(train_features[train_rows]), labels[train_rows])
        target_index = int(np.flatnonzero(model.classes_ == target)[0])
        probability = model.predict_proba(scaler.transform(eval_features[eval_rows]))[
            :, target_index
        ]
        rows = np.flatnonzero(eval_rows)
        better = probability > best_score[rows]
        best_score[rows[better]] = probability[better]
        best_target[rows[better]] = target
    return best_target, best_score


def predict(
    current: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> np.ndarray:
    output = current.copy()
    accepted = (target != current) & (score >= threshold)
    output[accepted] = target[accepted]
    return output


def evaluate(
    protocol_value,
    current: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    configuration: dict[str, object],
) -> dict[str, object]:
    prediction = predict(current, target, score, float(configuration["threshold"]))
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
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_rows = align_rows(all_ids, h1[0])
    h2_rows = align_rows(all_ids, h2[0])
    route = np.load(
        PROJECT_DIR / "runs/p89_route_vote_gate_v1/validation_predictions.npz"
    )
    h1_current = route["h1_prediction"]
    h2_current = route["h2_prediction"]
    confusion = Counter(
        (int(h1_current[row]), int(h1[1][row]))
        for row in np.flatnonzero(h1_current != h1[1])
    )
    pairs = [pair for pair, count in confusion.most_common(25) if count >= 2]
    fit_h1 = ~np.isin(metadata.users.astype(str), full40.H1_USERS)
    fit_h2 = ~np.isin(metadata.users.astype(str), full40.H2_USERS)
    cached_h1 = {}
    candidates = []
    for feature_kind in ("probability", "normalized_logp", "rank", "robust_combined"):
        train_features = full40.feature_matrix(train_probability, feature_kind)
        for regularization in (0.0003, 0.001, 0.003, 0.01, 0.03):
            key = (feature_kind, regularization)
            cached_h1[key] = fit_scores(
                train_features,
                labels,
                fit_h1,
                train_features[h1_rows],
                h1_current,
                pairs,
                regularization,
            )
            target, score = cached_h1[key]
            for threshold in (0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99):
                configuration = {
                    "feature_kind": feature_kind,
                    "regularization": regularization,
                    "threshold": threshold,
                }
                candidates.append(
                    evaluate(h1, h1_current, target, score, configuration)
                )
    candidates.sort(
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
    selected = candidates[0]
    configuration = selected["configuration"]
    train_features = full40.feature_matrix(
        train_probability, str(configuration["feature_kind"])
    )
    h2_target, h2_score = fit_scores(
        train_features,
        labels,
        fit_h2,
        train_features[h2_rows],
        h2_current,
        pairs,
        float(configuration["regularization"]),
    )
    confirmation = evaluate(
        h2, h2_current, h2_target, h2_score, configuration
    )

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
    test_probability, test_names = full40.test_probabilities(routed_ids)
    if test_names != expert_names:
        raise RuntimeError("Test expert order differs")
    test_features = full40.feature_matrix(
        test_probability, str(configuration["feature_kind"])
    )
    test_target, test_score = fit_scores(
        train_features,
        labels,
        np.ones(len(labels), dtype=bool),
        test_features,
        test_current[align_rows(test_ids, routed_ids)],
        pairs,
        float(configuration["regularization"]),
    )
    positions = align_rows(test_ids, routed_ids)
    prediction = test_current.copy()
    prediction[positions] = predict(
        test_current[positions],
        test_target,
        test_score,
        float(configuration["threshold"]),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_targeted_pair_specialists.csv"
    io.write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_H1_targeted_pair_specialists_v1",
        "protocol": (
            "Freeze directed confusion pairs occurring at least twice after the "
            "route gate on H1. Fit one subject-disjoint binary expert per pair, "
            "select feature/regularization/threshold on H1, transfer once to H2, "
            "and refit the frozen design on all labels for Test."
        ),
        "directed_pairs": [list(pair) for pair in pairs],
        "expert_names": expert_names,
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "Test": {
            "changes_vs_route_gate": int(np.sum(prediction != test_current)),
            "changes_vs_p87": int(np.sum(prediction != p87)),
            "path": str(submission.resolve()),
            "sha256": io.digest(submission),
        },
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
