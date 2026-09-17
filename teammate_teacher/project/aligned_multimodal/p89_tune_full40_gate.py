from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = PROJECT_DIR / "runs/p89_full40_scale_invariant_v1"
OUTPUT = PROJECT_DIR / "runs/p89_full40_tuned_gate_v1"


def evidence(
    protocol_value,
    holdout_users: list[str],
    all_ids: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    features: np.ndarray,
    expert_probability: np.ndarray,
    configuration: dict[str, object],
) -> dict[str, np.ndarray]:
    lookup = {value: index for index, value in enumerate(all_ids.astype(str))}
    rows = np.asarray(
        [lookup[value] for value in protocol_value[0].astype(str)], dtype=np.int64
    )
    fit = ~np.isin(users, holdout_users)
    scaler = StandardScaler()
    model = full40.make_model(
        str(configuration["model"]), float(configuration["regularization"])
    )
    model.fit(scaler.fit_transform(features[fit]), labels[fit])
    router_probability = full40.model_probability(
        model,
        scaler.transform(features[rows]),
        float(configuration["temperature"]),
    )
    weight = float(configuration["weight"])
    blended = (1.0 - weight) * protocol_value[2] + weight * router_probability
    blended /= blended.sum(axis=1, keepdims=True)
    candidate = full40.decode(blended, protocol_value)
    top = np.argmax(expert_probability[rows], axis=2)
    row_index = np.arange(len(candidate))
    candidate_votes = np.sum(top == candidate[:, None], axis=1)
    base_votes = np.sum(top == protocol_value[3][:, None], axis=1)
    router_delta = (
        router_probability[row_index, candidate]
        - router_probability[row_index, protocol_value[3]]
    )
    return {
        "labels": protocol_value[1],
        "base": protocol_value[3],
        "candidate": candidate,
        "candidate_votes": candidate_votes,
        "vote_margin": candidate_votes - base_votes,
        "router_delta": router_delta,
        "users": protocol_value[4].users.astype(str),
    }


def predict(values: dict[str, np.ndarray], config: dict[str, float]) -> np.ndarray:
    changed = values["candidate"] != values["base"]
    accepted = (
        changed
        & (values["candidate_votes"] >= int(config["minimum_candidate_votes"]))
        & (values["vote_margin"] >= int(config["minimum_vote_margin"]))
        & (values["router_delta"] >= float(config["minimum_router_delta"]))
    )
    output = values["base"].copy()
    output[accepted] = values["candidate"][accepted]
    return output


def evaluate(
    values: dict[str, np.ndarray], config: dict[str, float]
) -> dict[str, object]:
    output = predict(values, config)
    per_user = {}
    gains = []
    for user in sorted(set(values["users"].tolist())):
        selected = values["users"] == user
        base_correct = int(np.sum(values["base"][selected] == values["labels"][selected]))
        candidate_correct = int(np.sum(output[selected] == values["labels"][selected]))
        gains.append(candidate_correct - base_correct)
        per_user[user] = {
            "p87_correct": base_correct,
            "candidate_correct": candidate_correct,
            "gain": candidate_correct - base_correct,
            "changes": int(np.sum(output[selected] != values["base"][selected])),
        }
    return {
        "configuration": config,
        "metrics": classification_metrics(values["labels"], output),
        "rescue_harm_vs_p87": rescue_harm(values["labels"], values["base"], output),
        "minimum_user_gain": int(min(gains)),
        "positive_users": int(np.sum(np.asarray(gains) > 0)),
        "per_user": per_user,
    }


def main() -> None:
    source_summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    configuration = source_summary["selected"]
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
    features = full40.feature_matrix(
        train_probability, str(configuration["feature_kind"])
    )
    h1_protocol = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2_protocol = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1 = evidence(
        h1_protocol,
        full40.H1_USERS,
        all_ids,
        labels,
        metadata.users,
        features,
        train_probability,
        configuration,
    )
    h2 = evidence(
        h2_protocol,
        full40.H2_USERS,
        all_ids,
        labels,
        metadata.users,
        features,
        train_probability,
        configuration,
    )

    candidates = []
    best = None
    best_key = None
    for votes in range(0, 20):
        for margin in range(-19, 20):
            for delta in (-0.50, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
                item = evaluate(
                    h1,
                    {
                        "minimum_candidate_votes": votes,
                        "minimum_vote_margin": margin,
                        "minimum_router_delta": delta,
                    },
                )
                candidates.append(item)
                stable = item["minimum_user_gain"] >= 0
                changes = item["rescue_harm_vs_p87"]
                key = (
                    stable,
                    item["metrics"]["correct"],
                    item["positive_users"],
                    changes["net"],
                    -changes["harm"],
                    -changes["changed"],
                )
                if best_key is None or key > best_key:
                    best_key, best = key, item
    assert best is not None
    h2_result = evaluate(h2, best["configuration"])

    test = np.load(SOURCE / "predictions.npz")
    test_ids = test["sample_ids"].astype(str)
    routed_ids = test["routed_sample_ids"].astype(str)
    routed_probability = np.asarray(test["routed_probability"], dtype=np.float64)
    candidate = test["prediction"].astype(np.int64)
    test_expert_probability, test_names = full40.test_probabilities(routed_ids)
    if test_names != expert_names:
        raise RuntimeError("full40 expert order differs")
    routed_lookup = {value: index for index, value in enumerate(routed_ids)}
    all_positions = np.asarray(
        [index for index, value in enumerate(test_ids) if value in routed_lookup],
        dtype=np.int64,
    )
    routed_positions = np.asarray(
        [routed_lookup[test_ids[index]] for index in all_positions], dtype=np.int64
    )
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    partial_base = p87[all_positions]
    partial_candidate = candidate[all_positions]
    top = np.argmax(test_expert_probability[routed_positions], axis=2)
    row_index = np.arange(len(all_positions))
    test_values = {
        "base": partial_base,
        "candidate": partial_candidate,
        "candidate_votes": np.sum(top == partial_candidate[:, None], axis=1),
        "vote_margin": (
            np.sum(top == partial_candidate[:, None], axis=1)
            - np.sum(top == partial_base[:, None], axis=1)
        ),
        "router_delta": (
            routed_probability[routed_positions][row_index, partial_candidate]
            - routed_probability[routed_positions][row_index, partial_base]
        ),
    }
    partial_prediction = predict(test_values, best["configuration"])
    prediction = p87.copy()
    prediction[all_positions] = partial_prediction
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_full40_tuned_gate.csv"
    io.write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_full40_H1_tuned_stable_gate_v1",
        "protocol": (
            "Tune only three label-free gate thresholds on H1, require no H1 "
            "user to regress, then transfer the frozen thresholds once to H2 "
            "and Test. The underlying full40 router remains unchanged."
        ),
        "source_configuration": configuration,
        "selected_on_H1": best,
        "H2_confirmation": h2_result,
        "grid_size": len(candidates),
        "Test": {
            "changes_vs_p87": int(np.sum(prediction != p87)),
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
