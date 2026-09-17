from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_imu_rescue_gate_v1"


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    output = np.exp(scaled)
    return output / output.sum(axis=1, keepdims=True)


def aligned_imu(
    source_ids: np.ndarray,
    logits: np.ndarray,
    target_ids: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    present = np.asarray([value in lookup for value in target_ids.astype(str)])
    output = np.full((len(target_ids), 40), 1.0 / 40.0, dtype=np.float64)
    rows = np.flatnonzero(present)
    source_rows = np.asarray(
        [lookup[target_ids[row]] for row in rows], dtype=np.int64
    )
    output[rows] = softmax(logits[source_rows], temperature)
    return output, present


def predict(
    current: np.ndarray,
    base_probability: np.ndarray,
    imu_probability: np.ndarray,
    present: np.ndarray,
    configuration: dict[str, float],
) -> np.ndarray:
    top = np.argmax(imu_probability, axis=1)
    ordered = np.sort(imu_probability, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    rows = np.arange(len(current))
    delta = imu_probability[rows, top] - imu_probability[rows, current]
    accepted = (
        present
        & (top != current)
        & (confidence >= configuration["minimum_imu_confidence"])
        & (margin >= configuration["minimum_imu_margin"])
        & (delta >= configuration["minimum_imu_delta"])
        & (
            base_probability[rows, top]
            >= configuration["minimum_p87_target_probability"]
        )
    )
    output = current.copy()
    output[accepted] = top[accepted]
    return output


def evaluate(
    protocol_value,
    current: np.ndarray,
    imu_probability: np.ndarray,
    present: np.ndarray,
    configuration: dict[str, float],
) -> dict[str, object]:
    prediction = predict(
        current, protocol_value[2], imu_probability, present, configuration
    )
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
        "imu_present": int(np.sum(present)),
        "imu_raw_accuracy": float(
            np.mean(np.argmax(imu_probability[present], axis=1) == protocol_value[1][present])
        ),
    }


def main() -> None:
    oof = np.load(
        PROJECT_DIR
        / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    )
    oof_ids = oof["sample_ids"].astype(str)
    oof_logits = np.asarray(oof["imu_logits"], dtype=np.float64)
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    route = np.load(
        PROJECT_DIR / "runs/p89_route_vote_gate_v1/validation_predictions.npz"
    )
    h1_current = route["h1_prediction"]
    h2_current = route["h2_prediction"]
    candidates = []
    cache_h1 = {}
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        cache_h1[temperature] = aligned_imu(
            oof_ids, oof_logits, h1[0], temperature
        )
        probability, present = cache_h1[temperature]
        for confidence in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.10):
            for margin in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
                for delta in (0.05, 0.10, 0.20, 0.30, 0.40):
                    for target_probability in (0.001, 0.005, 0.01, 0.03, 0.05, 0.10):
                        configuration = {
                            "temperature": temperature,
                            "minimum_imu_confidence": confidence,
                            "minimum_imu_margin": margin,
                            "minimum_imu_delta": delta,
                            "minimum_p87_target_probability": target_probability,
                        }
                        candidates.append(
                            evaluate(
                                h1,
                                h1_current,
                                probability,
                                present,
                                configuration,
                            )
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
    h2_probability, h2_present = aligned_imu(
        oof_ids,
        oof_logits,
        h2[0],
        float(configuration["temperature"]),
    )
    confirmation = evaluate(
        h2, h2_current, h2_probability, h2_present, configuration
    )

    test_imu = np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz")
    test_ids = test_imu["sample_ids"].astype(str)
    test_probability = softmax(
        test_imu["imu_logits"], float(configuration["temperature"])
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
    base_ids = test_base["sample_ids"].astype(str)
    lookup = {value: index for index, value in enumerate(test_ids)}
    order = np.asarray([lookup[value] for value in base_ids], dtype=np.int64)
    test_probability = test_probability[order]
    prediction = predict(
        test_current,
        np.asarray(test_base["base_probability"], dtype=np.float64),
        test_probability,
        np.ones(len(test_current), dtype=bool),
        configuration,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_imu_rescue_gate.csv"
    io.write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_subject_disjoint_IMU_rescue_gate_v1",
        "protocol": (
            "Use the same P3 IMU RF contract for subject-disjoint OOF and Test. "
            "Select only temperature and high-confidence rescue thresholds on H1 "
            "under no-user-regression, then transfer once to H2 and Test."
        ),
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
