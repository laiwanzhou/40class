from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics, decode_unique_beam
from p88_train_depth_residual import rescue_harm
from p89_deterministic_triple_repeat import (
    TEST_SAFE,
    prepared_probability,
    test_protocol,
    triple_groups,
)
from p89_imu_rescue_gate import softmax
import p89_build_dual_consensus_submission as submission_io


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_peer_supported_triple_repair_v1"
SAFE_VALIDATION = PROJECT_DIR / "runs/p89_imu_probability_blend_v1/validation_predictions.npz"


def decode(protocol_value, probability: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    prediction = np.asarray(baseline, dtype=np.int64).copy()
    groups = triple_groups(protocol_value, "exact_three")
    changed_rows = supported_positions = 0
    for group in groups:
        stacked = np.stack([probability[session] for session in group])
        aggregate = stacked.mean(axis=0)
        shared = decode_unique_beam(
            np.log(np.maximum(aggregate, 1e-12)),
            protocol_value[7],
            protocol_value[8].transition_weight,
            protocol_value[8].beam_width,
        )
        for position, candidate in enumerate(shared):
            rows = np.asarray([session[position] for session in group], dtype=np.int64)
            # A shared-path proposal is allowed only if at least one independently
            # decoded take already predicted it at the aligned position.
            if not np.any(baseline[rows] == candidate):
                continue
            supported_positions += 1
            changed_rows += int(np.sum(prediction[rows] != candidate))
            prediction[rows] = candidate
    return prediction, {
        "candidate_groups": len(groups),
        "supported_positions": supported_positions,
        "changed_rows_with_duplicates": changed_rows,
    }


def evaluate(protocol_value, probability: np.ndarray, baseline: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    prediction, grouping = decode(protocol_value, probability, baseline)
    users = protocol_value[4].users.astype(str)
    per_user = {
        user: int(
            np.sum(protocol_value[1][users == user] == prediction[users == user])
            - np.sum(protocol_value[1][users == user] == baseline[users == user])
        )
        for user in sorted(set(users.tolist()))
    }
    return (
        {
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_safe": rescue_harm(protocol_value[1], baseline, prediction),
            "per_user_gain": per_user,
            "minimum_user_gain": int(min(per_user.values())),
            "grouping": grouping,
        },
        prediction,
    )


def test_probability() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz") as source:
        sample_ids = source["sample_ids"].astype(str)
        probability = np.asarray(source["base_probability"], dtype=np.float64)
    with np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz") as source:
        lookup = {sample_id: index for index, sample_id in enumerate(source["sample_ids"].astype(str))}
        rows = np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
        imu_probability = softmax(np.asarray(source["imu_logits"], dtype=np.float64)[rows], 3.0)
    probability = 0.95 * probability + 0.05 * imu_probability
    probability /= probability.sum(axis=1, keepdims=True)
    safe = submission_io.read_prediction(TEST_SAFE)
    return sample_ids, probability, safe


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(SAFE_VALIDATION) as source:
        if not np.array_equal(source["h1_sample_ids"].astype(str), h1[0]):
            raise RuntimeError("H1 alignment changed")
        if not np.array_equal(source["h2_sample_ids"].astype(str), h2[0]):
            raise RuntimeError("H2 alignment changed")
        h1_safe = source["h1_prediction"].astype(np.int64)
        h2_safe = source["h2_prediction"].astype(np.int64)
    h1_report, h1_prediction = evaluate(h1, prepared_probability(h1), h1_safe)
    h2_report, h2_prediction = evaluate(h2, prepared_probability(h2), h2_safe)

    sample_ids, probability, safe = test_probability()
    test = test_protocol(sample_ids, probability, safe)
    test_prediction, test_grouping = decode(test, probability, safe)
    changed = np.flatnonzero(test_prediction != safe)
    np.savez_compressed(
        OUTPUT / "validation_and_test_predictions.npz",
        h1_sample_ids=h1[0], h1_prediction=h1_prediction,
        h2_sample_ids=h2[0], h2_prediction=h2_prediction,
        test_sample_ids=sample_ids, test_prediction=test_prediction,
    )
    report = {
        "stage": "P89_peer_supported_exact_triple_repair_v1",
        "protocol": "Only maximal runs of exactly three equal-length sessions are used. A shared posterior path may repair a row only when at least one independently decoded take already supports that class at the same position. No Test labels or leaderboard feedback are used.",
        "H1": h1_report,
        "H2_confirmation": h2_report,
        "test": {
            "grouping": test_grouping,
            "changes_vs_0.85572_safe": int(len(changed)),
            "changed_ids": sample_ids[changed].tolist(),
            "changed_labels": [
                {"sample_id": sample_ids[index], "safe": int(safe[index]), "candidate": int(test_prediction[index])}
                for index in changed
            ],
        },
        "deployment_decision": (
            "eligible_for_additional_stress_tests_no_submission_yet"
            if h1_report["rescue_harm_vs_safe"]["net"] > 0
            and h1_report["minimum_user_gain"] >= 0
            and h2_report["rescue_harm_vs_safe"]["net"] > 0
            and h2_report["minimum_user_gain"] >= 0
            else "reject"
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
