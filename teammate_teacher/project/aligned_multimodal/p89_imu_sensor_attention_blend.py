from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm
from p89_deterministic_triple_repeat import prepared_probability
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig


PROJECT_DIR = Path(__file__).resolve().parent
EXPERT = PROJECT_DIR / "runs/p89_imu_sensor_attention_expert_v1"
OUTPUT = PROJECT_DIR / "runs/p89_imu_sensor_attention_blend_v1"
SAFE = PROJECT_DIR / "runs/p89_imu_probability_blend_v1/validation_predictions.npz"
METHODS = (
    "equal_probability",
    "entropy_probability",
    "class_log_t1",
    "class_log_t2",
    "class_entropy_log_t2",
)


def normalize_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    result = np.maximum(np.asarray(probability, dtype=np.float64), 1e-12) ** (1.0 / temperature)
    return result / result.sum(axis=1, keepdims=True)


def aligned(source_ids: np.ndarray, probability: np.ndarray, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids.astype(str))}
    result = np.full((len(target_ids), 40), 1.0 / 40.0, dtype=np.float64)
    present = np.zeros(len(target_ids), dtype=bool)
    for index, sample_id in enumerate(target_ids.astype(str)):
        if sample_id in lookup:
            result[index] = probability[lookup[sample_id]]
            present[index] = True
    return result, present


def per_user_gain(protocol_value, baseline: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    users = protocol_value[4].users.astype(str)
    return {
        user: int(
            np.sum(protocol_value[1][users == user] == prediction[users == user])
            - np.sum(protocol_value[1][users == user] == baseline[users == user])
        )
        for user in sorted(set(users.tolist()))
    }


def evaluate(
    protocol_value,
    safe: np.ndarray,
    expert_probability: np.ndarray,
    present: np.ndarray,
    grouping_config: GlobalRepeatConfig,
    weight: float,
) -> tuple[dict[str, object], np.ndarray]:
    base_probability = prepared_probability(protocol_value)
    adjusted = base_probability.copy()
    adjusted[present] = (1.0 - weight) * base_probability[present] + weight * expert_probability[present]
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    adjusted_protocol = list(protocol_value)
    adjusted_protocol[2] = adjusted
    prediction, grouping = joint_decode(
        adjusted,
        protocol_value[3],
        tuple(adjusted_protocol),
        grouping_config,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    gains = per_user_gain(protocol_value, safe, prediction)
    return (
        {
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_safe": rescue_harm(protocol_value[1], safe, prediction),
            "per_user_gain_vs_safe": gains,
            "minimum_user_gain_vs_safe": int(min(gains.values())),
            "positive_users_vs_safe": int(np.sum(np.asarray(list(gains.values())) > 0)),
            "grouping": grouping,
        },
        prediction,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(EXPERT / "oof_probabilities.npz") as source:
        expert_ids = source["sample_ids"].astype(str)
        expert_oof = {name: np.asarray(source[name], dtype=np.float64) for name in METHODS}
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(SAFE) as source:
        h1_safe = source["h1_prediction"].astype(np.int64)
        h2_safe = source["h2_prediction"].astype(np.int64)
        if not np.array_equal(source["h1_sample_ids"].astype(str), h1[0]):
            raise RuntimeError("H1 safe alignment changed")
        if not np.array_equal(source["h2_sample_ids"].astype(str), h2[0]):
            raise RuntimeError("H2 safe alignment changed")
    grouping_source = json.loads(
        (PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json").read_text(encoding="utf-8")
    )
    grouping_config = GlobalRepeatConfig(**grouping_source["H1_selected"]["configuration"])

    candidates = []
    predictions = []
    for method in METHODS:
        h1_probability, h1_present = aligned(expert_ids, expert_oof[method], h1[0])
        for temperature in (0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
            tempered = normalize_temperature(h1_probability, temperature)
            for weight in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10):
                item, prediction = evaluate(
                    h1, h1_safe, tempered, h1_present, grouping_config, weight
                )
                item["configuration"] = {
                    "method": method,
                    "temperature": temperature,
                    "weight": weight,
                }
                candidates.append(item)
                predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain_vs_safe"] >= 0,
            candidates[index]["rescue_harm_vs_safe"]["net"],
            candidates[index]["positive_users_vs_safe"],
            candidates[index]["metrics"]["balanced_accuracy"],
            -candidates[index]["rescue_harm_vs_safe"]["harm"],
            -candidates[index]["rescue_harm_vs_safe"]["changed"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    configuration = selected["configuration"]
    h2_probability, h2_present = aligned(
        expert_ids, expert_oof[str(configuration["method"])], h2[0]
    )
    confirmation, h2_prediction = evaluate(
        h2,
        h2_safe,
        normalize_temperature(h2_probability, float(configuration["temperature"])),
        h2_present,
        grouping_config,
        float(configuration["weight"]),
    )
    confirmation["configuration"] = configuration
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_sensor_attention_IMU_complementarity_blend_v1",
        "protocol": "Select only sensor-expert method, calibration temperature and a low outer weight on H1 under a no-user-regression preference; transfer all values unchanged to H2. The existing 0.85572 validation prediction is the comparison baseline.",
        "safe_H1": classification_metrics(h1[1], h1_safe),
        "safe_H2": classification_metrics(h2[1], h2_safe),
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "all_H1_candidates": [candidates[index] for index in order],
        "deployment_decision": "pending_H2_gate",
    }
    report["deployment_decision"] = (
        "eligible_for_further_stress_tests"
        if selected["rescue_harm_vs_safe"]["net"] > 0
        and selected["minimum_user_gain_vs_safe"] >= 0
        and confirmation["rescue_harm_vs_safe"]["net"] > 0
        and confirmation["minimum_user_gain_vs_safe"] >= 0
        else "reject_no_Test_submission"
    )
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
