from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics, decode_sessions
from p88_train_depth_residual import rescue_harm
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import aligned_imu


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_imu_probability_blend_v1"


def evaluate(
    protocol_value,
    imu_probability: np.ndarray,
    present: np.ndarray,
    weight: float,
    method: str,
    grouping_config: GlobalRepeatConfig,
) -> tuple[dict[str, object], np.ndarray]:
    adjusted = protocol_value[2].copy()
    adjusted[present] = (
        (1.0 - weight) * protocol_value[2][present]
        + weight * imu_probability[present]
    )
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    if method == "decoded":
        prediction = decode_sessions(
            np.log(np.maximum(adjusted, 1e-12)),
            protocol_value[6],
            protocol_value[7],
            protocol_value[8],
        )
        grouping = None
    else:
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
    users = protocol_value[4].users.astype(str)
    gains = []
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        base_correct = int(np.sum(protocol_value[3][rows] == protocol_value[1][rows]))
        candidate_correct = int(np.sum(prediction[rows] == protocol_value[1][rows]))
        gains.append(candidate_correct - base_correct)
        per_user[user] = candidate_correct - base_correct
    return (
        {
            "configuration": {"weight": weight, "method": method},
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_p87": rescue_harm(
                protocol_value[1], protocol_value[3], prediction
            ),
            "minimum_user_gain": int(min(gains)),
            "positive_users": int(np.sum(np.asarray(gains) > 0)),
            "per_user_gain": per_user,
            "grouping": grouping,
        },
        prediction,
    )


def main() -> None:
    oof = np.load(
        PROJECT_DIR
        / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    )
    oof_ids = oof["sample_ids"].astype(str)
    oof_logits = np.asarray(oof["imu_logits"], dtype=np.float64)
    grouping_source = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping_config = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    candidates = []
    predictions = []
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        h1_imu, h1_present = aligned_imu(
            oof_ids, oof_logits, h1[0], temperature
        )
        for weight in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
            for method in ("decoded", "joint"):
                item, prediction = evaluate(
                    h1,
                    h1_imu,
                    h1_present,
                    weight,
                    method,
                    grouping_config,
                )
                item["configuration"]["temperature"] = temperature
                candidates.append(item)
                predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["positive_users"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_p87"]["net"],
            -candidates[index]["rescue_harm_vs_p87"]["harm"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    configuration = selected["configuration"]
    h2_imu, h2_present = aligned_imu(
        oof_ids, oof_logits, h2[0], float(configuration["temperature"])
    )
    confirmation, h2_prediction = evaluate(
        h2,
        h2_imu,
        h2_present,
        float(configuration["weight"]),
        str(configuration["method"]),
        grouping_config,
    )
    confirmation["configuration"]["temperature"] = configuration["temperature"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_low_weight_IMU_probability_blend_v1",
        "protocol": (
            "Blend the contract-matched subject-disjoint P3 IMU posterior into "
            "P87 at low weight, select temperature/weight/decoder on H1 under "
            "no-user-regression, and transfer unchanged to H2."
        ),
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "all_H1_candidates": [candidates[index] for index in order],
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
