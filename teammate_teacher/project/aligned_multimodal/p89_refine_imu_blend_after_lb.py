from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
)
from p88_train_depth_residual import rescue_harm
from p89_deploy_supervised_router import load_decoder
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_probability_blend import evaluate
from p89_imu_rescue_gate import aligned_imu, softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_imu_blend_lb_refinement_v1"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_submission(path: Path, rows: list[dict[str, str]], prediction: np.ndarray):
    fieldnames = list(rows[0])
    label_key = "prediction"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, value in zip(rows, prediction, strict=True):
            output = dict(row)
            output[label_key] = str(int(value))
            writer.writerow(output)


def consensus_prediction(
    base: np.ndarray, predictions: np.ndarray, fraction: float
) -> np.ndarray:
    result = base.copy()
    required = int(np.ceil(fraction * len(predictions)))
    for row in range(len(base)):
        values, counts = np.unique(predictions[:, row], return_counts=True)
        order = np.argsort(counts)[::-1]
        best = int(values[order[0]])
        if best != int(base[row]) and int(counts[order[0]]) >= required:
            result[row] = best
    return result


def metrics(protocol_value, prediction: np.ndarray) -> dict:
    users = protocol_value[4].users.astype(str)
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        per_user[user] = int(
            np.sum(prediction[rows] == protocol_value[1][rows])
            - np.sum(protocol_value[3][rows] == protocol_value[1][rows])
        )
    return {
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "per_user_gain": per_user,
        "minimum_user_gain": min(per_user.values()),
    }


def test_prediction(
    base_probability: np.ndarray,
    imu_probability: np.ndarray,
    p87: np.ndarray,
    protocol_value,
    grouping: GlobalRepeatConfig,
    weight: float,
) -> np.ndarray:
    adjusted = (1.0 - weight) * base_probability + weight * imu_probability
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    adjusted_protocol = list(protocol_value)
    adjusted_protocol[2] = adjusted
    return joint_decode(
        adjusted,
        p87,
        tuple(adjusted_protocol),
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )[0]


def main() -> None:
    grouping_source = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(
        PROJECT_DIR
        / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    ) as oof:
        oof_ids = oof["sample_ids"].astype(str)
        oof_logits = np.asarray(oof["imu_logits"], dtype=np.float64)

    configurations = [
        (float(temperature), float(weight))
        for temperature in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
        for weight in (0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10)
    ]
    records = []
    h1_predictions = []
    h2_predictions = []
    for temperature, weight in configurations:
        h1_imu, h1_present = aligned_imu(oof_ids, oof_logits, h1[0], temperature)
        h2_imu, h2_present = aligned_imu(oof_ids, oof_logits, h2[0], temperature)
        h1_result, h1_prediction = evaluate(
            h1, h1_imu, h1_present, weight, "joint", grouping
        )
        h2_result, h2_prediction = evaluate(
            h2, h2_imu, h2_present, weight, "joint", grouping
        )
        records.append(
            {
                "temperature": temperature,
                "weight": weight,
                "H1": h1_result,
                "H2": h2_result,
                "worst_split_net": min(
                    h1_result["rescue_harm_vs_p87"]["net"],
                    h2_result["rescue_harm_vs_p87"]["net"],
                ),
                "sum_net": h1_result["rescue_harm_vs_p87"]["net"]
                + h2_result["rescue_harm_vs_p87"]["net"],
            }
        )
        h1_predictions.append(h1_prediction)
        h2_predictions.append(h2_prediction)
    h1_predictions = np.stack(h1_predictions)
    h2_predictions = np.stack(h2_predictions)
    stable_indices = [
        index
        for index, record in enumerate(records)
        if record["H1"]["rescue_harm_vs_p87"]["net"] >= 13
        and record["H2"]["rescue_harm_vs_p87"]["net"] >= 17
        and record["H1"]["minimum_user_gain"] >= 0
        and record["H2"]["minimum_user_gain"] >= 0
    ]
    if not stable_indices:
        raise RuntimeError("no stable IMU configurations")

    consensus = []
    for fraction in (0.50, 0.67, 0.80, 0.90):
        h1_prediction = consensus_prediction(
            h1[3], h1_predictions[stable_indices], fraction
        )
        h2_prediction = consensus_prediction(
            h2[3], h2_predictions[stable_indices], fraction
        )
        consensus.append(
            {
                "fraction": fraction,
                "H1": metrics(h1, h1_prediction),
                "H2": metrics(h2, h2_prediction),
                "h1_prediction": h1_prediction,
                "h2_prediction": h2_prediction,
            }
        )
    consensus.sort(
        key=lambda item: (
            min(
                item["H1"]["rescue_harm_vs_p87"]["net"],
                item["H2"]["rescue_harm_vs_p87"]["net"],
            ),
            item["H1"]["rescue_harm_vs_p87"]["net"]
            + item["H2"]["rescue_harm_vs_p87"]["net"],
            -item["H1"]["rescue_harm_vs_p87"]["harm"]
            - item["H2"]["rescue_harm_vs_p87"]["harm"],
        ),
        reverse=True,
    )
    selected_consensus = consensus[0]
    direct_order = sorted(
        range(len(records)),
        key=lambda index: (
            records[index]["worst_split_net"],
            records[index]["sum_net"],
            -records[index]["H1"]["rescue_harm_vs_p87"]["harm"]
            - records[index]["H2"]["rescue_harm_vs_p87"]["harm"],
        ),
        reverse=True,
    )
    selected_direct_index = direct_order[0]
    selected_direct = records[selected_direct_index]

    with np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    ) as test:
        test_ids = test["sample_ids"].astype(str)
        base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    with np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz") as imu:
        imu_ids = imu["sample_ids"].astype(str)
        lookup = {sample_id: index for index, sample_id in enumerate(imu_ids)}
        imu_logits = np.asarray(imu["imu_logits"], dtype=np.float64)[
            np.asarray([lookup[sample_id] for sample_id in test_ids], dtype=np.int64)
        ]
    transition, decoder = load_decoder()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        test_ids,
    )
    indices = np.arange(len(test_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = read_rows(p87_path)
    label_key = "prediction"
    p87 = np.asarray([int(row[label_key]) for row in source_rows], dtype=np.int64)
    reproduced = decode_sessions(
        np.log(np.maximum(base_probability, 1e-12)),
        sessions,
        transition,
        decoder,
    )
    if not np.array_equal(reproduced, p87):
        raise RuntimeError("failed to reproduce immutable P87")
    test_protocol = (
        test_ids,
        None,
        base_probability,
        p87,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        None,
    )
    test_predictions = []
    for index in stable_indices:
        record = records[index]
        imu_probability = softmax(imu_logits, record["temperature"])
        test_predictions.append(
            test_prediction(
                base_probability,
                imu_probability,
                p87,
                test_protocol,
                grouping,
                record["weight"],
            )
        )
    test_predictions = np.stack(test_predictions)
    consensus_test = consensus_prediction(
        p87, test_predictions, selected_consensus["fraction"]
    )
    direct_test = test_predictions[stable_indices.index(selected_direct_index)] if selected_direct_index in stable_indices else test_prediction(
        base_probability,
        softmax(imu_logits, selected_direct["temperature"]),
        p87,
        test_protocol,
        grouping,
        selected_direct["weight"],
    )
    known_best = np.asarray(
        [
            int(row[label_key])
            for row in read_rows(
                PROJECT_DIR
                / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
            )
        ],
        dtype=np.int64,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    consensus_path = OUTPUT / "submission_p89_imu_stable_consensus.csv"
    direct_path = OUTPUT / "submission_p89_imu_balanced_direct.csv"
    write_submission(consensus_path, source_rows, consensus_test)
    write_submission(direct_path, source_rows, direct_test)
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_consensus=selected_consensus["h1_prediction"],
        h2_consensus=selected_consensus["h2_prediction"],
        test_consensus=consensus_test,
        test_direct=direct_test,
    )
    clean_consensus = [
        {key: value for key, value in item.items() if not key.endswith("prediction")}
        for item in consensus
    ]
    report = {
        "stage": "P89_post_LB_conservative_IMU_blend_refinement_v1",
        "known_LB_anchor": {
            "configuration": {"temperature": 3.0, "weight": 0.05},
            "score": 0.85572,
        },
        "stable_configuration_count": len(stable_indices),
        "selected_direct": selected_direct,
        "selected_consensus": {
            key: value
            for key, value in selected_consensus.items()
            if not key.endswith("prediction")
        },
        "all_consensus": clean_consensus,
        "Test": {
            "consensus_changes_vs_p87": int(np.sum(consensus_test != p87)),
            "consensus_changes_vs_known_best": int(
                np.sum(consensus_test != known_best)
            ),
            "direct_changes_vs_p87": int(np.sum(direct_test != p87)),
            "direct_changes_vs_known_best": int(np.sum(direct_test != known_best)),
            "consensus_submission": str(consensus_path.resolve()),
            "direct_submission": str(direct_path.resolve()),
        },
        "all_configurations": [records[index] for index in direct_order],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "all_configurations"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
