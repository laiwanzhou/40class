from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import align_metadata, build_sessions, decode_sessions
from p89_deploy_supervised_router import load_decoder
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_probability_blend import evaluate
from p89_imu_rescue_gate import aligned_imu, softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_imu_multidomain_blend_v1"


def combine(
    first_probability: np.ndarray,
    first_present: np.ndarray,
    second_probability: np.ndarray,
    second_present: np.ndarray,
    first_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    present = first_present | second_present
    result = np.full_like(first_probability, 1.0 / first_probability.shape[1])
    both = first_present & second_present
    result[both] = (
        first_weight * first_probability[both]
        + (1.0 - first_weight) * second_probability[both]
    )
    first_only = first_present & ~second_present
    second_only = second_present & ~first_present
    result[first_only] = first_probability[first_only]
    result[second_only] = second_probability[second_only]
    result /= result.sum(axis=1, keepdims=True)
    return result, present


def main() -> None:
    p3 = np.load(
        PROJECT_DIR
        / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    )
    spectral = np.load(PROJECT_DIR / "runs/p89_imu_spectral_forest_v1/oof_logits.npz")
    p3_ids = p3["sample_ids"].astype(str)
    p3_logits = np.asarray(p3["imu_logits"], dtype=np.float64)
    spectral_ids = spectral["sample_ids"].astype(str)
    spectral_logits = np.asarray(spectral["imu_logits"], dtype=np.float64)
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
    h1_p3, h1_p3_present = aligned_imu(p3_ids, p3_logits, h1[0], 3.0)
    h2_p3, h2_p3_present = aligned_imu(p3_ids, p3_logits, h2[0], 3.0)
    candidates = []
    predictions = []
    for spectral_temperature in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        h1_spectral, h1_spectral_present = aligned_imu(
            spectral_ids, spectral_logits, h1[0], spectral_temperature
        )
        for p3_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
            h1_imu, h1_present = combine(
                h1_p3,
                h1_p3_present,
                h1_spectral,
                h1_spectral_present,
                p3_weight,
            )
            for weight in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10):
                item, prediction = evaluate(
                    h1, h1_imu, h1_present, weight, "joint", grouping
                )
                item["configuration"].update(
                    {
                        "p3_temperature": 3.0,
                        "spectral_temperature": spectral_temperature,
                        "p3_weight_within_imu": p3_weight,
                    }
                )
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
    config = selected["configuration"]
    h2_spectral, h2_spectral_present = aligned_imu(
        spectral_ids,
        spectral_logits,
        h2[0],
        float(config["spectral_temperature"]),
    )
    h2_imu, h2_present = combine(
        h2_p3,
        h2_p3_present,
        h2_spectral,
        h2_spectral_present,
        float(config["p3_weight_within_imu"]),
    )
    confirmation, h2_prediction = evaluate(
        h2, h2_imu, h2_present, float(config["weight"]), "joint", grouping
    )
    confirmation["configuration"].update(
        {
            "p3_temperature": 3.0,
            "spectral_temperature": config["spectral_temperature"],
            "p3_weight_within_imu": config["p3_weight_within_imu"],
        }
    )

    test = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    test_ids = test["sample_ids"].astype(str)
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    p3_test = np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz")
    spectral_test = np.load(
        PROJECT_DIR / "runs/p89_imu_spectral_forest_v1/test_logits.npz"
    )
    p3_lookup = {value: index for index, value in enumerate(p3_test["sample_ids"].astype(str))}
    spectral_lookup = {
        value: index
        for index, value in enumerate(spectral_test["sample_ids"].astype(str))
    }
    p3_test_probability = softmax(
        np.asarray(p3_test["imu_logits"], dtype=np.float64)[
            np.asarray([p3_lookup[value] for value in test_ids], dtype=np.int64)
        ],
        3.0,
    )
    spectral_test_probability = softmax(
        np.asarray(spectral_test["imu_logits"], dtype=np.float64)[
            np.asarray([spectral_lookup[value] for value in test_ids], dtype=np.int64)
        ],
        float(config["spectral_temperature"]),
    )
    imu_probability = (
        float(config["p3_weight_within_imu"]) * p3_test_probability
        + (1.0 - float(config["p3_weight_within_imu"]))
        * spectral_test_probability
    )
    weight = float(config["weight"])
    adjusted = (1.0 - weight) * base_probability + weight * imu_probability
    adjusted /= adjusted.sum(axis=1, keepdims=True)
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
    source_rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    reproduced = decode_sessions(
        np.log(np.maximum(base_probability, 1e-12)),
        sessions,
        transition,
        decoder,
    )
    if not np.array_equal(reproduced, p87):
        raise RuntimeError("failed to reproduce immutable P87")
    protocol_value = (
        test_ids,
        None,
        adjusted,
        p87,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        None,
    )
    test_prediction, test_grouping = joint_decode(
        adjusted,
        p87,
        protocol_value,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    known_best = io.read_prediction(
        PROJECT_DIR
        / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_imu_multidomain_blend.csv"
    io.write_submission(submission, source_rows, test_prediction)
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
        test_sample_ids=test_ids,
        test_prediction=test_prediction,
    )
    report = {
        "stage": "P89_multi_domain_IMU_probability_blend_v1",
        "protocol": (
            "H1 selects the mixture of the deployed P3 time-statistics forest and "
            "the new subject-disjoint time-frequency forest, plus one low outer "
            "weight. H2 confirms all frozen values once."
        ),
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "Test": {
            "grouping": test_grouping,
            "changes_vs_p87": int(np.sum(test_prediction != p87)),
            "changes_vs_known_0.85572": int(np.sum(test_prediction != known_best)),
            "submission": str(submission.resolve()),
            "sha256": io.digest(submission),
        },
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
