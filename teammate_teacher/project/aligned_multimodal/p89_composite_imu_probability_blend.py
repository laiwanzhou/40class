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
from p89_imu_rescue_gate import softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_composite_imu_probability_blend_v1"


def align_probability(source, target_ids: np.ndarray, temperature: float):
    source_ids = source["sample_ids"].astype(str)
    lookup = {value: index for index, value in enumerate(source_ids)}
    present = np.asarray(
        [value in lookup and bool(source["valid"][lookup[value]]) for value in target_ids.astype(str)]
    )
    output = np.full((len(target_ids), 40), 1.0 / 40.0, dtype=np.float64)
    rows = np.flatnonzero(present)
    source_rows = np.asarray([lookup[target_ids[row]] for row in rows], dtype=np.int64)
    output[rows] = softmax(source["imu_logits"][source_rows], temperature)
    return output, present


def main() -> None:
    oof = np.load(PROJECT_DIR / "runs/p86_imu_composite_teacher_v1/composite_imu_teacher.npz")
    grouping_source = json.loads(
        (PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json").read_text(encoding="utf-8")
    )
    grouping_config = GlobalRepeatConfig(**grouping_source["H1_selected"]["configuration"])
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    candidates = []
    predictions = []
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        h1_imu, h1_present = align_probability(oof, h1[0], temperature)
        for weight in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
            for method in ("decoded", "joint"):
                item, prediction = evaluate(
                    h1, h1_imu, h1_present, weight, method, grouping_config
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
    h2_imu, h2_present = align_probability(oof, h2[0], float(configuration["temperature"]))
    confirmation, h2_prediction = evaluate(
        h2,
        h2_imu,
        h2_present,
        float(configuration["weight"]),
        str(configuration["method"]),
        grouping_config,
    )
    confirmation["configuration"]["temperature"] = configuration["temperature"]

    test_source = np.load(PROJECT_DIR / "runs/p89_composite_imu_test_v1/composite_imu_test.npz")
    test_ids = test_source["sample_ids"].astype(str)
    test_logits = np.asarray(test_source["imu_logits"], dtype=np.float64).copy()
    event_valid = np.asarray(test_source["valid"], dtype=bool)
    test_logits[~event_valid] = np.asarray(test_source["p3_logits"], dtype=np.float64)[~event_valid]
    imu_probability = softmax(test_logits, float(configuration["temperature"]))
    test = np.load(PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz")
    base_ids = test["sample_ids"].astype(str)
    lookup = {value: index for index, value in enumerate(test_ids)}
    imu_probability = imu_probability[np.asarray([lookup[value] for value in base_ids], dtype=np.int64)]
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    weight = float(configuration["weight"])
    adjusted = (1.0 - weight) * base_probability + weight * imu_probability
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    transition, decoder = load_decoder()
    metadata = align_metadata(PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv", base_ids)
    indices = np.arange(len(base_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    p87_path = PROJECT_DIR / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    reproduced = decode_sessions(np.log(np.maximum(base_probability, 1e-12)), sessions, transition, decoder)
    if not np.array_equal(reproduced, p87):
        raise RuntimeError("failed to reproduce immutable P87 Test")
    protocol_value = (base_ids, None, adjusted, p87, metadata, indices, sessions, transition, decoder, None)
    prediction, grouping = joint_decode(
        adjusted, p87, protocol_value, grouping_config, evidence_weight=0.25, transition_scale=1.0
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_composite_imu_blend.csv"
    io.write_submission(submission, rows, prediction)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0], h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index], h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_composite_IMU_probability_blend_v1",
        "protocol": (
            "Use P86 composite subject-disjoint OOF on H1/H2 and the contract-matched "
            "full18-refit composite on Test; select only temperature/weight/decoder on H1."
        ),
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "Test": {
            "event_valid_rows": int(np.sum(event_valid)),
            "p3_fallback_rows": int(np.sum(~event_valid)),
            "grouping": grouping,
            "changes_vs_p87": int(np.sum(prediction != p87)),
            "path": str(submission.resolve()),
            "sha256": io.digest(submission),
        },
        "all_H1_candidates": [candidates[index] for index in order],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "all_H1_candidates"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
