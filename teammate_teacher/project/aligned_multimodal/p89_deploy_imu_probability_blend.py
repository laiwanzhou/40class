from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
from audit_p87_sequence_decoder import align_metadata, build_sessions, decode_sessions
from p89_deploy_supervised_router import load_decoder
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_imu_probability_blend_test_v1"


def main() -> None:
    source = json.loads(
        (PROJECT_DIR / "runs/p89_imu_probability_blend_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    configuration = source["H1_selected"]["configuration"]
    grouping_source = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping_config = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
    test = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    sample_ids = test["sample_ids"].astype(str)
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    imu = np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz")
    imu_ids = imu["sample_ids"].astype(str)
    lookup = {value: index for index, value in enumerate(imu_ids)}
    imu_rows = np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)
    imu_probability = softmax(
        np.asarray(imu["imu_logits"], dtype=np.float64)[imu_rows],
        float(configuration["temperature"]),
    )
    weight = float(configuration["weight"])
    adjusted = (1.0 - weight) * base_probability + weight * imu_probability
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    transition, decoder = load_decoder()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        sample_ids,
    )
    indices = np.arange(len(sample_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    reproduced = decode_sessions(
        np.log(np.maximum(base_probability, 1e-12)), sessions, transition, decoder
    )
    if not np.array_equal(reproduced, p87):
        raise RuntimeError("failed to reproduce immutable P87 Test prediction")
    protocol_value = (
        sample_ids,
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
    prediction, grouping = joint_decode(
        adjusted,
        p87,
        protocol_value,
        grouping_config,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_imu_probability_blend.csv"
    io.write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_low_weight_IMU_probability_blend_Test_v1",
        "protocol": (
            "Deploy the H1-selected/H2-confirmed low-weight P3 IMU posterior "
            "blend and shared repeated-take decoder on Test."
        ),
        "configuration": configuration,
        "grouping": grouping,
        "changes_vs_p87": int(np.sum(prediction != p87)),
        "submission": {
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
