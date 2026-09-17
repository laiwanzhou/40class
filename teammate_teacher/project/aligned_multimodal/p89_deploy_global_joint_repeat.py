from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
from audit_p87_sequence_decoder import align_metadata, build_sessions, decode_sessions
from p89_deploy_supervised_router import load_decoder
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_global_joint_repeat_test_v1"


def main() -> None:
    validation = json.loads(
        (PROJECT_DIR / "runs/p89_global_joint_repeat_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    grouping_config = GlobalRepeatConfig(**validation["grouping_configuration"])
    configuration = validation["H1_selected"]["configuration"]
    test = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    sample_ids = test["sample_ids"].astype(str)
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    transition, decoder = load_decoder()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        sample_ids,
    )
    indices = np.arange(len(sample_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    reproduced = decode_sessions(
        np.log(np.maximum(base_probability, 1e-12)),
        sessions,
        transition,
        decoder,
    )
    if not np.array_equal(reproduced, p87):
        raise RuntimeError("failed to reproduce immutable P87 Test prediction")
    protocol_value = (
        sample_ids,
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
    prediction, grouping = joint_decode(
        base_probability,
        p87,
        protocol_value,
        grouping_config,
        float(configuration["evidence_weight"]),
        float(configuration["transition_scale"]),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_global_joint_repeat.csv"
    io.write_submission(submission, source_rows, prediction)
    np.savez_compressed(
        OUTPUT / "test_predictions.npz",
        sample_ids=sample_ids,
        p87_prediction=p87,
        prediction=prediction,
    )
    report = {
        "stage": "P89_global_joint_repeated_take_Test_v1",
        "protocol": (
            "Deploy the H1-selected/H2-confirmed shared latent repeated-take "
            "path decoder on immutable P87 Test probabilities."
        ),
        "configuration": configuration,
        "grouping_configuration": validation["grouping_configuration"],
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
