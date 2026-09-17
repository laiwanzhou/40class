from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import align_metadata, build_sessions
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_session_template_decoder import apply_template_posterior, fit_templates
from p89_deploy_supervised_router import load_decoder


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_session_template_deploy_v1"


def validation_prediction(
    protocol_value,
    holdout_users: list[str],
    all_labels: np.ndarray,
    all_metadata,
    configuration: dict[str, float],
) -> np.ndarray:
    fit = np.flatnonzero(~np.isin(all_metadata.users.astype(str), holdout_users))
    templates = fit_templates(
        all_labels,
        build_sessions(
            fit, all_metadata, protocol_value[8].gap_seconds, "known_user"
        ),
        maximum_length=10,
    )
    adjusted, _ = apply_template_posterior(
        np.log(np.maximum(protocol_value[2], 1e-12)),
        protocol_value[6],
        templates,
        protocol_value[7],
        protocol_value[8],
        configuration,
    )
    return decode_aligned_repeat(
        adjusted,
        protocol_value[5],
        protocol_value[4],
        protocol_value[7],
        protocol_value[8],
        protocol_value[9],
    )[0]


def main() -> None:
    source = json.loads(
        (PROJECT_DIR / "runs/p88_session_template_h1_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    configuration = source["best"]["configuration"]
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_prediction = validation_prediction(
        h1, full40.H1_USERS, all_labels, all_metadata, configuration
    )
    h2_prediction = validation_prediction(
        h2, full40.H2_USERS, all_labels, all_metadata, configuration
    )

    test = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    test_ids = test["sample_ids"].astype(str)
    base_probability = np.asarray(test["base_probability"], dtype=np.float64)
    transition, decoder = load_decoder()
    test_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        test_ids,
    )
    indices = np.arange(len(test_ids), dtype=np.int64)
    sessions = build_sessions(indices, test_metadata, decoder.gap_seconds, "anonymous_date")
    fit_sessions = build_sessions(
        np.arange(len(all_labels)), all_metadata, decoder.gap_seconds, "known_user"
    )
    templates = fit_templates(all_labels, fit_sessions, maximum_length=10)
    adjusted, template_audit = apply_template_posterior(
        np.log(np.maximum(base_probability, 1e-12)),
        sessions,
        templates,
        transition,
        decoder,
        configuration,
    )
    repeat_config = h2[9]
    test_prediction, grouping = decode_aligned_repeat(
        adjusted,
        indices,
        test_metadata,
        transition,
        decoder,
        repeat_config,
    )
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    rows = io.read_rows(p87_path)
    p87 = io.read_prediction(p87_path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_session_template.csv"
    io.write_submission(submission, rows, test_prediction)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=h1_prediction,
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_session_template_Test_deployment_v1",
        "configuration": configuration,
        "H1_changes_vs_p87": int(np.sum(h1_prediction != h1[3])),
        "H2_changes_vs_p87": int(np.sum(h2_prediction != h2[3])),
        "Test": {
            "template_audit": template_audit,
            "grouping": grouping,
            "changes_vs_p87": int(np.sum(test_prediction != p87)),
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
