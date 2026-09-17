from __future__ import annotations

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
from p88_session_template_decoder import apply_template_posterior, fit_templates
from p88_train_depth_residual import rescue_harm
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_template_global_joint_v1"


def evaluate(
    protocol_value,
    holdout_users: list[str],
    all_ids: np.ndarray,
    all_labels: np.ndarray,
    all_metadata,
    template_configuration: dict[str, float],
    grouping_configuration: GlobalRepeatConfig,
) -> tuple[dict[str, object], np.ndarray]:
    decoder = protocol_value[8]
    fit_indices = np.flatnonzero(~np.isin(all_metadata.users, holdout_users))
    fit_sessions = build_sessions(
        fit_indices,
        all_metadata,
        decoder.gap_seconds,
        grouping="known_user",
    )
    templates = fit_templates(all_labels, fit_sessions, maximum_length=10)
    adjusted_logp, template_audit = apply_template_posterior(
        np.log(np.maximum(protocol_value[2], 1e-12)),
        protocol_value[6],
        templates,
        protocol_value[7],
        decoder,
        template_configuration,
    )
    adjusted_probability = np.exp(adjusted_logp)
    template_decoded = decode_sessions(
        adjusted_logp,
        protocol_value[6],
        protocol_value[7],
        decoder,
    )
    adjusted_protocol = list(protocol_value)
    adjusted_protocol[2] = adjusted_probability
    prediction, grouping = joint_decode(
        adjusted_probability,
        protocol_value[3],
        tuple(adjusted_protocol),
        grouping_configuration,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    return (
        {
            "template_decoded": classification_metrics(
                protocol_value[1], template_decoded
            ),
            "template_decoded_rescue_harm": rescue_harm(
                protocol_value[1], protocol_value[3], template_decoded
            ),
            "joint_metrics": classification_metrics(protocol_value[1], prediction),
            "joint_rescue_harm_vs_p87": rescue_harm(
                protocol_value[1], protocol_value[3], prediction
            ),
            "template_audit": template_audit,
            "joint_grouping": grouping,
        },
        prediction,
    )


def main() -> None:
    template_source = json.loads(
        (PROJECT_DIR / "runs/p88_session_template_h1_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    template_configuration = template_source["best"]["configuration"]
    grouping_source = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping_configuration = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
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
    h1_result, h1_prediction = evaluate(
        h1,
        full40.H1_USERS,
        all_ids,
        all_labels,
        all_metadata,
        template_configuration,
        grouping_configuration,
    )
    h2_result, h2_prediction = evaluate(
        h2,
        full40.H2_USERS,
        all_ids,
        all_labels,
        all_metadata,
        template_configuration,
        grouping_configuration,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=h1_prediction,
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_cross_subject_script_template_plus_global_joint_v1",
        "protocol": (
            "Fit exact short action-script templates without each holdout's "
            "subjects, apply the H1-frozen template posterior, then run the "
            "H1-selected/H2-confirmed shared repeated-take path decoder."
        ),
        "template_configuration": template_configuration,
        "grouping_configuration": grouping_source["H1_selected"]["configuration"],
        "H1": h1_result,
        "H2": h2_result,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
