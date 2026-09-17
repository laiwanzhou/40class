from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics, decode_unique_beam
from p88_aligned_repeat_holdout import align_probabilities
from p88_train_depth_residual import rescue_harm
from p89_global_repeat_decoder import (
    GlobalRepeatConfig,
    cluster_sessions,
    date_session_lists,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_global_joint_repeat_v1"


def joint_decode(
    probability: np.ndarray,
    base_prediction: np.ndarray,
    protocol_value,
    grouping_config: GlobalRepeatConfig,
    evidence_weight: float,
    transition_scale: float,
    initial_prediction: np.ndarray | None = None,
    grouping_prediction: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    metadata = protocol_value[4]
    indices = protocol_value[5]
    transition = protocol_value[7]
    decoder = protocol_value[8]
    prediction = np.asarray(
        base_prediction if initial_prediction is None else initial_prediction,
        dtype=np.int64,
    ).copy()
    group_source = np.asarray(
        base_prediction if grouping_prediction is None else grouping_prediction,
        dtype=np.int64,
    )
    log_probability = np.log(np.maximum(probability, 1e-12))
    date_lists = date_session_lists(indices, metadata, decoder.gap_seconds)
    groups = grouped_sessions = grouped_rows = aligned_pairs = assigned_rows = 0
    for sessions in date_lists:
        for group in cluster_sessions(
            sessions, probability, group_source, metadata, grouping_config
        ):
            reference = max(group, key=len)
            columns: list[list[int]] = [[int(index)] for index in reference]
            for session in group:
                if session is reference:
                    continue
                pairs, _ = align_probabilities(
                    probability[reference],
                    probability[session],
                    grouping_config.alignment_gap_penalty,
                )
                for reference_position, other_position in pairs:
                    columns[reference_position].append(int(session[other_position]))
                aligned_pairs += len(pairs)
            aggregate = np.empty(
                (len(columns), probability.shape[1]), dtype=np.float64
            )
            for position, rows in enumerate(columns):
                reference_logp = log_probability[rows[0]]
                if len(rows) == 1:
                    aggregate[position] = reference_logp
                else:
                    other_logp = log_probability[np.asarray(rows[1:])].mean(axis=0)
                    aggregate[position] = (
                        reference_logp + evidence_weight * other_logp
                    ) / (1.0 + evidence_weight)
            shared_path = decode_unique_beam(
                aggregate,
                transition,
                transition_weight=decoder.transition_weight * transition_scale,
                beam_width=decoder.beam_width,
            )
            for position, rows in enumerate(columns):
                prediction[np.asarray(rows, dtype=np.int64)] = int(shared_path[position])
                assigned_rows += len(rows)
            groups += 1
            grouped_sessions += len(group)
            grouped_rows += sum(map(len, group))
    return prediction, {
        "date_lists": len(date_lists),
        "groups": groups,
        "grouped_sessions": grouped_sessions,
        "grouped_rows": grouped_rows,
        "aligned_pairs": aligned_pairs,
        "assigned_rows_with_duplicates": assigned_rows,
    }


def evaluate(
    protocol_value,
    grouping_config: GlobalRepeatConfig,
    evidence_weight: float,
    transition_scale: float,
) -> dict[str, object]:
    prediction, grouping = joint_decode(
        protocol_value[2],
        protocol_value[3],
        protocol_value,
        grouping_config,
        evidence_weight,
        transition_scale,
    )
    return {
        "configuration": {
            "evidence_weight": evidence_weight,
            "transition_scale": transition_scale,
        },
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "grouping": grouping,
    }


def main() -> None:
    global_source = json.loads(
        (PROJECT_DIR / "runs/p89_global_repeat_h1_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    grouping_config = GlobalRepeatConfig(**global_source["selected_config"])
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    candidates = [
        evaluate(h1, grouping_config, evidence_weight, transition_scale)
        for evidence_weight in (0.10, 0.25, 0.50, 1.0, 2.0, 4.0)
        for transition_scale in (0.0, 0.5, 1.0, 1.5, 2.0)
    ]
    candidates.sort(
        key=lambda item: (
            item["metrics"]["correct"],
            item["metrics"]["balanced_accuracy"],
            item["rescue_harm_vs_p87"]["net"],
            -item["rescue_harm_vs_p87"]["harm"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    configuration = selected["configuration"]
    confirmation = evaluate(
        h2,
        grouping_config,
        float(configuration["evidence_weight"]),
        float(configuration["transition_scale"]),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=joint_decode(
            h1[2], h1[3], h1, grouping_config, **configuration
        )[0],
        h2_prediction=joint_decode(
            h2[2], h2[3], h2, grouping_config, **configuration
        )[0],
    )
    report = {
        "stage": "P89_global_joint_repeated_take_decoder_v1",
        "protocol": (
            "Use the frozen H1 global-repeat grouping, jointly decode one shared "
            "latent path for aligned repeated takes, select two decoding scalars "
            "on H1, and transfer them unchanged to H2."
        ),
        "grouping_configuration": global_source["selected_config"],
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "all_H1_candidates": candidates,
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
