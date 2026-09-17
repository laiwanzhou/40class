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
OUTPUT = PROJECT_DIR / "runs/p89_selective_global_joint_v1"
EVIDENCE_WEIGHT = 0.25
TRANSITION_SCALE = 1.0


def proposals(protocol_value, grouping_config: GlobalRepeatConfig) -> dict[str, np.ndarray]:
    probability = protocol_value[2]
    base = protocol_value[3]
    metadata = protocol_value[4]
    indices = protocol_value[5]
    transition = protocol_value[7]
    decoder = protocol_value[8]
    logp = np.log(np.maximum(probability, 1e-12))
    candidate = base.copy()
    column_size = np.ones(len(base), dtype=np.int64)
    base_support = np.ones(len(base), dtype=np.float64)
    probability_support = np.ones(len(base), dtype=np.float64)
    local_log_delta = np.zeros(len(base), dtype=np.float64)
    pooled_log_delta = np.zeros(len(base), dtype=np.float64)
    assigned = np.zeros(len(base), dtype=bool)
    groups = grouped_rows = aligned_pairs = 0
    for sessions in date_session_lists(indices, metadata, decoder.gap_seconds):
        for group in cluster_sessions(
            sessions, probability, base, metadata, grouping_config
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
            aggregate = np.empty((len(columns), probability.shape[1]), dtype=np.float64)
            for position, rows in enumerate(columns):
                reference_logp = logp[rows[0]]
                if len(rows) == 1:
                    aggregate[position] = reference_logp
                else:
                    other_logp = logp[np.asarray(rows[1:])].mean(axis=0)
                    aggregate[position] = (
                        reference_logp + EVIDENCE_WEIGHT * other_logp
                    ) / (1.0 + EVIDENCE_WEIGHT)
            shared_path = decode_unique_beam(
                aggregate,
                transition,
                transition_weight=decoder.transition_weight * TRANSITION_SCALE,
                beam_width=decoder.beam_width,
            )
            for position, rows_value in enumerate(columns):
                rows = np.asarray(rows_value, dtype=np.int64)
                shared = int(shared_path[position])
                deltas = logp[rows, shared] - logp[rows, base[rows]]
                candidate[rows] = shared
                column_size[rows] = len(rows)
                base_support[rows] = float(np.mean(base[rows] == shared))
                probability_support[rows] = float(np.mean(deltas >= 0.0))
                local_log_delta[rows] = deltas
                pooled_log_delta[rows] = float(np.mean(deltas))
                assigned[rows] = True
            groups += 1
            grouped_rows += sum(map(len, group))
    return {
        "candidate": candidate,
        "column_size": column_size,
        "base_support": base_support,
        "probability_support": probability_support,
        "local_log_delta": local_log_delta,
        "pooled_log_delta": pooled_log_delta,
        "assigned": assigned,
        "groups": np.asarray([groups]),
        "grouped_rows": np.asarray([grouped_rows]),
        "aligned_pairs": np.asarray([aligned_pairs]),
    }


def predict(protocol_value, values: dict[str, np.ndarray], config: dict[str, float]) -> np.ndarray:
    base = protocol_value[3]
    candidate = values["candidate"]
    accepted = (
        values["assigned"]
        & (candidate != base)
        & (values["column_size"] >= int(config["minimum_column_size"]))
        & (values["base_support"] >= float(config["minimum_base_support"]))
        & (
            values["probability_support"]
            >= float(config["minimum_probability_support"])
        )
        & (values["local_log_delta"] >= float(config["minimum_local_log_delta"]))
        & (values["pooled_log_delta"] >= float(config["minimum_pooled_log_delta"]))
    )
    output = base.copy()
    output[accepted] = candidate[accepted]
    return output


def evaluate(protocol_value, values: dict[str, np.ndarray], config: dict[str, float]) -> dict:
    prediction = predict(protocol_value, values, config)
    gains = []
    per_user = {}
    users = protocol_value[4].users.astype(str)
    for user in sorted(set(users.tolist())):
        rows = users == user
        base_correct = int(np.sum(protocol_value[3][rows] == protocol_value[1][rows]))
        candidate_correct = int(np.sum(prediction[rows] == protocol_value[1][rows]))
        gains.append(candidate_correct - base_correct)
        per_user[user] = {
            "gain": candidate_correct - base_correct,
            "changes": int(np.sum(prediction[rows] != protocol_value[3][rows])),
        }
    return {
        "configuration": config,
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "minimum_user_gain": int(min(gains)),
        "positive_users": int(np.sum(np.asarray(gains) > 0)),
        "per_user": per_user,
    }


def main() -> None:
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
    h1_values = proposals(h1, grouping_config)
    h2_values = proposals(h2, grouping_config)
    candidates = []
    for column_size in (2, 3):
        for base_support in (0.0, 0.34, 0.50, 0.67, 1.0):
            for probability_support in (0.0, 0.34, 0.50, 0.67, 1.0):
                for local_delta in (-8.0, -4.0, -2.0, -1.0, -0.5, 0.0):
                    for pooled_delta in (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5):
                        candidates.append(
                            evaluate(
                                h1,
                                h1_values,
                                {
                                    "minimum_column_size": column_size,
                                    "minimum_base_support": base_support,
                                    "minimum_probability_support": probability_support,
                                    "minimum_local_log_delta": local_delta,
                                    "minimum_pooled_log_delta": pooled_delta,
                                },
                            )
                        )
    candidates.sort(
        key=lambda item: (
            item["minimum_user_gain"] >= 0,
            item["metrics"]["correct"],
            item["positive_users"],
            item["metrics"]["balanced_accuracy"],
            item["rescue_harm_vs_p87"]["net"],
            -item["rescue_harm_vs_p87"]["harm"],
            -item["rescue_harm_vs_p87"]["changed"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    confirmation = evaluate(h2, h2_values, selected["configuration"])
    h1_prediction = predict(h1, h1_values, selected["configuration"])
    h2_prediction = predict(h2, h2_values, selected["configuration"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=h1_prediction,
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_selective_shared_path_assignment_v1",
        "protocol": (
            "Build shared-path proposals with frozen grouping/joint weights. "
            "Select only group-support and local-likelihood thresholds on H1 "
            "under a no-user-regression constraint, then transfer once to H2."
        ),
        "grouping_configuration": grouping_source["H1_selected"]["configuration"],
        "proposal_audit": {
            "H1": {
                "groups": int(h1_values["groups"][0]),
                "grouped_rows": int(h1_values["grouped_rows"][0]),
                "aligned_pairs": int(h1_values["aligned_pairs"][0]),
            },
            "H2": {
                "groups": int(h2_values["groups"][0]),
                "grouped_rows": int(h2_values["grouped_rows"][0]),
                "aligned_pairs": int(h2_values["aligned_pairs"][0]),
            },
        },
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
