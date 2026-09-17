from __future__ import annotations

import json
from dataclasses import asdict
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
OUTPUT = PROJECT_DIR / "runs/p89_multiexpert_repeat_grouping_v1"
EVIDENCE_WEIGHT = 0.25
TRANSITION_SCALE = 1.0


def normalize(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    return output / output.sum(axis=1, keepdims=True)


def grouping_sources(
    base_probability: np.ndarray, expert_probability: np.ndarray
) -> dict[str, np.ndarray]:
    top = np.argmax(expert_probability, axis=2)
    votes = np.zeros((len(top), 40), dtype=np.float64)
    for class_id in range(40):
        votes[:, class_id] = np.mean(top == class_id, axis=1)
    mean19 = expert_probability.mean(axis=1)
    sources = {
        "p87": base_probability,
        "mean19": mean19,
        "median19": np.median(expert_probability, axis=1),
        "mean17_visual": expert_probability[:, :17].mean(axis=1),
        "mean7_p85": expert_probability[:, :7].mean(axis=1),
        "mean10_p86": expert_probability[:, 7:17].mean(axis=1),
        "hybrid_p87_mean19": 0.5 * base_probability + 0.5 * mean19,
        "vote_distribution": votes + 1e-4,
    }
    return {name: normalize(values) for name, values in sources.items()}


def decode(
    protocol_value,
    grouping_probability: np.ndarray,
    grouping_prediction: np.ndarray,
    config: GlobalRepeatConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    probability = protocol_value[2]
    base = protocol_value[3]
    metadata = protocol_value[4]
    indices = protocol_value[5]
    transition = protocol_value[7]
    decoder = protocol_value[8]
    logp = np.log(np.maximum(probability, 1e-12))
    prediction = base.copy()
    groups = grouped_sessions = grouped_rows = aligned_pairs = assigned_rows = 0
    for sessions in date_session_lists(indices, metadata, decoder.gap_seconds):
        for group in cluster_sessions(
            sessions, grouping_probability, grouping_prediction, metadata, config
        ):
            reference = max(group, key=len)
            columns: list[list[int]] = [[int(index)] for index in reference]
            for session in group:
                if session is reference:
                    continue
                pairs, _ = align_probabilities(
                    grouping_probability[reference],
                    grouping_probability[session],
                    config.alignment_gap_penalty,
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
            for position, rows in enumerate(columns):
                prediction[np.asarray(rows, dtype=np.int64)] = int(shared_path[position])
                assigned_rows += len(rows)
            groups += 1
            grouped_sessions += len(group)
            grouped_rows += sum(map(len, group))
    return prediction, {
        "groups": groups,
        "grouped_sessions": grouped_sessions,
        "grouped_rows": grouped_rows,
        "aligned_pairs": aligned_pairs,
        "assigned_rows_with_duplicates": assigned_rows,
    }


def evaluate(
    protocol_value,
    source_name: str,
    grouping_probability: np.ndarray,
    path_source: str,
    config: GlobalRepeatConfig,
) -> tuple[dict, np.ndarray]:
    grouping_prediction = (
        protocol_value[3]
        if path_source == "p87"
        else np.argmax(grouping_probability, axis=1)
    )
    prediction, grouping = decode(
        protocol_value, grouping_probability, grouping_prediction, config
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
            "source": source_name,
            "path_source": path_source,
            "configuration": asdict(config),
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
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    expert_probability, expert_names = full40.train_probabilities(all_ids)
    lookup = {value: index for index, value in enumerate(all_ids)}
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_rows = np.asarray([lookup[value] for value in h1[0].astype(str)], dtype=np.int64)
    h2_rows = np.asarray([lookup[value] for value in h2[0].astype(str)], dtype=np.int64)
    h1_sources = grouping_sources(h1[2], expert_probability[h1_rows])
    h2_sources = grouping_sources(h2[2], expert_probability[h2_rows])
    configurations = [
        GlobalRepeatConfig(
            maximum_session_rank_distance=rank,
            maximum_start_gap_seconds=gap,
            minimum_probability_similarity=similarity,
            minimum_path_overlap=overlap,
            minimum_length_ratio=length_ratio,
            consensus_weight=0.5,
            alignment_gap_penalty=0.2,
            maximum_group_size=3,
        )
        for rank in (2, 3)
        for gap in (180.0, 300.0)
        for similarity in (0.75, 0.84, 0.90)
        for overlap in (0.0, 0.20)
        for length_ratio in (0.65, 0.80)
    ]
    candidates = []
    predictions = []
    total = len(h1_sources) * 2 * len(configurations)
    current = 0
    for source_name, probability in h1_sources.items():
        for path_source in ("p87", "source"):
            for config in configurations:
                item, prediction = evaluate(
                    h1, source_name, probability, path_source, config
                )
                candidates.append(item)
                predictions.append(prediction)
                current += 1
                if current % 100 == 0:
                    print(f"evaluated {current}/{total} grouping candidates", flush=True)
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
    source_name = selected["source"]
    selected_config = GlobalRepeatConfig(**selected["configuration"])
    confirmation, h2_prediction = evaluate(
        h2,
        source_name,
        h2_sources[source_name],
        selected["path_source"],
        selected_config,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_multiexpert_repeated_take_grouping_v1",
        "protocol": (
            "Use full40 expert aggregates only to identify and align repeated "
            "takes. Decode the shared label path exclusively from immutable P87 "
            "probabilities. Select source/group thresholds on H1 and transfer "
            "unchanged to H2."
        ),
        "expert_names": expert_names,
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
