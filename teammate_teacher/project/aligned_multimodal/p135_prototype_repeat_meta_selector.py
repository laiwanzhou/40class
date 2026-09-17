"""Outer-safe selector between P133 posterior prototypes and P134 repeats.

For a held outer cohort, each of the two source cohorts calibrates P133/P134 and
predicts the other source cohort. Those cross-cohort predictions train the final
selector, avoiding reuse of an in-sample source decision as meta supervision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import (
    fit_score,
    load_candidate_splits,
    loso_scores,
    one_hot,
    select_threshold,
    shared_features,
)
from p133_transductive_posterior_prototype import (
    build_lookup,
    choose_threshold,
    posterior_geometry,
)
from p134_frozen_repeat_consensus import (
    probability_lookup,
    repeat_proposal,
    select_rule,
)


HERE = Path(__file__).resolve().parent
P128 = HERE / "runs/p128_base_hierarchical_meta_selector_v1/predictions.npz"
P133 = HERE / "runs/p133_transductive_posterior_prototype_v1/predictions.npz"
P134 = HERE / "runs/p134_frozen_repeat_consensus_v1/predictions.npz"
OUTPUT = HERE / "runs/p135_prototype_repeat_meta_selector_v1"


def align(values: np.ndarray, ids: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(ids.astype(str))}
    return np.asarray(values)[
        np.asarray([lookup[value] for value in target.astype(str)], dtype=np.int64)
    ]


def apply_prototype(
    calibration_ids,
    calibration_base,
    calibration_labels,
    target_ids,
    target_base,
    posterior,
    votes,
):
    calibration_proposal, calibration_advantage = posterior_geometry(
        calibration_ids, calibration_base, posterior, votes
    )
    selected = choose_threshold(
        calibration_base,
        calibration_proposal,
        calibration_advantage,
        calibration_labels,
    )
    proposal, advantage = posterior_geometry(
        target_ids, target_base, posterior, votes
    )
    route = (proposal != target_base) & (advantage >= float(selected["threshold"]))
    prediction = target_base.copy()
    prediction[route] = proposal[route]
    return prediction, proposal, advantage, selected


def apply_repeat(
    calibration_ids,
    calibration_base,
    calibration_labels,
    target_ids,
    target_base,
    repeat_lookup,
):
    selected, _ = select_rule(
        calibration_ids, calibration_base, calibration_labels, repeat_lookup
    )
    proposal, advantage, peers, _ = repeat_proposal(
        target_ids, target_base, repeat_lookup, str(selected["mode"])
    )
    route = (
        (proposal != target_base)
        & (peers > 0)
        & (advantage >= float(selected["threshold"]))
    )
    prediction = target_base.copy()
    prediction[route] = proposal[route]
    return prediction, proposal, advantage, selected


def feature_lookup(data):
    result = {}
    for value in data.values():
        matrix = shared_features(value)
        for row, sample_id in enumerate(value.split.sample_ids.astype(str)):
            result[sample_id] = matrix[row]
    return result


def meta_features(
    sample_ids,
    p128,
    p133,
    p134,
    prototype_proposal,
    repeat_proposal_value,
    prototype_advantage,
    repeat_advantage,
    shared_lookup,
    hard_vote,
):
    shared = np.stack([shared_lookup[value] for value in sample_ids.astype(str)])
    teacher_support = np.asarray(
        [
            (
                np.mean(hard_vote[value] == p128[row]),
                np.mean(hard_vote[value] == p133[row]),
                np.mean(hard_vote[value] == p134[row]),
            )
            for row, value in enumerate(sample_ids.astype(str))
        ],
        dtype=np.float32,
    )
    scalar = np.column_stack(
        (
            p133 != p134,
            p133 != p128,
            p134 != p128,
            prototype_proposal != p128,
            repeat_proposal_value != p128,
            np.nan_to_num(prototype_advantage, nan=-1.0, neginf=-1.0),
            np.nan_to_num(repeat_advantage, nan=-1.0, neginf=-1.0),
            teacher_support,
        )
    ).astype(np.float32)
    return np.concatenate(
        (
            shared,
            one_hot(p128),
            one_hot(p133),
            one_hot(p134),
            one_hot(prototype_proposal),
            one_hot(repeat_proposal_value),
            scalar,
        ),
        axis=1,
    ).astype(np.float32)


def main() -> None:
    data = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=True,
        hand_object_bank=True,
        vjepa_dense_bank=True,
        nonvisual_bank=True,
        hierarchical_bank=True,
        epic_bank=True,
    )
    posterior, hard_vote = build_lookup(data)
    repeat_lookup = probability_lookup(data)
    shared_lookup = feature_lookup(data)
    base = np.load(P128)
    p133_run = np.load(P133)
    p134_run = np.load(P134)
    payload = {}
    cohorts = {}
    total_correct = 0

    split_names = list(data)
    for held_name in split_names:
        source_names = [name for name in split_names if name != held_name]
        source_ids = base[f"{held_name}_source_sample_ids"].astype(str)
        source_labels = base[f"{held_name}_source_labels"].astype(np.int64)
        source_users = base[f"{held_name}_source_users"].astype(str)
        source_base = base[f"{held_name}_source_prediction"].astype(np.int64)
        source_p133 = np.empty_like(source_base)
        source_p134 = np.empty_like(source_base)
        source_proto_proposal = np.empty_like(source_base)
        source_repeat_proposal = np.empty_like(source_base)
        source_proto_advantage = np.empty(len(source_base), dtype=np.float64)
        source_repeat_advantage = np.empty(len(source_base), dtype=np.float64)
        cross_rules = []

        for target_name in source_names:
            calibration_name = next(
                name for name in source_names if name != target_name
            )
            target_ids = data[target_name].split.sample_ids.astype(str)
            calibration_ids = data[calibration_name].split.sample_ids.astype(str)
            target_rows = np.asarray(
                [
                    {value: row for row, value in enumerate(source_ids)}[value]
                    for value in target_ids
                ],
                dtype=np.int64,
            )
            calibration_rows = np.asarray(
                [
                    {value: row for row, value in enumerate(source_ids)}[value]
                    for value in calibration_ids
                ],
                dtype=np.int64,
            )
            proto = apply_prototype(
                calibration_ids,
                source_base[calibration_rows],
                source_labels[calibration_rows],
                target_ids,
                source_base[target_rows],
                posterior,
                hard_vote,
            )
            repeat = apply_repeat(
                calibration_ids,
                source_base[calibration_rows],
                source_labels[calibration_rows],
                target_ids,
                source_base[target_rows],
                repeat_lookup,
            )
            source_p133[target_rows] = proto[0]
            source_proto_proposal[target_rows] = proto[1]
            source_proto_advantage[target_rows] = proto[2]
            source_p134[target_rows] = repeat[0]
            source_repeat_proposal[target_rows] = repeat[1]
            source_repeat_advantage[target_rows] = repeat[2]
            cross_rules.append(
                {
                    "calibration": calibration_name,
                    "target": target_name,
                    "prototype": proto[3],
                    "repeat": repeat[3],
                }
            )

        source_x = meta_features(
            source_ids,
            source_base,
            source_p133,
            source_p134,
            source_proto_proposal,
            source_repeat_proposal,
            source_proto_advantage,
            source_repeat_advantage,
            shared_lookup,
            hard_vote,
        )
        source_gain = (source_p133 == source_labels).astype(np.int8) - (
            source_p134 == source_labels
        ).astype(np.int8)
        source_disagreement = source_p133 != source_p134
        nested_score = loso_scores(source_x, source_gain, source_users)
        threshold = select_threshold(
            nested_score, source_gain, source_disagreement, source_users
        )

        sample_ids = p133_run[f"{held_name}_sample_ids"].astype(str)
        labels = p133_run[f"{held_name}_labels"].astype(np.int64)
        held_base = p133_run[f"{held_name}_base_prediction"].astype(np.int64)
        held_p133 = p133_run[f"{held_name}_prediction"].astype(np.int64)
        held_p134 = p134_run[f"{held_name}_prediction"].astype(np.int64)
        held_x = meta_features(
            sample_ids,
            held_base,
            held_p133,
            held_p134,
            p133_run[f"{held_name}_prototype_prediction"].astype(np.int64),
            p134_run[f"{held_name}_repeat_proposal"].astype(np.int64),
            p133_run[f"{held_name}_prototype_advantage"].astype(np.float64),
            p134_run[f"{held_name}_repeat_advantage"].astype(np.float64),
            shared_lookup,
            hard_vote,
        )
        held_score = fit_score(source_x, source_gain, held_x)
        route = (held_p133 != held_p134) & (
            held_score >= float(threshold["threshold"])
        )
        prediction = held_p134.copy()
        prediction[route] = held_p133[route]
        correct = int(np.sum(prediction == labels))
        total_correct += correct
        cohorts[held_name] = {
            "cross_source_rules": cross_rules,
            "source_threshold": threshold,
            "held": {
                "p133_correct": int(np.sum(held_p133 == labels)),
                "p134_correct": int(np.sum(held_p134 == labels)),
                "union_oracle_correct_analysis_only": int(
                    np.sum((held_p133 == labels) | (held_p134 == labels))
                ),
                "selected_correct": correct,
                "selected_p133_rows": int(route.sum()),
                "metrics": classification_metrics(labels, prediction),
            },
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_p133_prediction"] = held_p133
        payload[f"{held_name}_p134_prediction"] = held_p134
        payload[f"{held_name}_meta_score"] = held_score
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P135_prototype_repeat_meta_selector_v1",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "source_predictions": "two source cohorts predict each other",
            "held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "user_id_used_for_source_LOSO_only": True,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": cohorts,
        "aggregate": {
            "rows": rows,
            "correct": total_correct,
            "accuracy": total_correct / rows,
            "target_0.91_correct": int(np.ceil(0.91 * rows)),
            "gap_to_0.91_correct": int(np.ceil(0.91 * rows)) - total_correct,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
