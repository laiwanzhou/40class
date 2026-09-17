"""Frozen repeated-take geometry with outer-safe intervention calibration.

The grouping geometry was frozen from the previously confirmed P89 H1->H2
repeat experiment. For every held cohort, only a mode and intervention threshold
are selected on the other two cohorts. Anonymous date/time and posterior evidence
identify repeats; user identity is never an input feature.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import align_metadata, classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits
from p88_aligned_repeat_holdout import align_probabilities
from p89_global_repeat_decoder import (
    GlobalRepeatConfig,
    cluster_sessions,
    date_session_lists,
)


HERE = Path(__file__).resolve().parent
METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
BASE = HERE / "runs/p128_base_hierarchical_meta_selector_v1/predictions.npz"
OUTPUT = HERE / "runs/p134_frozen_repeat_consensus_v1"
CONFIG = GlobalRepeatConfig(
    maximum_session_rank_distance=2,
    maximum_start_gap_seconds=300.0,
    minimum_probability_similarity=0.75,
    minimum_path_overlap=0.20,
    minimum_length_ratio=0.80,
    consensus_weight=0.50,
    alignment_gap_penalty=0.20,
    maximum_group_size=3,
)
MODES = ("mean", "vote", "mix", "votemix")


def probability_lookup(data):
    result: dict[str, np.ndarray] = {}
    for value in data.values():
        teachers = [value.split.safe_probability, *value.candidates.values()]
        for row, sample_id in enumerate(value.split.sample_ids.astype(str)):
            result[sample_id] = np.stack(
                [teacher[row] for teacher in teachers]
            ).astype(np.float32)
    return result


def evidence(
    sample_ids: np.ndarray,
    base: np.ndarray,
    lookup: dict[str, np.ndarray],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    teachers = np.stack([lookup[value] for value in sample_ids.astype(str)])
    hard = teachers.argmax(axis=2)
    vote = np.stack([(hard == class_id).mean(axis=1) for class_id in range(40)], axis=1)
    vote += 1e-5
    vote /= vote.sum(axis=1, keepdims=True)
    mean = teachers.mean(axis=1)
    mean /= mean.sum(axis=1, keepdims=True)
    base_one_hot = np.zeros((len(base), 40), dtype=np.float32)
    base_one_hot[np.arange(len(base)), base] = 1.0
    choices = {
        "mean": mean,
        "vote": vote,
        "mix": 0.5 * mean + 0.5 * base_one_hot,
        "votemix": 0.5 * vote + 0.5 * base_one_hot,
    }
    return vote.astype(np.float64), choices[mode].astype(np.float64)


def repeat_proposal(
    sample_ids: np.ndarray,
    base: np.ndarray,
    lookup: dict[str, np.ndarray],
    mode: str,
    metadata_path: Path = METADATA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    grouping_probability, emission = evidence(sample_ids, base, lookup, mode)
    metadata = align_metadata(metadata_path, sample_ids)
    proposal = base.copy()
    advantage = np.full(len(base), -np.inf, dtype=np.float64)
    peer_count = np.zeros(len(base), dtype=np.int64)
    group_count = aligned_pairs = 0
    for sessions in date_session_lists(
        np.arange(len(base), dtype=np.int64), metadata, 30.0
    ):
        for group in cluster_sessions(
            sessions, grouping_probability, base, metadata, CONFIG
        ):
            reference = max(group, key=len)
            aligned = {int(row): [int(row)] for row in np.concatenate(group)}
            for session in group:
                if session is reference:
                    continue
                pairs, _ = align_probabilities(
                    grouping_probability[reference],
                    grouping_probability[session],
                    CONFIG.alignment_gap_penalty,
                )
                for reference_position, other_position in pairs:
                    left = int(reference[reference_position])
                    right = int(session[other_position])
                    aligned[left].append(right)
                    aligned[right].append(left)
                aligned_pairs += len(pairs)
            for row, peers in aligned.items():
                consensus = emission[np.asarray(peers, dtype=np.int64)].mean(axis=0)
                alternative = int(consensus.argmax())
                proposal[row] = alternative
                advantage[row] = consensus[alternative] - consensus[base[row]]
                peer_count[row] = len(peers) - 1
            group_count += 1
    return proposal, advantage, peer_count, {
        "groups": group_count,
        "aligned_pairs": aligned_pairs,
        "rows_with_peer": int(np.sum(peer_count > 0)),
    }


def select_rule(
    sample_ids: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    lookup: dict[str, np.ndarray],
) -> tuple[dict[str, float | int | str], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    cache = {}
    candidates = []
    for mode in MODES:
        proposal, advantage, peers, _ = repeat_proposal(
            sample_ids, base, lookup, mode
        )
        cache[mode] = (proposal, advantage, peers)
        disagreement = (proposal != base) & (peers > 0)
        values = np.unique(
            np.concatenate(
                (
                    np.linspace(0.0, 0.80, 161),
                    np.quantile(
                        advantage[disagreement], [0.2, 0.4, 0.6, 0.8, 0.9]
                    ),
                )
            )
        )
        for threshold in values:
            route = disagreement & (advantage >= threshold)
            prediction = base.copy()
            prediction[route] = proposal[route]
            rescue = int(np.sum((base != labels) & (prediction == labels)))
            harm = int(np.sum((base == labels) & (prediction != labels)))
            candidates.append(
                {
                    "mode": mode,
                    "threshold": float(threshold),
                    "changed": int(route.sum()),
                    "rescue": rescue,
                    "harm": harm,
                    "net": rescue - harm,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (row["net"], row["rescue"], -row["harm"], -row["changed"]),
    )
    return selected, cache


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
    lookup = probability_lookup(data)
    base_run = np.load(BASE)
    payload: dict[str, np.ndarray] = {}
    cohorts = {}
    totals = {"correct": 0, "rescue": 0, "harm": 0, "changed": 0}
    for held_name in data:
        source_ids = base_run[f"{held_name}_source_sample_ids"].astype(str)
        source_base = base_run[f"{held_name}_source_prediction"].astype(np.int64)
        source_labels = base_run[f"{held_name}_source_labels"].astype(np.int64)
        selected, _ = select_rule(source_ids, source_base, source_labels, lookup)

        sample_ids = base_run[f"{held_name}_sample_ids"].astype(str)
        labels = base_run[f"{held_name}_labels"].astype(np.int64)
        base = base_run[f"{held_name}_prediction"].astype(np.int64)
        proposal, advantage, peers, grouping = repeat_proposal(
            sample_ids, base, lookup, str(selected["mode"])
        )
        route = (
            (proposal != base)
            & (peers > 0)
            & (advantage >= float(selected["threshold"]))
        )
        prediction = base.copy()
        prediction[route] = proposal[route]
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        correct = int(np.sum(prediction == labels))
        changed = int(route.sum())
        totals["correct"] += correct
        totals["rescue"] += rescue
        totals["harm"] += harm
        totals["changed"] += changed
        cohorts[held_name] = {
            "source_selected": selected,
            "held": {
                "base_correct": int(np.sum(base == labels)),
                "repeat_correct": correct,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "changed": changed,
                "grouping": grouping,
                "metrics": classification_metrics(labels, prediction),
            },
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_base_prediction"] = base
        payload[f"{held_name}_repeat_proposal"] = proposal
        payload[f"{held_name}_repeat_advantage"] = advantage
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P134_frozen_repeat_consensus_v1",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "group_geometry": asdict(CONFIG),
            "group_geometry_source": "frozen P89 H1->H2 confirmed repeat rule",
            "anonymous_group_fields": ["recording_date", "start_seconds"],
            "held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": cohorts,
        "aggregate": {
            "rows": rows,
            **totals,
            "accuracy": totals["correct"] / rows,
            "net_vs_p128": totals["rescue"] - totals["harm"],
            "target_0.91_correct": int(np.ceil(0.91 * rows)),
            "gap_to_0.91_correct": int(np.ceil(0.91 * rows)) - totals["correct"],
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
