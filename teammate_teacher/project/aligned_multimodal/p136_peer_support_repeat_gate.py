"""Frozen repeat geometry with peer-majority proposals and support-gap gating."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import align_metadata, classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits
from p88_aligned_repeat_holdout import align_probabilities
from p89_global_repeat_decoder import cluster_sessions, date_session_lists
from p134_frozen_repeat_consensus import CONFIG, METADATA, evidence, probability_lookup


HERE = Path(__file__).resolve().parent
BASE = HERE / "runs/p128_base_hierarchical_meta_selector_v1/predictions.npz"
OUTPUT = HERE / "runs/p136_peer_support_repeat_gate_v1"
SCORE_NAMES = (
    "peer_vote_alt_minus_self_vote_base",
    "aligned_vote_alt_minus_base",
    "max_peer_vote_alt_minus_self_vote_base",
    "peer_meanprob_alt_minus_self_meanprob_base",
    "peer_and_self_vote_pair_gap",
    "peer_base_majority_margin",
    "self_vote_alt_minus_base",
    "self_meanprob_alt_minus_base",
)


def peer_candidate(sample_ids, base, lookup, metadata_path=METADATA):
    grouping_probability, mean_probability = evidence(
        sample_ids, base, lookup, "mean"
    )
    metadata = align_metadata(metadata_path, sample_ids)
    peers: dict[int, list[int]] = {}
    group_count = aligned_pairs = 0
    for sessions in date_session_lists(
        np.arange(len(sample_ids), dtype=np.int64), metadata, 30.0
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
            peers.update(aligned)
            group_count += 1

    proposal = base.copy()
    scores = np.full((len(base), len(SCORE_NAMES)), -9.0, dtype=np.float64)
    peer_count = np.zeros(len(base), dtype=np.int64)
    for row, aligned_rows in peers.items():
        other = [value for value in aligned_rows if value != row]
        if not other:
            continue
        count = np.bincount(base[other], minlength=40).astype(np.float64)
        alternative = int(count.argmax())
        proposal[row] = alternative
        peer_count[row] = len(other)
        all_rows = [row, *other]
        scores[row] = (
            grouping_probability[other, alternative].mean()
            - grouping_probability[row, base[row]],
            grouping_probability[all_rows, alternative].mean()
            - grouping_probability[all_rows, base[row]].mean(),
            grouping_probability[other, alternative].max()
            - grouping_probability[row, base[row]],
            mean_probability[other, alternative].mean()
            - mean_probability[row, base[row]],
            (
                grouping_probability[other, alternative]
                - grouping_probability[other, base[row]]
            ).mean()
            + grouping_probability[row, alternative]
            - grouping_probability[row, base[row]],
            count[alternative] / len(other) - count[base[row]] / len(other),
            grouping_probability[row, alternative]
            - grouping_probability[row, base[row]],
            mean_probability[row, alternative] - mean_probability[row, base[row]],
        )
    return proposal, scores, peer_count, {
        "groups": group_count,
        "aligned_pairs": aligned_pairs,
        "rows_with_peer": int(np.sum(peer_count > 0)),
    }


def select_rule(base, labels, proposal, scores, peer_count):
    disagreement = (proposal != base) & (peer_count > 0)
    candidates = []
    for score_index, score_name in enumerate(SCORE_NAMES):
        values = np.unique(
            np.concatenate(
                (
                    np.linspace(-1.0, 1.0, 201),
                    np.quantile(
                        scores[disagreement, score_index],
                        [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
                    ),
                )
            )
        )
        for threshold in values:
            route = disagreement & (scores[:, score_index] >= threshold)
            prediction = base.copy()
            prediction[route] = proposal[route]
            rescue = int(np.sum((base != labels) & (prediction == labels)))
            harm = int(np.sum((base == labels) & (prediction != labels)))
            candidates.append(
                {
                    "score_index": score_index,
                    "score_name": score_name,
                    "threshold": float(threshold),
                    "changed": int(route.sum()),
                    "rescue": rescue,
                    "harm": harm,
                    "net": rescue - harm,
                }
            )
    positive = [row for row in candidates if row["net"] > 0]
    if not positive:
        return {
            "score_index": 0,
            "score_name": SCORE_NAMES[0],
            "threshold": 10.0,
            "changed": 0,
            "rescue": 0,
            "harm": 0,
            "net": 0,
        }
    return max(
        positive,
        key=lambda row: (row["net"], row["rescue"], -row["harm"], -row["changed"]),
    )


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
    payload = {}
    cohorts = {}
    totals = {"correct": 0, "rescue": 0, "harm": 0, "changed": 0}
    for held_name in data:
        source_ids = base_run[f"{held_name}_source_sample_ids"].astype(str)
        source_labels = base_run[f"{held_name}_source_labels"].astype(np.int64)
        source_base = base_run[f"{held_name}_source_prediction"].astype(np.int64)
        source_proposal, source_scores, source_peers, _ = peer_candidate(
            source_ids, source_base, lookup
        )
        selected = select_rule(
            source_base,
            source_labels,
            source_proposal,
            source_scores,
            source_peers,
        )

        sample_ids = base_run[f"{held_name}_sample_ids"].astype(str)
        labels = base_run[f"{held_name}_labels"].astype(np.int64)
        base = base_run[f"{held_name}_prediction"].astype(np.int64)
        proposal, scores, peer_count, grouping = peer_candidate(
            sample_ids, base, lookup
        )
        route = (
            (proposal != base)
            & (peer_count > 0)
            & (
                scores[:, int(selected["score_index"])]
                >= float(selected["threshold"])
            )
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
                "peer_correct": correct,
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
        payload[f"{held_name}_peer_proposal"] = proposal
        payload[f"{held_name}_peer_scores"] = scores
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P136_peer_support_repeat_gate_v1",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "repeat_geometry": "frozen P89 H1->H2 confirmed geometry",
            "proposal": "majority of aligned peer champion predictions excluding self",
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
