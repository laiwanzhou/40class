"""Conservative unlabeled-batch posterior prototypes over the frozen candidate bank.

For each outer held cohort, the rule is calibrated only on the other two cohorts.
It never consumes raw identifiers as features: sample ids are alignment keys only.
The held batch contributes only frozen candidate probabilities and champion pseudo
labels to leave-one-out class prototypes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits


HERE = Path(__file__).resolve().parent
CHAMPION = HERE / "runs/p128_base_hierarchical_meta_selector_v1/predictions.npz"
OUTPUT = HERE / "runs/p133_transductive_posterior_prototype_v1"


def build_lookup(data):
    probability: dict[str, np.ndarray] = {}
    hard_vote: dict[str, np.ndarray] = {}
    for value in data.values():
        teachers = [value.split.safe_probability, *value.candidates.values()]
        for row, sample_id in enumerate(value.split.sample_ids.astype(str)):
            probability[sample_id] = np.stack(
                [teacher[row] for teacher in teachers]
            ).astype(np.float32)
            hard_vote[sample_id] = np.asarray(
                [teacher[row].argmax() for teacher in teachers], dtype=np.int64
            )
    return probability, hard_vote


def posterior_geometry(
    sample_ids: np.ndarray,
    pseudo_label: np.ndarray,
    probability: dict[str, np.ndarray],
    hard_vote: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return leave-one-out prototype proposal and its cosine advantage."""
    features = np.stack(
        [
            np.sqrt(np.clip(probability[sample_id], 0.0, 1.0)).reshape(-1)
            for sample_id in sample_ids.astype(str)
        ]
    ).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    weights = np.asarray(
        [
            (np.sum(hard_vote[sample_id] == label) + 1)
            / (len(hard_vote[sample_id]) + 1)
            for sample_id, label in zip(sample_ids.astype(str), pseudo_label)
        ],
        dtype=np.float32,
    )

    sums = np.zeros((40, features.shape[1]), dtype=np.float32)
    counts = np.zeros(40, dtype=np.float32)
    np.add.at(sums, pseudo_label, features * weights[:, None])
    np.add.at(counts, pseudo_label, weights)
    centroid = sums / np.maximum(counts[:, None], 1e-6)
    centroid /= np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-8)
    similarity = features @ centroid.T

    # Prevent the row's own pseudo label from winning through self-similarity.
    for row, label in enumerate(pseudo_label):
        leave_one_out = sums[label] - weights[row] * features[row]
        leave_one_out /= max(float(counts[label] - weights[row]), 1e-6)
        leave_one_out /= max(float(np.linalg.norm(leave_one_out)), 1e-8)
        similarity[row, label] = features[row] @ leave_one_out

    proposal = similarity.argmax(axis=1).astype(np.int64)
    rows = np.arange(len(proposal))
    advantage = similarity[rows, proposal] - similarity[rows, pseudo_label]
    return proposal, advantage.astype(np.float64)


def choose_threshold(
    base: np.ndarray,
    proposal: np.ndarray,
    advantage: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    disagreement = proposal != base
    values = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 0.30, 61),
                np.quantile(advantage[disagreement], [0.2, 0.4, 0.6, 0.8, 0.9]),
            )
        )
    )
    candidates = []
    for threshold in values:
        selected = disagreement & (advantage >= threshold)
        prediction = base.copy()
        prediction[selected] = proposal[selected]
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        candidates.append(
            {
                "threshold": float(threshold),
                "selected": int(selected.sum()),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
            }
        )
    # Fixed conservative tie-break: maximize net, then make fewer interventions.
    return max(
        candidates,
        key=lambda row: (row["net"], -row["selected"], row["threshold"]),
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
    probability, hard_vote = build_lookup(data)
    champion = np.load(CHAMPION)
    payload: dict[str, np.ndarray] = {}
    cohorts = {}
    total_correct = total_rescue = total_harm = total_changed = 0

    for held_name, value in data.items():
        source_ids = champion[f"{held_name}_source_sample_ids"].astype(str)
        source_labels = champion[f"{held_name}_source_labels"].astype(np.int64)
        source_base = champion[f"{held_name}_source_prediction"].astype(np.int64)
        source_proposal, source_advantage = posterior_geometry(
            source_ids, source_base, probability, hard_vote
        )
        selected = choose_threshold(
            source_base, source_proposal, source_advantage, source_labels
        )

        sample_ids = champion[f"{held_name}_sample_ids"].astype(str)
        labels = champion[f"{held_name}_labels"].astype(np.int64)
        base = champion[f"{held_name}_prediction"].astype(np.int64)
        proposal, advantage = posterior_geometry(
            sample_ids, base, probability, hard_vote
        )
        route = (proposal != base) & (advantage >= float(selected["threshold"]))
        prediction = base.copy()
        prediction[route] = proposal[route]
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        changed = int(route.sum())
        correct = int(np.sum(prediction == labels))
        total_correct += correct
        total_rescue += rescue
        total_harm += harm
        total_changed += changed
        cohorts[held_name] = {
            "source_selected": selected,
            "held": {
                "base_correct": int(np.sum(base == labels)),
                "prototype_correct": correct,
                "changed": changed,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "metrics": classification_metrics(labels, prediction),
            },
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_base_prediction"] = base
        payload[f"{held_name}_prototype_prediction"] = proposal
        payload[f"{held_name}_prototype_advantage"] = advantage
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P133_transductive_posterior_prototype_v1",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "candidate_count_including_safe": len(hard_vote[next(iter(hard_vote))]),
            "posterior_transform": "sqrt_probability_Hellinger_geometry",
            "prototype": "consensus_weighted_leave_one_out_held_batch",
            "held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": cohorts,
        "aggregate": {
            "rows": rows,
            "correct": total_correct,
            "accuracy": total_correct / rows,
            "rescue": total_rescue,
            "harm": total_harm,
            "net_vs_champion": total_rescue - total_harm,
            "changed": total_changed,
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
