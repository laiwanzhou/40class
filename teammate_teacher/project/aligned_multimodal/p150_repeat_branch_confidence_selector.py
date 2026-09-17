"""Source-safe confidence selector between P142 and P149 safe branches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import (
    load_candidate_splits,
    loso_scores,
    select_threshold,
)
from p134_frozen_repeat_consensus import probability_lookup
from p143_p142_source_user_safe_selector import build_features


HERE = Path(__file__).resolve().parent
P140 = HERE / "runs/p140_source_stable_expanded_sequence_v1/predictions.npz"
P142 = HERE / "runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz"
P149 = HERE / "runs/p149_vjepa_repeat_consistency_three_seed_v2/oof_predictions.npz"
P143_HELD = HERE / "runs/p143_p142_source_user_safe_selector_v1/predictions.npz"
P149_HELD = HERE / "runs/p149_repeat_source_user_safe_selector_v3/predictions.npz"
P145 = HERE / "runs/p145_dual_token_agreement_selector_v1/predictions.npz"
OUTPUT = HERE / "runs/p150_repeat_branch_confidence_selector_v1"


def source_branch(
    sample_ids,
    labels,
    base,
    probability,
    users,
    lookup,
):
    alternative = probability.argmax(axis=1).astype(np.int64)
    features = build_features(
        sample_ids, base, alternative, probability, lookup
    )
    gain = (alternative == labels).astype(np.int8) - (
        base == labels
    ).astype(np.int8)
    score = loso_scores(features, gain, users)
    selected = select_threshold(
        score, gain, alternative != base, users
    )
    eligible = selected["net"] > 0 and selected["minimum_user_gain"] >= 0
    route = (
        (alternative != base)
        & (score >= float(selected["threshold"]))
        & eligible
    )
    prediction = base.copy()
    prediction[route] = alternative[route]
    return prediction, selected


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
        expanded_bank=True,
    )
    lookup = probability_lookup(data)
    p140 = np.load(P140)
    p142 = np.load(P142)
    p149 = np.load(P149)
    p143_held = np.load(P143_HELD)
    p149_held = np.load(P149_HELD)
    p145 = np.load(P145)
    position = {
        value: row
        for row, value in enumerate(p142["sample_ids"].astype(str))
    }
    user_lookup = {
        sample_id: user
        for value in data.values()
        for sample_id, user in zip(
            value.split.sample_ids.astype(str), value.split.users.astype(str)
        )
    }
    payload = {}
    cohorts = {}
    total_correct = 0
    for held_name in data:
        source_ids = p140[f"{held_name}_source_sample_ids"].astype(str)
        source_labels = p140[f"{held_name}_source_labels"].astype(np.int64)
        source_base = p140[f"{held_name}_source_prediction"].astype(np.int64)
        source_rows = np.asarray([position[value] for value in source_ids], dtype=np.int64)
        source_users = np.asarray([user_lookup[value] for value in source_ids])
        all_branch, all_rule = source_branch(
            source_ids,
            source_labels,
            source_base,
            p142["probability"][source_rows].astype(np.float64),
            source_users,
            lookup,
        )
        repeat_branch, repeat_rule = source_branch(
            source_ids,
            source_labels,
            source_base,
            p149["probability"][source_rows].astype(np.float64),
            source_users,
            lookup,
        )
        repeat_probability = p149["probability"][source_rows].astype(np.float64)
        disagreement = repeat_branch != all_branch
        confidence = repeat_probability[
            np.arange(len(source_ids)), repeat_branch
        ]
        candidates = []
        values = (
            np.unique(
                np.concatenate(
                    (
                        np.linspace(0.0, 1.0, 201),
                        np.quantile(
                            confidence[disagreement],
                            [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
                        ),
                    )
                )
            )
            if disagreement.any()
            else np.asarray((2.0,))
        )
        for threshold in values:
            route = disagreement & (confidence >= threshold)
            prediction = all_branch.copy()
            prediction[route] = repeat_branch[route]
            rescue = int(
                np.sum(
                    (all_branch != source_labels)
                    & (prediction == source_labels)
                )
            )
            harm = int(
                np.sum(
                    (all_branch == source_labels)
                    & (prediction != source_labels)
                )
            )
            per_user = {
                user: int(
                    np.sum(prediction[source_users == user] == source_labels[source_users == user])
                    - np.sum(all_branch[source_users == user] == source_labels[source_users == user])
                )
                for user in sorted(set(source_users.tolist()))
            }
            candidates.append(
                {
                    "threshold": float(threshold),
                    "changed": int(route.sum()),
                    "rescue": rescue,
                    "harm": harm,
                    "net": rescue - harm,
                    "minimum_user_gain": int(min(per_user.values())),
                    "per_user_gain": per_user,
                }
            )
        selected = max(
            candidates,
            key=lambda row: (
                row["minimum_user_gain"] >= 0,
                row["net"],
                row["rescue"],
                -row["harm"],
                -row["changed"],
            ),
        )
        eligible = selected["net"] > 0 and selected["minimum_user_gain"] >= 0

        sample_ids = p145[f"{held_name}_sample_ids"].astype(str)
        labels = p145[f"{held_name}_labels"].astype(np.int64)
        base = p145[f"{held_name}_prediction"].astype(np.int64)
        all_held = p143_held[f"{held_name}_prediction"].astype(np.int64)
        repeat_held = p149_held[f"{held_name}_prediction"].astype(np.int64)
        held_rows = np.asarray([position[value] for value in sample_ids], dtype=np.int64)
        repeat_probability = p149["probability"][held_rows].astype(np.float64)
        confidence = repeat_probability[
            np.arange(len(sample_ids)), repeat_held
        ]
        route = (
            (repeat_held != all_held)
            & (confidence >= float(selected["threshold"]))
            & eligible
        )
        prediction = base.copy()
        prediction[route] = repeat_held[route]
        correct = int(np.sum(prediction == labels))
        total_correct += correct
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        cohorts[held_name] = {
            "all_branch_rule": all_rule,
            "repeat_branch_rule": repeat_rule,
            "source_meta_threshold": selected,
            "eligible": bool(eligible),
            "held": {
                "p145_correct": int(np.sum(base == labels)),
                "selected_correct": correct,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "changed": int(route.sum()),
                "metrics": classification_metrics(labels, prediction),
            },
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_p145_prediction"] = base
        payload[f"{held_name}_all_branch_prediction"] = all_held
        payload[f"{held_name}_repeat_branch_prediction"] = repeat_held
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P150_repeat_branch_confidence_selector_v1",
        "status": "complete_strict_outer_crossfit_exploratory",
        "protocol": {
            "default": "P145",
            "alternative": "P149 repeat-consistency safe branch",
            "meta_score": "P149 probability of repeat-branch class",
            "user_id_used_as_feature": False,
            "user_id_used_for_source_stability_only": True,
            "held_labels_used_for_selection": False,
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
