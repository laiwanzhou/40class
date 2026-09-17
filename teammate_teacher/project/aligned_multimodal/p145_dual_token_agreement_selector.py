"""Conservative agreement selector for all-token and hand-token Transformers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits


HERE = Path(__file__).resolve().parent
P140 = HERE / "runs/p140_source_stable_expanded_sequence_v1/predictions.npz"
P143 = HERE / "runs/p143_p142_source_user_safe_selector_v1/predictions.npz"
ALL_TOKEN = HERE / "runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz"
HAND_TOKEN = HERE / "runs/p144_vjepa_hand_interaction_transformer_three_seed_v1/oof_predictions.npz"
OUTPUT = HERE / "runs/p145_dual_token_agreement_selector_v1"


def main() -> None:
    data = load_candidate_splits()
    p140 = np.load(P140)
    p143 = np.load(P143)
    all_token = np.load(ALL_TOKEN)
    hand_token = np.load(HAND_TOKEN)
    position = {
        value: row
        for row, value in enumerate(all_token["sample_ids"].astype(str))
    }
    users = {
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
        all_probability = all_token["probability"][source_rows].astype(np.float64)
        hand_probability = hand_token["probability"][source_rows].astype(np.float64)
        all_prediction = all_probability.argmax(axis=1).astype(np.int64)
        hand_prediction = hand_probability.argmax(axis=1).astype(np.int64)
        agreement = (all_prediction == hand_prediction) & (
            all_prediction != source_base
        )
        confidence = np.minimum(
            all_probability[np.arange(len(source_ids)), all_prediction],
            hand_probability[np.arange(len(source_ids)), all_prediction],
        )
        source_users = np.asarray([users[value] for value in source_ids])
        candidates = []
        values = (
            np.unique(
                np.concatenate(
                    (
                        np.linspace(0.0, 1.0, 201),
                        np.quantile(
                            confidence[agreement],
                            [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
                        ),
                    )
                )
            )
            if agreement.any()
            else np.asarray((2.0,))
        )
        for threshold in values:
            route = agreement & (confidence >= threshold)
            prediction = source_base.copy()
            prediction[route] = all_prediction[route]
            rescue = int(
                np.sum((source_base != source_labels) & (prediction == source_labels))
            )
            harm = int(
                np.sum((source_base == source_labels) & (prediction != source_labels))
            )
            per_user = {
                user: int(
                    np.sum(prediction[source_users == user] == source_labels[source_users == user])
                    - np.sum(source_base[source_users == user] == source_labels[source_users == user])
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
        eligible = (
            selected["minimum_user_gain"] >= 0
            and selected["rescue"] >= 2
            and selected["harm"] == 0
        )

        sample_ids = p143[f"{held_name}_sample_ids"].astype(str)
        labels = p143[f"{held_name}_labels"].astype(np.int64)
        base = p143[f"{held_name}_prediction"].astype(np.int64)
        held_rows = np.asarray([position[value] for value in sample_ids], dtype=np.int64)
        all_probability = all_token["probability"][held_rows].astype(np.float64)
        hand_probability = hand_token["probability"][held_rows].astype(np.float64)
        all_prediction = all_probability.argmax(axis=1).astype(np.int64)
        hand_prediction = hand_probability.argmax(axis=1).astype(np.int64)
        confidence = np.minimum(
            all_probability[np.arange(len(sample_ids)), all_prediction],
            hand_probability[np.arange(len(sample_ids)), all_prediction],
        )
        route = (
            (all_prediction == hand_prediction)
            & (all_prediction != base)
            & (confidence >= float(selected["threshold"]))
            & eligible
        )
        prediction = base.copy()
        prediction[route] = all_prediction[route]
        correct = int(np.sum(prediction == labels))
        total_correct += correct
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        cohorts[held_name] = {
            "source_threshold": selected,
            "eligible": bool(eligible),
            "held": {
                "p143_correct": int(np.sum(base == labels)),
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
        payload[f"{held_name}_p143_prediction"] = base
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P145_dual_token_agreement_selector_v1",
        "status": "complete_strict_outer_crossfit_exploratory_low_support",
        "protocol": {
            "eligibility": "source rescue >= 2, harm == 0, minimum user gain >= 0",
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
