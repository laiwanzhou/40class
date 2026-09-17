"""Source-user-safe selector for the three-seed P142 token Transformer."""

from __future__ import annotations

import argparse
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
)
from p134_frozen_repeat_consensus import probability_lookup


HERE = Path(__file__).resolve().parent
P140 = HERE / "runs/p140_source_stable_expanded_sequence_v1/predictions.npz"
P142 = HERE / "runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz"
OUTPUT = HERE / "runs/p143_p142_source_user_safe_selector_v1"


def build_features(sample_ids, base, alternative, probability, lookup):
    hard = np.stack([lookup[value].argmax(axis=1) for value in sample_ids])
    vote = np.stack(
        [(hard == class_id).mean(axis=1) for class_id in range(40)], axis=1
    )
    mean_probability = np.stack(
        [lookup[value].mean(axis=0) for value in sample_ids]
    )
    rows = np.arange(len(sample_ids))
    ordered = np.sort(probability, axis=1)[:, ::-1]
    scalar = np.column_stack(
        (
            probability[rows, alternative] - probability[rows, base],
            probability[rows, alternative],
            probability[rows, base],
            ordered[:, 0] - ordered[:, 1],
            vote[rows, alternative],
            vote[rows, base],
            mean_probability[rows, alternative],
            mean_probability[rows, base],
            mean_probability[rows, alternative]
            - mean_probability[rows, base],
            alternative != base,
        )
    ).astype(np.float32)
    return np.concatenate(
        (
            probability.astype(np.float32),
            np.log(np.clip(probability, 1e-6, 1.0)).astype(np.float32),
            vote.astype(np.float32),
            mean_probability.astype(np.float32),
            one_hot(base),
            one_hot(alternative),
            scalar,
        ),
        axis=1,
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternative", type=Path, default=P142)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--alternative-name", type=str, default="p142_all24")
    parser.add_argument("--probability-key", type=str, default="probability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
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
    users = {
        sample_id: user
        for value in data.values()
        for sample_id, user in zip(
            value.split.sample_ids.astype(str), value.split.users.astype(str)
        )
    }
    p140 = np.load(P140)
    p142 = np.load(args.alternative.resolve())
    p142_position = {
        value: row for row, value in enumerate(p142["sample_ids"].astype(str))
    }
    payload = {}
    cohorts = {}
    total_correct = 0
    for held_name in data:
        source_ids = p140[f"{held_name}_source_sample_ids"].astype(str)
        source_labels = p140[f"{held_name}_source_labels"].astype(np.int64)
        source_base = p140[f"{held_name}_source_prediction"].astype(np.int64)
        source_probability = p142[args.probability_key][
            np.asarray([p142_position[value] for value in source_ids], dtype=np.int64)
        ].astype(np.float64)
        source_alternative = source_probability.argmax(axis=1).astype(np.int64)
        source_x = build_features(
            source_ids,
            source_base,
            source_alternative,
            source_probability,
            lookup,
        )
        gain = (source_alternative == source_labels).astype(np.int8) - (
            source_base == source_labels
        ).astype(np.int8)
        source_users = np.asarray([users[value] for value in source_ids])
        nested_score = loso_scores(source_x, gain, source_users)
        selected = select_threshold(
            nested_score,
            gain,
            source_alternative != source_base,
            source_users,
        )
        eligible = (
            int(selected["net"]) > 0
            and int(selected["minimum_user_gain"]) >= 0
        )

        sample_ids = p140[f"{held_name}_sample_ids"].astype(str)
        labels = p140[f"{held_name}_labels"].astype(np.int64)
        base = p140[f"{held_name}_prediction"].astype(np.int64)
        probability = p142[args.probability_key][
            np.asarray([p142_position[value] for value in sample_ids], dtype=np.int64)
        ].astype(np.float64)
        alternative = probability.argmax(axis=1).astype(np.int64)
        held_x = build_features(
            sample_ids, base, alternative, probability, lookup
        )
        held_score = fit_score(source_x, gain, held_x)
        route = (
            (alternative != base)
            & (held_score >= float(selected["threshold"]))
            & eligible
        )
        prediction = base.copy()
        prediction[route] = alternative[route]
        correct = int(np.sum(prediction == labels))
        total_correct += correct
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        cohorts[held_name] = {
            "source_threshold": selected,
            "eligible": bool(eligible),
            "held": {
                "p140_correct": int(np.sum(base == labels)),
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
        payload[f"{held_name}_p140_prediction"] = base
        payload[f"{held_name}_p142_prediction"] = alternative
        payload[f"{held_name}_selector_score"] = held_score
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P143_P142_source_user_safe_selector_v1",
        "status": "complete_strict_outer_crossfit_exploratory",
        "protocol": {
            "alternative": args.alternative_name,
            "alternative_path": str(args.alternative.resolve()),
            "probability_key": args.probability_key,
            "eligibility": "source net > 0 and minimum source-user gain >= 0",
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
