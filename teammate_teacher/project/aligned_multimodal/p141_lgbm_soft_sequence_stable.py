"""Shallow LightGBM group head with P140-style source-stable sequence gating."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

from audit_p87_sequence_decoder import (
    DecoderConfig,
    classification_metrics,
    decode_sessions,
    fit_transition_model,
)
from p117_transductive_multicandidate_router import load_candidate_splits
from p134_frozen_repeat_consensus import probability_lookup
from p137_group_classifier_selector import group_features
from p139_soft_sequence_gate import emission, gate_features, select_gate


HERE = Path(__file__).resolve().parent
P128 = HERE / "runs/p128_base_hierarchical_meta_selector_v1/predictions.npz"
P140 = HERE / "runs/p140_source_stable_expanded_sequence_v1/predictions.npz"
OUTPUT = HERE / "runs/p141_lgbm_soft_sequence_stable_v1"
MINIMUM_SOURCE_COHORT_NET = 11
DECODER = DecoderConfig(30.0, 0.30, 1.0, 50)


def fit_probability(train_x, train_y, predict_x):
    sample_weight = np.where(train_x[:, -1] > 0, 2.0, 1.0)
    model = LGBMClassifier(
        n_estimators=150,
        num_leaves=7,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=20,
        reg_lambda=10.0,
        colsample_bytree=0.30,
        n_jobs=-1,
        verbosity=-1,
        random_state=137,
    )
    model.fit(train_x, train_y, sample_weight=sample_weight)
    output = np.zeros((len(predict_x), 40), dtype=np.float64)
    output[:, model.classes_.astype(np.int64)] = model.predict_proba(predict_x)
    return output


def sessions_for(data, sample_ids, split_names):
    position = {
        value: row for row, value in enumerate(sample_ids.astype(str))
    }
    return [
        np.asarray(
            [position[data[name].split.sample_ids[row]] for row in session],
            dtype=np.int64,
        )
        for name in split_names
        for session in data[name].split.sessions
    ]


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
    p128 = np.load(P128)
    p140 = np.load(P140)
    feature_lookup = {}
    for name, value in data.items():
        matrix = group_features(
            value.split.sample_ids,
            p128[f"{name}_prediction"],
            lookup,
        )
        for row, sample_id in enumerate(value.split.sample_ids.astype(str)):
            feature_lookup[sample_id] = matrix[row]

    payload = {}
    cohorts = {}
    total_correct = 0
    split_names = list(data)
    for held_name in split_names:
        source_names = [name for name in split_names if name != held_name]
        source_ids = p140[f"{held_name}_source_sample_ids"].astype(str)
        source_labels = p140[f"{held_name}_source_labels"].astype(np.int64)
        source_base = p140[f"{held_name}_source_prediction"].astype(np.int64)
        source_position = {
            value: row for row, value in enumerate(source_ids)
        }
        source_probability = np.zeros((len(source_ids), 40), dtype=np.float64)
        for target_name in source_names:
            calibration_name = next(
                name for name in source_names if name != target_name
            )
            target_rows = np.asarray(
                [
                    source_position[value]
                    for value in data[target_name].split.sample_ids.astype(str)
                ],
                dtype=np.int64,
            )
            calibration_rows = np.asarray(
                [
                    source_position[value]
                    for value in data[calibration_name].split.sample_ids.astype(str)
                ],
                dtype=np.int64,
            )
            source_probability[target_rows] = fit_probability(
                np.stack(
                    [feature_lookup[value] for value in source_ids[calibration_rows]]
                ),
                source_labels[calibration_rows],
                np.stack([feature_lookup[value] for value in source_ids[target_rows]]),
            )

        source_sessions = sessions_for(data, source_ids, source_names)
        transition = fit_transition_model(
            source_labels, source_sessions, 40, DECODER.trigram_backoff
        )
        source_sequence = decode_sessions(
            emission(source_probability, source_base),
            source_sessions,
            transition,
            DECODER,
        )
        source_features = gate_features(
            source_probability, source_base, source_sequence
        )
        gate = select_gate(
            source_base, source_sequence, source_features, source_labels
        )
        source_route = (source_sequence != source_base) & (
            source_features[:, int(gate["score_index"])]
            >= float(gate["threshold"])
        )
        source_prediction = source_base.copy()
        source_prediction[source_route] = source_sequence[source_route]
        per_source_cohort = {}
        for name in source_names:
            rows = np.asarray(
                [
                    source_position[value]
                    for value in data[name].split.sample_ids.astype(str)
                ],
                dtype=np.int64,
            )
            rescue = int(
                np.sum(
                    (source_base[rows] != source_labels[rows])
                    & (source_prediction[rows] == source_labels[rows])
                )
            )
            harm = int(
                np.sum(
                    (source_base[rows] == source_labels[rows])
                    & (source_prediction[rows] != source_labels[rows])
                )
            )
            per_source_cohort[name] = {
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "changed": int(source_route[rows].sum()),
            }
        eligible = (
            min(row["net"] for row in per_source_cohort.values())
            >= MINIMUM_SOURCE_COHORT_NET
        )

        sample_ids = p140[f"{held_name}_sample_ids"].astype(str)
        labels = p140[f"{held_name}_labels"].astype(np.int64)
        base = p140[f"{held_name}_prediction"].astype(np.int64)
        held_probability = fit_probability(
            np.stack([feature_lookup[value] for value in source_ids]),
            source_labels,
            np.stack([feature_lookup[value] for value in sample_ids]),
        )
        held_sequence = decode_sessions(
            emission(held_probability, base),
            sessions_for(data, sample_ids, [held_name]),
            transition,
            DECODER,
        )
        held_features = gate_features(held_probability, base, held_sequence)
        held_route = (
            (held_sequence != base)
            & (
                held_features[:, int(gate["score_index"])]
                >= float(gate["threshold"])
            )
            & eligible
        )
        prediction = base.copy()
        prediction[held_route] = held_sequence[held_route]
        correct = int(np.sum(prediction == labels))
        total_correct += correct
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        cohorts[held_name] = {
            "source_gate": gate,
            "per_source_cohort": per_source_cohort,
            "eligible": bool(eligible),
            "held": {
                "p140_correct": int(np.sum(base == labels)),
                "selected_correct": correct,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "changed": int(held_route.sum()),
                "metrics": classification_metrics(labels, prediction),
            },
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_p140_prediction"] = base
        payload[f"{held_name}_lgbm_probability"] = held_probability.astype(np.float32)
        payload[f"{held_name}_sequence_prediction"] = held_sequence
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P141_lgbm_soft_sequence_stable_v1",
        "status": "complete_strict_outer_crossfit_exploratory",
        "protocol": {
            "head": "LightGBM leaves=7 depth=4 trees=150",
            "minimum_net_required_in_each_source_cohort": MINIMUM_SOURCE_COHORT_NET,
            "fallback": "P140",
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
