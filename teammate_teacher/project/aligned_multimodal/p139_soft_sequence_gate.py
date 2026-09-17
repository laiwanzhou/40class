"""Source-gated soft sequence decoding over P138-selected group probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DecoderConfig,
    classification_metrics,
    decode_sessions,
    fit_transition_model,
)
from p117_transductive_multicandidate_router import load_candidate_splits


HERE = Path(__file__).resolve().parent
P138 = HERE / "runs/p138_group_feature_mode_selector_v1/summary.json"
OUTPUT = HERE / "runs/p139_soft_sequence_gate_v1"
RUNS = {
    "sqrt": HERE / "runs/p137_group_classifier_selector_c003_w2_v3/predictions.npz",
    "sqrt_raw": HERE
    / "runs/p137_group_classifier_selector_c003_w2_sqrtraw_v7/predictions.npz",
}
SCORE_NAMES = (
    "sequence_minus_base_probability",
    "sequence_probability",
    "negative_base_probability",
    "group_max_probability",
    "group_probability_margin",
)
DECODER = DecoderConfig(
    gap_seconds=30.0,
    transition_weight=0.30,
    trigram_backoff=1.0,
    beam_width=50,
)


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


def emission(probability, base):
    output = 0.75 * np.asarray(probability, dtype=np.float64)
    output[np.arange(len(base)), base] += 0.25
    output /= output.sum(axis=1, keepdims=True)
    return np.log(np.clip(output, 1e-9, 1.0))


def gate_features(probability, base, sequence):
    rows = np.arange(len(base))
    ordered = np.sort(probability, axis=1)[:, ::-1]
    return np.column_stack(
        (
            probability[rows, sequence] - probability[rows, base],
            probability[rows, sequence],
            -probability[rows, base],
            ordered[:, 0],
            ordered[:, 0] - ordered[:, 1],
        )
    ).astype(np.float64)


def select_gate(base, sequence, features, labels):
    disagreement = sequence != base
    candidates = []
    for score_index, score_name in enumerate(SCORE_NAMES):
        values = np.unique(
            np.concatenate(
                (
                    np.linspace(-1.0, 1.0, 401),
                    np.quantile(
                        features[disagreement, score_index],
                        [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
                    ),
                )
            )
        )
        for threshold in values:
            route = disagreement & (features[:, score_index] >= threshold)
            prediction = base.copy()
            prediction[route] = sequence[route]
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
    return (
        max(
            positive,
            key=lambda row: (
                row["net"],
                row["rescue"],
                -row["harm"],
                -row["changed"],
            ),
        )
        if positive
        else {
            "score_index": 0,
            "score_name": SCORE_NAMES[0],
            "threshold": 2.0,
            "changed": 0,
            "rescue": 0,
            "harm": 0,
            "net": 0,
        }
    )


def main() -> None:
    data = load_candidate_splits()
    p138 = json.loads(P138.read_text(encoding="utf-8"))
    runs = {name: np.load(path) for name, path in RUNS.items()}
    payload = {}
    cohorts = {}
    totals = {"correct": 0, "rescue": 0, "harm": 0, "changed": 0}
    split_names = list(data)
    for held_name in split_names:
        mode = str(p138["cohorts"][held_name]["selected_mode"])
        run = runs[mode]
        source_names = [name for name in split_names if name != held_name]
        source_ids = run[f"{held_name}_source_sample_ids"].astype(str)
        source_labels = run[f"{held_name}_source_labels"].astype(np.int64)
        source_base = run[f"{held_name}_source_prediction"].astype(np.int64)
        source_probability = run[
            f"{held_name}_source_group_probability"
        ].astype(np.float64)
        source_sessions = sessions_for(data, source_ids, source_names)
        transition = fit_transition_model(
            source_labels,
            source_sessions,
            num_classes=40,
            trigram_backoff=DECODER.trigram_backoff,
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
        selected = select_gate(
            source_base, source_sequence, source_features, source_labels
        )

        sample_ids = run[f"{held_name}_sample_ids"].astype(str)
        labels = run[f"{held_name}_labels"].astype(np.int64)
        base = run[f"{held_name}_prediction"].astype(np.int64)
        probability = run[f"{held_name}_group_probability"].astype(np.float64)
        held_sessions = sessions_for(data, sample_ids, [held_name])
        sequence = decode_sessions(
            emission(probability, base), held_sessions, transition, DECODER
        )
        features = gate_features(probability, base, sequence)
        route = (sequence != base) & (
            features[:, int(selected["score_index"])]
            >= float(selected["threshold"])
        )
        prediction = base.copy()
        prediction[route] = sequence[route]
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        correct = int(np.sum(prediction == labels))
        changed = int(route.sum())
        totals["correct"] += correct
        totals["rescue"] += rescue
        totals["harm"] += harm
        totals["changed"] += changed
        cohorts[held_name] = {
            "selected_group_mode": mode,
            "source_gate": selected,
            "held": {
                "p138_correct": int(np.sum(base == labels)),
                "sequence_correct": correct,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "changed": changed,
                "metrics": classification_metrics(labels, prediction),
            },
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_p138_prediction"] = base
        payload[f"{held_name}_sequence_prediction"] = sequence
        payload[f"{held_name}_gate_features"] = features
        payload[f"{held_name}_prediction"] = prediction
        payload[f"{held_name}_source_sample_ids"] = source_ids
        payload[f"{held_name}_source_labels"] = source_labels
        payload[f"{held_name}_source_p138_prediction"] = source_base
        payload[f"{held_name}_source_sequence_prediction"] = source_sequence
        payload[f"{held_name}_source_gate_features"] = source_features

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P139_soft_sequence_gate_v1",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "decoder": {
                "group_probability_weight": 0.75,
                "p138_one_hot_weight": 0.25,
                "transition_weight": DECODER.transition_weight,
                "trigram_backoff": DECODER.trigram_backoff,
                "beam_width": DECODER.beam_width,
            },
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
            "net_vs_p138": totals["rescue"] - totals["harm"],
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
