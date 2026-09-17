"""Source-cohort-stable selector over fixed expanded P137 probability modes."""

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
from p139_soft_sequence_gate import emission, gate_features, select_gate


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p140_source_stable_expanded_sequence_v1"
P138 = HERE / "runs/p138_group_feature_mode_selector_v1/predictions.npz"
P138_SUMMARY = HERE / "runs/p138_group_feature_mode_selector_v1/summary.json"
MINIMUM_SOURCE_COHORT_NET = 11
RUNS = {
    "sqrt": HERE / "runs/p137_group_classifier_selector_c003_w2_v3",
    "sqrt_raw": HERE / "runs/p137_group_classifier_selector_c003_w2_sqrtraw_v7",
    "thermal": HERE / "runs/p137_expanded_plus_thermal_v23",
    "p12_imu": HERE / "runs/p137_expanded_plus_p12_imu_v23",
    "motion_front": HERE / "runs/p137_expanded_plus_motion_front_v23",
    "skeleton": HERE / "runs/p137_expanded_plus_skeleton_v23",
    "pose": HERE / "runs/p137_expanded_plus_pose_v23",
    "object": HERE / "runs/p137_expanded_plus_object_v23",
    "relation": HERE / "runs/p137_expanded_plus_relation_v23",
    "local_depth": HERE / "runs/p137_expanded_plus_local_depth_v23",
    "egovlp": HERE / "runs/p137_expanded_plus_egovlp_v23",
    "dense_blend": HERE / "runs/p137_expanded_plus_dense_blend_v23",
}
DECODER = DecoderConfig(30.0, 0.30, 1.0, 50)


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


def source_candidate(data, held_name, run):
    split_names = list(data)
    source_names = [name for name in split_names if name != held_name]
    sample_ids = run[f"{held_name}_source_sample_ids"].astype(str)
    labels = run[f"{held_name}_source_labels"].astype(np.int64)
    base = run[f"{held_name}_source_prediction"].astype(np.int64)
    probability = run[f"{held_name}_source_group_probability"].astype(np.float64)
    sessions = sessions_for(data, sample_ids, source_names)
    transition = fit_transition_model(
        labels, sessions, 40, DECODER.trigram_backoff
    )
    sequence = decode_sessions(
        emission(probability, base), sessions, transition, DECODER
    )
    features = gate_features(probability, base, sequence)
    gate = select_gate(base, sequence, features, labels)
    route = (sequence != base) & (
        features[:, int(gate["score_index"])] >= float(gate["threshold"])
    )
    prediction = base.copy()
    prediction[route] = sequence[route]
    position = {value: row for row, value in enumerate(sample_ids)}
    per_cohort = {}
    for name in source_names:
        rows = np.asarray(
            [position[value] for value in data[name].split.sample_ids.astype(str)],
            dtype=np.int64,
        )
        rescue = int(np.sum((base[rows] != labels[rows]) & (prediction[rows] == labels[rows])))
        harm = int(np.sum((base[rows] == labels[rows]) & (prediction[rows] != labels[rows])))
        per_cohort[name] = {
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "changed": int(route[rows].sum()),
        }
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "base": base,
        "sequence": sequence,
        "features": features,
        "transition": transition,
        "gate": gate,
        "per_source_cohort": per_cohort,
        "eligible": min(row["net"] for row in per_cohort.values())
        >= MINIMUM_SOURCE_COHORT_NET,
    }


def main() -> None:
    data = load_candidate_splits()
    runs = {
        name: np.load(path / "predictions.npz") for name, path in RUNS.items()
    }
    fallback = np.load(P138)
    fallback_summary = json.loads(P138_SUMMARY.read_text(encoding="utf-8"))
    payload = {}
    cohorts = {}
    total_correct = 0
    for held_name in data:
        candidates = []
        source_values = {}
        for mode, run in runs.items():
            source = source_candidate(data, held_name, run)
            source_values[mode] = source
            candidates.append(
                {
                    "mode": mode,
                    "eligible": bool(source["eligible"]),
                    "gate": source["gate"],
                    "per_source_cohort": source["per_source_cohort"],
                }
            )
        eligible = [row for row in candidates if row["eligible"]]
        selected = (
            max(
                eligible,
                key=lambda row: (
                    row["gate"]["net"],
                    -row["gate"]["harm"],
                    row["gate"]["rescue"],
                    -row["gate"]["changed"],
                ),
            )
            if eligible
            else None
        )

        if selected is None:
            fallback_mode = str(
                fallback_summary["cohorts"][held_name]["selected_mode"]
            )
            source_for_output = source_values[fallback_mode]
            sample_ids = fallback[f"{held_name}_sample_ids"].astype(str)
            labels = fallback[f"{held_name}_labels"].astype(np.int64)
            base = fallback[f"{held_name}_prediction"].astype(np.int64)
            prediction = base.copy()
            sequence = base.copy()
            features = np.zeros((len(base), 5), dtype=np.float64)
            route = np.zeros(len(base), dtype=bool)
            selected_mode = "p138_fallback"
        else:
            selected_mode = str(selected["mode"])
            run = runs[selected_mode]
            source = source_values[selected_mode]
            source_for_output = source
            sample_ids = run[f"{held_name}_sample_ids"].astype(str)
            labels = run[f"{held_name}_labels"].astype(np.int64)
            base = run[f"{held_name}_prediction"].astype(np.int64)
            probability = run[f"{held_name}_group_probability"].astype(np.float64)
            held_sessions = sessions_for(data, sample_ids, [held_name])
            sequence = decode_sessions(
                emission(probability, base),
                held_sessions,
                source["transition"],
                DECODER,
            )
            features = gate_features(probability, base, sequence)
            gate = source["gate"]
            route = (sequence != base) & (
                features[:, int(gate["score_index"])]
                >= float(gate["threshold"])
            )
            prediction = base.copy()
            prediction[route] = sequence[route]

        correct = int(np.sum(prediction == labels))
        total_correct += correct
        fallback_prediction = fallback[f"{held_name}_prediction"].astype(np.int64)
        cohorts[held_name] = {
            "candidate_count": len(candidates),
            "eligible_modes": [row["mode"] for row in eligible],
            "selected_mode": selected_mode,
            "selected_source": selected,
            "held": {
                "p138_correct": int(np.sum(fallback_prediction == labels)),
                "selected_correct": correct,
                "changed_from_selected_base": int(route.sum()),
                "metrics": classification_metrics(labels, prediction),
            },
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_selected_base_prediction"] = base
        payload[f"{held_name}_sequence_prediction"] = sequence
        payload[f"{held_name}_gate_features"] = features
        payload[f"{held_name}_prediction"] = prediction
        source_route = (
            source_for_output["sequence"] != source_for_output["base"]
        ) & (
            source_for_output["features"][:, int(source_for_output["gate"]["score_index"])]
            >= float(source_for_output["gate"]["threshold"])
        )
        source_prediction = source_for_output["base"].copy()
        if selected is not None:
            source_prediction[source_route] = source_for_output["sequence"][source_route]
        payload[f"{held_name}_source_sample_ids"] = source_for_output["sample_ids"]
        payload[f"{held_name}_source_labels"] = source_for_output["labels"]
        payload[f"{held_name}_source_prediction"] = source_prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P140_source_stable_expanded_sequence_v1",
        "status": "complete_strict_outer_crossfit_exploratory",
        "protocol": {
            "minimum_net_required_in_each_source_cohort": MINIMUM_SOURCE_COHORT_NET,
            "mode_selection_key": "source net, -harm, rescue, -changed",
            "fallback": "P138",
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
