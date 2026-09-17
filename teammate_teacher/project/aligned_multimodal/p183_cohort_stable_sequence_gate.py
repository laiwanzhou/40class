"""Cohort-minimax sequence gate over P177 group probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import decode_sessions, fit_transition_model
from p117_transductive_multicandidate_router import load_candidate_splits
from p139_soft_sequence_gate import DECODER, SCORE_NAMES, emission, gate_features, sessions_for


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p183_cohort_stable_sequence_gate_v1"
P177 = HERE / "runs/p177_p128_vjepa_group_teacher_v1/predictions.npz"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")


def select_stable_gate(base, sequence, features, labels, cohorts):
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
            prediction = base.copy(); prediction[route] = sequence[route]
            per_cohort = {}
            for cohort in sorted(set(cohorts.tolist())):
                rows = cohorts == cohort
                rescue = int(np.sum(rows & (base != labels) & (prediction == labels)))
                harm = int(np.sum(rows & (base == labels) & (prediction != labels)))
                per_cohort[cohort] = {"rescue": rescue, "harm": harm, "net": rescue-harm}
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
                    "net": rescue-harm,
                    "minimum_cohort_net": min(row["net"] for row in per_cohort.values()),
                    "per_cohort": per_cohort,
                }
            )
    eligible = [row for row in candidates if row["minimum_cohort_net"] > 0]
    if not eligible:
        return {
            "score_index": 0, "score_name": SCORE_NAMES[0], "threshold": 2.0,
            "changed": 0, "rescue": 0, "harm": 0, "net": 0,
            "minimum_cohort_net": 0, "per_cohort": {},
        }
    return max(
        eligible,
        key=lambda row: (
            row["minimum_cohort_net"], row["net"], row["rescue"],
            -row["harm"], -row["changed"],
        ),
    )


def main() -> None:
    data = load_candidate_splits()
    source = np.load(P177, allow_pickle=False)
    values = {
        name: {
            "ids": data[name].split.sample_ids.astype(str),
            "labels": data[name].split.labels.astype(np.int64),
            "base": source[f"{name}_held_prediction"].astype(np.int64),
            "probability": source[f"{name}_held_probability"].astype(np.float64),
        }
        for name in SPLITS
    }
    reports = {}; outputs = {}
    for held_name in SPLITS:
        source_names = [name for name in SPLITS if name != held_name]
        ids = np.concatenate([values[name]["ids"] for name in source_names])
        labels = np.concatenate([values[name]["labels"] for name in source_names])
        base = np.concatenate([values[name]["base"] for name in source_names])
        probability = np.concatenate([values[name]["probability"] for name in source_names])
        cohort = np.concatenate(
            [np.full(len(values[name]["ids"]), name, dtype=object) for name in source_names]
        )
        sessions = sessions_for(data, ids, source_names)
        transition = fit_transition_model(labels, sessions, 40, DECODER.trigram_backoff)
        sequence = decode_sessions(emission(probability, base), sessions, transition, DECODER)
        features = gate_features(probability, base, sequence)
        selected = select_stable_gate(base, sequence, features, labels, cohort)
        held = values[held_name]
        held_sessions = sessions_for(data, held["ids"], [held_name])
        held_sequence = decode_sessions(
            emission(held["probability"], held["base"]), held_sessions, transition, DECODER
        )
        held_features = gate_features(held["probability"], held["base"], held_sequence)
        route = (held_sequence != held["base"]) & (
            held_features[:, int(selected["score_index"])] >= float(selected["threshold"])
        )
        prediction = held["base"].copy(); prediction[route] = held_sequence[route]
        base_correct = held["base"] == held["labels"]
        final_correct = prediction == held["labels"]
        reports[held_name] = {
            "source_gate": selected,
            "held": {
                "base_correct": int(base_correct.sum()),
                "correct": int(final_correct.sum()),
                "net": int(final_correct.sum()-base_correct.sum()),
                "rescue": int(np.sum(~base_correct&final_correct)),
                "harm": int(np.sum(base_correct&~final_correct)),
                "changed": int(route.sum()),
            },
        }
        outputs[held_name] = prediction
    labels = np.concatenate([values[name]["labels"] for name in SPLITS])
    base = np.concatenate([values[name]["base"] for name in SPLITS])
    prediction = np.concatenate([outputs[name] for name in SPLITS])
    correct = int(np.sum(prediction==labels)); base_correct=int(np.sum(base==labels))
    report = {
        "stage":"P183_cohort_stable_sequence_gate",
        "status":"complete",
        "protocol":{"selection":"maximize minimum source-cohort net", "test_labels_read":False},
        "cohorts":reports,
        "aggregate":{"rows":len(labels),"base_correct":base_correct,"correct":correct,"accuracy":correct/len(labels),"net_vs_p177":correct-base_correct,"fold_nets":[int(reports[n]["held"]["net"]) for n in SPLITS]},
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUTPUT/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=prediction,**{f"{n}_held_prediction":outputs[n] for n in SPLITS})
    (OUTPUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
