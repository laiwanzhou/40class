"""Fresh nested session decoder for the P310/P315 teacher geometry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import DecoderConfig, decode_sessions, fit_transition_model
from p117_transductive_multicandidate_router import load_candidate_splits
from p139_soft_sequence_gate import SCORE_NAMES, gate_features, sessions_for
from p257_adaptive_physical_sequence import emission


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p373_fresh_p315_session_decoder_oof_v1"
P307 = HERE / "runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz"
P310 = HERE / "runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
EMISSION_WEIGHTS = (0.35, 0.50, 0.65, 0.80, 1.00)
TRANSITION_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50)
TRIGRAM_BACKOFFS = (0.5, 1.0)
BEAM_WIDTH = 50


def decoder(weight, backoff):
    return DecoderConfig(
        gap_seconds=30.0,
        transition_weight=float(weight),
        trigram_backoff=float(backoff),
        beam_width=BEAM_WIDTH,
    )


def load_parts():
    data = load_candidate_splits()
    group = np.load(P307)
    current = np.load(P310)
    offset = 0
    parts = {}
    for cohort in COHORTS:
        split = data[cohort].split
        rows = len(split.labels)
        labels = split.labels.astype(int)
        if not np.array_equal(labels, current["labels"][offset : offset + rows]):
            raise RuntimeError(f"P310 alignment failure: {cohort}")
        parts[cohort] = {
            "ids": split.sample_ids.astype(str),
            "labels": labels,
            "users": split.users.astype(str),
            "base": current["prediction"][offset : offset + rows].astype(int),
            "probability": group[f"{cohort}_group_probability"].astype(float),
        }
        offset += rows
    return data, parts


def fit_transition(data, parts, names, backoff):
    ids = np.concatenate([parts[name]["ids"] for name in names])
    labels = np.concatenate([parts[name]["labels"] for name in names])
    sessions = sessions_for(data, ids, names)
    return fit_transition_model(labels, sessions, 40, float(backoff))


def decode_part(data, parts, name, transition, emission_weight, transition_weight, backoff):
    part = parts[name]
    sessions = sessions_for(data, part["ids"], [name])
    sequence = decode_sessions(
        emission(part["probability"], part["base"], float(emission_weight)),
        sessions,
        transition,
        decoder(transition_weight, backoff),
    )
    return sequence


def select_gate(base, sequence, probability, labels, users, cohort_ids):
    features = gate_features(probability, base, sequence)
    disagreement = sequence != base
    best = None
    for score_index, score_name in enumerate(SCORE_NAMES):
        values = np.unique(
            np.concatenate(
                (
                    [-np.inf, np.inf],
                    np.linspace(-1.0, 1.0, 201),
                    np.quantile(features[disagreement, score_index], np.linspace(0.1, 0.95, 18)) if disagreement.any() else [np.inf],
                )
            )
        )
        for threshold in values:
            route = disagreement & (features[:, score_index] >= threshold)
            output = base.copy()
            output[route] = sequence[route]
            gain = (output == labels).astype(int) - (base == labels).astype(int)
            per_cohort = {cohort: int(gain[cohort_ids == cohort].sum()) for cohort in np.unique(cohort_ids)}
            per_user = {user: int(gain[users == user].sum()) for user in np.unique(users)}
            rescue = int(np.sum(route & (base != labels) & (output == labels)))
            harm = int(np.sum(route & (base == labels) & (output != labels)))
            result = {
                "score_index": score_index,
                "score_name": score_name,
                "threshold": float(threshold),
                "changed": int(route.sum()),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "minimum_cohort_gain": min(per_cohort.values()),
                "minimum_user_gain": min(per_user.values()),
                "positive_cohorts": sum(value > 0 for value in per_cohort.values()),
                "positive_users": sum(value > 0 for value in per_user.values()),
                "per_cohort": per_cohort,
            }
            valid = result["net"] > 0 and result["minimum_cohort_gain"] >= 0 and result["minimum_user_gain"] >= -1
            key = (
                valid,
                result["minimum_cohort_gain"],
                result["net"],
                result["rescue"],
                -result["harm"],
                -result["changed"],
                -score_index,
            )
            if best is None or key > best[0]:
                best = (key, result)
    if not best[0][0]:
        return {
            "score_index": 0,
            "score_name": SCORE_NAMES[0],
            "threshold": 2.0,
            "changed": 0,
            "rescue": 0,
            "harm": 0,
            "net": 0,
            "minimum_cohort_gain": 0,
            "minimum_user_gain": 0,
            "positive_cohorts": 0,
            "positive_users": 0,
            "per_cohort": {},
        }
    return best[1]


def apply_gate(part, sequence, gate):
    features = gate_features(part["probability"], part["base"], sequence)
    route = (sequence != part["base"]) & (
        features[:, int(gate["score_index"])] >= float(gate["threshold"])
    )
    output = part["base"].copy()
    output[route] = sequence[route]
    return output, route


def main():
    print(
        "P373 learns a fresh decoder and transition table for P310/P315 using P307 soft "
        "emissions; no P270 decoder parameter is reused.",
        flush=True,
    )
    data, parts = load_parts()
    configs = [
        (emission_weight, transition_weight, backoff)
        for emission_weight in EMISSION_WEIGHTS
        for transition_weight in TRANSITION_WEIGHTS
        for backoff in TRIGRAM_BACKOFFS
    ]
    report = {
        "stage": "P373_fresh_P315_session_decoder_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "soft_emission": "P307 matched 30-teacher group probability",
            "old_decoder_parameters_reused": False,
            "emission_weights": list(EMISSION_WEIGHTS),
            "transition_weights": list(TRANSITION_WEIGHTS),
            "trigram_backoffs": list(TRIGRAM_BACKOFFS),
            "inner_selection": "cross-decode each source cohort from the other source cohort",
            "transition_fit_excludes_held_labels": True,
            "held_labels_used_for_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source_base = np.concatenate([parts[name]["base"] for name in source_names])
        source_labels = np.concatenate([parts[name]["labels"] for name in source_names])
        source_users = np.concatenate([parts[name]["users"] for name in source_names])
        source_probability = np.concatenate([parts[name]["probability"] for name in source_names])
        cohort_ids = np.concatenate([np.full(len(parts[name]["labels"]), name, dtype=object) for name in source_names])
        best = None
        for emission_weight, transition_weight, backoff in configs:
            source_sequences = []
            for train_name, validation_name in (
                (source_names[0], source_names[1]),
                (source_names[1], source_names[0]),
            ):
                transition = fit_transition(data, parts, [train_name], backoff)
                value = decode_part(
                    data, parts, validation_name, transition,
                    emission_weight, transition_weight, backoff,
                )
                source_sequences.append((validation_name, value))
            sequence = np.empty(len(source_labels), dtype=int)
            offsets = {
                source_names[0]: (0, len(parts[source_names[0]]["labels"])),
                source_names[1]: (len(parts[source_names[0]]["labels"]), len(source_labels)),
            }
            for validation_name, value in source_sequences:
                lo, hi = offsets[validation_name]
                sequence[lo:hi] = value
            gate = select_gate(
                source_base, sequence, source_probability, source_labels,
                source_users, cohort_ids,
            )
            key = (
                gate["minimum_cohort_gain"] >= 0,
                gate["net"],
                gate["rescue"],
                -gate["harm"],
                -gate["changed"],
                -abs(emission_weight - 0.65),
                -abs(transition_weight - 0.30),
                -abs(backoff - 1.0),
            )
            if best is None or key > best[0]:
                best = (key, emission_weight, transition_weight, backoff, gate)

        _, emission_weight, transition_weight, backoff, gate = best
        transition = fit_transition(data, parts, source_names, backoff)
        held_sequence = decode_part(
            data, parts, held, transition,
            emission_weight, transition_weight, backoff,
        )
        output, route = apply_gate(parts[held], held_sequence, gate)
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
            "selected": {
                "emission_weight": emission_weight,
                "transition_weight": transition_weight,
                "trigram_backoff": backoff,
                "beam_width": BEAM_WIDTH,
            },
            "source_gate": gate,
            "held": {
                "rows": len(labels),
                "base_correct": int(np.sum(base == labels)),
                "correct": int(np.sum(output == labels)),
                "net": int(np.sum(output == labels) - np.sum(base == labels)),
                "changed": int(route.sum()),
                "rescue": int(np.sum(route & (base != labels) & (output == labels))),
                "harm": int(np.sum(route & (base == labels) & (output != labels))),
            },
        }
        print(json.dumps({"held": held, "selected": report["cohorts"][held]["selected"], "held_result": report["cohorts"][held]["held"]}), flush=True)

    labels = np.concatenate([parts[cohort]["labels"] for cohort in COHORTS])
    base = np.concatenate([parts[cohort]["base"] for cohort in COHORTS])
    prediction = np.concatenate(outputs)
    fold_nets = [report["cohorts"][cohort]["held"]["net"] for cohort in COHORTS]
    strict_pass = bool(
        all(net > 0 for net in fold_nets)
        or (sum(net > 0 for net in fold_nets) >= 2 and min(fold_nets) >= -1)
    )
    report["aggregate"] = {
        "rows": len(labels),
        "base_correct": int(np.sum(base == labels)),
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "net_vs_p310": int(np.sum(prediction == labels) - np.sum(base == labels)),
        "fold_nets": fold_nets,
        "strict_gate_pass": strict_pass,
        "decision": "eligible_for_test_refit" if strict_pass else "reject_before_test",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "oof_predictions.npz",
        labels=labels,
        base_prediction=base,
        prediction=prediction,
        **{f"{cohort}_held_prediction": outputs[index] for index, cohort in enumerate(COHORTS)},
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: decoder and transition grid newly selected for P310/P307 geometry.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
