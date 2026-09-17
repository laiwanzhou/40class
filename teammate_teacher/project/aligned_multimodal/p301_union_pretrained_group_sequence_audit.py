"""Audit whether the extra-122-row Thermal pretraining transfers to P270.

This is deliberately OOF-only.  P300 has no matched Test refit yet, so this
script must not manufacture a Test posterior or a submission from a mismatched
model.  A Test head is justified only if both the group layer and the frozen
sequence recipe pass this audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import decode_sessions, fit_transition_model
from p117_transductive_multicandidate_router import load_candidate_splits
from p139_soft_sequence_gate import gate_features, select_gate, sessions_for
from p165_deployable_group_teacher import SPLITS, crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank
from p257_adaptive_physical_sequence import dec, emission
from p255_repeat_augmented_physical_group import al


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p301_union_pretrained_group_sequence_audit_v1"
P128 = HERE / "runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz"
LAVILA = HERE / "runs/p158_lavila_frame_token_transformer_single_seed_v1/oof_predictions.npz"
PHYSICAL = HERE / "runs/p238_physical_token_transformer_oof_v1/oof_predictions.npz"
RIDGE = HERE / "runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz"
REPEAT = HERE / "runs/p253_repeat_physical_transformer_oof_v1/oof_predictions.npz"
UNION_PRETRAINED = HERE / "runs/p300_union_pretrained_physical_oof_v1/oof_predictions.npz"
P255 = HERE / "runs/p255_repeat_augmented_physical_group_v1/predictions.npz"
P270 = HERE / "runs/p270_fixed_emission065_transition045_v1/predictions.npz"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
EMISSION_WEIGHT = 0.65
TRANSITION_WEIGHT = 0.45


def append_expert(train: dict, probability: np.ndarray, sample_ids: np.ndarray) -> None:
    for name in SPLITS:
        split = train[name]
        extra = al(probability, sample_ids, split["ids"])
        split["bank"] = np.concatenate((split["bank"], extra[:, None, :]), axis=1)


def sequence_crossfit(train: dict, group_prediction: dict, group_probability: dict):
    candidates = load_candidate_splits()
    parts = {}
    for name in COHORTS:
        split = candidates[name].split
        parts[name] = {
            "ids": split.sample_ids.astype(str),
            "labels": split.labels.astype(int),
            "base": group_prediction[name].astype(int),
            "prob": group_probability[name].astype(float),
        }

    outputs = {}
    report = {}
    for held in COHORTS:
        source = [name for name in COHORTS if name != held]
        ids = np.concatenate([parts[name]["ids"] for name in source])
        labels = np.concatenate([parts[name]["labels"] for name in source])
        base = np.concatenate([parts[name]["base"] for name in source])
        probability = np.concatenate([parts[name]["prob"] for name in source])
        source_sessions = sessions_for(candidates, ids, source)
        transition = fit_transition_model(labels, source_sessions, 40, 1.0)
        source_sequence = decode_sessions(
            emission(probability, base, EMISSION_WEIGHT),
            source_sessions,
            transition,
            dec(TRANSITION_WEIGHT),
        )
        selected_gate = select_gate(
            base,
            source_sequence,
            gate_features(probability, base, source_sequence),
            labels,
        )

        held_sessions = sessions_for(candidates, parts[held]["ids"], [held])
        held_sequence = decode_sessions(
            emission(parts[held]["prob"], parts[held]["base"], EMISSION_WEIGHT),
            held_sessions,
            transition,
            dec(TRANSITION_WEIGHT),
        )
        held_features = gate_features(parts[held]["prob"], parts[held]["base"], held_sequence)
        route = (held_sequence != parts[held]["base"]) & (
            held_features[:, selected_gate["score_index"]] >= selected_gate["threshold"]
        )
        output = parts[held]["base"].copy()
        output[route] = held_sequence[route]
        outputs[held] = output
        base_correct = int(np.sum(parts[held]["base"] == parts[held]["labels"]))
        correct = int(np.sum(output == parts[held]["labels"]))
        report[held] = {
            "source": source,
            "source_gate": selected_gate,
            "held": {
                "rows": int(len(output)),
                "base_correct": base_correct,
                "correct": correct,
                "net": correct - base_correct,
                "changed": int(route.sum()),
            },
        }
    return outputs, report


def main() -> None:
    train, names = build_train_bank()
    sources = (
        (P128, "probabilities", "p128_hierarchical_multimodal"),
        (LAVILA, "probability", "p158_lavila_frame_token"),
        (PHYSICAL, "probability", "p238_physical_token"),
        (RIDGE, "ir_thermal_probability", "p231_ir_thermal"),
        (REPEAT, "probability", "p253_repeat_physical"),
        (UNION_PRETRAINED, "probability", "p300_union_pretrained_physical"),
    )
    for path, key, name in sources:
        artifact = np.load(path)
        append_expert(train, artifact[key], artifact["sample_ids"])
        names.append(name)

    group_report, group_prediction, group_probability, _ = crossfit(train)
    sequence_prediction, sequence_report = sequence_crossfit(
        train, group_prediction, group_probability
    )

    labels = np.concatenate([train[name]["labels"] for name in SPLITS])
    group = np.concatenate([group_prediction[name] for name in SPLITS])
    sequence = np.concatenate([sequence_prediction[name] for name in COHORTS])
    p255 = np.load(P255)
    p270 = np.load(P270)
    p255_prediction = np.concatenate(
        [p255[f"{name}_held_prediction"] for name in COHORTS]
    )
    p270_prediction = np.concatenate(
        [p270[f"{name}_held_prediction"] for name in COHORTS]
    )
    if not np.array_equal(
        labels,
        np.concatenate(
            [load_candidate_splits()[name].split.labels.astype(int) for name in COHORTS]
        ),
    ):
        raise RuntimeError("P301 cohort label order differs from the group order")

    group_correct = int(np.sum(group == labels))
    sequence_correct = int(np.sum(sequence == labels))
    p255_correct = int(np.sum(p255_prediction == labels))
    p270_correct = int(np.sum(p270_prediction == labels))
    report = {
        "stage": "P301_union_pretrained_group_sequence_audit",
        "status": "complete",
        "protocol": {
            "extra_training_rows_total": 122,
            "extra_thermal_available": 115,
            "balanced_extra_rows": 64,
            "extra_pretraining_source": "P300",
            "group_outer_crossfit": True,
            "sequence_recipe_frozen_from_p270": {
                "emission_probability_weight": EMISSION_WEIGHT,
                "transition_weight": TRANSITION_WEIGHT,
            },
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "experts": names,
        "group": {
            "correct": group_correct,
            "rows": int(len(labels)),
            "accuracy": float(np.mean(group == labels)),
            "p255_reference_correct": p255_correct,
            "net_vs_p255": group_correct - p255_correct,
            "cohorts": group_report,
        },
        "sequence": {
            "correct": sequence_correct,
            "rows": int(len(labels)),
            "accuracy": float(np.mean(sequence == labels)),
            "p270_reference_correct": p270_correct,
            "net_vs_p270": sequence_correct - p270_correct,
            "fold_nets_vs_group": [
                sequence_report[name]["held"]["net"] for name in COHORTS
            ],
            "cohorts": sequence_report,
        },
        "decision": "train_matched_test_head" if sequence_correct > p270_correct else "reject",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        labels=labels,
        group_prediction=group,
        sequence_prediction=sequence,
        **{f"{name}_held_prediction": sequence_prediction[name] for name in COHORTS},
        **{f"{name}_group_prediction": group_prediction[name] for name in COHORTS},
        **{f"{name}_held_probability": group_probability[name] for name in COHORTS},
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
