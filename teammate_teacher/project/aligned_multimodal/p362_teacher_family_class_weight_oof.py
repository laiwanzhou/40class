"""Class-conditional family reweighting over the existing P307 teacher bank.

Teacher families are fixed from model/modality provenance before labels are
examined.  Outer cross-fit then determines only where a family is reliable
enough to receive extra weight.  This script never loads Test data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from p361_bidirectional_existing_teacher_gate_oof import (
    ALPHAS,
    COHORTS,
    Candidate,
    apply_rules,
    candidate_outputs,
    concatenate,
    load_parts,
    normalized,
    select_rules,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p362_teacher_family_class_weight_oof_v1"

# These groups encode architectural/modality commonality only.  Overlap is
# intentional: a teacher may provide both temporal and physical evidence.
FAMILY_MEMBERS = {
    "visual_temporal": (
        "p85_teacher",
        "p85_head_early_logits",
        "p85_head_late_logits",
        "p85_head_window_mean_logits",
        "p85_head_early_late_logits",
        "p85_head_temporal_delta_logits",
        "p85_head_kinetics_logits",
        "p142_all_token",
    ),
    "visual_mechanism": (
        "p86_mechanism_baseline_logits",
        "p86_mechanism_drop_scene_logits",
        "p86_mechanism_drop_person_logits",
        "p86_mechanism_drop_workspace_logits",
        "p86_mechanism_drop_early_logits",
        "p86_mechanism_drop_late_logits",
        "p86_mechanism_swap_early_late_logits",
        "p86_mechanism_collapse_early_late_logits",
        "p86_mechanism_swap_person_workspace_logits",
        "p86_mechanism_collapse_view_identity_logits",
    ),
    "body_sensor": (
        "p89_safe_probability",
        "p12_skeleton",
        "p12_thermal",
        "a18_best_session",
        "p128_hierarchical_multimodal",
        "p238_physical_token",
    ),
    "semantic_interaction": (
        "p142_all_token",
        "p144_hand_interaction",
        "p128_hierarchical_multimodal",
        "p158_lavila_frame_token",
    ),
    "repeat_physical": (
        "p149_repeat_consistency",
        "p238_physical_token",
        "p231_ir_thermal",
        "p253_repeat_physical",
        "p306_union_repeat_physical",
    ),
    "object_state": (
        "p86_mechanism_drop_workspace_logits",
        "p12_thermal",
        "p144_hand_interaction",
        "p149_repeat_consistency",
        "p158_lavila_frame_token",
        "p231_ir_thermal",
    ),
}


def arithmetic_family(bank: np.ndarray, indices: list[int]) -> np.ndarray:
    return normalized(bank[:, indices, :].mean(axis=1))


def geometric_family(bank: np.ndarray, indices: list[int]) -> np.ndarray:
    logp = np.log(np.clip(bank[:, indices, :], 1e-7, 1.0)).mean(axis=1)
    probability = np.exp(logp - logp.max(axis=1, keepdims=True))
    return normalized(probability)


def main() -> None:
    print(
        "P362 tests whether provenance-defined teacher families provide more transferable "
        "class-conditional weight increases than individual teachers.",
        flush=True,
    )
    parts, teacher_names = load_parts()
    name_to_index = {name: index for index, name in enumerate(teacher_names)}
    missing = sorted(
        member
        for members in FAMILY_MEMBERS.values()
        for member in members
        if member not in name_to_index
    )
    if missing:
        raise RuntimeError(f"family teacher names missing from P307 bank: {missing}")

    family_names: list[str] = []
    family_members: dict[str, list[str]] = {}
    for family, members in FAMILY_MEMBERS.items():
        indices = [name_to_index[name] for name in members]
        for aggregation in ("arithmetic", "geometric"):
            family_name = f"{family}__{aggregation}"
            family_names.append(family_name)
            family_members[family_name] = list(members)
            for cohort in COHORTS:
                bank = parts[cohort]["bank"]
                probability = (
                    arithmetic_family(bank, indices)
                    if aggregation == "arithmetic"
                    else geometric_family(bank, indices)
                )
                parts[cohort].setdefault("family_bank", []).append(probability)
    for cohort in COHORTS:
        parts[cohort]["bank"] = np.stack(parts[cohort].pop("family_bank"), axis=1)

    candidates = [
        Candidate("positive_weight", index, family_name, alpha)
        for index, family_name in enumerate(family_names)
        for alpha in ALPHAS
    ]
    outputs = {
        cohort: candidate_outputs(parts[cohort], candidates) for cohort in COHORTS
    }
    report: dict[str, object] = {
        "stage": "P362_teacher_family_class_weight_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "family_definition_uses_labels": False,
            "family_count": len(FAMILY_MEMBERS),
            "aggregations": ["arithmetic", "geometric"],
            "alphas": list(ALPHAS),
            "outer_crossfit": True,
            "held_labels_used_for_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "families": family_members,
        "cohorts": {},
    }
    held_outputs: list[np.ndarray] = []
    rules_csv: list[dict[str, object]] = []
    changes_csv: list[dict[str, object]] = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source = concatenate([parts[cohort] for cohort in source_names])
        source_proposals = np.concatenate([outputs[cohort][0] for cohort in source_names], axis=0)
        source_scores = np.concatenate([outputs[cohort][1] for cohort in source_names], axis=0)
        cohort_ids = np.concatenate(
            [np.full(len(parts[cohort]["labels"]), cohort, dtype=object) for cohort in source_names]
        )
        rules = select_rules(source, source_proposals, source_scores, candidates, cohort_ids)
        prediction, selected_rule, strength = apply_rules(
            parts[held], outputs[held][0], outputs[held][1], rules
        )
        held_outputs.append(prediction)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        changed = prediction != base
        rescue = changed & (base != labels) & (prediction == labels)
        harm = changed & (base == labels) & (prediction != labels)
        report["cohorts"][held] = {
            "source": source_names,
            "selected_rule_count": len(rules),
            "held": {
                "rows": len(labels),
                "base_correct": int(np.sum(base == labels)),
                "correct": int(np.sum(prediction == labels)),
                "net": int(np.sum(prediction == labels) - np.sum(base == labels)),
                "changed": int(changed.sum()),
                "rescue": int(rescue.sum()),
                "harm": int(harm.sum()),
            },
            "rules": rules,
        }
        for rule_index, rule in enumerate(rules):
            rules_csv.append({"held_cohort": held, "rule_index": rule_index, **rule})
        for row in np.flatnonzero(changed):
            rule = rules[selected_rule[row]]
            changes_csv.append(
                {
                    "held_cohort": held,
                    "sample_id": parts[held]["ids"][row],
                    "base_prediction": int(base[row]),
                    "prediction": int(prediction[row]),
                    "label": int(labels[row]),
                    "gain": int(prediction[row] == labels[row]) - int(base[row] == labels[row]),
                    "family": rule["teacher_name"],
                    "alpha": rule["alpha"],
                    "strength": float(strength[row]),
                }
            )

    labels = np.concatenate([parts[cohort]["labels"] for cohort in COHORTS])
    base = np.concatenate([parts[cohort]["base"] for cohort in COHORTS])
    prediction = np.concatenate(held_outputs)
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
        "decision": "eligible_for_test_audit" if strict_pass else "reject_before_test",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "oof_predictions.npz",
        labels=labels,
        base_prediction=base,
        prediction=prediction,
        **{
            f"{cohort}_held_prediction": held_outputs[index]
            for index, cohort in enumerate(COHORTS)
        },
    )
    rule_fields = [
        "held_cohort", "rule_index", "base_class", "target_class", "teacher_name",
        "alpha", "threshold", "confusion_support", "changed", "rescue", "harm",
        "net", "minimum_cohort_gain", "positive_users",
    ]
    with (OUT / "selected_rules.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rule_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rules_csv)
    change_fields = [
        "held_cohort", "sample_id", "base_prediction", "prediction", "label",
        "gain", "family", "alpha", "strength",
    ]
    with (OUT / "changed_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=change_fields)
        writer.writeheader()
        writer.writerows(changes_csv)
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "notes.txt").write_text(
        "Run 1: fixed teacher-family consensus with class-conditional outer-cross-fit weighting.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
