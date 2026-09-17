"""Directed nonvisual teacher-by-class branches inside visual/P310 disagreement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p386_visual_scope_nonvisual_competence_gate import NONVISUAL_TEACHERS, load_data


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p389_nonvisual_teacher_class_branches_oof_v1"
VISUAL_REFERENCE = "strong_visual_mean"
CANDIDATE_KS = (3, 5)
TEACHER_SUPPORT_KS = (1, 3, 5)
MAX_RULES = 4


def concatenate(items):
    result = {}
    for key in (
        "ids", "users", "labels", "base", "group_probability", "nonvisual_probability", "cohort"
    ):
        result[key] = np.concatenate([item[key] for item in items], axis=0)
    result["visual_reference"] = np.concatenate(
        [item["visual_references"][VISUAL_REFERENCE] for item in items], axis=0
    )
    return result


def candidate_mask(part, k):
    visual = part["visual_reference"] if "visual_reference" in part else part["visual_references"][VISUAL_REFERENCE]
    visual_top = np.argsort(-visual, axis=1, kind="stable")[:, :k]
    group_top = np.argsort(-part["group_probability"], axis=1, kind="stable")[:, :k]
    mask = np.zeros((len(visual), 40), dtype=bool)
    rows = np.arange(len(mask))[:, None]
    mask[rows, visual_top] = True
    mask[rows, group_top] = True
    return mask, visual.argmax(axis=1) != part["base"]


def threshold_values(values):
    if len(values) == 0:
        return np.asarray([np.inf])
    return np.unique(np.concatenate(([-np.inf], np.quantile(values, (0.0, 0.25, 0.5, 0.75, 0.9)), [np.inf])))


def select_rules(source):
    probability = source["nonvisual_probability"]
    order = np.argsort(-probability, axis=2, kind="stable")
    rows = np.arange(len(source["base"]))
    best_by_target = {}
    audit_count = 0
    for candidate_k in CANDIDATE_KS:
        allowed, disagreement = candidate_mask(source, candidate_k)
        for teacher, teacher_name in enumerate(NONVISUAL_TEACHERS):
            teacher_probability = probability[:, teacher, :]
            for target_class in range(40):
                teacher_gap = teacher_probability[:, target_class] - teacher_probability[rows, source["base"]]
                group_gap = source["group_probability"][:, target_class] - source["group_probability"][rows, source["base"]]
                for support_k in TEACHER_SUPPORT_KS:
                    scope = (
                        disagreement
                        & (source["base"] != target_class)
                        & allowed[:, target_class]
                        & np.any(order[:, teacher, :support_k] == target_class, axis=1)
                    )
                    if int(scope.sum()) < 2:
                        continue
                    for teacher_threshold in threshold_values(teacher_gap[scope]):
                        for group_threshold in threshold_values(group_gap[scope]):
                            route = scope & (teacher_gap >= teacher_threshold) & (group_gap >= group_threshold)
                            gain = route.astype(int) * (
                                (source["labels"] == target_class).astype(int)
                                - (source["base"] == source["labels"]).astype(int)
                            )
                            per_cohort = {cohort: int(gain[source["cohort"] == cohort].sum()) for cohort in np.unique(source["cohort"])}
                            per_user = {user: int(gain[source["users"] == user].sum()) for user in np.unique(source["users"])}
                            rescue = int(np.sum(route & (source["base"] != source["labels"]) & (source["labels"] == target_class)))
                            harm = int(np.sum(route & (source["base"] == source["labels"])))
                            row = {
                                "teacher_index": teacher,
                                "teacher": teacher_name,
                                "target_class": target_class,
                                "candidate_k": candidate_k,
                                "teacher_support_k": support_k,
                                "teacher_gap_threshold": float(teacher_threshold),
                                "group_gap_threshold": float(group_threshold),
                                "changed": int(route.sum()),
                                "rescue": rescue,
                                "harm": harm,
                                "net": rescue - harm,
                                "minimum_cohort_gain": min(per_cohort.values()),
                                "minimum_user_gain": min(per_user.values()),
                                "positive_users": sum(value > 0 for value in per_user.values()),
                                "per_cohort": per_cohort,
                            }
                            audit_count += 1
                            valid = (
                                rescue >= 2
                                and harm == 0
                                and row["minimum_cohort_gain"] >= 1
                                and row["minimum_user_gain"] >= 0
                                and row["positive_users"] >= 2
                            )
                            key = (
                                valid,
                                row["minimum_cohort_gain"],
                                row["net"],
                                row["rescue"],
                                -row["changed"],
                                -candidate_k,
                                -support_k,
                                float(teacher_threshold),
                                float(group_threshold),
                            )
                            old = best_by_target.get(target_class)
                            if old is None or key > old[0]:
                                best_by_target[target_class] = (key, row)
    rules = [item[1] for item in best_by_target.values() if item[0][0]]
    rules.sort(
        key=lambda row: (
            row["minimum_cohort_gain"], row["net"], row["rescue"], -row["changed"]
        ),
        reverse=True,
    )
    return rules[:MAX_RULES], audit_count


def apply(part, rules):
    probability = part["nonvisual_probability"]
    order = np.argsort(-probability, axis=2, kind="stable")
    rows = np.arange(len(part["base"]))
    output = part["base"].copy()
    selected_rule = np.full(len(output), -1, dtype=int)
    best_strength = np.full(len(output), -np.inf, dtype=float)
    for rule_index, rule in enumerate(rules):
        target = int(rule["target_class"])
        teacher = int(rule["teacher_index"])
        allowed, disagreement = candidate_mask(part, int(rule["candidate_k"]))
        teacher_gap = probability[:, teacher, target] - probability[rows, teacher, part["base"]]
        group_gap = part["group_probability"][:, target] - part["group_probability"][rows, part["base"]]
        scope = (
            disagreement
            & (part["base"] != target)
            & allowed[:, target]
            & np.any(order[:, teacher, : int(rule["teacher_support_k"])] == target, axis=1)
            & (teacher_gap >= float(rule["teacher_gap_threshold"]))
            & (group_gap >= float(rule["group_gap_threshold"]))
        )
        strength = (
            teacher_gap - float(rule["teacher_gap_threshold"])
            + group_gap - float(rule["group_gap_threshold"])
        )
        route = scope & (strength > best_strength)
        output[route] = target
        selected_rule[route] = rule_index
        best_strength[route] = strength[route]
    return output, selected_rule


def main():
    print(
        "P389 learns source-safe nonvisual teacher-by-class branches inside the pure-visual/P310 "
        "disagreement region; one teacher may own several target classes.",
        flush=True,
    )
    parts = load_data()
    report = {
        "stage": "P389_nonvisual_teacher_class_branches_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_role": "strong visual mean disagreement and Top-K scope only",
            "nonvisual_teachers": list(NONVISUAL_TEACHERS),
            "rule_unit": "nonvisual teacher x target class",
            "candidate_k": list(CANDIDATE_KS),
            "teacher_support_k": list(TEACHER_SUPPORT_KS),
            "rule_requirements": "both source cohorts +1, zero harm, two positive users, no user regression",
            "held_labels_used_for_rule_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source = concatenate([parts[cohort] for cohort in source_names])
        rules, audit_count = select_rules(source)
        output, selected_rule = apply(parts[held], rules)
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        changed = output != base
        report["cohorts"][held] = {
            "source": source_names,
            "rules": rules,
            "source_audit_count": audit_count,
            "held": {
                "rows": len(labels),
                "base_correct": int(np.sum(base == labels)),
                "correct": int(np.sum(output == labels)),
                "net": int(np.sum(output == labels) - np.sum(base == labels)),
                "changed": int(changed.sum()),
                "rescue": int(np.sum(changed & (base != labels) & (output == labels))),
                "harm": int(np.sum(changed & (base == labels) & (output != labels))),
            },
        }
        print(json.dumps({"held": held, "rules": rules, "result": report["cohorts"][held]["held"]}, ensure_ascii=False), flush=True)
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
        "decision": "eligible_for_test_audit" if strict_pass else "reject_before_test",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "oof_predictions.npz", labels=labels, base_prediction=base, prediction=prediction)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: strict nonvisual teacher-by-class Top-K branches behind visual disagreement.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
