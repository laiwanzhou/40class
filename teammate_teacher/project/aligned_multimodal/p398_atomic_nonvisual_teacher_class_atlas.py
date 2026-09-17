"""Atomic outer-cross-fit atlas for each nonvisual teacher x small-action class."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p386_visual_scope_nonvisual_competence_gate import NONVISUAL_TEACHERS, load_data
from p389_nonvisual_teacher_class_branches_oof import (
    CANDIDATE_KS,
    TEACHER_SUPPORT_KS,
    candidate_mask,
    concatenate,
    threshold_values,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p398_atomic_nonvisual_teacher_class_atlas_v1"
SMALL_ACTIONS = (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39)


def select_atomic(source, teacher, target):
    probability = source["nonvisual_probability"][:, teacher, :]
    order = np.argsort(-probability, axis=1, kind="stable")
    rows = np.arange(len(source["base"]))
    teacher_gap = probability[:, target] - probability[rows, source["base"]]
    group_gap = source["group_probability"][:, target] - source["group_probability"][rows, source["base"]]
    best = None
    for candidate_k in CANDIDATE_KS:
        allowed, disagreement = candidate_mask(source, candidate_k)
        for support_k in TEACHER_SUPPORT_KS:
            scope = (
                disagreement
                & (source["base"] != target)
                & allowed[:, target]
                & np.any(order[:, :support_k] == target, axis=1)
            )
            if int(scope.sum()) < 2:
                continue
            for teacher_threshold in threshold_values(teacher_gap[scope]):
                for group_threshold in threshold_values(group_gap[scope]):
                    route = scope & (teacher_gap >= teacher_threshold) & (group_gap >= group_threshold)
                    gain = route.astype(int) * (
                        (source["labels"] == target).astype(int)
                        - (source["base"] == source["labels"]).astype(int)
                    )
                    per_cohort = {cohort: int(gain[source["cohort"] == cohort].sum()) for cohort in np.unique(source["cohort"])}
                    per_user = {user: int(gain[source["users"] == user].sum()) for user in np.unique(source["users"])}
                    rescue = int(np.sum(route & (source["base"] != source["labels"]) & (source["labels"] == target)))
                    harm = int(np.sum(route & (source["base"] == source["labels"])))
                    result = {
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
                    eligible = (
                        rescue >= 2
                        and harm == 0
                        and result["minimum_cohort_gain"] >= 0
                        and result["minimum_user_gain"] >= 0
                        and result["positive_users"] >= 2
                    )
                    key = (
                        eligible,
                        result["minimum_cohort_gain"],
                        result["net"],
                        result["rescue"],
                        -result["changed"],
                        -candidate_k,
                        -support_k,
                        float(teacher_threshold),
                        float(group_threshold),
                    )
                    if best is None or key > best[0]:
                        best = (key, result)
    if best is None or not best[0][0]:
        return None
    return best[1]


def apply_atomic(part, teacher, target, rule):
    probability = part["nonvisual_probability"][:, teacher, :]
    order = np.argsort(-probability, axis=1, kind="stable")
    rows = np.arange(len(part["base"]))
    allowed, disagreement = candidate_mask(part, int(rule["candidate_k"]))
    teacher_gap = probability[:, target] - probability[rows, part["base"]]
    group_gap = part["group_probability"][:, target] - part["group_probability"][rows, part["base"]]
    route = (
        disagreement
        & (part["base"] != target)
        & allowed[:, target]
        & np.any(order[:, : int(rule["teacher_support_k"])] == target, axis=1)
        & (teacher_gap >= float(rule["teacher_gap_threshold"]))
        & (group_gap >= float(rule["group_gap_threshold"]))
    )
    output = part["base"].copy()
    output[route] = target
    return output, route


def main():
    print(
        "P398 evaluates each fixed nonvisual teacher x pre-registered small-action class "
        "as an independent outer-cross-fit rule before any combination.",
        flush=True,
    )
    parts = load_data()
    for cohort in COHORTS:
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
    records = []
    for teacher, teacher_name in enumerate(NONVISUAL_TEACHERS):
        for target in SMALL_ACTIONS:
            fold_records = []
            for held in COHORTS:
                source_names = [cohort for cohort in COHORTS if cohort != held]
                source = concatenate([parts[cohort] for cohort in source_names])
                rule = select_atomic(source, teacher, target)
                if rule is None:
                    output = parts[held]["base"].copy()
                    route = np.zeros(len(output), dtype=bool)
                else:
                    output, route = apply_atomic(parts[held], teacher, target, rule)
                labels = parts[held]["labels"]
                base = parts[held]["base"]
                fold_records.append(
                    {
                        "held": held,
                        "rule": rule,
                        "net": int(np.sum(output == labels) - np.sum(base == labels)),
                        "changed": int(route.sum()),
                        "rescue": int(np.sum(route & (base != labels) & (output == labels))),
                        "harm": int(np.sum(route & (base == labels) & (output != labels))),
                    }
                )
            nets = [record["net"] for record in fold_records]
            stable = bool(
                all(net > 0 for net in nets)
                or (sum(net > 0 for net in nets) >= 2 and min(nets) >= -1)
            )
            records.append(
                {
                    "teacher_index": teacher,
                    "teacher": teacher_name,
                    "target_class": target,
                    "fold_nets": nets,
                    "total_net": sum(nets),
                    "total_changed": sum(record["changed"] for record in fold_records),
                    "total_rescue": sum(record["rescue"] for record in fold_records),
                    "total_harm": sum(record["harm"] for record in fold_records),
                    "positive_folds": sum(net > 0 for net in nets),
                    "stable_gate_pass": stable,
                    "folds": fold_records,
                }
            )
        print(json.dumps({"teacher": teacher_name, "completed_classes": len(SMALL_ACTIONS)}), flush=True)
    records.sort(
        key=lambda row: (
            row["stable_gate_pass"], row["positive_folds"], row["total_net"],
            row["total_rescue"], -row["total_harm"], -row["total_changed"],
        ),
        reverse=True,
    )
    stable = [record for record in records if record["stable_gate_pass"]]
    report = {
        "stage": "P398_atomic_nonvisual_teacher_class_atlas",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_role": "strong visual mean disagreement and Top-K scope only",
            "atomic_unit": "fixed nonvisual teacher x pre-registered small-action class",
            "source_rule": "zero harm, both source cohorts nonnegative, no source user regression, two positive users",
            "held_labels_used_for_source_rule": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "tested_atomic_rules": len(records),
        "stable_rule_count": len(stable),
        "stable_rules": stable,
        "top_rules": records[:30],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fields = (
        "teacher", "target_class", "fold_nets", "total_net", "total_changed",
        "total_rescue", "total_harm", "positive_folds", "stable_gate_pass",
    )
    with (OUT / "atomic_rule_atlas.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({**row, "fold_nets": ";".join(map(str, row["fold_nets"]))})
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: independent outer-cross-fit atlas before combining sparse teacher-class rules.\n"
        + json.dumps({"tested": len(records), "stable": len(stable)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tested": len(records), "stable": len(stable), "stable_rules": [{key: row[key] for key in ("teacher", "target_class", "fold_nets", "total_net", "total_rescue", "total_harm")} for row in stable]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
