"""Outer-cross-fit target-class boosts from fixed existing-teacher families."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from p361_bidirectional_existing_teacher_gate_oof import COHORTS, load_parts, normalized
from p362_teacher_family_class_weight_oof import FAMILY_MEMBERS


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p363_target_class_family_boost_oof_v1"
BOOSTS = (0.25, 0.50, 0.75, 1.00, 1.50)


def family_parts():
    parts, teacher_names = load_parts()
    lookup = {name: index for index, name in enumerate(teacher_names)}
    family_names = []
    members_by_name = {}
    for family, members in FAMILY_MEMBERS.items():
        indices = [lookup[name] for name in members]
        for aggregation in ("arithmetic", "geometric"):
            name = f"{family}__{aggregation}"
            family_names.append(name)
            members_by_name[name] = list(members)
            for cohort in COHORTS:
                bank = parts[cohort]["bank"][:, indices, :]
                if aggregation == "arithmetic":
                    probability = normalized(bank.mean(axis=1))
                else:
                    logp = np.log(np.clip(bank, 1e-7, 1.0)).mean(axis=1)
                    probability = normalized(np.exp(logp - logp.max(axis=1, keepdims=True)))
                parts[cohort].setdefault("families", []).append(probability)
    for cohort in COHORTS:
        parts[cohort]["families"] = np.stack(parts[cohort]["families"], axis=1)
    return parts, family_names, members_by_name


def anchored_group_logits(part):
    logits = np.log(np.clip(part["group_probability"], 1e-7, 1.0)).copy()
    rows = np.arange(len(logits))
    maximum = logits.max(axis=1)
    logits[rows, part["base"]] = np.maximum(logits[rows, part["base"]], maximum + 1e-6)
    if not np.array_equal(logits.argmax(axis=1), part["base"]):
        raise RuntimeError("anchored logits do not reproduce P310")
    return logits


def candidate(part, family_index, target_class, boost):
    logits = anchored_group_logits(part)
    group = np.log(np.clip(part["group_probability"][:, target_class], 1e-7, 1.0))
    family = np.log(np.clip(part["families"][:, family_index, target_class], 1e-7, 1.0))
    # A positive family route may add support but never subtract it.
    delta = boost * np.maximum(family - group, 0.0)
    logits[:, target_class] += delta
    proposal = logits.argmax(axis=1)
    rows = np.arange(len(logits))
    margin = logits[rows, proposal] - logits[rows, part["base"]]
    return proposal, margin


def metrics(parts, source_names, proposals, target_class):
    changed = rescue = harm = 0
    per_cohort = {}
    per_user = {}
    for cohort in source_names:
        part = parts[cohort]
        proposal = proposals[cohort]
        route = (proposal == target_class) & (proposal != part["base"])
        gain = route.astype(int) * (
            (proposal == part["labels"]).astype(int)
            - (part["base"] == part["labels"]).astype(int)
        )
        changed += int(route.sum())
        rescue += int(np.sum(route & (part["base"] != part["labels"]) & (proposal == part["labels"])))
        harm += int(np.sum(route & (part["base"] == part["labels"]) & (proposal != part["labels"])))
        per_cohort[cohort] = int(gain.sum())
        for user in np.unique(part["users"]):
            per_user[str(user)] = per_user.get(str(user), 0) + int(gain[part["users"] == user].sum())
    return {
        "changed": changed,
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "minimum_cohort_gain": min(per_cohort.values()),
        "positive_cohorts": sum(value > 0 for value in per_cohort.values()),
        "minimum_user_gain": min(per_user.values()),
        "positive_users": sum(value > 0 for value in per_user.values()),
        "per_cohort": per_cohort,
        "per_user": per_user,
    }


def select(parts, source_names, family_names):
    selected = []
    for target_class in range(40):
        best = None
        for family_index, family_name in enumerate(family_names):
            for boost in BOOSTS:
                proposals = {
                    cohort: candidate(parts[cohort], family_index, target_class, boost)[0]
                    for cohort in source_names
                }
                result = metrics(parts, source_names, proposals, target_class)
                if (
                    result["changed"] < 3
                    or result["net"] < 2
                    or result["minimum_cohort_gain"] < 0
                    or result["positive_cohorts"] < 1
                    or result["positive_users"] < 2
                    or result["minimum_user_gain"] < -1
                    or result["harm"] > max(1, result["rescue"] // 3)
                ):
                    continue
                key = (
                    result["minimum_cohort_gain"],
                    result["positive_cohorts"],
                    result["net"],
                    result["rescue"] / result["changed"],
                    -result["harm"],
                    -result["changed"],
                    -boost,
                )
                record = {
                    "target_class": target_class,
                    "family_index": family_index,
                    "family_name": family_name,
                    "boost": boost,
                    **result,
                }
                if best is None or key > best[0]:
                    best = (key, record)
        if best is not None:
            selected.append(best[1])
    selected.sort(
        key=lambda rule: (
            rule["minimum_cohort_gain"], rule["positive_cohorts"], rule["net"],
            rule["rescue"] / rule["changed"], -rule["harm"],
        ),
        reverse=True,
    )
    return selected[:8]


def apply(part, rules):
    output = part["base"].copy()
    winning_rule = np.full(len(output), -1, dtype=int)
    winning_margin = np.zeros(len(output), dtype=float)
    for index, rule in enumerate(rules):
        proposal, margin = candidate(
            part, int(rule["family_index"]), int(rule["target_class"]), float(rule["boost"])
        )
        route = (proposal != part["base"]) & (margin > winning_margin)
        output[route] = proposal[route]
        winning_rule[route] = index
        winning_margin[route] = margin[route]
    return output, winning_rule, winning_margin


def main():
    print(
        "P363 tests smooth target-class evidence boosts from fixed teacher families; "
        "it does not use source-prediction pair identities.",
        flush=True,
    )
    parts, family_names, member_map = family_parts()
    report = {
        "stage": "P363_target_class_family_boost_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "family_definition_uses_labels": False,
            "selection_unit": "target class, family, positive boost",
            "source_prediction_pair_used": False,
            "boosts": list(BOOSTS),
            "outer_crossfit": True,
            "held_labels_used_for_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "families": member_map,
        "cohorts": {},
    }
    outputs = []
    rule_rows = []
    changed_rows = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        rules = select(parts, source_names, family_names)
        prediction, winning_rule, margin = apply(parts[held], rules)
        outputs.append(prediction)
        part = parts[held]
        changed = prediction != part["base"]
        rescue = changed & (part["base"] != part["labels"]) & (prediction == part["labels"])
        harm = changed & (part["base"] == part["labels"]) & (prediction != part["labels"])
        report["cohorts"][held] = {
            "source": source_names,
            "selected_rule_count": len(rules),
            "held": {
                "rows": len(prediction),
                "base_correct": int(np.sum(part["base"] == part["labels"])),
                "correct": int(np.sum(prediction == part["labels"])),
                "net": int(np.sum(prediction == part["labels"]) - np.sum(part["base"] == part["labels"])),
                "changed": int(changed.sum()),
                "rescue": int(rescue.sum()),
                "harm": int(harm.sum()),
            },
            "rules": rules,
        }
        for index, rule in enumerate(rules):
            rule_rows.append({"held_cohort": held, "rule_index": index, **rule})
        for row in np.flatnonzero(changed):
            rule = rules[winning_rule[row]]
            changed_rows.append({
                "held_cohort": held,
                "sample_id": part["ids"][row],
                "base_prediction": int(part["base"][row]),
                "prediction": int(prediction[row]),
                "label": int(part["labels"][row]),
                "gain": int(prediction[row] == part["labels"][row]) - int(part["base"][row] == part["labels"][row]),
                "family_name": rule["family_name"],
                "target_class": rule["target_class"],
                "boost": rule["boost"],
                "margin": float(margin[row]),
            })

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
    rule_fields = [
        "held_cohort", "rule_index", "target_class", "family_name", "boost",
        "changed", "rescue", "harm", "net", "minimum_cohort_gain",
        "positive_cohorts", "minimum_user_gain", "positive_users",
    ]
    with (OUT / "selected_rules.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rule_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rule_rows)
    change_fields = [
        "held_cohort", "sample_id", "base_prediction", "prediction", "label", "gain",
        "family_name", "target_class", "boost", "margin",
    ]
    with (OUT / "changed_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=change_fields)
        writer.writeheader()
        writer.writerows(changed_rows)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: target-class-only boosts from provenance-defined teacher families.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
