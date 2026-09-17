"""Pair-restricted nonvisual family arbitration for visual/P310 disagreements."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p386_visual_scope_nonvisual_competence_gate import select_visual_reference
from p388_visual_scope_nonvisual_family_top3_gate import FAMILIES, load_family_data


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p391_visual_pair_nonvisual_family_arbitration_oof_v1"
KS = (3, 5)
SCORE_THRESHOLDS = (-0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
ALT_VOTE_THRESHOLDS = (1, 2, 3, 4, 5)
GROUP_GAP_THRESHOLDS = (-0.50, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40)


def pair_context(part, visual_name, k):
    visual_probability = part["visual_references"][visual_name]
    alternative = visual_probability.argmax(axis=1)
    base = part["base"]
    rows = np.arange(len(base))
    group_top = np.argsort(-part["group_probability"], axis=1, kind="stable")[:, :k]
    return {
        "alternative": alternative,
        "disagreement": alternative != base,
        "alternative_in_group_topk": np.any(group_top == alternative[:, None], axis=1),
        "group_gap": part["group_probability"][rows, alternative] - part["group_probability"][rows, base],
    }


def competence(part, visual_name, k):
    context = pair_context(part, visual_name, k)
    probability = part["nonvisual_probability"]
    rows = np.arange(len(part["base"]))
    result = np.full((len(FAMILIES), 40), 0.5, dtype=np.float64)
    for family in range(len(FAMILIES)):
        choose_alternative = (
            probability[rows, family, context["alternative"]]
            > probability[rows, family, part["base"]]
        )
        choice = np.where(choose_alternative, context["alternative"], part["base"])
        for class_id in range(40):
            selected = (
                context["disagreement"]
                & context["alternative_in_group_topk"]
                & (choice == class_id)
                & (
                    (part["labels"] == part["base"])
                    | (part["labels"] == context["alternative"])
                )
            )
            success = int(np.sum(selected & (part["labels"] == class_id)))
            result[family, class_id] = (success + 1.0) / (int(selected.sum()) + 2.0)
    return result


def score(part, visual_name, k, reliability):
    context = pair_context(part, visual_name, k)
    probability = part["nonvisual_probability"]
    rows = np.arange(len(part["base"]))
    alt_score = np.zeros(len(rows), dtype=np.float64)
    base_score = np.zeros(len(rows), dtype=np.float64)
    alt_votes = np.zeros(len(rows), dtype=np.int16)
    for family in range(len(FAMILIES)):
        choose_alt = (
            probability[rows, family, context["alternative"]]
            > probability[rows, family, part["base"]]
        )
        alt_score += choose_alt * reliability[family, context["alternative"]]
        base_score += (~choose_alt) * reliability[family, part["base"]]
        alt_votes += choose_alt.astype(np.int16)
    context.update(
        {
            "score_gap": (alt_score - base_score) / len(FAMILIES),
            "alt_votes": alt_votes,
        }
    )
    return context


def concatenate(items):
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in items[0]}


def select_threshold(scored, labels, base, users, cohorts):
    best = None
    for score_threshold in SCORE_THRESHOLDS:
        for vote_threshold in ALT_VOTE_THRESHOLDS:
            for group_threshold in GROUP_GAP_THRESHOLDS:
                route = (
                    scored["disagreement"]
                    & scored["alternative_in_group_topk"]
                    & (scored["score_gap"] >= score_threshold)
                    & (scored["alt_votes"] >= vote_threshold)
                    & (scored["group_gap"] >= group_threshold)
                )
                output = base.copy()
                output[route] = scored["alternative"][route]
                gain = (output == labels).astype(int) - (base == labels).astype(int)
                per_cohort = {cohort: int(gain[cohorts == cohort].sum()) for cohort in np.unique(cohorts)}
                per_user = {user: int(gain[users == user].sum()) for user in np.unique(users)}
                rescue = int(np.sum(route & (base != labels) & (output == labels)))
                harm = int(np.sum(route & (base == labels) & (output != labels)))
                row = {
                    "score_threshold": score_threshold,
                    "alt_vote_threshold": vote_threshold,
                    "group_gap_threshold": group_threshold,
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
                valid = row["net"] > 0 and row["minimum_cohort_gain"] >= 0 and row["minimum_user_gain"] >= -1
                key = (
                    valid,
                    row["minimum_cohort_gain"],
                    row["net"],
                    row["rescue"],
                    -row["harm"],
                    -row["changed"],
                    score_threshold,
                    vote_threshold,
                    group_threshold,
                )
                if best is None or key > best[0]:
                    best = (key, row)
    if not best[0][0]:
        best[1].update({"score_threshold": 2.0, "alt_vote_threshold": 99, "group_gap_threshold": 2.0})
    return best[1]


def apply(scored, base, rule):
    route = (
        scored["disagreement"]
        & scored["alternative_in_group_topk"]
        & (scored["score_gap"] >= float(rule["score_threshold"]))
        & (scored["alt_votes"] >= int(rule["alt_vote_threshold"]))
        & (scored["group_gap"] >= float(rule["group_gap_threshold"]))
    )
    output = base.copy()
    output[route] = scored["alternative"][route]
    return output, route


def main():
    print(
        "P391 locks visual/P310 agreements and lets five nonvisual families arbitrate only "
        "between the P310 class and the pure-visual Top-1 alternative.",
        flush=True,
    )
    parts = load_family_data()
    report = {
        "stage": "P391_visual_pair_nonvisual_family_arbitration_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_role": "agreement lock, alternative class, and Top-K scope only",
            "nonvisual_families": {key: list(value) for key, value in FAMILIES.items()},
            "arbitration": "pair restricted to {P310 Top-1, pure-visual Top-1}; no third class",
            "family_class_competence_crossfit": True,
            "held_labels_used_for_visual_selection_competence_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        visual_name, visual_audit = select_visual_reference(parts, source_names)
        best = None
        for k in KS:
            source_scores = []
            for target_name, calibration_name in (
                (source_names[0], source_names[1]),
                (source_names[1], source_names[0]),
            ):
                source_scores.append(score(parts[target_name], visual_name, k, competence(parts[calibration_name], visual_name, k)))
            joined = concatenate(source_scores)
            labels = np.concatenate([parts[cohort]["labels"] for cohort in source_names])
            base = np.concatenate([parts[cohort]["base"] for cohort in source_names])
            users = np.concatenate([parts[cohort]["users"] for cohort in source_names])
            cohorts = np.concatenate([np.full(len(parts[cohort]["labels"]), cohort, dtype=object) for cohort in source_names])
            rule = select_threshold(joined, labels, base, users, cohorts)
            key = (
                rule["minimum_cohort_gain"] >= 0,
                rule["net"],
                rule["rescue"],
                -rule["harm"],
                -rule["changed"],
                -k,
            )
            if best is None or key > best[0]:
                best = (key, k, rule)
        _, k, rule = best
        reliability = np.minimum(
            competence(parts[source_names[0]], visual_name, k),
            competence(parts[source_names[1]], visual_name, k),
        )
        held_score = score(parts[held], visual_name, k, reliability)
        output, route = apply(held_score, parts[held]["base"], rule)
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
            "visual_reference_audit": visual_audit,
            "selected_visual_reference": visual_name,
            "selected_k": k,
            "source_rule": rule,
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
        print(json.dumps({"held": held, "visual": visual_name, "k": k, "rule": rule, "result": report["cohorts"][held]["held"]}), flush=True)
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
        "Run 1: pair-restricted five-family arbitration in visual/P310 disagreement rows.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
