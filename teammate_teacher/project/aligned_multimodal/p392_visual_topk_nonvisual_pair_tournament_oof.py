"""Nonvisual family tournament over the visual/P307 Top-K intersection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p386_visual_scope_nonvisual_competence_gate import select_visual_reference
from p388_visual_scope_nonvisual_family_top3_gate import FAMILIES, load_family_data
from p391_visual_pair_nonvisual_family_arbitration_oof import (
    ALT_VOTE_THRESHOLDS,
    GROUP_GAP_THRESHOLDS,
    SCORE_THRESHOLDS,
    apply,
    concatenate,
    select_threshold,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p392_visual_topk_nonvisual_pair_tournament_oof_v1"
KS = (3, 5)


def intersection_mask(part, visual_name, k):
    visual = part["visual_references"][visual_name]
    visual_top = np.argsort(-visual, axis=1, kind="stable")[:, :k]
    group_top = np.argsort(-part["group_probability"], axis=1, kind="stable")[:, :k]
    visual_mask = np.zeros((len(visual), 40), dtype=bool)
    group_mask = np.zeros((len(visual), 40), dtype=bool)
    rows = np.arange(len(visual))[:, None]
    visual_mask[rows, visual_top] = True
    group_mask[rows, group_top] = True
    return visual_mask & group_mask


def competence(part, visual_name, k):
    candidate = intersection_mask(part, visual_name, k)
    visual_top1 = part["visual_references"][visual_name].argmax(axis=1)
    disagreement = visual_top1 != part["base"]
    probability = part["nonvisual_probability"]
    rows = np.arange(len(part["base"]))
    result = np.full((len(FAMILIES), 40), 0.5, dtype=np.float64)
    for family in range(len(FAMILIES)):
        for class_id in range(40):
            support = probability[:, family, class_id] > probability[rows, family, part["base"]]
            selected = disagreement & (part["base"] != class_id) & candidate[:, class_id] & support
            success = int(np.sum(selected & (part["labels"] == class_id)))
            result[family, class_id] = (success + 1.0) / (int(selected.sum()) + 2.0)
    return result


def score(part, visual_name, k, reliability):
    candidate = intersection_mask(part, visual_name, k)
    visual_top1 = part["visual_references"][visual_name].argmax(axis=1)
    disagreement = visual_top1 != part["base"]
    probability = part["nonvisual_probability"]
    rows = np.arange(len(part["base"]))
    score_gap = np.full((len(rows), 40), -np.inf, dtype=np.float64)
    vote_count = np.zeros((len(rows), 40), dtype=np.int16)
    for class_id in range(40):
        valid = candidate[:, class_id] & (part["base"] != class_id)
        if not valid.any():
            continue
        alt_score = np.zeros(len(rows), dtype=np.float64)
        base_score = np.zeros(len(rows), dtype=np.float64)
        for family in range(len(FAMILIES)):
            choose_alt = probability[:, family, class_id] > probability[rows, family, part["base"]]
            alt_score += choose_alt * reliability[family, class_id]
            base_score += (~choose_alt) * reliability[family, part["base"]]
            vote_count[:, class_id] += choose_alt.astype(np.int16)
        score_gap[valid, class_id] = ((alt_score - base_score) / len(FAMILIES))[valid]
    proposal = score_gap.argmax(axis=1)
    candidate_score = score_gap[rows, proposal]
    candidate_votes = vote_count[rows, proposal]
    group_gap = (
        part["group_probability"][rows, proposal]
        - part["group_probability"][rows, part["base"]]
    )
    return {
        "alternative": proposal,
        "disagreement": disagreement,
        "alternative_in_group_topk": np.isfinite(candidate_score),
        "score_gap": candidate_score,
        "alt_votes": candidate_votes,
        "group_gap": group_gap,
    }


def main():
    print(
        "P392 locks visual/P310 agreements and runs a five-family pairwise tournament over "
        "classes shared by the pure-visual and P307 Top-K sets.",
        flush=True,
    )
    parts = load_family_data()
    report = {
        "stage": "P392_visual_TopK_nonvisual_pair_tournament_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_role": "agreement lock and Top-K candidate set only",
            "candidate_set": "intersection of pure-visual Top-K and P307 Top-K",
            "nonvisual_families": {key: list(value) for key, value in FAMILIES.items()},
            "arbitration": "each candidate versus P310 base, then highest competence-weighted family score",
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
        "Run 1: nonvisual family pair tournament over visual/P307 Top-K intersection.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
