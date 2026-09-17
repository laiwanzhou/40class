"""Cross-fitted per-class visual-teacher competence voting over P310."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p382_visual_multimodal_disagreement_gate_oof import VISUAL_NAMES, build_parts


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p384_crossfit_class_teacher_competence_gate_v1"
SMALL_ACTIONS = (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39)
COMPETENCE_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
VOTE_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)
GROUP_GAP_THRESHOLDS = (-0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
MINIMUM_USER_GAIN = -1


def competence(part):
    predictions = part["visual_probability"].argmax(axis=2)
    result = np.full((predictions.shape[1], 40), 0.5, dtype=np.float64)
    for teacher in range(predictions.shape[1]):
        for class_id in range(40):
            selected = (predictions[:, teacher] == class_id) & (part["base"] != class_id)
            success = int(np.sum(selected & (part["labels"] == class_id)))
            result[teacher, class_id] = (success + 1.0) / (int(selected.sum()) + 2.0)
    return result


def score(part, reliability):
    predictions = part["visual_probability"].argmax(axis=2)
    class_score = np.zeros((len(predictions), 40), dtype=np.float64)
    vote = np.zeros((len(predictions), 40), dtype=np.float64)
    rows = np.arange(len(predictions))
    for teacher in range(predictions.shape[1]):
        proposed = predictions[:, teacher]
        class_score[rows, proposed] += reliability[teacher, proposed]
        vote[rows, proposed] += 1.0
    class_score /= predictions.shape[1]
    vote /= predictions.shape[1]
    proposal = class_score.argmax(axis=1)
    candidate_score = class_score[rows, proposal]
    candidate_vote = vote[rows, proposal]
    group_gap = (
        part["group_probability"][rows, proposal]
        - part["group_probability"][rows, part["base"]]
    )
    top5 = np.argsort(-part["group_probability"], axis=1, kind="stable")[:, :5]
    in_top5 = np.any(top5 == proposal[:, None], axis=1)
    global_disagreement = part["proposal"] != part["base"]
    return {
        "proposal": proposal,
        "competence": candidate_score,
        "vote": candidate_vote,
        "group_gap": group_gap,
        "in_top5": in_top5,
        "global_disagreement": global_disagreement,
    }


def concatenate(values):
    return {key: np.concatenate([value[key] for value in values], axis=0) for key in values[0]}


def select(source, labels, base, users, cohorts):
    best = None
    for competence_threshold in COMPETENCE_THRESHOLDS:
        for vote_threshold in VOTE_THRESHOLDS:
            for gap_threshold in GROUP_GAP_THRESHOLDS:
                route = (
                    source["global_disagreement"]
                    & source["in_top5"]
                    & np.isin(source["proposal"], SMALL_ACTIONS)
                    & (source["proposal"] != base)
                    & (source["competence"] >= competence_threshold)
                    & (source["vote"] >= vote_threshold)
                    & (source["group_gap"] >= gap_threshold)
                )
                output = base.copy()
                output[route] = source["proposal"][route]
                gain = (output == labels).astype(int) - (base == labels).astype(int)
                per_cohort = {cohort: int(gain[cohorts == cohort].sum()) for cohort in np.unique(cohorts)}
                per_user = {user: int(gain[users == user].sum()) for user in np.unique(users)}
                rescue = int(np.sum(route & (base != labels) & (output == labels)))
                harm = int(np.sum(route & (base == labels) & (output != labels)))
                row = {
                    "competence_threshold": competence_threshold,
                    "vote_threshold": vote_threshold,
                    "group_gap_threshold": gap_threshold,
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
                valid = row["net"] > 0 and row["minimum_cohort_gain"] >= 0 and row["minimum_user_gain"] >= MINIMUM_USER_GAIN
                key = (
                    valid,
                    row["minimum_cohort_gain"],
                    row["net"],
                    row["rescue"],
                    -row["harm"],
                    -row["changed"],
                    competence_threshold,
                    vote_threshold,
                    gap_threshold,
                )
                if best is None or key > best[0]:
                    best = (key, row)
    if not best[0][0]:
        best[1].update({"competence_threshold": 2.0, "vote_threshold": 2.0, "group_gap_threshold": 2.0})
    return best[1]


def apply(scored, base, rule):
    route = (
        scored["global_disagreement"]
        & scored["in_top5"]
        & np.isin(scored["proposal"], SMALL_ACTIONS)
        & (scored["proposal"] != base)
        & (scored["competence"] >= float(rule["competence_threshold"]))
        & (scored["vote"] >= float(rule["vote_threshold"]))
        & (scored["group_gap"] >= float(rule["group_gap_threshold"]))
    )
    output = base.copy()
    output[route] = scored["proposal"][route]
    return output, route


def main():
    print(
        "P384 estimates each visual teacher's correction precision per class by cross-cohort "
        "prediction, then gates competence-weighted candidates behind global disagreement.",
        flush=True,
    )
    parts = build_parts()
    report = {
        "stage": "P384_crossfit_class_teacher_competence_gate",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_teachers": list(VISUAL_NAMES),
            "competence": "Beta(1,1)-smoothed precision conditional on teacher Top-1=c and base!=c",
            "competence_cross_prediction": True,
            "scope": "pre-registered small actions, global visual/P310 disagreement, P307 Top-5",
            "held_labels_used_for_competence_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source_scores = []
        for target_name, calibration_name in (
            (source_names[0], source_names[1]),
            (source_names[1], source_names[0]),
        ):
            source_scores.append(score(parts[target_name], competence(parts[calibration_name])))
        joined = concatenate(source_scores)
        source_labels = np.concatenate([parts[cohort]["labels"] for cohort in source_names])
        source_base = np.concatenate([parts[cohort]["base"] for cohort in source_names])
        source_users = np.concatenate([parts[cohort]["users"] for cohort in source_names])
        source_cohorts = np.concatenate([np.full(len(parts[cohort]["labels"]), cohort, dtype=object) for cohort in source_names])
        rule = select(joined, source_labels, source_base, source_users, source_cohorts)
        held_reliability = np.minimum(competence(parts[source_names[0]]), competence(parts[source_names[1]]))
        held_score = score(parts[held], held_reliability)
        output, route = apply(held_score, parts[held]["base"], rule)
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
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
        print(json.dumps({"held": held, "rule": rule, "result": report["cohorts"][held]["held"]}), flush=True)
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
        "Run 1: cross-cohort per-class visual-teacher competence voting.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
