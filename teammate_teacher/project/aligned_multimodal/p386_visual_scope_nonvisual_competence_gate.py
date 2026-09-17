"""Visual scope and Top-K, nonvisual per-class competence arbitration over P310."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from p117_transductive_multicandidate_router import load_candidate_splits
from p361_bidirectional_existing_teacher_gate_oof import COHORTS, load_parts


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p386_visual_scope_nonvisual_competence_gate_v1"
VISUAL_REFERENCES = {
    "p90_visual_equal": ("p90_visual_equal",),
    "p123_old_ir_dense": ("expanded_p123_old_ir_dense",),
    "strong_visual_mean": (
        "p90_visual_equal",
        "p90_videomaev2_distilled_base",
        "p90_internvideo2_l_early_late",
        "p90_internvideo2_l_early_late_plus_k400",
        "p123_dense24_group8",
        "expanded_p142_token",
        "expanded_p144_hand_token",
        "expanded_p146_workspace_token",
        "expanded_p123_old_ir_dense",
    ),
}
NONVISUAL_TEACHERS = (
    "p12_thermal_candidate",
    "p90_deep_imu",
    "p90_motionbert_3view",
    "expanded_thermal",
    "expanded_p12_imu",
    "expanded_motionbert_front",
    "expanded_skeleton_invariant",
    "a18_best_session",
    "p87_sequence",
    "p88_repeat",
    "p88_latent_prefix",
    "p128_hierarchical_multimodal",
)
KS = (3, 5)
COMPETENCE_THRESHOLDS = (0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30)
VOTE_THRESHOLDS = (1, 2, 3, 4, 5)
GROUP_GAP_THRESHOLDS = (-0.50, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40)
NONVISUAL_SUPPORT_K = 1


def load_data():
    candidates = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=True,
        hand_object_bank=True,
        vjepa_dense_bank=True,
        nonvisual_bank=True,
        hierarchical_bank=True,
        expanded_bank=True,
    )
    parts, _ = load_parts()
    for cohort in COHORTS:
        parts[cohort]["visual_references"] = {
            name: np.mean(
                np.stack([candidates[cohort].candidates[item] for item in members], axis=1),
                axis=1,
            ).astype(np.float32)
            for name, members in VISUAL_REFERENCES.items()
        }
        parts[cohort]["nonvisual_probability"] = np.stack(
            [candidates[cohort].candidates[name] for name in NONVISUAL_TEACHERS], axis=1
        ).astype(np.float32)
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
    return parts


def select_visual_reference(parts, source_names):
    best = None
    audits = []
    for name in VISUAL_REFERENCES:
        agreement_accuracies = []
        disagreement_errors = 0
        agreement_rows = 0
        for cohort in source_names:
            part = parts[cohort]
            visual = part["visual_references"][name].argmax(axis=1)
            agreement = visual == part["base"]
            agreement_accuracies.append(float(np.mean(part["base"][agreement] == part["labels"][agreement])))
            disagreement_errors += int(np.sum((~agreement) & (part["base"] != part["labels"])))
            agreement_rows += int(agreement.sum())
        row = {
            "name": name,
            "minimum_agreement_accuracy": min(agreement_accuracies),
            "mean_agreement_accuracy": float(np.mean(agreement_accuracies)),
            "disagreement_errors": disagreement_errors,
            "agreement_rows": agreement_rows,
        }
        audits.append(row)
        key = (
            row["minimum_agreement_accuracy"],
            row["mean_agreement_accuracy"],
            row["disagreement_errors"],
            row["agreement_rows"],
        )
        if best is None or key > best[0]:
            best = (key, name)
    return best[1], audits


def candidate_mask(part, visual_name, k):
    visual_top = np.argsort(-part["visual_references"][visual_name], axis=1, kind="stable")[:, :k]
    group_top = np.argsort(-part["group_probability"], axis=1, kind="stable")[:, :k]
    mask = np.zeros((len(visual_top), 40), dtype=bool)
    rows = np.arange(len(mask))[:, None]
    mask[rows, visual_top] = True
    mask[rows, group_top] = True
    return mask


def competence(part, visual_name, k):
    probability = part["nonvisual_probability"]
    order = np.argsort(-probability, axis=2, kind="stable")[:, :, :NONVISUAL_SUPPORT_K]
    allowed = candidate_mask(part, visual_name, k)
    visual_top1 = part["visual_references"][visual_name].argmax(axis=1)
    difficult = visual_top1 != part["base"]
    result = np.full((len(NONVISUAL_TEACHERS), 40), 0.5, dtype=np.float64)
    for teacher in range(len(NONVISUAL_TEACHERS)):
        for class_id in range(40):
            selected = (
                difficult
                & (part["base"] != class_id)
                & np.any(order[:, teacher] == class_id, axis=1)
                & allowed[:, class_id]
            )
            success = int(np.sum(selected & (part["labels"] == class_id)))
            result[teacher, class_id] = (success + 1.0) / (int(selected.sum()) + 2.0)
    return result


def score(part, visual_name, k, reliability):
    probability = part["nonvisual_probability"]
    order = np.argsort(-probability, axis=2, kind="stable")[:, :, :NONVISUAL_SUPPORT_K]
    allowed = candidate_mask(part, visual_name, k)
    class_score = np.zeros((len(probability), 40), dtype=np.float64)
    votes = np.zeros((len(probability), 40), dtype=np.int16)
    rows = np.arange(len(probability))
    for teacher in range(len(NONVISUAL_TEACHERS)):
        for rank in range(NONVISUAL_SUPPORT_K):
            proposed = order[:, teacher, rank]
            accepted = allowed[rows, proposed] & (proposed != part["base"])
            selected_rows = rows[accepted]
            selected_classes = proposed[accepted]
            rank_weight = (NONVISUAL_SUPPORT_K - rank) / NONVISUAL_SUPPORT_K
            class_score[selected_rows, selected_classes] += (
                reliability[teacher, selected_classes] * rank_weight
            )
            votes[selected_rows, selected_classes] += 1
    proposal = class_score.argmax(axis=1)
    candidate_score = class_score[rows, proposal] / len(NONVISUAL_TEACHERS)
    candidate_votes = votes[rows, proposal]
    has_vote = candidate_votes > 0
    group_gap = (
        part["group_probability"][rows, proposal]
        - part["group_probability"][rows, part["base"]]
    )
    visual_top1 = part["visual_references"][visual_name].argmax(axis=1)
    return {
        "proposal": proposal,
        "competence": candidate_score,
        "votes": candidate_votes,
        "group_gap": group_gap,
        "has_vote": has_vote,
        "visual_disagreement": visual_top1 != part["base"],
    }


def concatenate(values):
    return {key: np.concatenate([value[key] for value in values], axis=0) for key in values[0]}


def select_threshold(scored, labels, base, users, cohorts):
    best = None
    for competence_threshold in COMPETENCE_THRESHOLDS:
        for vote_threshold in VOTE_THRESHOLDS:
            for gap_threshold in GROUP_GAP_THRESHOLDS:
                route = (
                    scored["visual_disagreement"]
                    & scored["has_vote"]
                    & (scored["proposal"] != base)
                    & (scored["competence"] >= competence_threshold)
                    & (scored["votes"] >= vote_threshold)
                    & (scored["group_gap"] >= gap_threshold)
                )
                output = base.copy()
                output[route] = scored["proposal"][route]
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
                valid = row["net"] > 0 and row["minimum_cohort_gain"] >= 0 and row["minimum_user_gain"] >= -1
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
        best[1].update({"competence_threshold": 2.0, "vote_threshold": 99, "group_gap_threshold": 2.0})
    return best[1]


def apply(scored, base, rule):
    route = (
        scored["visual_disagreement"]
        & scored["has_vote"]
        & (scored["proposal"] != base)
        & (scored["competence"] >= float(rule["competence_threshold"]))
        & (scored["votes"] >= int(rule["vote_threshold"]))
        & (scored["group_gap"] >= float(rule["group_gap_threshold"]))
    )
    output = base.copy()
    output[route] = scored["proposal"][route]
    return output, route


def main():
    print(
        "P386 uses pure visual models only to define agreement and Top-K scope; all final "
        "candidate votes come from Thermal/IMU/Skeleton/Session/multimodal teachers.",
        flush=True,
    )
    parts = load_data()
    report = {
        "stage": "P386_visual_scope_nonvisual_competence_gate",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_role": "agreement scope and Top-K candidates only",
            "visual_references": {key: list(value) for key, value in VISUAL_REFERENCES.items()},
            "nonvisual_voters": list(NONVISUAL_TEACHERS),
            "nonvisual_competence": "cross-cohort Beta-smoothed teacher-by-class correction precision",
            "nonvisual_support_k": NONVISUAL_SUPPORT_K,
            "candidate_k": list(KS),
            "held_labels_used_for_visual_reference_competence_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    capability_rows = []
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
                source_scores.append(
                    score(parts[target_name], visual_name, k, competence(parts[calibration_name], visual_name, k))
                )
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
        left = competence(parts[source_names[0]], visual_name, k)
        right = competence(parts[source_names[1]], visual_name, k)
        reliability = np.minimum(left, right)
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
        for teacher_index, teacher in enumerate(NONVISUAL_TEACHERS):
            for class_id in range(40):
                capability_rows.append({
                    "held_cohort": held,
                    "teacher": teacher,
                    "class_id": class_id,
                    "minimum_source_competence": float(reliability[teacher_index, class_id]),
                    "source_a_competence": float(left[teacher_index, class_id]),
                    "source_b_competence": float(right[teacher_index, class_id]),
                })
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
    with (OUT / "teacher_class_competence.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(capability_rows[0]))
        writer.writeheader()
        writer.writerows(capability_rows)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: visual agreement/Top-K scope with exclusively nonvisual competence voting.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
