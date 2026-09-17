"""Two-stage visual-consensus versus P310 disagreement gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p117_transductive_multicandidate_router import load_candidate_splits
from p361_bidirectional_existing_teacher_gate_oof import COHORTS, load_parts


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p382_visual_multimodal_disagreement_gate_oof_v1"
VISUAL_NAMES = (
    "p90_visual_equal",
    "p90_videomaev2_distilled_base",
    "p90_internvideo2_l_early_late",
    "p90_internvideo2_l_early_late_plus_k400",
    "p85_window_mean",
    "p85_early",
    "p86_drop_person",
    "p122_hand_object_all",
    "p123_dense24_group8",
    "p123_dense24_group8_ssv2",
    "p130_epic_slowfast",
    "p131_egovlp",
    "expanded_p122_pose",
    "expanded_p122_object",
    "expanded_p122_relation",
    "expanded_local_depth",
    "expanded_egovlp",
    "expanded_p142_token",
    "expanded_p144_hand_token",
    "expanded_p146_workspace_token",
    "expanded_p158_lavila_frame_token",
    "expanded_p123_old_ir_dense",
)
VOTE_THRESHOLDS = (0.30, 0.40, 0.50, 0.60)
GAP_THRESHOLDS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def build_parts():
    candidates = load_candidate_splits(
        full_visual_bank=True,
        legacy_visual_bank=True,
        hand_object_bank=True,
        vjepa_dense_bank=True,
        epic_bank=True,
        egovlp_bank=True,
        expanded_bank=True,
    )
    parts, _ = load_parts()
    for cohort in COHORTS:
        probability = np.stack(
            [candidates[cohort].candidates[name] for name in VISUAL_NAMES], axis=1
        ).astype(np.float32)
        average = probability.mean(axis=1)
        proposal = average.argmax(axis=1)
        teacher_prediction = probability.argmax(axis=2)
        parts[cohort]["visual_probability"] = probability
        parts[cohort]["visual_average"] = average
        parts[cohort]["proposal"] = proposal
        parts[cohort]["visual_vote"] = (teacher_prediction == proposal[:, None]).mean(axis=1)
        rows = np.arange(len(proposal))
        parts[cohort]["group_gap"] = (
            parts[cohort]["group_probability"][rows, proposal]
            - parts[cohort]["group_probability"][rows, parts[cohort]["base"]]
        )
        top5 = np.argsort(-parts[cohort]["group_probability"], axis=1, kind="stable")[:, :5]
        parts[cohort]["in_group_top5"] = np.any(top5 == proposal[:, None], axis=1)
        parts[cohort]["cohort"] = np.full(len(proposal), cohort, dtype=object)
    return parts


def concatenate(items):
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in (
            "ids", "users", "labels", "base", "proposal", "visual_vote",
            "group_gap", "in_group_top5", "cohort",
        )
    }


def hard_classes(source):
    result = []
    for class_id in range(40):
        selected = source["labels"] == class_id
        support = int(selected.sum())
        recall = float(np.mean(source["base"][selected] == class_id)) if support else 1.0
        if support >= 8 and recall < 0.80:
            result.append(class_id)
    return result


def select(source, difficult):
    best = None
    for vote_threshold in VOTE_THRESHOLDS:
        for gap_threshold in GAP_THRESHOLDS:
            route = (
                (source["proposal"] != source["base"])
                & np.isin(source["proposal"], difficult)
                & source["in_group_top5"]
                & (source["visual_vote"] >= vote_threshold)
                & (source["group_gap"] >= gap_threshold)
            )
            output = source["base"].copy()
            output[route] = source["proposal"][route]
            gain = (output == source["labels"]).astype(int) - (source["base"] == source["labels"]).astype(int)
            per_cohort = {cohort: int(gain[source["cohort"] == cohort].sum()) for cohort in np.unique(source["cohort"])}
            per_user = {user: int(gain[source["users"] == user].sum()) for user in np.unique(source["users"])}
            rescue = int(np.sum(route & (source["base"] != source["labels"]) & (output == source["labels"])))
            harm = int(np.sum(route & (source["base"] == source["labels"]) & (output != source["labels"])))
            row = {
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
                vote_threshold,
                gap_threshold,
            )
            if best is None or key > best[0]:
                best = (key, row)
    if not best[0][0]:
        best[1].update({"vote_threshold": 2.0, "group_gap_threshold": 2.0})
    return best[1]


def apply(part, difficult, rule):
    route = (
        (part["proposal"] != part["base"])
        & np.isin(part["proposal"], difficult)
        & part["in_group_top5"]
        & (part["visual_vote"] >= float(rule["vote_threshold"]))
        & (part["group_gap"] >= float(rule["group_gap_threshold"]))
    )
    output = part["base"].copy()
    output[route] = part["proposal"][route]
    return output, route


def main():
    print(
        "P382 tests source-defined hard classes behind a global visual/P310 disagreement "
        "filter, followed by visual vote and P307 Top-5 support thresholds.",
        flush=True,
    )
    parts = build_parts()
    report = {
        "stage": "P382_visual_multimodal_disagreement_gate_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "pure_visual_teachers": list(VISUAL_NAMES),
            "pure_visual_proposal": "arithmetic mean posterior Top-1",
            "scope": "source-defined recall<0.80 classes and proposal in P307 Top-5",
            "global_filter": "visual ensemble disagrees with P310",
            "thresholds": {"visual_vote": list(VOTE_THRESHOLDS), "p307_group_gap": list(GAP_THRESHOLDS)},
            "held_labels_used_for_class_or_threshold_selection": False,
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
        difficult = hard_classes(source)
        rule = select(source, difficult)
        output, route = apply(parts[held], difficult, rule)
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
            "source_hard_classes": difficult,
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
        print(json.dumps({"held": held, "hard": difficult, "rule": rule, "result": report["cohorts"][held]["held"]}), flush=True)
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
    np.savez_compressed(
        OUT / "oof_predictions.npz",
        labels=labels,
        base_prediction=base,
        prediction=prediction,
        **{f"{cohort}_held_prediction": outputs[index] for index, cohort in enumerate(COHORTS)},
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: visual-consensus disagreement scope plus source-defined hard classes and P307 support.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
