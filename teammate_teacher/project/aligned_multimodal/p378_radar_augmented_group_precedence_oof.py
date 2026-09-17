"""Add the existing small Radar expert to P307 and audit P310-style precedence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p165_deployable_group_teacher import SPLITS, crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank
from p255_repeat_augmented_physical_group import al
from p307_union_repeat_group_sequence_audit import SOURCES


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p378_radar_augmented_group_precedence_oof_v1"
RADAR = HERE / "runs/p89_radar_temporal_expert_v1/oof_logits.npz"
P307 = HERE / "runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz"
P310 = HERE / "runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
RADAR_RESIDUAL_WEIGHT = 0.25


def radar_key(sample_id):
    action, user, trial = str(sample_id).split("/")
    return int(action.split("_", 1)[0]), user, trial


def current_key(sample_id):
    _, class_code, user, trial = str(sample_id).split("__")
    return int(class_code[1:]), user, trial


def main():
    print(
        "P378 adds the existing low-capacity Radar expert at fixed 0.25 residual strength "
        "and audits P310-style new-group precedence.",
        flush=True,
    )
    train, names = build_train_bank()
    for path, key, name in SOURCES:
        archive = np.load(path)
        for cohort in SPLITS:
            split = train[cohort]
            probability = al(archive[key], archive["sample_ids"], split["ids"])
            split["bank"] = np.concatenate((split["bank"], probability[:, None, :]), axis=1)
        names.append(name)

    radar = np.load(RADAR)
    radar_probability = np.exp(radar["radar_logits"].astype(float))
    radar_probability /= radar_probability.sum(axis=1, keepdims=True)
    lookup = {radar_key(sample_id): index for index, sample_id in enumerate(radar["sample_ids"].astype(str))}
    availability = {}
    for cohort in SPLITS:
        split = train[cohort]
        weak = split["bank"][:, 0, :].astype(float).copy()
        available = np.zeros(len(split["ids"]), dtype=bool)
        for row, sample_id in enumerate(split["ids"]):
            position = lookup.get(current_key(sample_id))
            if position is None:
                continue
            available[row] = True
            weak[row] = (
                (1.0 - RADAR_RESIDUAL_WEIGHT) * weak[row]
                + RADAR_RESIDUAL_WEIGHT * radar_probability[position]
            )
        weak /= weak.sum(axis=1, keepdims=True)
        split["bank"] = np.concatenate((split["bank"], weak[:, None, :]), axis=1)
        availability[cohort] = available
    names.append("p89_radar_temporal_residual025")

    group_report, group_prediction, group_probability, thresholds = crossfit(train)
    old = np.load(P307)
    current = np.load(P310)
    outputs = []
    report = {
        "stage": "P378_Radar_augmented_group_precedence_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "group": "P307 30-teacher bank plus existing P89 Radar expert",
            "radar_residual_weight": RADAR_RESIDUAL_WEIGHT,
            "radar_missing_fallback": "P89 safe probability",
            "precedence": "new Radar group replaces P310 only where it differs from old P307 group",
            "held_labels_used_for_group_fit_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "experts": names,
        "cohorts": {},
    }
    offset = 0
    for cohort in COHORTS:
        labels = train[cohort]["labels"].astype(int)
        rows = len(labels)
        base = current["prediction"][offset : offset + rows].astype(int)
        old_group = old[f"{cohort}_group_prediction"].astype(int)
        new_group = group_prediction[cohort].astype(int)
        route = new_group != old_group
        output = base.copy()
        output[route] = new_group[route]
        outputs.append(output)
        changed = output != base
        report["cohorts"][cohort] = {
            "radar_available": int(availability[cohort].sum()),
            "new_group_correct": int(np.sum(new_group == labels)),
            "old_group_correct": int(np.sum(old_group == labels)),
            "new_group_net_vs_old_group": int(np.sum(new_group == labels) - np.sum(old_group == labels)),
            "group_report": group_report[cohort],
            "precedence": {
                "rows": rows,
                "base_correct": int(np.sum(base == labels)),
                "correct": int(np.sum(output == labels)),
                "net": int(np.sum(output == labels) - np.sum(base == labels)),
                "changed": int(changed.sum()),
                "rescue": int(np.sum(changed & (base != labels) & (output == labels))),
                "harm": int(np.sum(changed & (base == labels) & (output != labels))),
            },
        }
        offset += rows
    labels = current["labels"].astype(int)
    base = current["prediction"].astype(int)
    prediction = np.concatenate(outputs)
    fold_nets = [report["cohorts"][cohort]["precedence"]["net"] for cohort in COHORTS]
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
        **{f"{cohort}_held_probability": group_probability[cohort] for cohort in COHORTS},
        **{f"{cohort}_held_group_prediction": group_prediction[cohort] for cohort in COHORTS},
        **{f"{cohort}_radar_available": availability[cohort] for cohort in COHORTS},
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: existing Radar expert added with fixed 0.25 residual and P310 precedence.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": report["aggregate"], "cohorts": {cohort: report["cohorts"][cohort]["precedence"] for cohort in COHORTS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
