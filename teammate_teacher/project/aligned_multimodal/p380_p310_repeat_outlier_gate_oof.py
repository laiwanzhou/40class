"""Repeat-consensus outlier correction on the current 30-teacher P310 base."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p134_frozen_repeat_consensus import repeat_proposal, select_rule
from p136_peer_support_repeat_gate import peer_candidate, select_rule as select_peer
from p173_vjepa_augmented_group_teacher import build_train_bank
from p255_repeat_augmented_physical_group import al
from p307_union_repeat_group_sequence_audit import SOURCES


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p380_p310_repeat_outlier_gate_oof_v1"
P310 = HERE / "runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz"
TRAIN_META = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")


def main():
    print(
        "P380 re-audits frozen repeat alignment with the complete 30-teacher bank and "
        "current P310 base; Test is not loaded.",
        flush=True,
    )
    train, names = build_train_bank()
    for path, key, name in SOURCES:
        archive = np.load(path)
        for cohort in COHORTS:
            split = train[cohort]
            probability = al(archive[key], archive["sample_ids"], split["ids"])
            split["bank"] = np.concatenate((split["bank"], probability[:, None, :]), axis=1)
        names.append(name)
    lookup = {
        sample_id: probability
        for cohort in COHORTS
        for sample_id, probability in zip(train[cohort]["ids"], train[cohort]["bank"], strict=True)
    }
    current = np.load(P310)
    labels = current["labels"].astype(int)
    base = current["prediction"].astype(int)
    label_lookup = {}
    base_lookup = {}
    offset = 0
    for cohort in COHORTS:
        rows = len(train[cohort]["ids"])
        for row, sample_id in enumerate(train[cohort]["ids"]):
            label_lookup[sample_id] = int(labels[offset + row])
            base_lookup[sample_id] = int(base[offset + row])
        offset += rows

    report = {
        "stage": "P380_P310_repeat_outlier_gate_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "expert_count": len(names),
            "repeat_geometry": "frozen anonymous date/session alignment",
            "candidates": ["all-expert repeat consensus", "peer-majority support"],
            "held_labels_used_for_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source_ids = np.concatenate([train[cohort]["ids"] for cohort in source_names])
        source_base = np.asarray([base_lookup[sample_id] for sample_id in source_ids])
        source_labels = np.asarray([label_lookup[sample_id] for sample_id in source_ids])
        consensus, _ = select_rule(source_ids, source_base, source_labels, lookup)
        peer_proposal, peer_scores, peer_count, _ = peer_candidate(
            source_ids, source_base, lookup, TRAIN_META
        )
        peer = select_peer(source_base, source_labels, peer_proposal, peer_scores, peer_count)
        selected_kind = "consensus" if consensus["net"] >= peer["net"] else "peer"

        held_ids = train[held]["ids"]
        held_base = np.asarray([base_lookup[sample_id] for sample_id in held_ids])
        held_labels = np.asarray([label_lookup[sample_id] for sample_id in held_ids])
        if selected_kind == "consensus":
            proposal, advantage, peers, grouping = repeat_proposal(
                held_ids, held_base, lookup, str(consensus["mode"]), TRAIN_META
            )
            route = (proposal != held_base) & (peers > 0) & (advantage >= consensus["threshold"])
            selected = consensus
        else:
            proposal, scores, peers, grouping = peer_candidate(
                held_ids, held_base, lookup, TRAIN_META
            )
            route = (
                (proposal != held_base)
                & (peers > 0)
                & (scores[:, int(peer["score_index"])] >= float(peer["threshold"]))
            )
            selected = peer
        output = held_base.copy()
        output[route] = proposal[route]
        outputs.append(output)
        report["cohorts"][held] = {
            "source": source_names,
            "selected_kind": selected_kind,
            "source_consensus": consensus,
            "source_peer": peer,
            "selected": selected,
            "held": {
                "rows": len(held_labels),
                "base_correct": int(np.sum(held_base == held_labels)),
                "correct": int(np.sum(output == held_labels)),
                "net": int(np.sum(output == held_labels) - np.sum(held_base == held_labels)),
                "changed": int(route.sum()),
                "rescue": int(np.sum(route & (held_base != held_labels) & (output == held_labels))),
                "harm": int(np.sum(route & (held_base == held_labels) & (output != held_labels))),
                "grouping": grouping,
            },
        }
        print(json.dumps({"held": held, "kind": selected_kind, "selected": selected, "result": report["cohorts"][held]["held"]}, ensure_ascii=False), flush=True)

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
        "Run 1: complete-bank repeat outlier gate over P310.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
