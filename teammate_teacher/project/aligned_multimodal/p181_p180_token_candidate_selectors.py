"""Source-user-safe P142/P144/P149 selectors on top of P180."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p117_transductive_multicandidate_router import (
    fit_score,
    load_candidate_splits,
    loso_scores,
    select_threshold,
)
from p134_frozen_repeat_consensus import probability_lookup
from p143_p142_source_user_safe_selector import build_features


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p181_p180_token_candidate_selectors_v1"
P180 = HERE / "runs/p180_sequence_micro_teacher_v1/oof_predictions.npz"
ALTERNATIVES = {
    "p142_all_token": HERE / "runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz",
    "p144_hand_interaction": HERE / "runs/p144_vjepa_hand_interaction_transformer_three_seed_v1/oof_predictions.npz",
    "p149_repeat_consistency": HERE / "runs/p149_vjepa_repeat_consistency_three_seed_v2/oof_predictions.npz",
}
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")


def main() -> None:
    data = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=True,
        hand_object_bank=True,
        vjepa_dense_bank=True,
        nonvisual_bank=True,
        hierarchical_bank=True,
        epic_bank=True,
        expanded_bank=True,
    )
    lookup = probability_lookup(data)
    with np.load(P180, allow_pickle=False) as saved:
        p180_ids = saved["sample_ids"].astype(str)
        p180_labels = saved["labels"].astype(np.int64)
        p180_prediction = saved["prediction"].astype(np.int64)
    p180_map = {
        sample_id: (int(label), int(prediction))
        for sample_id, label, prediction in zip(p180_ids, p180_labels, p180_prediction)
    }
    users = {
        sample_id: user
        for split in data.values()
        for sample_id, user in zip(
            split.split.sample_ids.astype(str), split.split.users.astype(str)
        )
    }
    alternatives = {}
    for name, path in ALTERNATIVES.items():
        with np.load(path, allow_pickle=False) as saved:
            alternatives[name] = {
                sample_id: probability
                for sample_id, probability in zip(
                    saved["sample_ids"].astype(str), saved["probability"].astype(np.float64)
                )
            }
    report = {
        "stage": "P181_P180_token_candidate_selectors",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "base": "P180",
            "eligibility": "source net > 0 and minimum source-user gain >= 0",
            "user_id_used_as_feature": False,
            "held_labels_used_for_selection": False,
            "test_labels_read": False,
        },
        "alternatives": {},
    }
    payload = {
        "sample_ids": p180_ids,
        "labels": p180_labels,
        "base_prediction": p180_prediction,
    }
    for alternative_name, alternative_map in alternatives.items():
        held_outputs = {}
        cohorts = {}
        for held_name in SPLITS:
            source_names = [name for name in SPLITS if name != held_name]
            source_ids = np.concatenate(
                [data[name].split.sample_ids.astype(str) for name in source_names]
            )
            source_labels = np.asarray([p180_map[value][0] for value in source_ids], dtype=np.int64)
            source_base = np.asarray([p180_map[value][1] for value in source_ids], dtype=np.int64)
            source_probability = np.stack([alternative_map[value] for value in source_ids])
            source_alternative = source_probability.argmax(axis=1).astype(np.int64)
            source_x = build_features(
                source_ids, source_base, source_alternative, source_probability, lookup
            )
            gain = (source_alternative == source_labels).astype(np.int8) - (
                source_base == source_labels
            ).astype(np.int8)
            source_users = np.asarray([users[value] for value in source_ids])
            nested = loso_scores(source_x, gain, source_users)
            selected = select_threshold(
                nested, gain, source_alternative != source_base, source_users
            )
            eligible = selected["net"] > 0 and selected["minimum_user_gain"] >= 0
            held_ids = data[held_name].split.sample_ids.astype(str)
            labels = np.asarray([p180_map[value][0] for value in held_ids], dtype=np.int64)
            base = np.asarray([p180_map[value][1] for value in held_ids], dtype=np.int64)
            probability = np.stack([alternative_map[value] for value in held_ids])
            alternative = probability.argmax(axis=1).astype(np.int64)
            held_x = build_features(held_ids, base, alternative, probability, lookup)
            score = fit_score(source_x, gain, held_x)
            route = eligible & (alternative != base) & (score >= float(selected["threshold"]))
            prediction = base.copy()
            prediction[route] = alternative[route]
            base_correct = base == labels
            final_correct = prediction == labels
            cohorts[held_name] = {
                "source_threshold": selected,
                "eligible": bool(eligible),
                "held": {
                    "base_correct": int(base_correct.sum()),
                    "correct": int(final_correct.sum()),
                    "net": int(final_correct.sum() - base_correct.sum()),
                    "rescue": int(np.sum(~base_correct & final_correct)),
                    "harm": int(np.sum(base_correct & ~final_correct)),
                    "changed": int(route.sum()),
                },
            }
            held_outputs[held_name] = prediction
        position = {value: row for row, value in enumerate(p180_ids)}
        prediction = p180_prediction.copy()
        for held_name in SPLITS:
            ids = data[held_name].split.sample_ids.astype(str)
            rows = np.asarray([position[value] for value in ids], dtype=np.int64)
            prediction[rows] = held_outputs[held_name]
        correct = int(np.sum(prediction == p180_labels))
        fold_nets = [int(cohorts[name]["held"]["net"]) for name in SPLITS]
        report["alternatives"][alternative_name] = {
            "cohorts": cohorts,
            "aggregate": {
                "rows": len(prediction),
                "base_correct": int(np.sum(p180_prediction == p180_labels)),
                "correct": correct,
                "accuracy": correct / len(prediction),
                "net_vs_p180": correct - int(np.sum(p180_prediction == p180_labels)),
                "fold_nets": fold_nets,
            },
        }
        payload[f"{alternative_name}_prediction"] = prediction
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
