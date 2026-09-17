"""Conservative filter that may only remove frozen A18 confidence-gap routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p169_relative_reliability_harm_guard import (
    OUTPUT as P169_OUTPUT,
    SPLITS,
    apply_tail,
    concatenate,
    fit_score,
    load_oof,
    loso_score,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "runs/p170_conservative_a18_harm_filter_v1"
FROZEN = REPO / "runs/a18_p89_selective_replacement_v1/crossfit_predictions.npz"
SIZES = (663, 834, 973)


def split_masks():
    with np.load(FROZEN, allow_pickle=False) as saved:
        values = saved["confidence_gap_only_replacement_mask"].astype(bool)
    output = {}
    low = 0
    for name, size in zip(SPLITS, SIZES, strict=True):
        output[name] = values[low : low + size]
        low += size
    return output


def current_prediction(part, mask):
    guarded = part["base"].copy()
    guarded[mask] = part["a18"][mask]
    return apply_tail(part["base"], guarded, part["group"], part["micro"])


def select_filter(source, source_mask, score):
    reference = current_prediction(source, source_mask)
    reference_correct = reference == source["labels"]
    values = np.unique(
        np.concatenate(
            (
                np.asarray((-np.inf,)),
                np.arange(0.30, 0.901, 0.025),
                np.quantile(score[source_mask], np.linspace(0.1, 0.9, 9))
                if source_mask.any()
                else np.asarray((np.inf,)),
            )
        )
    )
    candidates = []
    for threshold in values:
        kept = source_mask & (score >= threshold)
        prediction = current_prediction(source, kept)
        correct = prediction == source["labels"]
        per_user = {
            user: int(
                np.sum(correct[source["users"] == user])
                - np.sum(reference_correct[source["users"] == user])
            )
            for user in sorted(set(source["users"].tolist()))
        }
        candidates.append(
            {
                "threshold": float(threshold),
                "kept_routes": int(kept.sum()),
                "removed_routes": int(source_mask.sum() - kept.sum()),
                "correct": int(correct.sum()),
                "reference_correct": int(reference_correct.sum()),
                "delta_correct": int(correct.sum() - reference_correct.sum()),
                "minimum_user_delta": int(min(per_user.values())),
                "per_user_delta": per_user,
            }
        )
    eligible = [
        row
        for row in candidates
        if row["delta_correct"] > 0 and row["minimum_user_delta"] >= 0
    ]
    if not eligible:
        return next(row for row in candidates if np.isneginf(row["threshold"]))
    return max(
        eligible,
        key=lambda row: (
            row["delta_correct"],
            -row["removed_routes"],
            row["kept_routes"],
        ),
    )


def main() -> None:
    data = load_oof()
    masks = split_masks()
    reports = {}
    predictions = {}
    for held_name in SPLITS:
        source_names = [name for name in SPLITS if name != held_name]
        source_parts = [data[name] for name in source_names]
        source = concatenate(source_parts)
        source_mask = np.concatenate([masks[name] for name in source_names])
        gain = (source["a18"] == source["labels"]).astype(np.int8) - (
            source["base"] == source["labels"]
        ).astype(np.int8)
        nested = loso_score(source["x"], gain, source["users"])
        selected = select_filter(source, source_mask, nested)
        held = data[held_name]
        held_gain = gain
        held_score = fit_score(source["x"], held_gain, held["x"])
        threshold = float(selected["threshold"])
        kept = masks[held_name] if np.isneginf(threshold) else (
            masks[held_name] & (held_score >= threshold)
        )
        prediction = current_prediction(held, kept)
        base_correct = held["base"] == held["labels"]
        final_correct = prediction == held["labels"]
        reports[held_name] = {
            "source_cohorts": source_names,
            "source_filter": selected,
            "held": {
                "base_correct": int(base_correct.sum()),
                "correct": int(final_correct.sum()),
                "net": int(final_correct.sum() - base_correct.sum()),
                "kept_a18_routes": int(kept.sum()),
                "removed_a18_routes": int(masks[held_name].sum() - kept.sum()),
                "changed": int(np.sum(prediction != held["base"])),
            },
        }
        predictions[held_name] = prediction
    all_data = concatenate([data[name] for name in SPLITS])
    prediction = np.concatenate([predictions[name] for name in SPLITS])
    correct = int(np.sum(prediction == all_data["labels"]))
    base_correct = int(np.sum(all_data["base"] == all_data["labels"]))
    folds = [int(reports[name]["held"]["net"]) for name in SPLITS]
    report = {
        "stage": "P170_conservative_A18_harm_filter",
        "status": "passed" if correct > 2146 and min(folds) > 0 else "rejected_keep_p168",
        "protocol": {
            "may_add_A18_routes": False,
            "default_if_source_filter_unstable": "keep all frozen confidence-gap routes",
            "user_id_used_as_feature": False,
            "test_labels_read": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": len(prediction),
            "base_correct": base_correct,
            "correct": correct,
            "accuracy": correct / len(prediction),
            "net_vs_p89": correct - base_correct,
            "fold_nets": folds,
            "p168_correct_to_beat": 2146,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        sample_ids=all_data["ids"],
        labels=all_data["labels"],
        base_prediction=all_data["base"],
        prediction=prediction,
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
