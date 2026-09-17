"""Outer-cross-fit harm filter for existing P179 sequence routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p117_transductive_multicandidate_router import one_hot, select_threshold
from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p182_sequence_route_harm_filter_v1"
P177 = HERE / "runs/p177_p128_vjepa_group_teacher_v1/predictions.npz"
P179 = HERE / "runs/p179_p177_soft_sequence_gate_v1/predictions.npz"
MICRO = HERE / "runs/p89_verified_micro_union_audit_v1/validation_predictions.npz"
A18 = HERE.parent / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
P142 = HERE / "runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz"
P144 = HERE / "runs/p144_vjepa_hand_interaction_transformer_three_seed_v1/oof_predictions.npz"
P149 = HERE / "runs/p149_vjepa_repeat_consistency_three_seed_v2/oof_predictions.npz"
P128 = HERE / "runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")


def align(values, source_ids, target_ids):
    lookup = {value: row for row, value in enumerate(source_ids.astype(str))}
    rows = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(values)[rows]


def probability_margin(probability):
    top = np.partition(probability, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def build_features(ids, base, sequence, probability, candidate_predictions):
    rows = np.arange(len(ids), dtype=np.int64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0)
    hard = np.stack(candidate_predictions, axis=1)
    scalar = np.column_stack(
        (
            probability[rows, sequence] - probability[rows, base],
            probability[rows, sequence],
            probability[rows, base],
            probability.max(axis=1),
            probability_margin(probability),
            np.mean(hard == sequence[:, None], axis=1),
            np.mean(hard == base[:, None], axis=1),
            sequence != base,
        )
    ).astype(np.float32)
    return np.concatenate(
        (
            probability.astype(np.float32),
            np.log(probability).astype(np.float32),
            one_hot(base),
            one_hot(sequence),
            scalar,
        ),
        axis=1,
    ).astype(np.float32)


def fit_score(train_x, gain, predict_x):
    decisive = gain != 0
    target = (gain[decisive] > 0).astype(np.int64)
    if len(np.unique(target)) < 2:
        return np.zeros(len(predict_x), dtype=np.float64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.03,
            solver="liblinear",
            max_iter=1200,
            class_weight="balanced",
        ),
    )
    model.fit(train_x[decisive], target)
    return model.predict_proba(predict_x)[:, 1]


def loso_score(x, gain, users):
    score = np.zeros(len(x), dtype=np.float64)
    for user in sorted(set(users.astype(str).tolist())):
        held = users.astype(str) == user
        score[held] = fit_score(x[~held], gain[~held], x[held])
    return score


def main() -> None:
    splits = load_splits()
    p177 = np.load(P177, allow_pickle=False)
    p179 = np.load(P179, allow_pickle=False)
    micro = np.load(MICRO, allow_pickle=False)
    sources = []
    for path, key in (
        (A18, "best_session_probability"),
        (P142, "probability"),
        (P144, "probability"),
        (P149, "probability"),
        (P128, "probabilities"),
    ):
        saved = np.load(path, allow_pickle=False)
        sources.append((saved["sample_ids"].astype(str), saved[key].argmax(axis=1).astype(np.int64)))
    prefix = {
        "H1_selection": "h1",
        "H2_confirmation": "h2",
        "H3_independent_fold0": "h3",
    }
    data = {}
    for name in SPLITS:
        ids = splits[name].sample_ids.astype(str)
        labels = splits[name].labels.astype(np.int64)
        base = p177[f"{name}_held_prediction"].astype(np.int64)
        sequence = p179[f"{name}_held_prediction"].astype(np.int64)
        probability = p177[f"{name}_held_probability"].astype(np.float64)
        candidates = [align(values, source_ids, ids) for source_ids, values in sources]
        micro_prediction = micro[f"{prefix[name]}_union"].astype(np.int64)
        data[name] = {
            "ids": ids,
            "labels": labels,
            "users": splits[name].users.astype(str),
            "base": base,
            "sequence": sequence,
            "micro": micro_prediction,
            "x": build_features(ids, base, sequence, probability, candidates),
        }
    reports = {}
    outputs = {}
    for held_name in SPLITS:
        source_names = [name for name in SPLITS if name != held_name]
        source = {
            key: np.concatenate([data[name][key] for name in source_names])
            for key in ("ids", "labels", "users", "base", "sequence", "micro", "x")
        }
        source_route = source["sequence"] != source["base"]
        gain = (source["sequence"] == source["labels"]).astype(np.int8) - (
            source["base"] == source["labels"]
        ).astype(np.int8)
        nested = loso_score(source["x"], gain, source["users"])
        selected = select_threshold(nested, gain, source_route, source["users"])
        eligible = selected["net"] > 0 and selected["minimum_user_gain"] >= 0
        held = data[held_name]
        score = fit_score(source["x"], gain, held["x"])
        original_route = held["sequence"] != held["base"]
        kept = eligible & original_route & (score >= float(selected["threshold"]))
        guarded = held["base"].copy()
        guarded[kept] = held["sequence"][kept]
        prediction = np.where(
            guarded != held["base"],
            guarded,
            np.where(held["micro"] != held["base"], held["micro"], held["base"]),
        )
        base_correct = held["base"] == held["labels"]
        final_correct = prediction == held["labels"]
        reports[held_name] = {
            "source_threshold": selected,
            "eligible": bool(eligible),
            "held": {
                "base_correct": int(base_correct.sum()),
                "correct": int(final_correct.sum()),
                "net": int(final_correct.sum() - base_correct.sum()),
                "original_sequence_routes": int(original_route.sum()),
                "kept_sequence_routes": int(kept.sum()),
                "removed_sequence_routes": int(original_route.sum() - kept.sum()),
                "rescue": int(np.sum(~base_correct & final_correct)),
                "harm": int(np.sum(base_correct & ~final_correct)),
            },
        }
        outputs[held_name] = prediction
    labels = np.concatenate([data[name]["labels"] for name in SPLITS])
    base = np.concatenate([data[name]["base"] for name in SPLITS])
    prediction = np.concatenate([outputs[name] for name in SPLITS])
    correct = int(np.sum(prediction == labels))
    base_correct = int(np.sum(base == labels))
    report = {
        "stage": "P182_P179_sequence_route_harm_filter",
        "status": "passed" if correct > 2185 else "rejected_keep_p180",
        "protocol": {
            "may_add_sequence_routes": False,
            "source_LOSO_threshold": True,
            "user_id_used_as_feature": False,
            "test_labels_read": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": len(labels),
            "base_correct": base_correct,
            "correct": correct,
            "accuracy": correct / len(labels),
            "net_vs_p177": correct - base_correct,
            "p180_correct_to_beat": 2185,
            "fold_nets": [int(reports[name]["held"]["net"]) for name in SPLITS],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        sample_ids=np.concatenate([data[name]["ids"] for name in SPLITS]),
        labels=labels,
        base_prediction=base,
        prediction=prediction,
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
