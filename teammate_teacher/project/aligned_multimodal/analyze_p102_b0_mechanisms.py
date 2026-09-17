"""Mechanism audit for the negative P102-B0 candidate reranker."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_p102_session_closure import load_npz, true_rank


HERE = Path(__file__).resolve().parent
DEFAULT_B0 = HERE / "runs/p102_b0_candidate_reranker_oof_v1/b0_oof_predictions.npz"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_OUTPUT = HERE / "runs/p102_b0_candidate_reranker_oof_v1/mechanism_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0)
    parser.add_argument("--hard-set", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def transition_rows(
    labels: np.ndarray, a_prediction: np.ndarray, b_prediction: np.ndarray, selected: np.ndarray
) -> list[dict[str, int]]:
    counts = Counter(
        (int(labels[row]), int(a_prediction[row]), int(b_prediction[row]))
        for row in np.flatnonzero(selected)
    )
    return [
        {"true": true, "a_prediction": a, "b_prediction": b, "rows": count}
        for (true, a, b), count in counts.most_common(30)
    ]


def outcome_breakdown(
    labels: np.ndarray,
    a_prediction: np.ndarray,
    b_prediction: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    a_correct = a_prediction == labels
    b_correct = b_prediction == labels
    changed = a_prediction != b_prediction
    rescue = selected & (~a_correct) & b_correct
    harm = selected & a_correct & (~b_correct)
    wrong_to_wrong = selected & (~a_correct) & (~b_correct) & changed
    return {
        "rows": int(selected.sum()),
        "changed": int(np.sum(selected & changed)),
        "rescue": int(rescue.sum()),
        "harm": int(harm.sum()),
        "net": int(rescue.sum() - harm.sum()),
        "wrong_to_wrong": int(wrong_to_wrong.sum()),
        "correct_change_precision": float(rescue.sum() / max(np.sum(selected & changed), 1)),
        "top_transitions": transition_rows(labels, a_prediction, b_prediction, selected & changed),
    }


def metric_by_group(values: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float | None]:
    return {
        name: float(values[selected].mean()) if selected.any() else None
        for name, selected in masks.items()
    }


def per_value_net(
    values: np.ndarray,
    labels: np.ndarray,
    a_prediction: np.ndarray,
    b_prediction: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(set(values.tolist()), key=str):
        selected = values == value
        result = outcome_breakdown(labels, a_prediction, b_prediction, selected)
        rows.append({"value": str(value), **{key: result[key] for key in ("rows", "changed", "rescue", "harm", "net", "wrong_to_wrong")}})
    return rows


def duplicate_audit(
    session_id: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    a_prediction: np.ndarray,
) -> dict[str, Any]:
    duplicate_sessions = 0
    duplicate_rows = np.zeros(len(prediction), dtype=bool)
    for value in sorted(set(map(int, session_id[session_id >= 0]))):
        selected = np.flatnonzero(session_id == value)
        if len(np.unique(prediction[selected])) < len(selected):
            duplicate_sessions += 1
            duplicate_rows[selected] = True
    rescue = (a_prediction != labels) & (prediction == labels)
    harm = (a_prediction == labels) & (prediction != labels)
    return {
        "duplicate_sessions": duplicate_sessions,
        "rows_in_duplicate_sessions": int(duplicate_rows.sum()),
        "rescues_in_duplicate_sessions": int(np.sum(rescue & duplicate_rows)),
        "harms_in_duplicate_sessions": int(np.sum(harm & duplicate_rows)),
    }


def main() -> None:
    args = parse_args()
    b0 = load_npz(args.b0.resolve())
    hard = load_npz(args.hard_set.resolve())
    if not np.array_equal(b0["sample_ids"].astype(str), hard["sample_ids"].astype(str)):
        raise RuntimeError("B0/hard sample order differs")
    labels = np.asarray(b0["labels"], dtype=np.int64)
    users = b0["users"].astype(str)
    folds = np.asarray(b0["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(b0["a_probability"], dtype=np.float64)
    a_prediction = a_probability.argmax(axis=1)
    full_probability = np.asarray(b0["full_probability"], dtype=np.float64)
    full_prediction = full_probability.argmax(axis=1)
    category = hard["category"].astype(str)
    margin = np.asarray(hard["margin"], dtype=np.float64)
    entropy = np.asarray(hard["entropy"], dtype=np.float64)
    session_harm = np.asarray(hard["session_harm"], dtype=bool)
    session_id = np.asarray(load_npz(HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz")["sequence_session_id"], dtype=np.int64)
    a_correct = a_prediction == labels
    full_correct = full_prediction == labels
    changed = a_prediction != full_prediction
    masks = {
        "rescue": (~a_correct) & full_correct,
        "harm": a_correct & (~full_correct),
        "wrong_to_wrong": (~a_correct) & (~full_correct) & changed,
        "unchanged": ~changed,
    }

    variants: dict[str, Any] = {}
    for name in ("full", "shuffle_all", "zero_visual", "zero_skeleton", "zero_imu", "zero_all"):
        probability = np.asarray(b0[f"{name}_probability"], dtype=np.float64)
        prediction = probability.argmax(axis=1)
        variants[name] = {
            **outcome_breakdown(labels, a_prediction, prediction, np.ones(len(labels), dtype=bool)),
            "correct": int(np.sum(prediction == labels)),
            "rank_improved": int(np.sum(true_rank(probability, labels) < true_rank(a_probability, labels))),
            "rank_harmed": int(np.sum(true_rank(probability, labels) > true_rank(a_probability, labels))),
        }

    category_report = {
        name: outcome_breakdown(labels, a_prediction, full_prediction, category == name)
        for name in ("correct", "A", "B", "C")
    }
    class_values = labels.astype(str)
    result = {
        "status": "complete",
        "core": {
            "candidate_error_rows": int(np.sum(np.isin(category, ["A", "B"]))),
            "candidate_miss_rows": int(np.sum(category == "C")),
            "full": variants["full"],
            "aligned_minus_shuffle_correct": variants["full"]["correct"] - variants["shuffle_all"]["correct"],
            "aligned_minus_zero_skeleton_correct": variants["full"]["correct"] - variants["zero_skeleton"]["correct"],
            "aligned_minus_zero_imu_correct": variants["full"]["correct"] - variants["zero_imu"]["correct"],
        },
        "variants": variants,
        "candidate_category": category_report,
        "fold": per_value_net(folds.astype(str), labels, a_prediction, full_prediction),
        "subject": per_value_net(users, labels, a_prediction, full_prediction),
        "true_class": per_value_net(class_values, labels, a_prediction, full_prediction),
        "uncertainty": {
            "a_margin": metric_by_group(margin, masks),
            "a_entropy": metric_by_group(entropy, masks),
        },
        "session_harm": outcome_breakdown(labels, a_prediction, full_prediction, session_harm),
        "session_consistency": {
            "a": duplicate_audit(session_id, a_prediction, labels, a_prediction),
            "b0_full": duplicate_audit(session_id, full_prediction, labels, a_prediction),
            "shuffle_all": duplicate_audit(
                session_id,
                np.asarray(b0["shuffle_all_probability"]).argmax(axis=1),
                labels,
                a_prediction,
            ),
        },
        "mechanism_decision": {
            "candidate_bottleneck": False,
            "prototype_correspondence_supported": False,
            "motion_correspondence_supported": False,
            "independent_rerank_breaks_session_constraint": True,
            "evidence": [
                "289 candidate-hit A errors remain, but full B0 corrects only 30",
                "shuffle_all has 16 more correct rows than aligned full",
                "zero Skeleton and zero IMU have 10 and 5 more correct rows than full",
                "full fold nets 0/-7/+8/-2 show subject instability",
                "B0 independently reranks rows after Session and creates duplicate-label sessions",
            ],
            "authorized_revision": (
                "B1 unified listwise discriminative visual candidate scorer. Remove "
                "Skeleton/IMU prototype compatibility, learn candidate-specific visual "
                "directions rather than one prototype distance, and pass local candidate "
                "scores through the same source-only Session decoder before final output."
            ),
            "forbidden_posthoc": [
                "B0 PCA dimension scan",
                "B0 tree/iteration/weight scan",
                "blend or trigger threshold scan",
                "oracle family routing",
            ],
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

