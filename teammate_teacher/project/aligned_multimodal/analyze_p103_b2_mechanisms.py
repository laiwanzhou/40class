"""Formal mechanism audit for P103-B2 rich-global candidate attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_p102_b0_mechanisms import outcome_breakdown, per_value_net
from audit_p102_session_closure import load_npz


HERE = Path(__file__).resolve().parent
DEFAULT_B2 = HERE / "runs/p103_b2_rich_visual_oof_v1/b2_oof_predictions.npz"
DEFAULT_SUMMARY = HERE / "runs/p103_b2_rich_visual_oof_v1/summary.json"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_OUTPUT = HERE / "runs/p103_b2_rich_visual_oof_v1/mechanism_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2", type=Path, default=DEFAULT_B2)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--hard-set", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalized_entropy(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    probability = probability / np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=-1)
    return entropy / np.log(probability.shape[-1])


def attention_structure(
    attention: np.ndarray, candidate_ids: np.ndarray, names: np.ndarray
) -> dict[str, Any]:
    lookup = {str(name): index for index, name in enumerate(names.astype(str))}
    valid = candidate_ids >= 0
    query_rows = attention[valid]

    axes = {
        "encoder": [lookup["encoder_videomaev2"], lookup["encoder_internvideo2"]],
        "view": [lookup[f"view_{name}"] for name in ("scene", "person", "workspace")],
        "window": [lookup[f"window_{name}"] for name in ("early", "late")],
        "time": [lookup[f"time_{index}"] for index in range(8)],
        "type": [lookup[f"type_{name}"] for name in ("temporal", "pooled", "action")],
    }
    entropy = {
        name: {
            "mean_normalized_entropy": float(normalized_entropy(query_rows[:, indices]).mean()),
            "p10_normalized_entropy": float(np.quantile(normalized_entropy(query_rows[:, indices]), 0.10)),
        }
        for name, indices in axes.items()
    }

    dispersion: dict[str, float] = {}
    for axis, indices in axes.items():
        row_values: list[float] = []
        for row in range(len(attention)):
            selected = attention[row, valid[row]][:, indices]
            row_values.append(float(np.mean(np.std(selected, axis=0))))
        dispersion[axis] = float(np.mean(row_values))
    return {
        "normalized_entropy": entropy,
        "mean_candidate_query_dispersion": dispersion,
        "interpretation": (
            "Entropy near one and small between-candidate dispersion mean that class queries "
            "read nearly the same global-pooled evidence; attention weights are descriptive, "
            "while aligned/shuffle/zero remains the causal test."
        ),
    }


def main() -> None:
    args = parse_args()
    b2 = load_npz(args.b2.resolve())
    hard = load_npz(args.hard_set.resolve())
    summary = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    if not np.array_equal(b2["sample_ids"].astype(str), hard["sample_ids"].astype(str)):
        raise RuntimeError("B2/hard sample order differs")

    labels = np.asarray(b2["labels"], dtype=np.int64)
    users = b2["users"].astype(str)
    folds = np.asarray(b2["fold_ids"], dtype=np.int64)
    candidates = np.asarray(b2["candidate_ids"], dtype=np.int64)
    evaluated = np.asarray(b2["evaluated"], dtype=bool)
    if len(labels) != 1941 or not evaluated.all():
        raise RuntimeError("formal B2 OOF coverage is incomplete")
    if summary["data"]["h3_rows_selected"] != 0 or summary["data"]["h3_users_loaded"]:
        raise RuntimeError("H3 contract failed")

    probabilities = {
        name: np.asarray(b2[f"{name}_probability"], dtype=np.float64)
        for name in (
            "full_local",
            "full_session",
            "shuffle_local",
            "shuffle_session",
            "zero_local",
            "zero_session",
        )
    }
    a_probability = np.asarray(b2["a_session_probability"], dtype=np.float64)
    for name, probability in {"a": a_probability, **probabilities}.items():
        if probability.shape != (1941, 40) or not np.isfinite(probability).all():
            raise RuntimeError(f"invalid probability: {name}")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=2e-6):
            raise RuntimeError(f"probability mass failed: {name}")

    a_prediction = a_probability.argmax(axis=1)
    predictions = {name: value.argmax(axis=1) for name, value in probabilities.items()}
    all_rows = np.ones(len(labels), dtype=bool)
    candidate_hit = np.any(candidates == labels[:, None], axis=1)
    hard_hit = (a_prediction != labels) & candidate_hit
    if int(hard_hit.sum()) != 289:
        raise RuntimeError("P103 candidate-hit hard population drifted")

    outcomes = {
        name: outcome_breakdown(labels, a_prediction, prediction, all_rows)
        for name, prediction in predictions.items()
    }
    aligned = predictions["full_session"]
    category = hard["category"].astype(str)
    attention = np.asarray(b2["aligned_attention_groups"], dtype=np.float32)
    attention_names = b2["attention_group_names"].astype(str)
    if attention.shape != (1941, 8, 18):
        raise RuntimeError("unexpected attention audit shape")

    training_rows = []
    for fold in summary["folds"]:
        final = fold["training"]["history"][-1]
        training_rows.append(
            {
                "fold": int(fold["fold"]),
                "selected_source_rows": int(fold["population"]["selected_rows"]),
                "final_training_accuracy": float(final["training_accuracy"]),
                "held_full_session_net": int(fold["variants"]["full_session"]["vs_a"]["net"]),
            }
        )

    result = {
        "status": "complete",
        "integrity": {
            "rows": 1941,
            "subjects": sorted(set(users.tolist())),
            "folds": sorted(set(map(int, folds.tolist()))),
            "evaluated_rows": int(evaluated.sum()),
            "candidate_hit_a_errors": int(hard_hit.sum()),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "level1": summary["level1_conditional"],
        "level2": summary["level2_full_system"],
        "outcomes": outcomes,
        "candidate_category": {
            name: outcome_breakdown(labels, a_prediction, aligned, category == name)
            for name in ("correct", "A", "B", "C")
        },
        "fold": per_value_net(folds.astype(str), labels, a_prediction, aligned),
        "subject": per_value_net(users, labels, a_prediction, aligned),
        "causal_correspondence": {
            "aligned_minus_shuffle_full_system_correct": int(
                np.sum(aligned == labels) - np.sum(predictions["shuffle_session"] == labels)
            ),
            "aligned_minus_zero_full_system_correct": int(
                np.sum(aligned == labels) - np.sum(predictions["zero_session"] == labels)
            ),
            "aligned_minus_shuffle_local_correct": int(
                np.sum(predictions["full_local"] == labels)
                - np.sum(predictions["shuffle_local"] == labels)
            ),
            "aligned_minus_zero_local_correct": int(
                np.sum(predictions["full_local"] == labels)
                - np.sum(predictions["zero_local"] == labels)
            ),
        },
        "attention_structure": attention_structure(attention, candidates, attention_names),
        "optimization_transfer": training_rows,
        "mechanism_decision": {
            "b2_pass": False,
            "student_authorized": False,
            "compressed_global_b_route_exhausted": True,
            "hard_case_b_teacher_route_exhausted": False,
            "b_teacher_hypothesis_closed": False,
            "failure_type": "representation locality plus source-transfer failure",
            "evidence": [
                "B2 full-session conditional Top-1 is 64/289 (22.15%), far below the 50% capability reference.",
                "The full system loses 68 rows versus frozen A; every outer fold is negative (-15/-19/-15/-19).",
                "Aligned is 22 rows worse than within-subject shuffled visual and 55 worse than zero visual.",
                "All folds reach about 99% source-population training accuracy, so capacity/optimization is not the limiting failure.",
                "P101 temporal extraction averages spatial patches before caching; B2 therefore has view/time identity but no within-crop hand/object patch location.",
                "Attention is close to uniform across views/windows/times and does not establish candidate-specific correspondence.",
            ],
            "authorized_revision": (
                "P103-B3 uses explicit full/early/late/motion-peak left-hand, right-hand, "
                "interaction, and workspace crops from frozen VideoMAEv2/V-JEPA2 caches. "
                "It keeps direct candidate-query scoring and aligned/shuffle/zero, and does "
                "not scan learning rate, seed, PCA, blend, threshold, or residual scale."
            ),
            "b3_assets": {
                "videomaev2": "full/peak-motion x left/right/interaction (6 local views)",
                "vjepa2": "full/early/late/motion-peak x left/right/interaction plus workspace temporal views",
                "h3_usage": "label-free cache may contain all rows, but B3 resolves the 1941-row dev allowlist before selecting feature rows; no H3 labels/results are loaded",
            },
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["mechanism_decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
