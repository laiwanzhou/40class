"""P99-T1 minimal two-expert Teacher: source-safe anchor + P99-D0 Depth.

The only fitted fusion parameter is one nonnegative scalar per outer split.  It
is optimized on inner-user-OOF probabilities, never on the evaluated user.
Removing Depth returns the anchor exactly.  H3 is intentionally unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from p99_depth_oof_expert import (
    DEFAULT_CONFIG as DEFAULT_D0_CONFIG,
    DEFAULT_DEPTH,
    DEFAULT_E0_BASE,
    DEFAULT_SPLITS,
    NUM_CLASSES,
    build_cohort,
    calibrated_outer_prediction,
    canonical_hash,
    change_audit,
    decision_scores,
    metrics,
    softmax,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/p99_depth_anchor_t1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99 anchor + Depth two-expert Teacher")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--d0-config", type=Path, default=DEFAULT_D0_CONFIG)
    parser.add_argument("--d0-h1-summary", type=Path, required=True)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--e0-base", type=Path, default=DEFAULT_E0_BASE)
    parser.add_argument("--h1-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def anchor_probability(prediction: np.ndarray, confidence: float) -> np.ndarray:
    if not 1.0 / NUM_CLASSES < confidence < 1.0:
        raise ValueError("anchor confidence must be between chance and one")
    output = np.full(
        (len(prediction), NUM_CLASSES),
        (1.0 - confidence) / (NUM_CLASSES - 1),
        dtype=np.float64,
    )
    output[np.arange(len(prediction)), prediction.astype(np.int64)] = confidence
    return output


def geometric_pool(
    anchor: np.ndarray, depth: np.ndarray, weight: float
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("mixture weight must be in [0, 1]")
    scores = (
        (1.0 - weight) * np.log(np.clip(anchor, 1e-12, 1.0))
        + weight * np.log(np.clip(depth, 1e-12, 1.0))
    )
    return softmax(scores)


def fit_weight(
    anchor: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    def objective(weight: float) -> float:
        probability = geometric_pool(anchor, depth, float(weight))
        return float(
            np.mean(
                -np.log(
                    np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1.0)
                )
            )
        )

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success:
        raise RuntimeError("inner-OOF mixture optimization failed")
    return float(result.x)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    d0_summary = json.loads(args.d0_h1_summary.resolve().read_text(encoding="utf-8"))
    if d0_summary.get("stage") != "P99_D0_H1":
        raise ValueError("--d0-h1-summary is not a P99-D0 H1 result")
    if d0_summary.get("config_sha256") != canonical_hash(d0_config):
        raise ValueError("D0 summary/config mismatch")
    recipe_name = str(d0_summary["selected_recipe"])
    if recipe_name not in d0_config["recipes"]:
        raise ValueError("frozen D0 recipe is absent from its config")
    combined_hash = canonical_hash(
        {"fusion": config, "d0": d0_config, "d0_recipe": recipe_name}
    )
    if args.stage == "h2_confirmation":
        if args.h1_summary is None:
            raise ValueError("H2 requires --h1-summary")
        frozen = json.loads(args.h1_summary.resolve().read_text(encoding="utf-8"))
        if frozen.get("stage") != "P99_T1_H1" or frozen.get("config_sha256") != combined_hash:
            raise ValueError("T1 H1 summary/config mismatch")
    elif args.h1_summary is not None:
        raise ValueError("--h1-summary is only valid for H2")

    cohort, eval_indices, eval_ids = build_cohort(
        args.stage, d0_config, args.depth_features, args.split_source, args.e0_base
    )
    values = cohort.features[recipe_name]
    recipe = d0_config["recipes"][recipe_name]
    evaluation_users = cohort.users[eval_indices]
    folds = sorted(set(evaluation_users.tolist())) if args.stage == "h1" else ["H2_all"]
    direct = np.zeros((len(eval_indices), NUM_CLASSES), dtype=np.float64)
    zero = np.zeros_like(direct)
    shuffled = np.zeros_like(direct)
    depth_direct = np.zeros_like(direct)
    weights: dict[str, float] = {}
    temperatures: dict[str, float] = {}
    confidence = float(config["anchor_confidence"])
    bounds = tuple(map(float, config["weight_bounds"]))
    for fold_number, held_user in enumerate(folds):
        if args.stage == "h1":
            local_eval = np.flatnonzero(evaluation_users == held_user)
            outer_eval = eval_indices[local_eval]
            train = np.flatnonzero(cohort.users != held_user)
        else:
            local_eval = np.arange(len(eval_indices))
            outer_eval = eval_indices
            train = np.flatnonzero(~np.isin(np.arange(len(cohort.labels)), eval_indices))
        depth_logits, inner_logits, temperature, model = calibrated_outer_prediction(
            values, cohort.labels, cohort.users, train, outer_eval, recipe
        )
        train_anchor = anchor_probability(cohort.anchor_prediction[train], confidence)
        weight = fit_weight(
            train_anchor, softmax(inner_logits), cohort.labels[train], bounds
        )
        eval_anchor = anchor_probability(cohort.anchor_prediction[outer_eval], confidence)
        direct_probability = geometric_pool(eval_anchor, softmax(depth_logits), weight)
        training_mean = values[train].mean(axis=0, keepdims=True)
        zero_logits = decision_scores(
            model, np.repeat(training_mean, len(outer_eval), axis=0)
        )
        zero_probability = softmax(zero_logits / temperature)
        rng = np.random.default_rng(int(config["seed"]) + fold_number * 1009)
        permutation = np.arange(len(outer_eval))
        for user in sorted(set(cohort.users[outer_eval].tolist())):
            selected = np.flatnonzero(cohort.users[outer_eval] == user)
            permutation[selected] = selected[rng.permutation(len(selected))]
        shuffled_probability = softmax(depth_logits[permutation])
        direct[local_eval] = np.log(np.clip(direct_probability, 1e-12, 1.0))
        zero[local_eval] = np.log(
            np.clip(geometric_pool(eval_anchor, zero_probability, weight), 1e-12, 1.0)
        )
        shuffled[local_eval] = np.log(
            np.clip(geometric_pool(eval_anchor, shuffled_probability, weight), 1e-12, 1.0)
        )
        depth_direct[local_eval] = depth_logits
        weights[str(held_user)] = weight
        temperatures[str(held_user)] = temperature

    labels = cohort.labels[eval_indices]
    anchor_prediction = cohort.anchor_prediction[eval_indices]
    prediction = direct.argmax(axis=1)
    per_user: dict[str, Any] = {}
    for user in sorted(set(evaluation_users.tolist())):
        selected = evaluation_users == user
        per_user[user] = {
            "rows": int(selected.sum()),
            "anchor_correct": int(np.sum(anchor_prediction[selected] == labels[selected])),
            "teacher_correct": int(np.sum(prediction[selected] == labels[selected])),
            "delta": int(
                np.sum(prediction[selected] == labels[selected])
                - np.sum(anchor_prediction[selected] == labels[selected])
            ),
        }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    direct_probability = softmax(direct).astype(np.float32)
    np.savez_compressed(
        output / ("h1_predictions.npz" if args.stage == "h1" else "h2_predictions.npz"),
        sample_ids=eval_ids,
        labels=labels,
        users=evaluation_users,
        anchor_prediction=anchor_prediction,
        anchor_probability=anchor_probability(anchor_prediction, confidence).astype(np.float32),
        depth_probability=softmax(depth_direct).astype(np.float32),
        direct_probability=direct_probability,
        direct_logits=direct.astype(np.float32),
        zero_probability=softmax(zero).astype(np.float32),
        shuffle_probability=softmax(shuffled).astype(np.float32),
        selected_depth_recipe=np.asarray(recipe_name),
    )
    summary = {
        "stage": "P99_T1_H1" if args.stage == "h1" else "P99_T1_H2_confirmation",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": combined_hash,
        "selected_depth_recipe": recipe_name,
        "protocol": (
            "one class-agnostic geometric-pool weight fit on inner-user-OOF rows for every held H1 user"
            if args.stage == "h1"
            else "frozen D0/T1 recipe; one source-user-OOF weight; H2 evaluated once"
        ),
        "metrics": metrics(direct, labels),
        "anchor_metrics": {
            "correct": int(np.sum(anchor_prediction == labels)),
            "total": int(len(labels)),
            "accuracy": float(np.mean(anchor_prediction == labels)),
        },
        "zero_metrics": metrics(zero, labels),
        "shuffle_metrics": metrics(shuffled, labels),
        "vs_anchor": change_audit(labels, anchor_prediction, prediction),
        "per_user": per_user,
        "depth_weight_by_outer_fold": weights,
        "depth_temperature_by_outer_fold": temperatures,
        "matched_train_without_depth": {
            "implementation": "exact anchor probability; no trainable Depth path remains",
            "correct": int(np.sum(anchor_prediction == labels)),
        },
        "leakage_audit": {
            "outer_user_disjoint": True,
            "depth_temperature_inner_user_oof": True,
            "fusion_weight_inner_user_oof": True,
            "h2_requires_frozen_h1_summary": True,
            "h3_code_path_present": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
