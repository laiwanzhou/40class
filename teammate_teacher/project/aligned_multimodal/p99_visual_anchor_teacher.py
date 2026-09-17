"""P99-VT1 source-safe anchor + InternVideo2 minimal two-expert Teacher.

Only one class-agnostic geometric-pool weight is learned per outer split. The
weight is fitted on inner-user-OOF predictions, while the evaluated user's
labels are inaccessible to model, calibration and fusion fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from p99_depth_anchor_teacher import anchor_probability, fit_weight, geometric_pool
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
from p99_visual_oof_experts import (
    DEFAULT_CONFIG as DEFAULT_VISUAL_CONFIG,
    extended_audit,
    load_visual_matrices,
)
from p99_visual_transfer_probe import paired_exact_pvalue


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/p99_visual_anchor_vt1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99 anchor + InternVideo2 Teacher")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--visual-config", type=Path, default=DEFAULT_VISUAL_CONFIG)
    parser.add_argument("--v0-h1-summary", type=Path, required=True)
    parser.add_argument("--d0-config", type=Path, default=DEFAULT_D0_CONFIG)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--e0-base", type=Path, default=DEFAULT_E0_BASE)
    parser.add_argument("--h1-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_frozen_inputs(
    stage: str,
    config: dict[str, Any],
    visual_config: dict[str, Any],
    v0_summary: dict[str, Any],
    h1_summary: dict[str, Any] | None,
) -> str:
    expert = str(config["visual_expert"])
    if expert not in visual_config["experts"]:
        raise ValueError(f"frozen Visual expert is absent from config: {expert}")
    if v0_summary.get("stage") != "P99_V0_H1_visual_single_expert_pool":
        raise ValueError("--v0-h1-summary is not a P99-V0 H1 result")
    if v0_summary.get("config_sha256") != canonical_hash(visual_config):
        raise ValueError("V0 summary/Visual config mismatch")
    if expert not in v0_summary.get("experts", {}):
        raise ValueError("frozen Visual expert is absent from V0 summary")
    combined_hash = canonical_hash(
        {"fusion": config, "visual": visual_config, "visual_expert": expert}
    )
    if stage == "h2_confirmation":
        if h1_summary is None:
            raise ValueError("H2 confirmation requires the frozen VT1 H1 summary")
        if (
            h1_summary.get("stage") != "P99_VT1_H1"
            or h1_summary.get("config_sha256") != combined_hash
            or h1_summary.get("selected_visual_expert") != expert
        ):
            raise ValueError("VT1 H1 summary/config mismatch")
    elif h1_summary is not None:
        raise ValueError("--h1-summary is only valid for H2 confirmation")
    return combined_hash


def run_teacher(
    stage: str,
    config: dict[str, Any],
    visual_config: dict[str, Any],
    combined_hash: str,
    depth_config: dict[str, Any],
    depth_features: Path,
    split_source: Path,
    e0_base: Path,
    output: Path,
) -> dict[str, Any]:
    cohort, eval_indices, eval_ids = build_cohort(
        stage, depth_config, depth_features, split_source, e0_base
    )
    matrices, cache_audit = load_visual_matrices(visual_config, cohort.sample_ids)
    expert = str(config["visual_expert"])
    values = matrices[expert]
    recipe = visual_config["experts"][expert]
    evaluation_users = cohort.users[eval_indices]
    folds = sorted(set(evaluation_users.tolist())) if stage == "h1" else ["H2_all"]
    direct = np.zeros((len(eval_indices), NUM_CLASSES), dtype=np.float64)
    zero = np.zeros_like(direct)
    shuffled = np.zeros_like(direct)
    visual_direct = np.zeros_like(direct)
    weights: dict[str, float] = {}
    temperatures: dict[str, float] = {}
    confidence = float(config["anchor_confidence"])
    bounds = tuple(map(float, config["weight_bounds"]))

    for fold_number, held_user in enumerate(folds):
        if stage == "h1":
            local_eval = np.flatnonzero(evaluation_users == held_user)
            outer_eval = eval_indices[local_eval]
            train = np.flatnonzero(cohort.users != held_user)
        else:
            local_eval = np.arange(len(eval_indices))
            outer_eval = eval_indices
            train = np.flatnonzero(~np.isin(np.arange(len(cohort.labels)), eval_indices))
        if set(cohort.users[train]) & set(cohort.users[outer_eval]):
            raise RuntimeError("outer train/evaluation users overlap")
        visual_logits, inner_logits, temperature, model = calibrated_outer_prediction(
            values, cohort.labels, cohort.users, train, outer_eval, recipe
        )
        train_anchor = anchor_probability(cohort.anchor_prediction[train], confidence)
        weight = fit_weight(
            train_anchor, softmax(inner_logits), cohort.labels[train], bounds
        )
        eval_anchor = anchor_probability(cohort.anchor_prediction[outer_eval], confidence)
        direct_probability = geometric_pool(eval_anchor, softmax(visual_logits), weight)

        training_mean = values[train].mean(axis=0, keepdims=True)
        zero_probability = softmax(
            decision_scores(model, np.repeat(training_mean, len(outer_eval), axis=0))
            / temperature
        )
        rng = np.random.default_rng(int(config["seed"]) + fold_number * 1009)
        permutation = np.arange(len(outer_eval))
        for user in sorted(set(cohort.users[outer_eval].tolist())):
            selected = np.flatnonzero(cohort.users[outer_eval] == user)
            permutation[selected] = selected[rng.permutation(len(selected))]
        shuffle_probability = softmax(visual_logits[permutation])

        direct[local_eval] = np.log(np.clip(direct_probability, 1e-12, 1.0))
        zero[local_eval] = np.log(
            np.clip(geometric_pool(eval_anchor, zero_probability, weight), 1e-12, 1.0)
        )
        shuffled[local_eval] = np.log(
            np.clip(geometric_pool(eval_anchor, shuffle_probability, weight), 1e-12, 1.0)
        )
        visual_direct[local_eval] = visual_logits
        weights[str(held_user)] = weight
        temperatures[str(held_user)] = temperature

    labels = cohort.labels[eval_indices]
    anchor_prediction = cohort.anchor_prediction[eval_indices]
    prediction = direct.argmax(axis=1)
    per_user: dict[str, Any] = {}
    for user in sorted(set(evaluation_users.tolist())):
        selected = evaluation_users == user
        anchor_correct = int(np.sum(anchor_prediction[selected] == labels[selected]))
        teacher_correct = int(np.sum(prediction[selected] == labels[selected]))
        per_user[user] = {
            "rows": int(selected.sum()),
            "anchor_correct": anchor_correct,
            "teacher_correct": teacher_correct,
            "delta": teacher_correct - anchor_correct,
        }

    direct_probability = softmax(direct).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    prediction_name = "h1_predictions.npz" if stage == "h1" else "h2_predictions.npz"
    np.savez_compressed(
        output / prediction_name,
        sample_ids=eval_ids,
        labels=labels,
        users=evaluation_users,
        anchor_prediction=anchor_prediction,
        anchor_probability=anchor_probability(anchor_prediction, confidence).astype(np.float32),
        visual_probability=softmax(visual_direct).astype(np.float32),
        direct_probability=direct_probability,
        direct_logits=direct.astype(np.float32),
        zero_probability=softmax(zero).astype(np.float32),
        shuffle_probability=softmax(shuffled).astype(np.float32),
        selected_visual_expert=np.asarray(expert),
    )
    change = change_audit(labels, anchor_prediction, prediction)
    change["mcnemar_exact_pvalue"] = paired_exact_pvalue(
        labels, anchor_prediction, prediction
    )
    user_nonnegative = sum(value["delta"] >= 0 for value in per_user.values())
    user_positive = sum(value["delta"] > 0 for value in per_user.values())
    student_gate_passed = bool(
        change["net"] >= int(config["student_gate_min_net"])
        and user_positive >= int(config["student_gate_min_positive_users"])
        and user_nonnegative >= int(config["student_gate_min_nonnegative_users"])
    )
    report = {
        "stage": "P99_VT1_H1" if stage == "h1" else "P99_VT1_H2_confirmation",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": combined_hash,
        "selected_visual_expert": expert,
        "protocol": (
            "E0+H1 outer LOUO; one class-agnostic fusion weight fitted on inner-user-OOF NLL"
            if stage == "h1"
            else "frozen VT1 recipe; source-user-OOF weight; one H2 confirmation"
        ),
        "metrics": metrics(direct, labels),
        "visual_metrics": metrics(visual_direct, labels),
        "anchor_metrics": metrics(
            np.log(np.clip(anchor_probability(anchor_prediction, confidence), 1e-12, 1.0)),
            labels,
        ),
        "zero_metrics": metrics(zero, labels),
        "shuffle_metrics": metrics(shuffled, labels),
        "vs_anchor": change,
        "extended_audit": extended_audit(
            labels, anchor_prediction, direct_probability,
            visual_config["focus_groups"], evaluation_users,
        ),
        "per_user": per_user,
        "visual_weight_by_outer_fold": weights,
        "visual_temperature_by_outer_fold": temperatures,
        "matched_train_without_visual": {
            "implementation": "exact anchor probability; no trainable Visual path remains",
            "correct": int(np.sum(anchor_prediction == labels)),
            "prediction_exact_match": True,
        },
        "student_gate": {
            "rule": config["student_gate"],
            "positive_users": int(user_positive),
            "nonnegative_users": int(user_nonnegative),
            "passed": student_gate_passed,
        },
        "cache_audit": cache_audit,
        "leakage_audit": {
            "backbone_features_label_free": True,
            "embedded_cache_labels_used": False,
            "outer_user_disjoint": True,
            "temperature_inner_user_oof": True,
            "fusion_weight_inner_user_oof": True,
            "matched_removal_exact": True,
            "h2_requires_frozen_h1_summary": True,
            "h3_code_path_present": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    visual_config = json.loads(args.visual_config.resolve().read_text(encoding="utf-8"))
    v0_summary = json.loads(args.v0_h1_summary.resolve().read_text(encoding="utf-8"))
    h1_summary = (
        None
        if args.h1_summary is None
        else json.loads(args.h1_summary.resolve().read_text(encoding="utf-8"))
    )
    combined_hash = validate_frozen_inputs(
        args.stage, config, visual_config, v0_summary, h1_summary
    )
    depth_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    report = run_teacher(
        args.stage,
        config,
        visual_config,
        combined_hash,
        depth_config,
        args.depth_features,
        args.split_source,
        args.e0_base,
        args.output_dir.resolve(),
    )
    compact = {
        "stage": report["stage"],
        "teacher_correct": report["metrics"]["correct"],
        "teacher_top5": report["metrics"]["top5"],
        "anchor_correct": report["anchor_metrics"]["correct"],
        "visual_correct": report["visual_metrics"]["correct"],
        "zero_correct": report["zero_metrics"]["correct"],
        "shuffle_correct": report["shuffle_metrics"]["correct"],
        "vs_anchor": report["vs_anchor"],
        "per_user": report["per_user"],
        "weights": report["visual_weight_by_outer_fold"],
        "student_gate": report["student_gate"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
