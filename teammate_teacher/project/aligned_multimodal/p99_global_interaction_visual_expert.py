"""P99-HV1 global + hand-interaction Visual structural revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from p99_depth_oof_expert import (
    DEFAULT_CONFIG as DEFAULT_D0_CONFIG,
    DEFAULT_DEPTH,
    DEFAULT_E0_BASE,
    DEFAULT_SPLITS,
    Cohort,
    align,
    build_cohort,
    calibrated_outer_prediction,
    canonical_hash,
    change_audit,
    decision_scores,
    metrics,
    softmax,
)
from p99_hand_object_visual_expert import hand_feature_families
from p99_visual_oof_experts import extended_audit, videomae_matrix


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_global_interaction_visual_hv1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99-HV1 global/local Visual audit")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--d0-config", type=Path, default=DEFAULT_D0_CONFIG)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--e0-base", type=Path, default=DEFAULT_E0_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def load_global_local(
    config: dict[str, Any], cohort_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sources = config["sources"]
    global_path = resolve(sources["global_videomae"])
    local_path = resolve(sources["local_hand"])
    with np.load(global_path, allow_pickle=False) as source:
        global_ids = np.asarray(source["sample_ids"]).astype(str)
        global_features = align(global_ids, np.asarray(source["features"]), cohort_ids)
    with np.load(local_path, allow_pickle=False) as source:
        local_ids = np.asarray(source["sample_ids"]).astype(str)
        local_features = align(local_ids, np.asarray(source["features"]), cohort_ids)
    global_matrix = videomae_matrix(global_features)
    local_matrix = hand_feature_families(local_features)["interaction_only"]
    if not np.isfinite(global_matrix).all() or not np.isfinite(local_matrix).all():
        raise ValueError("HV1 features contain non-finite values")
    return global_matrix, local_matrix, {
        "global_cache": str(global_path),
        "local_cache": str(local_path),
        "global_cache_rows": int(len(global_ids)),
        "local_cache_rows": int(len(local_ids)),
        "cohort_rows": int(len(cohort_ids)),
        "global_dimension": int(global_matrix.shape[1]),
        "local_dimension": int(local_matrix.shape[1]),
        "joint_dimension": int(global_matrix.shape[1] + local_matrix.shape[1]),
        "embedded_cache_labels_used": False,
        "global_and_crop_before_classifier": True,
    }


def load_matched_global(
    path: Path, eval_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path.resolve(), allow_pickle=False) as source:
        sample_ids = np.asarray(source["sample_ids"]).astype(str)
        probability = align(
            sample_ids,
            np.asarray(source["videomaev2_early_late_direct_probability"]),
            eval_ids,
        ).astype(np.float64)
    if probability.shape != (len(eval_ids), 40):
        raise ValueError(f"invalid matched global probability shape: {probability.shape}")
    probability = np.clip(probability, 1e-12, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, {
        "path": str(path.resolve()),
        "rows": int(len(sample_ids)),
        "labels_read": False,
        "matched_recipe": "P99-V0 videomaev2_early_late source-OOF",
    }


def paired_change(
    labels: np.ndarray, base_prediction: np.ndarray, candidate_prediction: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = change_audit(labels, base_prediction, candidate_prediction)
    discordant = result["rescue"] + result["harm"]
    result["mcnemar_exact_pvalue"] = (
        float(binomtest(result["rescue"], discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return result


def build_selection_gate(
    config: dict[str, Any],
    labels: np.ndarray,
    users: np.ndarray,
    direct_prediction: np.ndarray,
    global_prediction: np.ndarray,
    direct_correct: int,
    zero_correct: int,
    shuffle_correct: int,
) -> dict[str, Any]:
    rule = config["selection_gate"]
    user_delta: dict[str, dict[str, float | int]] = {}
    for user in sorted(set(users.tolist())):
        selected = users == user
        global_correct = int(np.sum(global_prediction[selected] == labels[selected]))
        direct_user_correct = int(np.sum(direct_prediction[selected] == labels[selected]))
        rows = int(selected.sum())
        user_delta[str(user)] = {
            "rows": rows,
            "global_correct": global_correct,
            "direct_correct": direct_user_correct,
            "delta": direct_user_correct - global_correct,
            "accuracy_delta_pp": 100.0 * (direct_user_correct - global_correct) / rows,
        }
    deltas = [int(value["delta"]) for value in user_delta.values()]
    accuracy_deltas = [float(value["accuracy_delta_pp"]) for value in user_delta.values()]
    checks = {
        "minimum_net_gain": direct_correct - int(np.sum(global_prediction == labels))
        >= int(rule["minimum_net_gain"]),
        "minimum_positive_users": sum(value > 0 for value in deltas)
        >= int(rule["minimum_positive_users"]),
        "minimum_nonnegative_users": sum(value >= 0 for value in deltas)
        >= int(rule["minimum_nonnegative_users"]),
        "worst_user_risk": min(accuracy_deltas)
        >= -float(rule["maximum_worst_user_accuracy_drop_pp"]),
        "direct_above_local_zero": direct_correct > zero_correct,
        "direct_above_local_shuffle": direct_correct > shuffle_correct,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "per_user": user_delta,
        "positive_users": int(sum(value > 0 for value in deltas)),
        "nonnegative_users": int(sum(value >= 0 for value in deltas)),
        "worst_user_accuracy_delta_pp": float(min(accuracy_deltas)),
    }


def main() -> None:
    args = parse_args()
    if args.stage != "h1":
        raise ValueError("HV1 is H1-only until a joint Teacher/Student recipe is frozen")
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    base, eval_indices, eval_ids = build_cohort(
        "h1", d0_config, args.depth_features, args.split_source, args.e0_base
    )
    global_matrix, local_matrix, cache_audit = load_global_local(
        config, base.sample_ids
    )
    joint = np.concatenate((global_matrix, local_matrix), axis=1)
    cohort = Cohort(
        sample_ids=base.sample_ids,
        labels=base.labels,
        users=base.users,
        anchor_prediction=base.anchor_prediction,
        features={"global_plus_interaction": joint},
    )
    direct_logits = np.zeros((len(eval_indices), 40), dtype=np.float64)
    zero_logits = np.zeros_like(direct_logits)
    shuffle_logits = np.zeros_like(direct_logits)
    temperatures: dict[str, float] = {}
    evaluation_users = cohort.users[eval_indices]
    recipe = config["expert"]
    global_dim = global_matrix.shape[1]
    for fold_number, held_user in enumerate(sorted(set(evaluation_users.tolist()))):
        local_eval = np.flatnonzero(evaluation_users == held_user)
        outer_eval = eval_indices[local_eval]
        train = np.flatnonzero(cohort.users != held_user)
        if set(cohort.users[train]) & set(cohort.users[outer_eval]):
            raise RuntimeError("outer train/evaluation users overlap")
        logits, _, temperature, model = calibrated_outer_prediction(
            joint, cohort.labels, cohort.users, train, outer_eval, recipe
        )
        direct_logits[local_eval] = logits
        local_mean = local_matrix[train].mean(axis=0, keepdims=True)
        zero_values = np.concatenate(
            (
                global_matrix[outer_eval],
                np.repeat(local_mean, len(outer_eval), axis=0),
            ),
            axis=1,
        )
        zero_logits[local_eval] = decision_scores(model, zero_values) / temperature
        rng = np.random.default_rng(int(config["seed"]) + fold_number * 1009)
        permutation = rng.permutation(len(outer_eval))
        shuffle_values = np.concatenate(
            (global_matrix[outer_eval], local_matrix[outer_eval][permutation]), axis=1
        )
        shuffle_logits[local_eval] = decision_scores(model, shuffle_values) / temperature
        temperatures[str(held_user)] = float(temperature)
    if global_dim != 4608:
        raise RuntimeError(f"unexpected global dimension: {global_dim}")

    labels = cohort.labels[eval_indices]
    users = cohort.users[eval_indices]
    anchor = cohort.anchor_prediction[eval_indices]
    direct_probability = softmax(direct_logits)
    direct_prediction = direct_probability.argmax(axis=1)
    zero_prediction = zero_logits.argmax(axis=1)
    shuffle_prediction = shuffle_logits.argmax(axis=1)
    global_probability, global_audit = load_matched_global(
        resolve(config["sources"]["matched_global_h1"]), eval_ids
    )
    global_prediction = global_probability.argmax(axis=1)
    direct_metrics = metrics(direct_logits, labels)
    zero_metrics = metrics(zero_logits, labels)
    shuffle_metrics = metrics(shuffle_logits, labels)
    global_metrics = metrics(np.log(global_probability), labels)
    gate = build_selection_gate(
        config,
        labels,
        users,
        direct_prediction,
        global_prediction,
        int(direct_metrics["correct"]),
        int(zero_metrics["correct"]),
        int(shuffle_metrics["correct"]),
    )
    hv0 = json.loads(resolve(config["sources"]["hv0_summary"]).read_text(encoding="utf-8"))
    if hv0.get("stage") != "P99_HV0_H1_hand_workspace_visual_expert":
        raise ValueError("HV0 summary stage mismatch")
    local_only = hv0["experts"]["interaction_only"]
    report = {
        "stage": "P99_HV1_H1_global_interaction_visual",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": canonical_hash(config),
        "protocol": "E0+H1 outer LOUO; one joint head; only local representation added",
        "cache_audit": cache_audit,
        "direct_metrics": direct_metrics,
        "matched_global_metrics": global_metrics,
        "local_zero_metrics": zero_metrics,
        "local_shuffle_metrics": shuffle_metrics,
        "local_only_metrics": local_only["metrics"],
        "vs_matched_global": paired_change(labels, global_prediction, direct_prediction),
        "vs_local_zero": paired_change(labels, zero_prediction, direct_prediction),
        "vs_local_shuffle": paired_change(labels, shuffle_prediction, direct_prediction),
        "extended_audit": extended_audit(
            labels, anchor, direct_probability, config["focus_groups"], users
        ),
        "temperature_by_outer_fold": temperatures,
        "selection_gate": gate,
        "matched_removal_audit": global_audit,
        "leakage_audit": {
            "backbone_features_label_free": True,
            "embedded_cache_labels_used": False,
            "matched_global_labels_read": False,
            "outer_user_disjoint": True,
            "temperature_inner_user_oof": True,
            "only_local_feature_changed": True,
            "h2_h3_accessed": False,
            "h3_code_path_present": False,
        },
        "h2_policy": config["h2_policy"],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "h1_predictions.npz",
        sample_ids=eval_ids,
        labels=labels,
        users=users,
        anchor_prediction=anchor,
        direct_probability=direct_probability.astype(np.float32),
        matched_global_probability=global_probability.astype(np.float32),
        local_zero_probability=softmax(zero_logits).astype(np.float32),
        local_shuffle_probability=softmax(shuffle_logits).astype(np.float32),
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "direct": {key: direct_metrics[key] for key in ("correct", "top5", "balanced_accuracy", "macro_f1", "log_loss")},
                "matched_global": {key: global_metrics[key] for key in ("correct", "top5", "balanced_accuracy", "macro_f1", "log_loss")},
                "local_zero_correct": zero_metrics["correct"],
                "local_shuffle_correct": shuffle_metrics["correct"],
                "vs_matched_global": report["vs_matched_global"],
                "selection_gate": gate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
