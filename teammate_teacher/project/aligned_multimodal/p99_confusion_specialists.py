"""P99-C0 conditional Skeleton/IMU confusion specialist lower bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from p99_depth_oof_expert import (
    DEFAULT_CONFIG as DEFAULT_D0_CONFIG,
    DEFAULT_DEPTH,
    DEFAULT_E0_BASE,
    DEFAULT_SPLITS,
    NUM_CLASSES,
    build_cohort,
    canonical_hash,
    change_audit,
    fit_temperature,
    metrics,
    softmax,
)
from p99_imu_statistics_expert import load_descriptor as load_imu_descriptor
from p99_skeleton_relation_expert import load_descriptor as load_skeleton_descriptor
from p99_visual_transfer_probe import paired_exact_pvalue


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_confusion_specialists_c0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99 conditional confusion specialists")
    parser.add_argument("--stage", choices=("h1",), default="h1")
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


def make_model(model_config: dict[str, Any], seed: int) -> ExtraTreesClassifier:
    if model_config["family"] != "extra_trees":
        raise ValueError("C0 reuses only frozen ExtraTrees modality heads")
    return ExtraTreesClassifier(
        n_estimators=int(model_config["n_estimators"]),
        max_depth=int(model_config["max_depth"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        max_features=model_config["max_features"],
        class_weight=model_config["class_weight"],
        random_state=int(seed),
        n_jobs=-1,
    )


def full_probability(model: ExtraTreesClassifier, values: np.ndarray) -> np.ndarray:
    partial = np.asarray(model.predict_proba(values), dtype=np.float64)
    output = np.full((len(values), NUM_CLASSES), 1e-12, dtype=np.float64)
    output[:, np.asarray(model.classes_, dtype=np.int64)] = partial
    output /= output.sum(axis=1, keepdims=True)
    return output


def prior_probability(labels: np.ndarray, classes: list[int], rows: int) -> np.ndarray:
    counts = np.asarray([np.sum(labels == class_id) for class_id in classes], dtype=np.float64)
    counts += 1.0
    counts /= counts.sum()
    output = np.full((rows, NUM_CLASSES), 1e-12, dtype=np.float64)
    output[:, np.asarray(classes, dtype=np.int64)] = counts
    output /= output.sum(axis=1, keepdims=True)
    return output


def calibrated_specialist_prediction(
    values: np.ndarray,
    available: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    train_group: np.ndarray,
    evaluation: np.ndarray,
    classes: list[int],
    model_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, float, ExtraTreesClassifier]:
    eligible = train_group[available[train_group]]
    if len(set(labels[eligible].tolist())) != len(classes):
        raise RuntimeError("conditional training split does not cover every frozen class")
    inner_log_probability = np.zeros((len(eligible), NUM_CLASSES), dtype=np.float64)
    eligible_users = users[eligible]
    for inner_number, inner_user in enumerate(sorted(set(eligible_users.tolist()))):
        inner_eval = np.flatnonzero(eligible_users == inner_user)
        inner_fit = np.flatnonzero(eligible_users != inner_user)
        model = make_model(model_config, seed + inner_number * 101)
        model.fit(values[eligible[inner_fit]], labels[eligible[inner_fit]])
        probability = full_probability(model, values[eligible[inner_eval]])
        inner_log_probability[inner_eval] = np.log(np.clip(probability, 1e-12, 1.0))
    temperature = fit_temperature(inner_log_probability, labels[eligible])
    model = make_model(model_config, seed + 997)
    model.fit(values[eligible], labels[eligible])
    probability = np.full((len(evaluation), NUM_CLASSES), 1e-12, dtype=np.float64)
    probability[:, np.asarray(classes, dtype=np.int64)] = 1.0 / len(classes)
    present = available[evaluation]
    if present.any():
        probability[present] = full_probability(model, values[evaluation[present]])
    probability /= probability.sum(axis=1, keepdims=True)
    return np.log(np.clip(probability, 1e-12, 1.0)) / temperature, temperature, model


def run_specialist(
    name: str,
    spec: dict[str, Any],
    values: np.ndarray,
    available: np.ndarray,
    model_config: dict[str, Any],
    labels: np.ndarray,
    users: np.ndarray,
    anchor: np.ndarray,
    eval_indices: np.ndarray,
    sample_ids: np.ndarray,
    seed: int,
    gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    classes = list(map(int, spec["classes"]))
    eval_group = eval_indices[np.isin(labels[eval_indices], classes)]
    group_users = users[eval_group]
    direct = np.zeros((len(eval_group), NUM_CLASSES), dtype=np.float64)
    zero = np.zeros_like(direct)
    shuffled = np.zeros_like(direct)
    prior = np.zeros_like(direct)
    temperatures: dict[str, float] = {}
    for fold_number, held_user in enumerate(sorted(set(group_users.tolist()))):
        local_eval = np.flatnonzero(group_users == held_user)
        outer_eval = eval_group[local_eval]
        train_group = np.flatnonzero(
            (users != held_user) & np.isin(labels, np.asarray(classes, dtype=np.int64))
        )
        logits, temperature, model = calibrated_specialist_prediction(
            values,
            available,
            labels,
            users,
            train_group,
            outer_eval,
            classes,
            model_config,
            seed + fold_number * 1009,
        )
        direct[local_eval] = logits
        eligible = train_group[available[train_group]]
        median = np.median(values[eligible], axis=0, keepdims=True)
        zero_probability = full_probability(model, np.repeat(median, len(outer_eval), axis=0))
        zero[local_eval] = np.log(np.clip(zero_probability, 1e-12, 1.0)) / temperature
        permutation = np.arange(len(outer_eval))
        present_local = available[outer_eval]
        selected = np.flatnonzero(present_local)
        rng = np.random.default_rng(seed + fold_number * 2027)
        permutation[selected] = selected[rng.permutation(len(selected))]
        shuffled_probability = np.full((len(outer_eval), NUM_CLASSES), 1e-12)
        shuffled_probability[:, np.asarray(classes)] = 1.0 / len(classes)
        if present_local.any():
            shuffled_probability[present_local] = full_probability(
                model, values[outer_eval][permutation][present_local]
            )
        shuffled_probability /= shuffled_probability.sum(axis=1, keepdims=True)
        shuffled[local_eval] = (
            np.log(np.clip(shuffled_probability, 1e-12, 1.0)) / temperature
        )
        prior[local_eval] = np.log(
            np.clip(
                prior_probability(labels[train_group], classes, len(outer_eval)),
                1e-12,
                1.0,
            )
        )
        temperatures[str(held_user)] = temperature

    group_labels = labels[eval_group]
    prediction = direct.argmax(axis=1)
    prior_prediction = prior.argmax(axis=1)
    shuffle_prediction = shuffled.argmax(axis=1)
    direct_metrics = metrics(direct, group_labels)
    prior_metrics = metrics(prior, group_labels)
    shuffle_metrics = metrics(shuffled, group_labels)
    zero_metrics = metrics(zero, group_labels)
    per_user: dict[str, Any] = {}
    for user in sorted(set(group_users.tolist())):
        selected = group_users == user
        per_user[user] = {
            "rows": int(selected.sum()),
            "available_rows": int(np.sum(available[eval_group][selected])),
            "direct_correct": int(np.sum(prediction[selected] == group_labels[selected])),
            "prior_correct": int(np.sum(prior_prediction[selected] == group_labels[selected])),
            "shuffle_correct": int(np.sum(shuffle_prediction[selected] == group_labels[selected])),
            "anchor_raw_correct": int(np.sum(anchor[eval_group][selected] == group_labels[selected])),
        }
    nonnegative = sum(
        value["direct_correct"] >= value["prior_correct"] for value in per_user.values()
    )
    gain_prior_pp = 100.0 * (direct_metrics["accuracy"] - prior_metrics["accuracy"])
    gain_shuffle_pp = 100.0 * (direct_metrics["accuracy"] - shuffle_metrics["accuracy"])
    checks = {
        "accuracy": direct_metrics["accuracy"] >= float(gate["minimum_accuracy"]),
        "balanced_accuracy": direct_metrics["balanced_accuracy"]
        >= float(gate["minimum_balanced_accuracy"]),
        "gain_over_prior": gain_prior_pp >= float(gate["minimum_gain_over_prior_pp"]),
        "gain_over_shuffle": gain_shuffle_pp >= float(gate["minimum_gain_over_shuffle_pp"]),
        "nonnegative_users": nonnegative
        >= int(gate["minimum_nonnegative_users_vs_prior"]),
    }
    comparison_prior = change_audit(group_labels, prior_prediction, prediction)
    comparison_prior["mcnemar_exact_pvalue"] = paired_exact_pvalue(
        group_labels, prior_prediction, prediction
    )
    result = {
        "name": name,
        "modality": spec["modality"],
        "classes": classes,
        "class_names": spec["class_names"],
        "scope": "oracle-known candidate group; not a deployable router",
        "metrics": direct_metrics,
        "prior_metrics": prior_metrics,
        "zero_metrics": zero_metrics,
        "shuffle_metrics": shuffle_metrics,
        "raw_anchor_metrics": metrics(
            np.log(
                np.clip(
                    np.eye(NUM_CLASSES, dtype=np.float64)[anchor[eval_group]] * 0.94
                    + (1.0 - np.eye(NUM_CLASSES)[anchor[eval_group]])
                    * (0.06 / (NUM_CLASSES - 1)),
                    1e-12,
                    1.0,
                )
            ),
            group_labels,
        ),
        "vs_prior": comparison_prior,
        "gain_over_prior_pp": gain_prior_pp,
        "gain_over_shuffle_pp": gain_shuffle_pp,
        "per_user": per_user,
        "temperature_by_outer_fold": temperatures,
        "selection_gate": {"passed": bool(all(checks.values())), "checks": checks},
    }
    arrays = {
        "sample_ids": sample_ids[eval_group],
        "labels": group_labels,
        "users": group_users,
        "available": available[eval_group],
        "direct_probability": softmax(direct).astype(np.float32),
        "prior_probability": softmax(prior).astype(np.float32),
        "zero_probability": softmax(zero).astype(np.float32),
        "shuffle_probability": softmax(shuffled).astype(np.float32),
    }
    return result, arrays


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if bool(config.get("h3_code_path")):
        raise ValueError("P99-C0 cannot expose H3")
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    cohort, eval_indices, _ = build_cohort(
        "h1", d0_config, args.depth_features, args.split_source, args.e0_base
    )
    skeleton_config = json.loads(resolve(config["skeleton_config"]).read_text(encoding="utf-8"))
    imu_config = json.loads(resolve(config["imu_config"]).read_text(encoding="utf-8"))
    skeleton_values, skeleton_audit = load_skeleton_descriptor(
        skeleton_config, cohort.sample_ids, cohort.users
    )
    imu_values, imu_available, _, imu_audit = load_imu_descriptor(
        imu_config, cohort.sample_ids, cohort.users
    )
    sources = {
        "skeleton": (
            skeleton_values,
            np.ones(len(cohort.labels), dtype=bool),
            skeleton_config["classifier"],
        ),
        "imu": (imu_values, imu_available, imu_config["classifier"]),
    }
    results: dict[str, Any] = {}
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for number, (name, spec) in enumerate(config["specialists"].items()):
        values, available, model_config = sources[spec["modality"]]
        result, arrays = run_specialist(
            name,
            spec,
            values,
            available,
            model_config,
            cohort.labels,
            cohort.users,
            cohort.anchor_prediction,
            eval_indices,
            cohort.sample_ids,
            int(config["seed"]) + number * 10000,
            config["selection_gate"],
        )
        results[name] = result
        artifacts[name] = arrays

    report = {
        "stage": "P99_C0_H1_conditional_confusion_lower_bound",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": canonical_hash(config),
        "scope": config["scope"],
        "specialists": results,
        "descriptor_audit": {"skeleton": skeleton_audit, "imu": imu_audit},
        "leakage_audit": {
            "candidate_groups_pre_registered": True,
            "outer_user_disjoint": True,
            "temperature_inner_user_oof": True,
            "true_group_used_only_for_conditional_lower_bound": True,
            "router_trained_or_evaluated": False,
            "h2_h3_accessed": False,
            "h3_code_path_present": False,
        },
        "h2_policy": config["h2_policy"],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, arrays in artifacts.items():
        np.savez_compressed(output / f"{name}_h1_predictions.npz", **arrays)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {
        name: {
            "rows": value["metrics"]["total"],
            "correct": value["metrics"]["correct"],
            "accuracy": value["metrics"]["accuracy"],
            "balanced_accuracy": value["metrics"]["balanced_accuracy"],
            "prior_correct": value["prior_metrics"]["correct"],
            "shuffle_correct": value["shuffle_metrics"]["correct"],
            "gain_prior_pp": value["gain_over_prior_pp"],
            "gain_shuffle_pp": value["gain_over_shuffle_pp"],
            "per_user": value["per_user"],
            "gate": value["selection_gate"],
        }
        for name, value in results.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
