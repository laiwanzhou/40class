"""P99-I0 left/right intensity-duration IMU source-OOF expert."""

from __future__ import annotations

import argparse
import csv
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
from p99_visual_oof_experts import extended_audit


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_imu_statistics_i0.json"
DEVICE_COUNT = 5
SIGNALS_PER_DEVICE = 36
STATISTICS_PER_SIGNAL = 25
NON_SPECTRAL_STATISTICS = 21
PHASE_AND_MASK_PER_DEVICE = 114
DEVICE_FEATURES = SIGNALS_PER_DEVICE * STATISTICS_PER_SIGNAL + PHASE_AND_MASK_PER_DEVICE
PAIR_AND_DURATION_FEATURES = 152


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99 time-domain IMU expert")
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


def non_spectral_feature_indices() -> np.ndarray:
    selected: list[int] = []
    for device in range(DEVICE_COUNT):
        base = device * DEVICE_FEATURES
        for signal in range(SIGNALS_PER_DEVICE):
            start = base + signal * STATISTICS_PER_SIGNAL
            selected.extend(range(start, start + NON_SPECTRAL_STATISTICS))
        selected.extend(
            range(
                base + SIGNALS_PER_DEVICE * STATISTICS_PER_SIGNAL,
                base + DEVICE_FEATURES,
            )
        )
    pair_start = DEVICE_COUNT * DEVICE_FEATURES
    selected.extend(range(pair_start, pair_start + PAIR_AND_DURATION_FEATURES))
    return np.asarray(selected, dtype=np.int64)


def read_index_without_labels(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # class_id is intentionally not accessed.
    return (
        np.asarray([row["sample_id"] for row in rows]),
        np.asarray([row["user_id"] for row in rows]),
        np.asarray([int(row["cache_index"]) for row in rows], dtype=np.int64),
        np.asarray([row["usable"] == "1" for row in rows], dtype=bool),
        np.asarray([int(row["device_count"]) for row in rows], dtype=np.int64),
    )


def load_descriptor(
    config: dict[str, Any], cohort_ids: np.ndarray, cohort_users: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    feature_path = resolve(config["orientation_feature_cache"])
    index_path = resolve(config["orientation_index"])
    source_ids, source_users, cache_indices, usable, device_count = read_index_without_labels(
        index_path
    )
    source = np.load(feature_path, mmap_mode="r")
    if len(source) <= int(cache_indices.max()):
        raise RuntimeError("IMU feature cache/index contract changed")
    lookup = {value: index for index, value in enumerate(source_ids)}
    missing = [value for value in cohort_ids if value not in lookup]
    if missing:
        raise KeyError(f"IMU cache misses {len(missing)} P99 cohort rows")
    row_order = np.asarray([lookup[value] for value in cohort_ids], dtype=np.int64)
    if not np.array_equal(source_users[row_order], cohort_users):
        raise RuntimeError("IMU cache/P99 cohort user alignment differs")
    include_spectral = bool(config.get("include_spectral_statistics", False))
    columns = (
        np.arange(source.shape[1], dtype=np.int64)
        if include_spectral
        else non_spectral_feature_indices()
    )
    descriptor = np.asarray(source[np.ix_(cache_indices[row_order], columns)], dtype=np.float32)
    descriptor = np.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)
    available = usable[row_order] & (device_count[row_order] > 0)
    audit = {
        "feature_cache": str(feature_path),
        "index": str(index_path),
        "cache_rows": int(len(source)),
        "cohort_rows": int(len(cohort_ids)),
        "original_dimension": int(source.shape[1]),
        "descriptor_dimension": int(descriptor.shape[1]),
        "time_domain_core_dimension": int(len(non_spectral_feature_indices())),
        "include_spectral_statistics": include_spectral,
        "removed_spectral_features_per_signal": 0 if include_spectral else 4,
        "available_rows": int(available.sum()),
        "missing_rows": int((~available).sum()),
        "device_count_histogram": np.bincount(
            device_count[row_order], minlength=DEVICE_COUNT + 1
        ).tolist(),
        "all_finite": bool(np.isfinite(descriptor).all()),
        "embedded_index_class_id_used": False,
    }
    return descriptor, available, device_count[row_order], audit


def make_model(config: dict[str, Any], seed: int) -> ExtraTreesClassifier:
    model = config["classifier"]
    if model["family"] != "extra_trees":
        raise ValueError("I0 supports only the frozen ExtraTrees family")
    return ExtraTreesClassifier(
        n_estimators=int(model["n_estimators"]),
        max_depth=int(model["max_depth"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        max_features=model["max_features"],
        class_weight=model["class_weight"],
        random_state=int(seed),
        n_jobs=-1,
    )


def full_probability(model: ExtraTreesClassifier, values: np.ndarray) -> np.ndarray:
    partial = np.asarray(model.predict_proba(values), dtype=np.float64)
    output = np.full((len(values), NUM_CLASSES), 1e-12, dtype=np.float64)
    output[:, np.asarray(model.classes_, dtype=np.int64)] = partial
    output /= output.sum(axis=1, keepdims=True)
    return output


def calibrated_outer_prediction(
    values: np.ndarray,
    available: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, float, ExtraTreesClassifier]:
    eligible_train = train[available[train]]
    if not len(eligible_train):
        raise RuntimeError("outer IMU training split has no available rows")
    inner_log_probability = np.zeros((len(eligible_train), NUM_CLASSES), dtype=np.float64)
    eligible_users = users[eligible_train]
    for inner_number, inner_user in enumerate(sorted(set(eligible_users.tolist()))):
        inner_eval = np.flatnonzero(eligible_users == inner_user)
        inner_fit = np.flatnonzero(eligible_users != inner_user)
        model = make_model(config, seed + inner_number * 101)
        model.fit(values[eligible_train[inner_fit]], labels[eligible_train[inner_fit]])
        probability = full_probability(model, values[eligible_train[inner_eval]])
        inner_log_probability[inner_eval] = np.log(np.clip(probability, 1e-12, 1.0))
    temperature = fit_temperature(inner_log_probability, labels[eligible_train])
    model = make_model(config, seed + 997)
    model.fit(values[eligible_train], labels[eligible_train])
    probability = np.full((len(evaluation), NUM_CLASSES), 1.0 / NUM_CLASSES)
    present_local = available[evaluation]
    if present_local.any():
        probability[present_local] = full_probability(
            model, values[evaluation[present_local]]
        )
    return np.log(np.clip(probability, 1e-12, 1.0)) / temperature, temperature, model


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if bool(config.get("h3_code_path")):
        raise ValueError("P99-I0 cannot expose H3")
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    cohort, eval_indices, eval_ids = build_cohort(
        "h1", d0_config, args.depth_features, args.split_source, args.e0_base
    )
    values, available, device_count, descriptor_audit = load_descriptor(
        config, cohort.sample_ids, cohort.users
    )
    evaluation_users = cohort.users[eval_indices]
    direct = np.zeros((len(eval_indices), NUM_CLASSES), dtype=np.float64)
    zero = np.zeros_like(direct)
    shuffled = np.zeros_like(direct)
    temperatures: dict[str, float] = {}
    for fold_number, held_user in enumerate(sorted(set(evaluation_users.tolist()))):
        local_eval = np.flatnonzero(evaluation_users == held_user)
        outer_eval = eval_indices[local_eval]
        train = np.flatnonzero(cohort.users != held_user)
        if set(cohort.users[train]) & set(cohort.users[outer_eval]):
            raise RuntimeError("outer train/evaluation IMU users overlap")
        logits, temperature, model = calibrated_outer_prediction(
            values,
            available,
            cohort.labels,
            cohort.users,
            train,
            outer_eval,
            config,
            int(config["seed"]) + fold_number * 1009,
        )
        direct[local_eval] = logits
        eligible_train = train[available[train]]
        training_median = np.median(values[eligible_train], axis=0, keepdims=True)
        zero_probability = full_probability(
            model, np.repeat(training_median, len(outer_eval), axis=0)
        )
        zero[local_eval] = np.log(np.clip(zero_probability, 1e-12, 1.0)) / temperature
        permutation = np.arange(len(outer_eval))
        rng = np.random.default_rng(int(config["seed"]) + fold_number * 2027)
        for count in sorted(set(device_count[outer_eval].tolist())):
            selected = np.flatnonzero(device_count[outer_eval] == count)
            permutation[selected] = selected[rng.permutation(len(selected))]
        shuffled_probability = np.full((len(outer_eval), NUM_CLASSES), 1.0 / NUM_CLASSES)
        present_local = available[outer_eval]
        if present_local.any():
            shuffled_probability[present_local] = full_probability(
                model, values[outer_eval][permutation][present_local]
            )
        shuffled[local_eval] = (
            np.log(np.clip(shuffled_probability, 1e-12, 1.0)) / temperature
        )
        temperatures[str(held_user)] = temperature

    labels = cohort.labels[eval_indices]
    anchor = cohort.anchor_prediction[eval_indices]
    eval_available = available[eval_indices]
    direct_probability = softmax(direct).astype(np.float32)
    prediction = direct_probability.argmax(axis=1)
    per_user: dict[str, Any] = {}
    for user in sorted(set(evaluation_users.tolist())):
        selected = evaluation_users == user
        anchor_correct = anchor[selected] == labels[selected]
        expert_correct = prediction[selected] == labels[selected]
        per_user[user] = {
            "rows": int(selected.sum()),
            "available_rows": int(np.sum(eval_available[selected])),
            "correct": int(expert_correct.sum()),
            "anchor_correct": int(anchor_correct.sum()),
            "rescue": int(np.sum(~anchor_correct & expert_correct)),
            "harm": int(np.sum(anchor_correct & ~expert_correct)),
        }
    available_metrics = (
        metrics(direct[eval_available], labels[eval_available])
        if eval_available.any()
        else None
    )
    result = {
        "stage": "P99_I0_H1_time_domain_IMU_expert",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": canonical_hash(config),
        "protocol": "E0+H1 outer LOUO; missing IMU excluded from fit and mapped to uniform; no spectral features or old head",
        "metrics": metrics(direct, labels),
        "available_row_metrics": available_metrics,
        "zero_metrics": metrics(zero, labels),
        "shuffle_metrics": metrics(shuffled, labels),
        "vs_anchor": change_audit(labels, anchor, prediction),
        "extended_audit": extended_audit(
            labels, anchor, direct_probability, config["focus_groups"], evaluation_users
        ),
        "per_user": per_user,
        "temperature_by_outer_fold": temperatures,
        "descriptor_audit": descriptor_audit,
        "leakage_audit": {
            "orientation_feature_cache_label_free": True,
            "embedded_index_class_id_used": False,
            "outer_user_disjoint": True,
            "temperature_inner_user_oof": True,
            "missing_rows_excluded_from_training": True,
            "shuffle_device_count_stratified": True,
            "old_imu_head_used": False,
            "h2_h3_accessed": False,
            "h3_code_path_present": False,
        },
        "selection_rule": config["selection"],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "h1_predictions.npz",
        sample_ids=eval_ids,
        labels=labels,
        users=evaluation_users,
        available=eval_available,
        device_count=device_count[eval_indices],
        anchor_prediction=anchor,
        direct_probability=direct_probability,
        direct_logits=direct.astype(np.float32),
        zero_probability=softmax(zero).astype(np.float32),
        shuffle_probability=softmax(shuffled).astype(np.float32),
    )
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {
        "correct": result["metrics"]["correct"],
        "top5": result["metrics"]["top5"],
        "balanced_accuracy": result["metrics"]["balanced_accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
        "available_correct": available_metrics["correct"] if available_metrics else None,
        "available_rows": int(eval_available.sum()),
        "zero_correct": result["zero_metrics"]["correct"],
        "shuffle_correct": result["shuffle_metrics"]["correct"],
        "vs_anchor": result["vs_anchor"],
        "per_user": result["per_user"],
        "temperatures": temperatures,
        "descriptor_audit": descriptor_audit,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
