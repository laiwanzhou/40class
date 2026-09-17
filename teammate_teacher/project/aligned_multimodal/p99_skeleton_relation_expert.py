"""P99-S0 hand/arm relation Skeleton source-OOF expert."""

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
DEFAULT_CONFIG = HERE / "configs/p99_skeleton_relation_s0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99 Skeleton relation expert")
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


def read_cache_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # class_id is deliberately not accessed; controlled cohort labels come
    # from the P99 split contract instead of an embedded cache field.
    return (
        np.asarray([row["sample_id"] for row in rows]),
        np.asarray([row["user_id"] for row in rows]),
    )


def masked_robust_statistics(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    count = np.maximum(valid.sum(axis=1), 1)
    safe = np.where(valid, values, 0.0)
    mean = safe.sum(axis=1) / count
    centered = np.where(valid, values - mean[:, None], 0.0)
    std = np.sqrt(np.sum(centered**2, axis=1) / count)
    minimum = np.min(np.where(valid, values, np.inf), axis=1)
    maximum = np.max(np.where(valid, values, -np.inf), axis=1)
    minimum[~np.isfinite(minimum)] = 0.0
    maximum[~np.isfinite(maximum)] = 0.0
    quantile_source = np.where(valid, values, np.nan)
    with np.errstate(all="ignore"):
        quantiles = np.nanquantile(
            quantile_source, (0.10, 0.25, 0.50, 0.75, 0.90), axis=1
        )
    pair_valid = valid[:, 1:] & valid[:, :-1]
    difference = np.where(pair_valid, values[:, 1:] - values[:, :-1], 0.0)
    pair_count = np.maximum(pair_valid.sum(axis=1), 1)
    mean_abs_difference = np.sum(np.abs(difference), axis=1) / pair_count
    rms_difference = np.sqrt(np.sum(difference**2, axis=1) / pair_count)
    first = np.argmax(valid, axis=1)
    last = values.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    batch = np.arange(len(values))[:, None]
    signal = np.arange(values.shape[2])[None, :]
    endpoint_delta = safe[batch, last, signal] - safe[batch, first, signal]
    endpoint_delta[~valid.any(axis=1)] = 0.0
    blocks = (
        mean,
        std,
        np.sqrt(np.sum(safe**2, axis=1) / count),
        minimum,
        maximum,
        maximum - minimum,
        *quantiles,
        quantiles[3] - quantiles[1],
        mean_abs_difference,
        rms_difference,
        endpoint_delta,
    )
    return np.nan_to_num(
        np.stack(blocks, axis=-1).reshape(len(values), -1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def masked_phase_statistics(
    values: np.ndarray, valid: np.ndarray, bins: int = 8
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for indices in np.array_split(np.arange(values.shape[1]), bins):
        part = np.where(valid[:, indices], values[:, indices], np.nan)
        with np.errstate(all="ignore"):
            blocks.extend((np.nanmean(part, axis=1), np.nanstd(part, axis=1)))
    return np.nan_to_num(
        np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def relation_descriptor(
    skeleton: np.ndarray,
    feature_mask: np.ndarray,
    relations: np.ndarray,
    relation_mask: np.ndarray,
    frame_quality: np.ndarray,
    arm_joints: list[int],
    relation_indices: list[int],
    include_global_context: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = len(skeleton)
    skeleton = np.asarray(skeleton, dtype=np.float32).reshape(count, 32, 17, 13)
    feature_mask = np.asarray(feature_mask, dtype=bool).reshape(count, 32, 17, 13)
    relations = np.asarray(relations, dtype=np.float32).reshape(count, 32, 18)
    relation_mask = np.asarray(relation_mask, dtype=bool).reshape(count, 32, 18)
    frame_quality = np.asarray(frame_quality, dtype=np.float32).reshape(count, 32)
    joints = np.asarray(arm_joints, dtype=np.int64)
    relation_ids = np.asarray(relation_indices, dtype=np.int64)
    arm_values = skeleton[:, :, joints, :12].reshape(count, 32, -1)
    arm_valid = feature_mask[:, :, joints, :12].reshape(count, 32, -1)
    relation_values = relations[:, :, relation_ids]
    relation_valid = relation_mask[:, :, relation_ids]
    signals = np.concatenate((arm_values, relation_values), axis=2)
    signal_valid = np.concatenate((arm_valid, relation_valid), axis=2)

    blocks: list[tuple[str, np.ndarray]] = [
        ("robust_hand_arm_relation", masked_robust_statistics(signals, signal_valid)),
        ("phase8_hand_arm_relation", masked_phase_statistics(signals, signal_valid, 8)),
        (
            "early_relation",
            masked_robust_statistics(relation_values[:, :16], relation_valid[:, :16]),
        ),
        (
            "late_relation",
            masked_robust_statistics(relation_values[:, 16:], relation_valid[:, 16:]),
        ),
    ]
    if include_global_context:
        stream = skeleton[..., :12].reshape(count, 32, 17, 4, 3)
        stream_valid = feature_mask[..., :12].reshape(count, 32, 17, 4, 3)
        global_values = np.linalg.norm(stream, axis=-1).reshape(count, 32, -1)
        global_valid = np.all(stream_valid, axis=-1).reshape(count, 32, -1)
        remaining_relations = np.asarray([7, 8, 10], dtype=np.int64)
        global_values = np.concatenate(
            (global_values, relations[:, :, remaining_relations]), axis=2
        )
        global_valid = np.concatenate(
            (global_valid, relation_mask[:, :, remaining_relations]), axis=2
        )
        blocks.extend(
            (
                (
                    "robust_global_body_context",
                    masked_robust_statistics(global_values, global_valid),
                ),
                (
                    "phase8_global_body_context",
                    masked_phase_statistics(global_values, global_valid, 8),
                ),
            )
        )
    joint_valid = feature_mask[:, :, joints, 0]
    coverage = np.concatenate(
        (
            joint_valid.mean(axis=1),
            joint_valid.std(axis=1),
            relation_valid.mean(axis=1),
            relation_valid.std(axis=1),
            frame_quality.mean(axis=1, keepdims=True),
            frame_quality.std(axis=1, keepdims=True),
        ),
        axis=1,
    ).astype(np.float32)
    blocks.append(("coverage", coverage))
    if include_global_context:
        all_joint_valid = feature_mask[..., 0]
        blocks.append(
            (
                "global_joint_coverage",
                np.concatenate(
                    (all_joint_valid.mean(axis=1), all_joint_valid.std(axis=1)), axis=1
                ).astype(np.float32),
            )
        )
    descriptor = np.concatenate([value for _, value in blocks], axis=1)
    descriptor = np.nan_to_num(
        descriptor, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    offset = 0
    slices: dict[str, list[int]] = {}
    for name, value in blocks:
        slices[name] = [offset, offset + value.shape[1]]
        offset += value.shape[1]
    audit = {
        "rows": int(count),
        "feature_dimension": int(descriptor.shape[1]),
        "arm_signal_count": int(arm_values.shape[2]),
        "relation_signal_count": int(relation_values.shape[2]),
        "include_global_context": bool(include_global_context),
        "global_context_signal_count": 71 if include_global_context else 0,
        "feature_slices": slices,
        "mean_arm_validity": float(arm_valid.mean()),
        "mean_relation_validity": float(relation_valid.mean()),
        "mean_frame_quality": float(frame_quality.mean()),
        "all_finite": bool(np.isfinite(descriptor).all()),
    }
    return descriptor, audit


def load_descriptor(
    config: dict[str, Any], cohort_ids: np.ndarray, cohort_users: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    cache = resolve(config["motion_cache"])
    cache_ids, cache_users = read_cache_rows(cache / "rows.csv")
    lookup = {value: index for index, value in enumerate(cache_ids)}
    missing = [value for value in cohort_ids if value not in lookup]
    if missing:
        raise KeyError(f"Skeleton cache misses {len(missing)} P99 cohort rows")
    order = np.asarray([lookup[value] for value in cohort_ids], dtype=np.int64)
    if not np.array_equal(cache_users[order], cohort_users):
        raise RuntimeError("Skeleton cache/P99 cohort user alignment differs")
    arrays = {
        "skeleton": np.load(cache / "skeleton_features.npy", mmap_mode="r")[order],
        "feature_mask": np.load(cache / "skeleton_feature_mask.npy", mmap_mode="r")[order],
        "relations": np.load(cache / "skeleton_relations.npy", mmap_mode="r")[order],
        "relation_mask": np.load(cache / "skeleton_relation_mask.npy", mmap_mode="r")[order],
        "frame_quality": np.load(cache / "skeleton_frame_quality.npy", mmap_mode="r")[order],
    }
    descriptor, audit = relation_descriptor(
        arrays["skeleton"],
        arrays["feature_mask"],
        arrays["relations"],
        arrays["relation_mask"],
        arrays["frame_quality"],
        config["arm_joint_indices"],
        config["relation_indices"],
        bool(config.get("include_global_context", False)),
    )
    audit.update(
        {
            "cache": str(cache),
            "cache_rows": int(len(cache_ids)),
            "cohort_rows": int(len(cohort_ids)),
            "body_local_contract": True,
            "embedded_cache_class_id_used": False,
        }
    )
    return descriptor, audit


def make_model(config: dict[str, Any], seed: int) -> ExtraTreesClassifier:
    model = config["classifier"]
    if model["family"] != "extra_trees":
        raise ValueError("S0 supports only the frozen ExtraTrees family")
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
    labels: np.ndarray,
    users: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, float, ExtraTreesClassifier]:
    inner_log_probability = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    train_users = users[train]
    for inner_number, inner_user in enumerate(sorted(set(train_users.tolist()))):
        inner_eval_local = np.flatnonzero(train_users == inner_user)
        inner_fit_local = np.flatnonzero(train_users != inner_user)
        model = make_model(config, seed + inner_number * 101)
        model.fit(values[train[inner_fit_local]], labels[train[inner_fit_local]])
        inner_probability = full_probability(model, values[train[inner_eval_local]])
        inner_log_probability[inner_eval_local] = np.log(
            np.clip(inner_probability, 1e-12, 1.0)
        )
    temperature = fit_temperature(inner_log_probability, labels[train])
    model = make_model(config, seed + 997)
    model.fit(values[train], labels[train])
    outer_probability = full_probability(model, values[evaluation])
    return np.log(np.clip(outer_probability, 1e-12, 1.0)) / temperature, temperature, model


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if bool(config.get("h3_code_path")):
        raise ValueError("P99-S0 cannot expose H3")
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    cohort, eval_indices, eval_ids = build_cohort(
        "h1", d0_config, args.depth_features, args.split_source, args.e0_base
    )
    values, descriptor_audit = load_descriptor(config, cohort.sample_ids, cohort.users)
    evaluation_users = cohort.users[eval_indices]
    direct = np.zeros((len(eval_indices), NUM_CLASSES), dtype=np.float64)
    zero = np.zeros_like(direct)
    shuffled = np.zeros_like(direct)
    temperatures: dict[str, float] = {}
    importance_by_fold: dict[str, dict[str, float]] = {}
    for fold_number, held_user in enumerate(sorted(set(evaluation_users.tolist()))):
        local_eval = np.flatnonzero(evaluation_users == held_user)
        outer_eval = eval_indices[local_eval]
        train = np.flatnonzero(cohort.users != held_user)
        if set(cohort.users[train]) & set(cohort.users[outer_eval]):
            raise RuntimeError("outer train/evaluation Skeleton users overlap")
        logits, temperature, model = calibrated_outer_prediction(
            values,
            cohort.labels,
            cohort.users,
            train,
            outer_eval,
            config,
            int(config["seed"]) + fold_number * 1009,
        )
        direct[local_eval] = logits
        training_median = np.median(values[train], axis=0, keepdims=True)
        zero_probability = full_probability(
            model, np.repeat(training_median, len(outer_eval), axis=0)
        )
        zero[local_eval] = np.log(np.clip(zero_probability, 1e-12, 1.0)) / temperature
        rng = np.random.default_rng(int(config["seed"]) + fold_number * 2027)
        permutation = rng.permutation(len(outer_eval))
        shuffled_probability = full_probability(model, values[outer_eval][permutation])
        shuffled[local_eval] = (
            np.log(np.clip(shuffled_probability, 1e-12, 1.0)) / temperature
        )
        temperatures[str(held_user)] = temperature
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
        importance_by_fold[str(held_user)] = {
            name: float(importance[start:end].sum())
            for name, (start, end) in descriptor_audit["feature_slices"].items()
        }

    labels = cohort.labels[eval_indices]
    anchor = cohort.anchor_prediction[eval_indices]
    direct_probability = softmax(direct).astype(np.float32)
    prediction = direct_probability.argmax(axis=1)
    per_user: dict[str, Any] = {}
    for user in sorted(set(evaluation_users.tolist())):
        selected = evaluation_users == user
        anchor_correct = anchor[selected] == labels[selected]
        expert_correct = prediction[selected] == labels[selected]
        per_user[user] = {
            "rows": int(selected.sum()),
            "correct": int(expert_correct.sum()),
            "anchor_correct": int(anchor_correct.sum()),
            "rescue": int(np.sum(~anchor_correct & expert_correct)),
            "harm": int(np.sum(anchor_correct & ~expert_correct)),
        }
    result = {
        "stage": "P99_S0_H1_skeleton_relation_expert",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": canonical_hash(config),
        "protocol": "E0+H1 outer LOUO; inner-user-OOF probability temperature; no old head/H2/H3 selection",
        "metrics": metrics(direct, labels),
        "zero_metrics": metrics(zero, labels),
        "shuffle_metrics": metrics(shuffled, labels),
        "vs_anchor": change_audit(labels, anchor, prediction),
        "extended_audit": extended_audit(
            labels, anchor, direct_probability, config["focus_groups"], evaluation_users
        ),
        "per_user": per_user,
        "temperature_by_outer_fold": temperatures,
        "feature_importance_by_outer_fold": importance_by_fold,
        "descriptor_audit": descriptor_audit,
        "leakage_audit": {
            "raw_motion_cache_label_free": True,
            "embedded_cache_class_id_used": False,
            "outer_user_disjoint": True,
            "temperature_inner_user_oof": True,
            "old_skeleton_head_used": False,
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
