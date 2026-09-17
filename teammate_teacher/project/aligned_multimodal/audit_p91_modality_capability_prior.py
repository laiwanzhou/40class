"""Mine a source-only soft modality prior for recurrent Top-K confusions.

This is a diagnostic capability audit, not a deployable reranker.  It expands
the P96 primary-visual OOF hard pool to every confusion pair repeated across
at least two source users.  One feature recipe and one Ridge alpha per modality
are reused from the preceding P91 feature-capability audit; no pair receives a
separately selected recipe.  Session posteriors are rebuilt strictly
leave-one-source-user-out.  The resulting source prior is frozen before H2 is
used once for confirmation.  H3 is never read.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from audit_p87_sequence_decoder import (
    DEFAULT_TRAIN_METADATA,
    align_metadata,
    build_sessions,
    decode_unique_beam_posterior,
    fit_transition_model,
)
from audit_p91_confusion_feature_capability import (
    IMU_DEVICES,
    JOINT_NAMES,
    MOTION_CACHE,
    P96_CACHE,
    FeatureBlock,
    concatenate_blocks,
    imu_feature_builders,
    l2_normalize,
    load_audit_data,
    make_probe,
    pair_key,
    read_csv,
    skeleton_feature_builders,
    skeleton_signal_block,
    temporal_summary,
    visual_feature_builders,
)
from audit_p91_p87_session_candidate_probe import (
    BEAM_WIDTH,
    NUM_CLASSES,
    P85_TEACHER,
    SESSION_GAP_SECONDS,
    TRANSITION_WEIGHT,
    TRIGRAM_BACKOFF,
)
from p90_teacher_common import REPO_ROOT, load_protocol
from p96_visual_top5_hard_pool_audit import (
    align_indices,
    load_class_names,
    load_primary_visual_scores,
)


OUTPUT = REPO_ROOT / "runs/p91_modality_capability_prior_v1"
POOL_ROOT = REPO_ROOT / "runs/p96_primary_visual_top5_hard_pools_v1"
PREVIOUS_AUDIT = REPO_ROOT / "runs/p91_confusion_feature_capability_v1/summary.json"

MODALITIES = ("visual", "skeleton", "imu", "session")
FROZEN_RECIPES = {
    "visual": ("visual_workspace_temporal", 10000.0),
    "skeleton": ("skeleton_hand_head_early_late", 1000.0),
    "imu": ("imu_arms_full_trial_statistics", 10.0),
}
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260821


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    left: int
    right: int
    left_name: str
    right_name: str
    source_confusions: int
    source_confusion_users: int
    h2_confusions: int
    h2_confusion_users: int


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def safe_balanced_accuracy(labels: np.ndarray, prediction: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float(np.mean(labels == prediction))
    return float(balanced_accuracy_score(labels, prediction))


def load_pairs() -> list[PairSpec]:
    rows = read_csv(POOL_ROOT / "source_pair_pools.csv")
    pairs = [
        PairSpec(
            pair_id=row["pair_pool_id"],
            left=int(row["class_a_id"]),
            right=int(row["class_b_id"]),
            left_name=row["class_a_name"],
            right_name=row["class_b_name"],
            source_confusions=int(row["source_samples"]),
            source_confusion_users=int(row["source_users"]),
            h2_confusions=int(row["H2_samples"]),
            h2_confusion_users=int(row["H2_users"]),
        )
        for row in rows
        if row["pool_tier"] == "shared_cross_user"
    ]
    if len(pairs) != 25:
        raise RuntimeError(f"expected 25 frozen cross-user pairs, got {len(pairs)}")
    return pairs


def validate_frozen_recipes() -> None:
    summary = json.loads(PREVIOUS_AUDIT.read_text(encoding="utf-8"))
    for modality, (feature_group, alpha) in FROZEN_RECIPES.items():
        selected = summary["selected_recipes"][modality]
        if selected["feature_group"] != feature_group or float(selected["alpha"]) != alpha:
            raise RuntimeError(f"frozen {modality} recipe differs from previous audit")


def source_evidence_tier(pair: PairSpec) -> str:
    if pair.source_confusions >= 5 and pair.source_confusion_users >= 3:
        return "strong"
    if pair.source_confusions >= 3 and pair.source_confusion_users >= 2:
        return "medium"
    return "weak"


def primary_visual_scores(data: Any) -> np.ndarray:
    score_ids, scores = load_primary_visual_scores()
    indices = align_indices(score_ids, data.sample_ids)
    return np.asarray(scores[indices], dtype=np.float64)


def _flat_block(values: np.ndarray, prefix: str) -> FeatureBlock:
    flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    return FeatureBlock(flat, [f"{prefix}:dim_{index}" for index in range(flat.shape[1])])


def visual_mechanism_blocks(data: Any) -> dict[str, FeatureBlock]:
    dense = np.asarray(
        np.load(P96_CACHE / "features.npy", mmap_mode="r")[data.analysis_indices],
        dtype=np.float32,
    )
    dense = l2_normalize(dense)
    return {
        "scene_temporal": _flat_block(dense[:, (0, 3, 6, 9)], "scene"),
        "person_temporal": _flat_block(dense[:, (1, 4, 7, 10)], "person"),
        "workspace_temporal": _flat_block(dense[:, (2, 5, 8, 11)], "workspace"),
        "hand_interaction_temporal": _flat_block(
            dense[:, (14, 17, 20, 23)], "hand_interaction"
        ),
    }


def _motion_cache_index(sample_ids: np.ndarray) -> np.ndarray:
    rows = read_csv(MOTION_CACHE / "rows.csv")
    lookup = {row["sample_id"]: index for index, row in enumerate(rows)}
    return np.asarray([lookup[str(value)] for value in sample_ids], dtype=np.int64)


def skeleton_mechanism_blocks(data: Any) -> dict[str, FeatureBlock]:
    index = _motion_cache_index(data.sample_ids)
    skeleton = np.asarray(
        np.load(MOTION_CACHE / "skeleton_features.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )
    joint_mask = np.asarray(
        np.load(MOTION_CACHE / "skeleton_joint_mask.npy", mmap_mode="r")[index],
        dtype=bool,
    )
    relations = np.asarray(
        np.load(MOTION_CACHE / "skeleton_relations.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )
    relation_mask = np.asarray(
        np.load(MOTION_CACHE / "skeleton_relation_mask.npy", mmap_mode="r")[index],
        dtype=bool,
    )

    def recipe(joints: list[int], relation_ids: list[int], prefix: str) -> FeatureBlock:
        signals, mask, names = skeleton_signal_block(
            skeleton, joint_mask, relations, relation_mask, joints, relation_ids
        )
        return concatenate_blocks(
            (
                temporal_summary(signals[:, 0], mask[:, 0], names, prefix="early"),
                temporal_summary(signals[:, 1], mask[:, 1], names, prefix="late"),
            )
        )

    return {
        "wrists": recipe([13, 16], [0, 1, 2, 3, 4, 15, 16], "wrists"),
        "elbows": recipe([12, 15], [5, 6], "elbows"),
        "shoulders": recipe([11, 14], [9], "shoulders"),
        "hand_head_relation": recipe([9, 10, 13, 16], [0, 1, 2, 3, 4], "hand_head"),
        "body": recipe([0, 7, 8, 9, 10], [14, 17], "body"),
    }


def imu_mechanism_blocks(data: Any) -> dict[str, FeatureBlock]:
    index = _motion_cache_index(data.sample_ids)
    global_statistics = np.asarray(
        np.load(MOTION_CACHE / "imu_global_statistics.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )
    global_mask = np.asarray(
        np.load(MOTION_CACHE / "imu_global_mask.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )
    stat_names = (
        "mean",
        "std",
        "rms",
        "minimum",
        "maximum",
        "range",
        "mean_abs_difference",
        "mean_squared_difference",
    )
    channels = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

    def global_block(devices: list[int], prefix: str) -> FeatureBlock:
        values = global_statistics[:, devices].reshape(len(global_statistics), -1)
        coverage = global_mask[:, devices].reshape(len(global_mask), -1)
        names = [
            f"full_trial:{IMU_DEVICES[device]}:{channel}:{statistic}"
            for device in devices
            for channel in channels
            for statistic in stat_names
        ]
        names.extend(
            f"coverage:{IMU_DEVICES[device]}:{kind}"
            for device in devices
            for kind in ("available", "camera_span_fraction")
        )
        return FeatureBlock(
            np.nan_to_num(np.concatenate((values, coverage), axis=1)).astype(np.float32),
            names,
        )

    base = imu_feature_builders(data)
    return {
        "left_arm_trial_statistics": global_block([1], "left"),
        "right_arm_trial_statistics": global_block([2], "right"),
        "both_arms_trial_statistics": global_block([1, 2], "arms"),
        "both_arms_temporal_frequency": base["imu_arms_full_temporal"](),
    }


def louo_probe_predictions(
    values: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    pair: PairSpec,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(np.isin(labels, (pair.left, pair.right)))
    prediction = np.full(len(indices), -1, dtype=np.int64)
    pair_users = users[indices]
    for held_user in np.unique(pair_users):
        held = pair_users == held_user
        train = indices[~held]
        if len(np.unique(labels[train])) != 2:
            raise RuntimeError(f"source fold lacks both classes: {pair.pair_id}/{held_user}")
        model = make_probe(alpha)
        model.fit(values[train], labels[train])
        prediction[held] = model.predict(values[indices[held]]).astype(np.int64)
    if np.any(prediction < 0):
        raise RuntimeError(f"incomplete LOUO prediction: {pair.pair_id}")
    return indices, prediction


def h2_probe_predictions(
    block: FeatureBlock,
    data: Any,
    pair: PairSpec,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_values = block.values[: data.source_count]
    h2_values = block.values[data.source_count :]
    source_mask = np.isin(data.source_labels, (pair.left, pair.right))
    h2_indices = np.flatnonzero(np.isin(data.h2_labels, (pair.left, pair.right)))
    model = make_probe(alpha)
    model.fit(source_values[source_mask], data.source_labels[source_mask])
    return h2_indices, model.predict(h2_values[h2_indices]).astype(np.int64)


def session_probabilities(
    data: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(P85_TEACHER, allow_pickle=False) as source:
        teacher_ids = source["oof_sample_ids"].astype(str)
        teacher_labels = source["oof_labels"].astype(np.int64)
        emission_probability = source["oof_teacher_probability"].astype(np.float64)
        emission_log_probability = source["oof_teacher_log_probability"].astype(np.float64)
    metadata = align_metadata(DEFAULT_TRAIN_METADATA, teacher_ids)
    lookup = {sample_id: row for row, sample_id in enumerate(teacher_ids)}
    source_global = np.asarray(
        [lookup[str(value)] for value in data.sample_ids[: data.source_count]], dtype=np.int64
    )
    h2_global = np.asarray(
        [lookup[str(value)] for value in data.h2_sample_ids], dtype=np.int64
    )
    source_users = np.asarray(data.source_users).astype(str)
    source_emission = emission_probability[source_global].copy()
    h2_emission = emission_probability[h2_global].copy()
    source_posterior = source_emission.copy()
    source_available = np.zeros(len(source_global), dtype=bool)
    source_local = {int(global_row): local for local, global_row in enumerate(source_global)}

    for held_user in np.unique(source_users):
        train_local = source_users != held_user
        train_global = source_global[train_local]
        train_sessions = build_sessions(
            train_global,
            metadata,
            gap_seconds=SESSION_GAP_SECONDS,
            grouping="known_user",
        )
        model = fit_transition_model(
            teacher_labels,
            train_sessions,
            num_classes=NUM_CLASSES,
            trigram_backoff=TRIGRAM_BACKOFF,
        )
        held_global = source_global[source_users == held_user]
        target_sessions = build_sessions(
            held_global,
            metadata,
            gap_seconds=SESSION_GAP_SECONDS,
            grouping="anonymous_date",
        )
        for session in target_sessions:
            result = decode_unique_beam_posterior(
                emission_log_probability[session],
                model,
                transition_weight=TRANSITION_WEIGHT,
                beam_width=BEAM_WIDTH,
            )
            for position, global_row in enumerate(session):
                local = source_local[int(global_row)]
                source_posterior[local] = result.marginals[position]
                source_available[local] = True

    train_sessions = build_sessions(
        source_global,
        metadata,
        gap_seconds=SESSION_GAP_SECONDS,
        grouping="known_user",
    )
    model = fit_transition_model(
        teacher_labels,
        train_sessions,
        num_classes=NUM_CLASSES,
        trigram_backoff=TRIGRAM_BACKOFF,
    )
    h2_posterior = h2_emission.copy()
    h2_available = np.zeros(len(h2_global), dtype=bool)
    h2_local = {int(global_row): local for local, global_row in enumerate(h2_global)}
    target_sessions = build_sessions(
        h2_global,
        metadata,
        gap_seconds=SESSION_GAP_SECONDS,
        grouping="anonymous_date",
    )
    for session in target_sessions:
        result = decode_unique_beam_posterior(
            emission_log_probability[session],
            model,
            transition_weight=TRANSITION_WEIGHT,
            beam_width=BEAM_WIDTH,
        )
        for position, global_row in enumerate(session):
            local = h2_local[int(global_row)]
            h2_posterior[local] = result.marginals[position]
            h2_available[local] = True
    return source_emission, source_posterior, h2_emission, h2_posterior, {
        "source_rows": int(len(source_global)),
        "source_users": sorted(np.unique(source_users).tolist()),
        "source_session_available": int(source_available.sum()),
        "source_emission_fallback": int((~source_available).sum()),
        "h2_rows": int(len(h2_global)),
        "h2_users": sorted(np.unique(data.h2_users).tolist()),
        "h2_session_available": int(h2_available.sum()),
        "h2_emission_fallback": int((~h2_available).sum()),
        "strict_source_louo": True,
    }


def restricted_prediction(scores: np.ndarray, pair: PairSpec) -> np.ndarray:
    candidates = np.asarray((pair.left, pair.right), dtype=np.int64)
    return candidates[np.argmax(scores[:, candidates], axis=1)]


def bootstrap_user_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
    seed: int,
) -> dict[str, float]:
    unique_users = np.unique(users)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    per_user_indices = {user: np.flatnonzero(users == user) for user in unique_users}
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        indices = np.concatenate([per_user_indices[user] for user in sampled])
        values[replicate] = safe_balanced_accuracy(labels[indices], prediction[indices])
    return {
        "bootstrap_mean": float(values.mean()),
        "bootstrap_se": float(values.std(ddof=1)),
        "bootstrap_q10": float(np.quantile(values, 0.10)),
        "bootstrap_q90": float(np.quantile(values, 0.90)),
    }


def bootstrap_user_delta(
    labels: np.ndarray,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    users: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Paired user-block bootstrap for session posterior minus clip emission."""
    unique_users = np.unique(users)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    per_user_indices = {user: np.flatnonzero(users == user) for user in unique_users}
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        indices = np.concatenate([per_user_indices[user] for user in sampled])
        values[replicate] = safe_balanced_accuracy(
            labels[indices], prediction[indices]
        ) - safe_balanced_accuracy(labels[indices], baseline_prediction[indices])
    return {
        "context_delta_bootstrap_mean": float(values.mean()),
        "context_delta_bootstrap_se": float(values.std(ddof=1)),
        "context_delta_bootstrap_q10": float(np.quantile(values, 0.10)),
        "context_delta_bootstrap_q90": float(np.quantile(values, 0.90)),
    }


def prediction_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    per_user = {
        str(user): float(np.mean(prediction[users == user] == labels[users == user]))
        for user in np.unique(users)
    }
    result: dict[str, Any] = {
        "samples": int(len(labels)),
        "users": int(len(per_user)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": safe_balanced_accuracy(labels, prediction),
        "users_above_random": int(sum(value > 0.5 for value in per_user.values())),
        "per_user_accuracy": per_user,
    }
    if bootstrap_seed is not None:
        result.update(bootstrap_user_metrics(labels, prediction, users, bootstrap_seed))
    return result


def hard_masks(
    data: Any,
    pair: PairSpec,
) -> dict[str, np.ndarray]:
    source_ids = {
        row["sample_id"]
        for row in read_csv(POOL_ROOT / "source_hard_samples.csv")
        if row["pair_pool_id"] == pair.pair_id
    }
    h2_ids = {
        row["sample_id"]
        for row in read_csv(POOL_ROOT / "h2_hard_samples_confirmation.csv")
        if row["pair_pool_id"] == pair.pair_id
    }
    source = np.isin(data.sample_ids[: data.source_count], list(source_ids))
    h2 = np.isin(data.h2_sample_ids, list(h2_ids))
    p91 = (
        data.h2_primary_target
        & (np.minimum(data.h2_labels, data.h2_p91_prediction) == pair.left)
        & (np.maximum(data.h2_labels, data.h2_p91_prediction) == pair.right)
    )
    return {"source": source, "h2": h2, "p91": p91}


def pair_predictions_to_global(
    pair_indices: np.ndarray, pair_prediction: np.ndarray, total: int
) -> np.ndarray:
    result = np.full(total, -1, dtype=np.int64)
    result[pair_indices] = pair_prediction
    return result


def evaluate_mechanisms(
    modality: str,
    blocks: dict[str, FeatureBlock],
    alpha: float,
    pairs: list[PairSpec],
    data: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism, block in blocks.items():
        for pair_index, pair in enumerate(pairs):
            source_indices, source_prediction = louo_probe_predictions(
                block.values[: data.source_count],
                data.source_labels,
                data.source_users,
                pair,
                alpha,
            )
            h2_indices, h2_prediction = h2_probe_predictions(block, data, pair, alpha)
            source_metrics = prediction_metrics(
                data.source_labels[source_indices],
                source_prediction,
                data.source_users[source_indices],
            )
            h2_metrics = prediction_metrics(
                data.h2_labels[h2_indices], h2_prediction, data.h2_users[h2_indices]
            )
            rows.append(
                {
                    "modality": modality,
                    "mechanism": mechanism,
                    "pair_id": pair.pair_id,
                    "pair": f"{pair.left_name} / {pair.right_name}",
                    "dimensions": int(block.values.shape[1]),
                    "source_balanced_accuracy": source_metrics["balanced_accuracy"],
                    "source_accuracy": source_metrics["accuracy"],
                    "source_users_above_random": source_metrics["users_above_random"],
                    "h2_balanced_accuracy": h2_metrics["balanced_accuracy"],
                    "h2_accuracy": h2_metrics["accuracy"],
                    "h2_users_above_random": h2_metrics["users_above_random"],
                }
            )
    return rows


def connected_families(pairs: list[PairSpec]) -> list[tuple[str, set[int], list[str]]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    core_pairs = [pair for pair in pairs if source_evidence_tier(pair) != "weak"]
    for pair in core_pairs:
        adjacency[pair.left].add(pair.right)
        adjacency[pair.right].add(pair.left)
    remaining = set(adjacency)
    components: list[set[int]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        remaining -= component
        components.append(component)
    components.sort(key=lambda values: (-len(values), min(values)))
    result = []
    for index, component in enumerate(components, start=1):
        edge_ids = [
            pair.pair_id
            for pair in core_pairs
            if pair.left in component and pair.right in component
        ]
        result.append((f"family_{index:02d}", component, edge_ids))
    for pair in pairs:
        if pair not in core_pairs:
            result.append(
                (
                    f"weak_{pair.pair_id}",
                    {pair.left, pair.right},
                    [pair.pair_id],
                )
            )
    return result


def prior_weights(
    pair_rows: list[dict[str, Any]], pairs: list[PairSpec]
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    interval_half_widths = []
    for row in pair_rows:
        if row["modality"] in MODALITIES:
            if row["modality"] == "session":
                interval_half_widths.append(
                    0.5
                    * (
                        row["context_delta_bootstrap_q90"]
                        - row["context_delta_bootstrap_q10"]
                    )
                )
            else:
                interval_half_widths.append(
                    0.5
                    * (row["source_bootstrap_q90"] - row["source_bootstrap_q10"])
                )
    temperature = max(float(np.median(interval_half_widths)), 0.02)
    effective_evidence = np.asarray(
        [pair.source_confusions * pair.source_confusion_users for pair in pairs],
        dtype=np.float64,
    )
    evidence_pivot = float(np.median(effective_evidence))
    by_pair = {(row["pair_id"], row["modality"]): row for row in pair_rows}
    weights: dict[str, dict[str, float]] = {}
    for pair in pairs:
        scores = []
        for modality in MODALITIES:
            row = by_pair[(pair.pair_id, modality)]
            if modality == "session":
                scores.append(0.5 + row["context_delta_bootstrap_q10"])
            else:
                scores.append(row["source_bootstrap_q10"])
        scores = np.asarray(scores, dtype=np.float64)
        capability = softmax((scores - scores.max()) / temperature)
        evidence = float(pair.source_confusions * pair.source_confusion_users)
        reliability = evidence / (evidence + evidence_pivot)
        final = reliability * capability + (1.0 - reliability) / len(MODALITIES)
        weights[pair.pair_id] = {
            modality: float(final[index]) for index, modality in enumerate(MODALITIES)
        }
    audit = {
        "source_only": True,
        "capability_statistic": (
            "10th percentile of source user-block bootstrap balanced accuracy; "
            "session uses 0.5 + paired (posterior - P85 clip emission) delta"
        ),
        "temperature": temperature,
        "temperature_rule": "median source bootstrap 80% interval half-width, floor 0.02",
        "evidence": "source confusion samples multiplied by source confusion users",
        "evidence_pivot": evidence_pivot,
        "reliability_rule": "evidence / (evidence + median evidence)",
        "uncertain_prior": "blend capability softmax toward uniform according to source reliability",
        "h2_used": False,
    }
    return weights, audit


def source_feature_importance(
    modality: str,
    block: FeatureBlock,
    alpha: float,
    pairs: list[PairSpec],
    data: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = block.values[: data.source_count]
    for pair in pairs:
        selected = np.isin(data.source_labels, (pair.left, pair.right))
        model = make_probe(alpha)
        model.fit(values[selected], data.source_labels[selected])
        coefficient = np.abs(np.asarray(model.named_steps["ridge"].coef_).reshape(-1))
        order = np.argsort(-coefficient)[:20]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "modality": modality,
                    "pair_id": pair.pair_id,
                    "rank": rank,
                    "feature": block.names[int(feature_index)],
                    "absolute_standardized_coefficient": float(coefficient[feature_index]),
                }
            )
    return rows


def split_stability(source_best: str, h2_scores: dict[str, float]) -> str:
    h2_best = max(h2_scores, key=h2_scores.get)
    gap = h2_scores[h2_best] - h2_scores[source_best]
    if h2_scores[source_best] >= 0.60 and gap <= 0.05:
        return "confirmed"
    if gap > 0.10 or h2_scores[source_best] < 0.50:
        return "contradicted"
    return "mixed"


def main() -> None:
    validate_frozen_recipes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_audit_data()
    pairs = load_pairs()
    names = load_class_names()

    score = primary_visual_scores(data)
    source_score = score[: data.source_count]
    h2_score = score[data.source_count :]
    source_visual_prediction = source_score.argmax(axis=1).astype(np.int64)
    h2_visual_prediction = h2_score.argmax(axis=1).astype(np.int64)

    feature_builders = {
        "visual": visual_feature_builders(data),
        "skeleton": skeleton_feature_builders(data),
        "imu": imu_feature_builders(data),
    }
    fixed_blocks: dict[str, FeatureBlock] = {}
    for modality, (recipe, _) in FROZEN_RECIPES.items():
        fixed_blocks[modality] = feature_builders[modality][recipe]()

    (
        source_session_emission,
        source_session,
        h2_session_emission,
        h2_session,
        session_audit,
    ) = session_probabilities(data)
    prediction_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
        modality: {}
        for modality in ("primary_visual", "session_emission") + MODALITIES
    }
    pair_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for pair_index, pair in enumerate(pairs):
        source_pair_indices = np.flatnonzero(
            np.isin(data.source_labels, (pair.left, pair.right))
        )
        h2_pair_indices = np.flatnonzero(np.isin(data.h2_labels, (pair.left, pair.right)))
        source_primary_pair = restricted_prediction(source_score[source_pair_indices], pair)
        h2_primary_pair = restricted_prediction(h2_score[h2_pair_indices], pair)
        prediction_cache["primary_visual"][pair.pair_id] = (
            source_pair_indices,
            source_primary_pair,
            h2_pair_indices,
            h2_primary_pair,
        )

        source_session_prediction = restricted_prediction(
            source_session[source_pair_indices], pair
        )
        h2_session_prediction = restricted_prediction(h2_session[h2_pair_indices], pair)
        prediction_cache["session"][pair.pair_id] = (
            source_pair_indices,
            source_session_prediction,
            h2_pair_indices,
            h2_session_prediction,
        )
        prediction_cache["session_emission"][pair.pair_id] = (
            source_pair_indices,
            restricted_prediction(source_session_emission[source_pair_indices], pair),
            h2_pair_indices,
            restricted_prediction(h2_session_emission[h2_pair_indices], pair),
        )

        for modality in ("visual", "skeleton", "imu"):
            alpha = FROZEN_RECIPES[modality][1]
            source_indices, source_prediction = louo_probe_predictions(
                fixed_blocks[modality].values[: data.source_count],
                data.source_labels,
                data.source_users,
                pair,
                alpha,
            )
            h2_indices, h2_prediction = h2_probe_predictions(
                fixed_blocks[modality], data, pair, alpha
            )
            prediction_cache[modality][pair.pair_id] = (
                source_indices,
                source_prediction,
                h2_indices,
                h2_prediction,
            )

        masks = hard_masks(data, pair)
        for modality in ("primary_visual", "session_emission") + MODALITIES:
            source_indices, source_prediction, h2_indices, h2_prediction = (
                prediction_cache[modality][pair.pair_id]
            )
            source_labels = data.source_labels[source_indices]
            source_users = data.source_users[source_indices]
            h2_labels = data.h2_labels[h2_indices]
            h2_users = data.h2_users[h2_indices]
            source_metrics = prediction_metrics(
                source_labels,
                source_prediction,
                source_users,
                bootstrap_seed=BOOTSTRAP_SEED + 101 * pair_index + 17 * len(pair_rows),
            )
            h2_metrics = prediction_metrics(h2_labels, h2_prediction, h2_users)
            source_global_prediction = pair_predictions_to_global(
                source_indices, source_prediction, data.source_count
            )
            h2_global_prediction = pair_predictions_to_global(
                h2_indices, h2_prediction, len(data.h2_labels)
            )
            source_hard = masks["source"]
            h2_hard = masks["h2"]
            p91_hard = masks["p91"]
            source_protected = (
                np.isin(data.source_labels, (pair.left, pair.right))
                & (source_visual_prediction == data.source_labels)
            )
            h2_protected = (
                np.isin(data.h2_labels, (pair.left, pair.right))
                & (h2_visual_prediction == data.h2_labels)
            )
            row = {
                "pair_id": pair.pair_id,
                "pair": f"{pair.left_name} / {pair.right_name}",
                "left": pair.left,
                "right": pair.right,
                "modality": modality,
                "source_confusions": pair.source_confusions,
                "source_confusion_users": pair.source_confusion_users,
                "source_evidence_tier": source_evidence_tier(pair),
                "h2_confusions": pair.h2_confusions,
                "h2_confusion_users": pair.h2_confusion_users,
                "source_probe_samples": source_metrics["samples"],
                "source_probe_users": source_metrics["users"],
                "source_accuracy": source_metrics["accuracy"],
                "source_balanced_accuracy": source_metrics["balanced_accuracy"],
                "source_users_above_random": source_metrics["users_above_random"],
                "source_bootstrap_se": source_metrics["bootstrap_se"],
                "source_bootstrap_q10": source_metrics["bootstrap_q10"],
                "source_bootstrap_q90": source_metrics["bootstrap_q90"],
                "source_hard_rescue": int(
                    np.sum(source_hard & (source_global_prediction == data.source_labels))
                ),
                "source_hard_targets": int(source_hard.sum()),
                "source_protected_harm": int(
                    np.sum(source_protected & (source_global_prediction != data.source_labels))
                ),
                "source_protected": int(source_protected.sum()),
                "h2_probe_samples": h2_metrics["samples"],
                "h2_probe_users": h2_metrics["users"],
                "h2_accuracy": h2_metrics["accuracy"],
                "h2_balanced_accuracy": h2_metrics["balanced_accuracy"],
                "h2_users_above_random": h2_metrics["users_above_random"],
                "h2_hard_rescue": int(
                    np.sum(h2_hard & (h2_global_prediction == data.h2_labels))
                ),
                "h2_hard_targets": int(h2_hard.sum()),
                "h2_protected_harm": int(
                    np.sum(h2_protected & (h2_global_prediction != data.h2_labels))
                ),
                "h2_protected": int(h2_protected.sum()),
                "p91_hard_rescue": int(
                    np.sum(p91_hard & (h2_global_prediction == data.h2_labels))
                ),
                "p91_hard_targets": int(p91_hard.sum()),
                "source_per_user_accuracy": json.dumps(
                    source_metrics["per_user_accuracy"], sort_keys=True
                ),
                "h2_per_user_accuracy": json.dumps(
                    h2_metrics["per_user_accuracy"], sort_keys=True
                ),
            }
            if modality == "session":
                emission_source_prediction = prediction_cache["session_emission"][
                    pair.pair_id
                ][1]
                emission_h2_prediction = prediction_cache["session_emission"][
                    pair.pair_id
                ][3]
                emission_source_global = pair_predictions_to_global(
                    source_indices, emission_source_prediction, data.source_count
                )
                emission_h2_global = pair_predictions_to_global(
                    h2_indices, emission_h2_prediction, len(data.h2_labels)
                )
                row.update(
                    bootstrap_user_delta(
                        source_labels,
                        source_prediction,
                        emission_source_prediction,
                        source_users,
                        BOOTSTRAP_SEED + 10007 + pair_index,
                    )
                )
                row.update(
                    {
                        "source_context_delta_balanced_accuracy": float(
                            source_metrics["balanced_accuracy"]
                            - safe_balanced_accuracy(
                                source_labels, emission_source_prediction
                            )
                        ),
                        "h2_context_delta_balanced_accuracy": float(
                            h2_metrics["balanced_accuracy"]
                            - safe_balanced_accuracy(h2_labels, emission_h2_prediction)
                        ),
                        "source_context_hard_net_rescue": int(
                            np.sum(
                                source_hard
                                & (source_global_prediction == data.source_labels)
                                & (emission_source_global != data.source_labels)
                            )
                            - np.sum(
                                source_hard
                                & (source_global_prediction != data.source_labels)
                                & (emission_source_global == data.source_labels)
                            )
                        ),
                        "h2_context_hard_net_rescue": int(
                            np.sum(
                                h2_hard
                                & (h2_global_prediction == data.h2_labels)
                                & (emission_h2_global != data.h2_labels)
                            )
                            - np.sum(
                                h2_hard
                                & (h2_global_prediction != data.h2_labels)
                                & (emission_h2_global == data.h2_labels)
                            )
                        ),
                    }
                )
            if modality == "primary_visual":
                source_order = np.argsort(-source_score[source_pair_indices], axis=1)
                h2_order = np.argsort(-h2_score[h2_pair_indices], axis=1)
                row.update(
                    {
                        "source_40class_top1_accuracy": float(
                            np.mean(source_visual_prediction[source_pair_indices] == source_labels)
                        ),
                        "source_40class_top5_coverage": float(
                            np.mean(
                                np.any(source_order[:, :5] == source_labels[:, None], axis=1)
                            )
                        ),
                        "h2_40class_top1_accuracy": float(
                            np.mean(h2_visual_prediction[h2_pair_indices] == h2_labels)
                        ),
                        "h2_40class_top5_coverage": float(
                            np.mean(np.any(h2_order[:, :5] == h2_labels[:, None], axis=1))
                        ),
                    }
                )
            pair_rows.append(row)
            for local, global_index in enumerate(source_indices):
                sample_rows.append(
                    {
                        "split": "source_oof",
                        "pair_id": pair.pair_id,
                        "modality": modality,
                        "sample_id": data.sample_ids[global_index],
                        "user": data.source_users[global_index],
                        "true_class_id": int(data.source_labels[global_index]),
                        "prediction_id": int(source_prediction[local]),
                        "correct": int(source_prediction[local] == data.source_labels[global_index]),
                        "primary_visual_hard_pair": int(source_hard[global_index]),
                    }
                )
            for local, global_index in enumerate(h2_indices):
                sample_rows.append(
                    {
                        "split": "H2_confirmation",
                        "pair_id": pair.pair_id,
                        "modality": modality,
                        "sample_id": data.h2_sample_ids[global_index],
                        "user": data.h2_users[global_index],
                        "true_class_id": int(data.h2_labels[global_index]),
                        "prediction_id": int(h2_prediction[local]),
                        "correct": int(h2_prediction[local] == data.h2_labels[global_index]),
                        "primary_visual_hard_pair": int(h2_hard[global_index]),
                        "p91_hard_pair": int(p91_hard[global_index]),
                    }
                )

    weights, prior_audit = prior_weights(pair_rows, pairs)
    by_pair_modality = {
        (row["pair_id"], row["modality"]): row for row in pair_rows
    }
    prior_rows: list[dict[str, Any]] = []
    for pair in pairs:
        source_scores: dict[str, float] = {}
        h2_scores: dict[str, float] = {}
        for modality in MODALITIES:
            modality_row = by_pair_modality[(pair.pair_id, modality)]
            if modality == "session":
                source_scores[modality] = 0.5 + modality_row[
                    "context_delta_bootstrap_q10"
                ]
                h2_scores[modality] = 0.5 + modality_row[
                    "h2_context_delta_balanced_accuracy"
                ]
            else:
                source_scores[modality] = modality_row["source_bootstrap_q10"]
                h2_scores[modality] = modality_row["h2_balanced_accuracy"]
        source_best = max(weights[pair.pair_id], key=weights[pair.pair_id].get)
        h2_best = max(h2_scores, key=h2_scores.get)
        stability = split_stability(source_best, h2_scores)
        if source_evidence_tier(pair) == "strong" and stability == "confirmed":
            confidence = "high"
        elif source_evidence_tier(pair) != "weak" and stability != "contradicted":
            confidence = "medium"
        else:
            confidence = "low"
        prior_rows.append(
            {
                "pair_id": pair.pair_id,
                "pair": f"{pair.left_name} / {pair.right_name}",
                "source_confusions": pair.source_confusions,
                "source_confusion_users": pair.source_confusion_users,
                "source_evidence_tier": source_evidence_tier(pair),
                "h2_confusions": pair.h2_confusions,
                "h2_confusion_users": pair.h2_confusion_users,
                "visual_prior": weights[pair.pair_id]["visual"],
                "skeleton_prior": weights[pair.pair_id]["skeleton"],
                "imu_prior": weights[pair.pair_id]["imu"],
                "session_prior": weights[pair.pair_id]["session"],
                "source_best_modality": source_best,
                "h2_best_modality": h2_best,
                "source_best_q10_bacc": source_scores[source_best],
                "h2_source_best_bacc": h2_scores[source_best],
                "h2_best_bacc": h2_scores[h2_best],
                "source_session_absolute_bacc": by_pair_modality[
                    (pair.pair_id, "session")
                ]["source_balanced_accuracy"],
                "source_session_emission_bacc": by_pair_modality[
                    (pair.pair_id, "session_emission")
                ]["source_balanced_accuracy"],
                "source_session_context_delta_bacc": by_pair_modality[
                    (pair.pair_id, "session")
                ]["source_context_delta_balanced_accuracy"],
                "h2_session_context_delta_bacc": by_pair_modality[
                    (pair.pair_id, "session")
                ]["h2_context_delta_balanced_accuracy"],
                "split_stability": stability,
                "confidence": confidence,
            }
        )

    mechanism_rows: list[dict[str, Any]] = []
    visual_mechanisms = visual_mechanism_blocks(data)
    skeleton_mechanisms = skeleton_mechanism_blocks(data)
    imu_mechanisms = imu_mechanism_blocks(data)
    mechanism_rows.extend(
        evaluate_mechanisms("visual", visual_mechanisms, 10000.0, pairs, data)
    )
    mechanism_rows.extend(
        evaluate_mechanisms("skeleton", skeleton_mechanisms, 1000.0, pairs, data)
    )
    mechanism_rows.extend(
        evaluate_mechanisms("imu", imu_mechanisms, 10.0, pairs, data)
    )

    importance_rows: list[dict[str, Any]] = []
    for modality in ("skeleton", "imu"):
        importance_rows.extend(
            source_feature_importance(
                modality,
                fixed_blocks[modality],
                FROZEN_RECIPES[modality][1],
                pairs,
                data,
            )
        )

    family_rows: list[dict[str, Any]] = []
    for family_id, class_ids, edge_ids in connected_families(pairs):
        class_text = ";".join(f"{class_id}:{names[class_id]}" for class_id in sorted(class_ids))
        row: dict[str, Any] = {
            "family_id": family_id,
            "classes": class_text,
            "class_count": len(class_ids),
            "pair_count": len(edge_ids),
            "pairs": ";".join(edge_ids),
            "source_confusions": int(
                sum(pair.source_confusions for pair in pairs if pair.pair_id in edge_ids)
            ),
            "source_confusion_users_union": len(
                {
                    item["user"]
                    for item in read_csv(POOL_ROOT / "source_hard_samples.csv")
                    if item["pair_pool_id"] in edge_ids
                }
            ),
        }
        for modality in MODALITIES:
            family_pair_rows = [by_pair_modality[(edge, modality)] for edge in edge_ids]
            row[f"source_{modality}_macro_bacc"] = float(
                np.mean([item["source_balanced_accuracy"] for item in family_pair_rows])
            )
            row[f"h2_{modality}_macro_bacc"] = float(
                np.mean([item["h2_balanced_accuracy"] for item in family_pair_rows])
            )
            row[f"{modality}_prior"] = float(
                np.mean([weights[edge][modality] for edge in edge_ids])
            )
        session_family_rows = [
            by_pair_modality[(edge, "session")] for edge in edge_ids
        ]
        row["source_session_context_delta_bacc"] = float(
            np.mean(
                [
                    item["source_context_delta_balanced_accuracy"]
                    for item in session_family_rows
                ]
            )
        )
        row["h2_session_context_delta_bacc"] = float(
            np.mean(
                [
                    item["h2_context_delta_balanced_accuracy"]
                    for item in session_family_rows
                ]
            )
        )
        family_rows.append(row)

    aggregate: dict[str, Any] = {}
    for modality in ("primary_visual", "session_emission") + MODALITIES:
        rows = [row for row in pair_rows if row["modality"] == modality]
        aggregate[modality] = {
            "source_macro_pair_balanced_accuracy": float(
                np.mean([row["source_balanced_accuracy"] for row in rows])
            ),
            "h2_macro_pair_balanced_accuracy": float(
                np.mean([row["h2_balanced_accuracy"] for row in rows])
            ),
            "source_pairs_at_or_above_60pct": int(
                sum(row["source_balanced_accuracy"] >= 0.60 for row in rows)
            ),
            "h2_pairs_at_or_above_60pct": int(
                sum(row["h2_balanced_accuracy"] >= 0.60 for row in rows)
            ),
            "source_hard_rescue": int(sum(row["source_hard_rescue"] for row in rows)),
            "source_hard_targets": int(sum(row["source_hard_targets"] for row in rows)),
            "h2_hard_rescue": int(sum(row["h2_hard_rescue"] for row in rows)),
            "h2_hard_targets": int(sum(row["h2_hard_targets"] for row in rows)),
            "p91_hard_rescue": int(sum(row["p91_hard_rescue"] for row in rows)),
            "p91_hard_targets": int(sum(row["p91_hard_targets"] for row in rows)),
        }

    session_rows = [row for row in pair_rows if row["modality"] == "session"]
    aggregate["session_context_increment"] = {
        "source_macro_pair_delta_balanced_accuracy": float(
            np.mean(
                [row["source_context_delta_balanced_accuracy"] for row in session_rows]
            )
        ),
        "h2_macro_pair_delta_balanced_accuracy": float(
            np.mean([row["h2_context_delta_balanced_accuracy"] for row in session_rows])
        ),
        "source_pairs_positive_delta": int(
            sum(row["source_context_delta_balanced_accuracy"] > 0 for row in session_rows)
        ),
        "source_pairs_positive_q10_delta": int(
            sum(row["context_delta_bootstrap_q10"] > 0 for row in session_rows)
        ),
        "h2_pairs_positive_delta": int(
            sum(row["h2_context_delta_balanced_accuracy"] > 0 for row in session_rows)
        ),
        "source_hard_net_rescue": int(
            sum(row["source_context_hard_net_rescue"] for row in session_rows)
        ),
        "h2_hard_net_rescue": int(
            sum(row["h2_context_hard_net_rescue"] for row in session_rows)
        ),
    }

    stability_counts: dict[str, int] = defaultdict(int)
    confidence_counts: dict[str, int] = defaultdict(int)
    for row in prior_rows:
        stability_counts[row["split_stability"]] += 1
        confidence_counts[row["confidence"]] += 1
    summary = {
        "experiment_id": "p91_modality_capability_prior_v1",
        "status": "complete_source_oof_discovery_h2_frozen_confirmation",
        "protocol": {
            "source": "H1 plus established embargo users; subject-disjoint leave-one-user-out",
            "pair_universe": "all P96 primary-visual OOF confusion pairs repeated across >=2 source users",
            "pair_count": len(pairs),
            "source_confusion_samples_covered": int(sum(pair.source_confusions for pair in pairs)),
            "h2_confirmation": "single frozen evaluation; does not change source prior",
            "recipes": {
                modality: {"feature_group": recipe, "alpha": alpha}
                for modality, (recipe, alpha) in FROZEN_RECIPES.items()
            },
            "pair_specific_feature_or_alpha_selection": False,
            "diagnostic_pair_probes_only": True,
            "final_model_trained": False,
            "p91_modified": False,
            "h3_read": False,
        },
        "session_audit": session_audit,
        "prior_audit": prior_audit,
        "aggregate": aggregate,
        "split_stability_counts": dict(stability_counts),
        "confidence_counts": dict(confidence_counts),
        "high_confidence_pairs": [
            row["pair_id"] for row in prior_rows if row["confidence"] == "high"
        ],
    }

    write_csv(OUTPUT / "confusion_modality_capability.csv", pair_rows)
    write_csv(OUTPUT / "modality_soft_prior.csv", prior_rows)
    write_csv(OUTPUT / "confusion_families.csv", family_rows)
    write_csv(OUTPUT / "mechanism_capability.csv", mechanism_rows)
    write_csv(OUTPUT / "source_feature_importance.csv", importance_rows)
    write_csv(OUTPUT / "pair_sample_predictions.csv", sample_rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
