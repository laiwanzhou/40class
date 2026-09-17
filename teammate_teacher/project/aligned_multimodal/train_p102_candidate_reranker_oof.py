"""Train the initial P102 candidate-conditioned Large B Teacher.

B is a pointwise candidate scorer, not a 40-class classifier.  Each outer fold
uses source-only class prototypes over four privileged representation blocks:
fine VideoMAEv2, fine InternVideo2, Skeleton relation/parts, and IMU device-time.
The scorer receives A probability/rank/uncertainty and candidate identity.  It
only outputs scores for the deployment candidate set frozen by P102 hard-set
analysis.  No learned trigger or held-label calibration is used.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from analyze_p102_hard_set import (
    CandidateRecipe,
    build_candidate_ids,
    confusion_neighbors,
    source_crossfit_session_probability,
)
from audit_p102_session_closure import (
    classification_metrics,
    comparison,
    load_npz,
    topk_correct,
    true_rank,
)
from audit_p87_sequence_decoder import DecoderConfig, align_metadata
from p100a_global_teacher_data import FOLD_USERS, H3_USERS
from p101_finegrained_teacher_data import load_p101_data


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_HARD_SUMMARY = HERE / "runs/p102_hard_set_v1/summary.json"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p102_b0_candidate_reranker_oof_v1"
BLOCKS = ("videomae", "internvideo", "skeleton", "imu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-oof", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--session-summary", type=Path, default=DEFAULT_SESSION_SUMMARY)
    parser.add_argument("--hard-set", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--hard-summary", type=Path, default=DEFAULT_HARD_SUMMARY)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--pca-components", type=int, default=40)
    return parser.parse_args()


def masked_moments(values: np.ndarray, mask: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    count = np.maximum(mask.sum(axis=axes), 1.0)
    mean = (values * mask).sum(axis=axes) / count
    centered = values - np.expand_dims(mean, axis=axes[0]) if len(axes) == 1 else values
    # General masked variance avoids brittle broadcasting across several axes.
    expanded_mean = mean
    for axis in sorted(axes):
        expanded_mean = np.expand_dims(expanded_mean, axis=axis)
    variance = (((values - expanded_mean) ** 2) * mask).sum(axis=axes) / count
    return mean, np.sqrt(np.maximum(variance, 0.0))


def visual_summary(temporal: np.ndarray, action: np.ndarray) -> np.ndarray:
    temporal = np.asarray(temporal, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    mean = temporal.mean(axis=(1, 3))
    std = temporal.std(axis=(1, 3))
    window_delta = temporal[:, 1].mean(axis=2) - temporal[:, 0].mean(axis=2)
    time_delta = (temporal[:, :, :, -1] - temporal[:, :, :, 0]).mean(axis=1)
    action_mean = action.mean(axis=1)
    action_delta = action[:, 1] - action[:, 0]
    return np.concatenate(
        [value.reshape(len(temporal), -1) for value in (mean, std, window_delta, time_delta, action_mean, action_delta)],
        axis=1,
    ).astype(np.float32)


def skeleton_summary(data: Any) -> np.ndarray:
    rows = data.motion_rows
    features = np.asarray(data.motion["skeleton_features"][rows], dtype=np.float32)
    feature_mask = np.asarray(data.motion["skeleton_feature_mask"][rows], dtype=np.float32)
    relations = np.asarray(data.motion["skeleton_relations"][rows], dtype=np.float32)
    relation_mask = np.asarray(data.motion["skeleton_relation_mask"][rows], dtype=np.float32)
    quality = np.asarray(data.motion["skeleton_frame_quality"][rows], dtype=np.float32)
    feature_mean, feature_std = masked_moments(features, feature_mask, axes=(1, 2))
    relation_mean, relation_std = masked_moments(relations, relation_mask, axes=(1, 2))
    feature_window = []
    relation_window = []
    for window in (0, 1):
        value, _ = masked_moments(features[:, window], feature_mask[:, window], axes=(1,))
        feature_window.append(value)
        value, _ = masked_moments(relations[:, window], relation_mask[:, window], axes=(1,))
        relation_window.append(value)
    feature_time_delta = (features[:, :, -1] - features[:, :, 0]).mean(axis=1)
    relation_time_delta = (relations[:, :, -1] - relations[:, :, 0]).mean(axis=1)
    values = (
        feature_mean,
        feature_std,
        feature_window[1] - feature_window[0],
        feature_time_delta,
        relation_mean,
        relation_std,
        relation_window[1] - relation_window[0],
        relation_time_delta,
        quality.mean(axis=(1, 2))[:, None],
        quality.std(axis=(1, 2))[:, None],
        (quality[:, 1].mean(axis=1) - quality[:, 0].mean(axis=1))[:, None],
    )
    return np.concatenate([value.reshape(len(rows), -1) for value in values], axis=1).astype(np.float32)


def imu_summary(data: Any) -> np.ndarray:
    rows = data.motion_rows
    statistics = np.asarray(data.motion["imu_bin_statistics"][rows], dtype=np.float32)
    bin_mask = np.asarray(data.motion["imu_bin_mask"][rows], dtype=np.float32)
    sequence = np.asarray(data.motion["imu_sequences"][rows], dtype=np.float32)
    sequence_mask = np.asarray(data.motion["imu_sequence_mask"][rows], dtype=np.float32)
    sequence_bin_mask = (sequence_mask > 0).any(axis=4).astype(np.float32)
    global_statistics = np.asarray(data.motion["imu_global_statistics"][rows], dtype=np.float32)
    global_mask = np.asarray(data.motion["imu_global_mask"][rows], dtype=np.float32)
    stat_mean, stat_std = masked_moments(statistics, bin_mask, axes=(1, 2))
    sequence_point_mean = sequence.mean(axis=4)
    sequence_mean, sequence_std = masked_moments(
        sequence_point_mean, sequence_bin_mask, axes=(1, 2)
    )
    stat_window = []
    sequence_window = []
    for window in (0, 1):
        value, _ = masked_moments(statistics[:, window], bin_mask[:, window], axes=(1,))
        stat_window.append(value)
        value, _ = masked_moments(
            sequence_point_mean[:, window], sequence_bin_mask[:, window], axes=(1,)
        )
        sequence_window.append(value)
    stat_time_delta = (statistics[:, :, -1] - statistics[:, :, 0]).mean(axis=1)
    sequence_time_delta = (sequence_point_mean[:, :, -1] - sequence_point_mean[:, :, 0]).mean(axis=1)
    values = (
        stat_mean,
        stat_std,
        stat_window[1] - stat_window[0],
        stat_time_delta,
        sequence_mean,
        sequence_std,
        sequence_window[1] - sequence_window[0],
        sequence_time_delta,
        global_statistics,
        global_mask,
    )
    return np.concatenate([value.reshape(len(rows), -1) for value in values], axis=1).astype(np.float32)


def load_raw_blocks(sample_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = load_p101_data()
    if not np.array_equal(data.sample_ids.astype(str), sample_ids.astype(str)):
        raise RuntimeError("P101 privileged representation order differs from P102")
    contract = data.summary()
    if int(contract["h3_rows"]) != 0 or set(data.users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 reached B representation loader")
    blocks = {
        "videomae": visual_summary(data.visual_vmae_temporal, data.visual_vmae_action),
        "internvideo": visual_summary(data.visual_iv2_temporal, data.visual_iv2_action),
        "skeleton": skeleton_summary(data),
        "imu": imu_summary(data),
    }
    for name, values in blocks.items():
        if len(values) != 1941 or not np.isfinite(values).all():
            raise RuntimeError(f"invalid privileged block: {name}")
    return blocks, {"p101_contract": contract, "raw_shapes": {name: list(value.shape) for name, value in blocks.items()}}


def fit_embeddings(
    raw_blocks: dict[str, np.ndarray], source: np.ndarray, components: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    embeddings: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    for offset, name in enumerate(BLOCKS):
        raw = raw_blocks[name]
        scaler = StandardScaler(copy=True)
        source_scaled = scaler.fit_transform(raw[source]).astype(np.float32)
        all_scaled = scaler.transform(raw).astype(np.float32)
        count = min(int(components), source_scaled.shape[0] - 1, source_scaled.shape[1])
        pca = PCA(n_components=count, svd_solver="randomized", random_state=seed + offset)
        pca.fit(source_scaled)
        transformed = pca.transform(all_scaled).astype(np.float32)
        component_scale = np.maximum(transformed[source].std(axis=0, keepdims=True), 1e-4)
        transformed /= component_scale
        embeddings[name] = transformed
        audit[name] = {
            "raw_dim": int(raw.shape[1]),
            "components": int(count),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "fit_rows": int(source.sum()),
        }
    return embeddings, audit


def fit_prototypes(
    embeddings: dict[str, np.ndarray], labels: np.ndarray, fit: np.ndarray
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    prototypes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    counts = np.bincount(labels[fit], minlength=40).astype(np.float64)
    for name, values in embeddings.items():
        global_mean = values[fit].mean(axis=0)
        global_std = np.maximum(values[fit].std(axis=0), 0.25)
        means = np.zeros((40, values.shape[1]), dtype=np.float32)
        stds = np.zeros_like(means)
        for class_id in range(40):
            selected = fit & (labels == class_id)
            if selected.any():
                means[class_id] = values[selected].mean(axis=0)
                stds[class_id] = np.maximum(values[selected].std(axis=0), 0.25)
            else:
                means[class_id] = global_mean
                stds[class_id] = global_std
        prototypes[name] = (means, stds)
    return prototypes, counts


def candidate_ranks(probability: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(probability), axis=1)[:, ::-1]
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(1, order.shape[1] + 1)
    return ranks


def point_features(
    rows: np.ndarray,
    candidate_ids: np.ndarray,
    probability: np.ndarray,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[str, tuple[np.ndarray, np.ndarray]],
    class_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    rows = np.asarray(rows, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    ranks = candidate_ranks(probability)
    prediction = probability.argmax(axis=1)
    ordered = np.sort(probability, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    uncertainty = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
    features: list[np.ndarray] = []
    pair_rows: list[int] = []
    pair_candidates: list[int] = []
    block_offsets: dict[str, slice] = {}
    total_count = float(class_counts.sum() + 40.0)
    for row in rows:
        valid = candidate_ids[row][candidate_ids[row] >= 0]
        size = len(valid)
        for position, candidate in enumerate(valid):
            candidate = int(candidate)
            base = [
                math.log(max(float(probability[row, candidate]), 1e-12)),
                float(probability[row, candidate]),
                float(ranks[row, candidate] / 40.0),
                float(position / max(size - 1, 1)),
                float(size / 8.0),
                float(candidate == prediction[row]),
                float(margin[row]),
                float(uncertainty[row] / math.log(40.0)),
                math.log((float(class_counts[candidate]) + 1.0) / total_count),
            ]
            candidate_onehot = np.zeros(40, dtype=np.float32)
            candidate_onehot[candidate] = 1.0
            prediction_onehot = np.zeros(40, dtype=np.float32)
            prediction_onehot[int(prediction[row])] = 1.0
            values = list(map(float, base)) + candidate_onehot.tolist() + prediction_onehot.tolist()
            for name in BLOCKS:
                embedding = embeddings[name][row]
                means, stds = prototypes[name]
                prototype = means[candidate]
                standard = stds[candidate]
                cosine = float(
                    np.dot(embedding, prototype)
                    / max(np.linalg.norm(embedding) * np.linalg.norm(prototype), 1e-6)
                )
                squared = float(np.mean((embedding - prototype) ** 2))
                diagonal = float(np.mean(((embedding - prototype) / standard) ** 2 + 2.0 * np.log(standard)))
                values.extend((cosine, -squared, -diagonal))
            features.append(np.asarray(values, dtype=np.float32))
            pair_rows.append(int(row))
            pair_candidates.append(candidate)
    matrix = np.stack(features)
    base_dim = 9 + 80
    for index, name in enumerate(BLOCKS):
        block_offsets[name] = slice(base_dim + index * 3, base_dim + (index + 1) * 3)
    return matrix, np.asarray(pair_rows), np.asarray(pair_candidates), block_offsets


def fit_scorer(
    matrix: np.ndarray,
    pair_rows: np.ndarray,
    pair_candidates: np.ndarray,
    labels: np.ndarray,
    a_probability: np.ndarray,
    seed: int,
) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    target = (pair_candidates == labels[pair_rows]).astype(np.int64)
    if target.sum() == 0:
        raise RuntimeError("B source pairs contain no positive candidate")
    a_error = a_probability[pair_rows].argmax(axis=1) != labels[pair_rows]
    candidate_size = np.bincount(pair_rows, minlength=len(labels))[pair_rows]
    weight = np.where(target == 1, np.maximum(candidate_size - 1, 1), 1).astype(np.float64)
    weight *= np.where(a_error, 2.0, 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(matrix, target, sample_weight=weight)
    return model, {
        "pairs": int(len(target)),
        "positive_pairs": int(target.sum()),
        "rows": int(len(np.unique(pair_rows))),
        "positive_weighting": "candidate_size_minus_one; source A-error rows x2",
    }


def rerank(
    model: HistGradientBoostingClassifier,
    matrix: np.ndarray,
    pair_rows: np.ndarray,
    pair_candidates: np.ndarray,
    candidate_ids: np.ndarray,
    a_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.clip(model.predict_proba(matrix)[:, 1], 1e-5, 1.0 - 1e-5)
    pair_score = np.log(np.maximum(a_probability[pair_rows, pair_candidates], 1e-12)) + np.log(q)
    score = np.full_like(a_probability, -1e9, dtype=np.float64)
    score[pair_rows, pair_candidates] = pair_score
    prediction = score.argmax(axis=1)
    # Rows outside the requested evaluation subset remain A predictions.
    present = np.zeros(len(a_probability), dtype=bool)
    present[np.unique(pair_rows)] = True
    prediction[~present] = a_probability[~present].argmax(axis=1)
    probability = np.zeros_like(a_probability, dtype=np.float64)
    for row in np.flatnonzero(present):
        valid = candidate_ids[row][candidate_ids[row] >= 0]
        values = score[row, valid]
        values = np.exp(values - values.max())
        values /= values.sum()
        probability[row, valid] = values
    probability[~present] = a_probability[~present]
    return prediction, probability


def shuffle_embeddings(
    embeddings: dict[str, np.ndarray], users: np.ndarray, selected: np.ndarray, blocks: tuple[str, ...]
) -> dict[str, np.ndarray]:
    output = {name: values.copy() for name, values in embeddings.items()}
    for user in sorted(set(users[selected].tolist())):
        rows = np.flatnonzero(selected & (users == user))
        if len(rows) <= 1:
            continue
        shifted = np.roll(rows, 1)
        for name in blocks:
            output[name][rows] = embeddings[name][shifted]
    return output


def zero_embeddings(
    embeddings: dict[str, np.ndarray], selected: np.ndarray, blocks: tuple[str, ...]
) -> dict[str, np.ndarray]:
    output = {name: values.copy() for name, values in embeddings.items()}
    for name in blocks:
        output[name][selected] = 0.0
    return output


def full_order_ranks(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return true_rank(probability, labels)


def evaluate_variant(
    labels: np.ndarray,
    users: np.ndarray,
    selected: np.ndarray,
    a_probability: np.ndarray,
    final_probability: np.ndarray,
) -> dict[str, Any]:
    a = a_probability[selected]
    final = final_probability[selected]
    y = labels[selected]
    u = users[selected]
    comp = comparison(y, u, a, final)
    a_prediction = a.argmax(axis=1)
    final_prediction = final.argmax(axis=1)
    changed = final_prediction != a_prediction
    correct_change = changed & (final_prediction == y)
    return {
        "metrics": classification_metrics(final, y, u),
        "vs_a": comp,
        "trigger": {
            "coverage_rows": int(changed.sum()),
            "coverage": float(changed.mean()),
            "correct_change_rows": int(correct_change.sum()),
            "precision": float(correct_change.sum() / max(changed.sum(), 1)),
            "wrong_to_wrong": int(np.sum(changed & (final_prediction != y) & (a_prediction != y))),
        },
    }


def main() -> None:
    args = parse_args()
    folds_requested = tuple(sorted(set(map(int, args.folds))))
    if not set(folds_requested) <= {0, 1, 2, 3}:
        raise ValueError("folds must be a subset of 0..3")
    session = load_npz(args.session_oof.resolve())
    hard = load_npz(args.hard_set.resolve())
    session_summary = json.loads(args.session_summary.resolve().read_text(encoding="utf-8"))
    hard_summary = json.loads(args.hard_summary.resolve().read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    target_candidates = np.asarray(hard["candidate_ids"], dtype=np.int64)
    if not np.array_equal(hard["sample_ids"].astype(str), sample_ids):
        raise RuntimeError("hard/session row order differs")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached B")
    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    raw_blocks, representation_audit = load_raw_blocks(sample_ids)

    variants = ("full", "shuffle_all", "zero_visual", "zero_skeleton", "zero_imu", "zero_all")
    variant_probability = {name: a_probability.copy() for name in variants}
    fold_reports: list[dict[str, Any]] = []
    for outer_fold in folds_requested:
        source = fold_ids != outer_fold
        held = fold_ids == outer_fold
        selected_session = session_summary["folds"][outer_fold]["selected"]
        config = DecoderConfig(
            gap_seconds=float(selected_session["gap_seconds"]),
            transition_weight=float(selected_session["transition_weight"]),
            trigram_backoff=float(selected_session["trigram_backoff"]),
            beam_width=int(selected_session["beam_width"]),
        )
        source_probability, source_session_audit = source_crossfit_session_probability(
            outer_fold,
            args.nested_root.resolve(),
            sample_ids,
            users,
            fold_ids,
            labels,
            metadata,
            config,
        )
        recipe_values = hard_summary["folds"][outer_fold]["selected_recipe"]
        recipe = CandidateRecipe(**{key: int(value) for key, value in recipe_values.items()})
        source_prediction = source_probability.argmax(axis=1)
        source_candidates = np.full((len(labels), 8), -1, dtype=np.int64)
        for user in sorted(set(users[source].tolist())):
            validation = source & (users == user)
            graph_fit = np.flatnonzero(source & (users != user)).astype(np.int64)
            neighbors, _ = confusion_neighbors(
                labels, source_prediction, users, graph_fit, recipe
            )
            values = build_candidate_ids(source_probability[validation], neighbors, recipe)
            source_candidates[validation, : recipe.max_size] = values
        source_hit = np.any(source_candidates == labels[:, None], axis=1)
        train_rows = np.flatnonzero(source & source_hit).astype(np.int64)

        embeddings, embedding_audit = fit_embeddings(
            raw_blocks, source, int(args.pca_components), seed=20260823 + outer_fold * 100
        )
        prototypes, class_counts = fit_prototypes(embeddings, labels, source)
        train_matrix, train_pair_rows, train_pair_candidates, feature_slices = point_features(
            train_rows, source_candidates, source_probability, embeddings, prototypes, class_counts
        )
        model, train_audit = fit_scorer(
            train_matrix,
            train_pair_rows,
            train_pair_candidates,
            labels,
            source_probability,
            seed=20260823 + outer_fold,
        )
        held_rows = np.flatnonzero(held).astype(np.int64)
        fold_variants: dict[str, Any] = {}
        embedding_variants = {
            "full": embeddings,
            "shuffle_all": shuffle_embeddings(embeddings, users, held, BLOCKS),
            "zero_visual": zero_embeddings(embeddings, held, ("videomae", "internvideo")),
            "zero_skeleton": zero_embeddings(embeddings, held, ("skeleton",)),
            "zero_imu": zero_embeddings(embeddings, held, ("imu",)),
            "zero_all": zero_embeddings(embeddings, held, BLOCKS),
        }
        for name, variant_embeddings in embedding_variants.items():
            matrix, pair_rows, pair_candidates, _ = point_features(
                held_rows,
                target_candidates,
                a_probability,
                variant_embeddings,
                prototypes,
                class_counts,
            )
            _, probability = rerank(
                model,
                matrix,
                pair_rows,
                pair_candidates,
                target_candidates,
                a_probability,
            )
            variant_probability[name][held] = probability[held]
            fold_variants[name] = evaluate_variant(
                labels, users, held, a_probability, probability
            )
        fold_reports.append(
            {
                "fold": outer_fold,
                "held_users": list(FOLD_USERS[outer_fold]),
                "source_users": sorted(set(users[source].tolist())),
                "source_held_overlap": sorted(set(users[source].tolist()) & set(FOLD_USERS[outer_fold])),
                "candidate_recipe": asdict(recipe),
                "source_candidate": {
                    "rows": int(source.sum()),
                    "hits": int(np.sum(source & source_hit)),
                    "recall": float(np.mean(source_hit[source])),
                    "train_rows_with_positive_candidate": int(len(train_rows)),
                },
                "source_session": source_session_audit,
                "embedding": embedding_audit,
                "training": train_audit,
                "feature_dim": int(train_matrix.shape[1]),
                "feature_slices": {name: [value.start, value.stop] for name, value in feature_slices.items()},
                "variants": fold_variants,
            }
        )

    evaluated = np.isin(fold_ids, np.asarray(folds_requested))
    system_variants = {
        name: evaluate_variant(labels, users, evaluated, a_probability, probability)
        for name, probability in variant_probability.items()
    }
    full_probability = variant_probability["full"]
    a_prediction = a_probability.argmax(axis=1)
    final_prediction = full_probability.argmax(axis=1)
    candidate_hit = np.asarray(hard["candidate_hit"], dtype=bool)
    a_error = a_prediction != labels
    attackable = evaluated & a_error & candidate_hit
    conditional_correct = final_prediction == labels
    raw_rank = true_rank(a_probability[evaluated], labels[evaluated])
    final_rank = true_rank(full_probability[evaluated], labels[evaluated])
    summary = {
        "status": "complete" if folds_requested == (0, 1, 2, 3) else "smoke_complete",
        "protocol": (
            "P102-B0 unified candidate-conditioned point scorer; four source-only "
            "privileged prototype blocks; fixed A-log-prob + B-log-prob ranking; no "
            "oracle family, held-label calibration, learned trigger or 40-class B head."
        ),
        "folds_requested": list(folds_requested),
        "data": {
            "rows": int(evaluated.sum()),
            "users": sorted(set(users[evaluated].tolist())),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "representation": representation_audit,
        "a_baseline": classification_metrics(a_probability[evaluated], labels[evaluated], users[evaluated]),
        "variants": system_variants,
        "candidate": {
            "attackable_a_error_rows": int(attackable.sum()),
            "conditional_rerank_correct": int(np.sum(attackable & conditional_correct)),
            "conditional_rerank_accuracy": float(np.mean(conditional_correct[attackable])),
            "candidate_miss_a_errors": int(np.sum(evaluated & a_error & (~candidate_hit))),
        },
        "true_rank_movement": {
            "improved": int(np.sum(final_rank < raw_rank)),
            "harmed": int(np.sum(final_rank > raw_rank)),
            "mean_delta": float(np.mean(final_rank - raw_rank)),
        },
        "folds": fold_reports,
        "student_started": False,
    }
    np.savez_compressed(
        output / "b0_oof_predictions.npz",
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=fold_ids,
        evaluated=evaluated,
        candidate_ids=target_candidates,
        a_probability=a_probability.astype(np.float32),
        **{f"{name}_probability": probability.astype(np.float32) for name, probability in variant_probability.items()},
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
