"""Shared candidate-conditioned outer-cross-fit router for the full visual bank.

Unlike P117's one-router-per-candidate design, this model pools decisive
rescue/harm examples across candidates and performs one row-level arbitration.
All held-cohort labels remain excluded from model and threshold selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import csv

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import (
    CandidateSplit,
    candidate_features,
    fit_score as default_fit_score,
    gain_and_disagreement,
    load_candidate_splits,
    shared_features,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p118_candidate_conditioned_router_v1"
P120_CACHE = HERE / "runs/p120_yolo11n_coco_object_teacher_v1"
P119_OOF = HERE / "runs/p119_clip_vitb32_semantic_teacher_v1/oof_predictions.npz"
P12_MULTIMODAL = HERE / "runs/p12_complete_oof/complete_oof.npz"
P90_IMU_DEEP = HERE.parent / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz"
P90_MOTIONBERT = HERE.parent / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front-side-top_linear_oof.npz"
P128_OOF = HERE / "runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz"
P86_VISUAL_FEATURES = HERE / "runs/p86_visual_student_cache_v1/features.npy"
P86_VISUAL_ROWS = HERE / "runs/p86_visual_student_cache_v1/rows.csv"
P119_CLIP_FEATURES = HERE / "runs/p119_clip_vitb32_semantic_teacher_v1/frame_features.npy"
P130_EPIC_FEATURES = HERE / "runs/p130_epic_slowfast_teacher_v1/features.npy"
OBJECT_CLASS_IDS = (
    0, 16, 25, 26, 27, 28, 32, 35, 38, 39, 40, 41, 45, 56, 57, 59,
    60, 61, 62, 63, 65, 66, 67, 69, 71, 72, 73, 74, 75, 79,
)


def router_fit_score(
    train_x: np.ndarray,
    gain: np.ndarray,
    predict_x: np.ndarray,
    users: np.ndarray,
    weighting: str,
) -> np.ndarray:
    if weighting == "default":
        return default_fit_score(train_x, gain, predict_x)
    if weighting != "user_target_balanced":
        raise ValueError(f"unknown router weighting: {weighting}")
    decisive = gain != 0
    target = (gain[decisive] > 0).astype(np.int64)
    decisive_users = users.astype(str)[decisive]
    weights = np.zeros(len(target), dtype=np.float64)
    for user in sorted(set(decisive_users.tolist())):
        for label in (0, 1):
            selected = (decisive_users == user) & (target == label)
            if selected.any():
                weights[selected] = 1.0 / float(selected.sum())
    weights *= len(weights) / max(float(weights.sum()), 1e-12)
    values = train_x[decisive]
    predictions = []
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.03, max_iter=1200, solver="liblinear"),
    )
    logistic.fit(values, target, logisticregression__sample_weight=weights)
    predictions.append(logistic.predict_proba(predict_x)[:, 1])
    for model in (
        ExtraTreesClassifier(
            n_estimators=240,
            max_depth=7,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=11701,
            n_jobs=-1,
        ),
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=140,
            max_leaf_nodes=7,
            min_samples_leaf=15,
            l2_regularization=10.0,
            random_state=11702,
        ),
    ):
        model.fit(values, target, sample_weight=weights)
        predictions.append(model.predict_proba(predict_x)[:, 1])
    return np.mean(np.stack(predictions, axis=1), axis=1)


def object_router_features() -> dict[str, np.ndarray]:
    with (HERE / "runs/p86_visual_pixel_cache_t16_r160_v12/rows.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        sample_ids = [row["sample_id"] for row in csv.DictReader(handle)]
    raw = np.load(P120_CACHE / "frame_object_features.npy", mmap_mode="r")
    values = np.asarray(raw, dtype=np.float32).reshape(
        len(sample_ids), 2, 2, 2, 3, 80
    )[..., list(OBJECT_CLASS_IDS)]
    mean = values.mean(axis=2)  # rows, windows, views, statistics, classes
    maximum = values.max(axis=2)
    delta = mean[:, 1] - mean[:, 0]
    vector = np.concatenate(
        (
            mean.transpose(0, 2, 1, 3, 4).reshape(len(sample_ids), -1),
            maximum.transpose(0, 2, 1, 3, 4).reshape(len(sample_ids), -1),
            delta.reshape(len(sample_ids), -1),
        ),
        axis=1,
    ).astype(np.float32)
    return {sample_id: vector[index] for index, sample_id in enumerate(sample_ids)}


def clip_router_features() -> dict[str, np.ndarray]:
    source = np.load(P119_OOF)
    sample_ids = source["sample_ids"].astype(str)
    vector = np.concatenate(
        [
            source[f"{name}_probability"].astype(np.float32)
            for name in ("scene", "person", "workspace", "all_views", "zero_shot")
        ],
        axis=1,
    )
    return {sample_id: vector[index] for index, sample_id in enumerate(sample_ids)}


def nonvisual_router_features() -> dict[str, np.ndarray]:
    with np.load(P12_MULTIMODAL) as thermal, np.load(P90_IMU_DEEP) as imu, np.load(
        P90_MOTIONBERT
    ) as motion:
        sample_ids = thermal["sample_ids"].astype(str)
        imu_lookup = {
            value: index for index, value in enumerate(imu["sample_ids"].astype(str))
        }
        motion_lookup = {
            value: index for index, value in enumerate(motion["sample_ids"].astype(str))
        }
        imu_positions = np.asarray([imu_lookup[value] for value in sample_ids], dtype=np.int64)
        motion_positions = np.asarray(
            [motion_lookup[value] for value in sample_ids], dtype=np.int64
        )
        thermal_logits = thermal["thermal_candidate_logits"].astype(np.float64)
        thermal_logits -= thermal_logits.max(axis=1, keepdims=True)
        thermal_probability = np.exp(thermal_logits)
        thermal_probability /= thermal_probability.sum(axis=1, keepdims=True)
        vector = np.concatenate(
            (
                thermal_probability.astype(np.float32),
                imu["probabilities"][imu_positions].astype(np.float32),
                motion["probabilities"][motion_positions].astype(np.float32),
            ),
            axis=1,
        )
    return {sample_id: vector[index] for index, sample_id in enumerate(sample_ids)}


def hierarchical_router_features() -> dict[str, np.ndarray]:
    source = np.load(P128_OOF)
    sample_ids = source["sample_ids"].astype(str)
    reliability = 1.0 / (
        1.0 + np.exp(-np.clip(source["reliability_logits"].astype(np.float32), -20, 20))
    )
    vector = np.concatenate(
        (source["probabilities"].astype(np.float32), reliability[:, None]), axis=1
    )
    return {sample_id: vector[index] for index, sample_id in enumerate(sample_ids)}


def _fixed_sparse_projection(
    values: np.ndarray, components: int, seed: int
) -> np.ndarray:
    """Label-free deterministic projection for frozen embedding geometry."""
    values = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    projector = SparseRandomProjection(
        n_components=components,
        density="auto",
        random_state=seed,
    )
    projected = np.asarray(projector.fit_transform(values), dtype=np.float32)
    projected /= np.maximum(
        np.linalg.norm(projected, axis=1, keepdims=True), 1e-8
    )
    return projected


def frozen_embedding_router_features() -> dict[str, np.ndarray]:
    """Compact, label-free appearance/motion context from three frozen teachers."""
    with P86_VISUAL_ROWS.open("r", encoding="utf-8-sig", newline="") as handle:
        sample_ids = [row["sample_id"] for row in csv.DictReader(handle)]

    visual = np.load(P86_VISUAL_FEATURES, mmap_mode="r")
    # Preserve window/view identity and temporal dispersion without exposing IDs.
    visual_summary = np.concatenate(
        (
            np.asarray(visual, dtype=np.float32).mean(axis=2),
            np.asarray(visual, dtype=np.float32).std(axis=2),
        ),
        axis=-1,
    )
    clip = np.load(P119_CLIP_FEATURES, mmap_mode="r")
    epic = np.load(P130_EPIC_FEATURES, mmap_mode="r")
    if not (len(sample_ids) == len(visual) == len(clip) == len(epic)):
        raise RuntimeError("frozen embedding row counts differ")

    vector = np.concatenate(
        (
            _fixed_sparse_projection(visual_summary, 128, 13201),
            _fixed_sparse_projection(clip, 64, 13202),
            _fixed_sparse_projection(epic, 64, 13203),
        ),
        axis=1,
    ).astype(np.float32)
    return {sample_id: vector[index] for index, sample_id in enumerate(sample_ids)}


def feature_bank(
    data: dict[str, CandidateSplit],
    include_object_features: bool = False,
    include_clip_features: bool = False,
    include_nonvisual_features: bool = False,
    include_hierarchical_features: bool = False,
    include_frozen_embedding_features: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    candidate_names = list(next(iter(data.values())).candidates)
    result: dict[str, dict[str, np.ndarray]] = {}
    object_lookup = object_router_features() if include_object_features else None
    clip_lookup = clip_router_features() if include_clip_features else None
    nonvisual_lookup = (
        nonvisual_router_features() if include_nonvisual_features else None
    )
    hierarchical_lookup = (
        hierarchical_router_features() if include_hierarchical_features else None
    )
    embedding_lookup = (
        frozen_embedding_router_features()
        if include_frozen_embedding_features
        else None
    )
    for split_name, value in data.items():
        if list(value.candidates) != candidate_names:
            raise ValueError("candidate order differs by split")
        shared = shared_features(value)
        if object_lookup is not None:
            object_values = np.stack(
                [object_lookup[sample_id] for sample_id in value.split.sample_ids.astype(str)]
            ).astype(np.float32)
            shared = np.concatenate((shared, object_values), axis=1)
        if clip_lookup is not None:
            clip_values = np.stack(
                [clip_lookup[sample_id] for sample_id in value.split.sample_ids.astype(str)]
            ).astype(np.float32)
            shared = np.concatenate((shared, clip_values), axis=1)
        if nonvisual_lookup is not None:
            nonvisual_values = np.stack(
                [
                    nonvisual_lookup[sample_id]
                    for sample_id in value.split.sample_ids.astype(str)
                ]
            ).astype(np.float32)
            shared = np.concatenate((shared, nonvisual_values), axis=1)
        if hierarchical_lookup is not None:
            hierarchical_values = np.stack(
                [
                    hierarchical_lookup[sample_id]
                    for sample_id in value.split.sample_ids.astype(str)
                ]
            ).astype(np.float32)
            shared = np.concatenate((shared, hierarchical_values), axis=1)
        if embedding_lookup is not None:
            embedding_values = np.stack(
                [
                    embedding_lookup[sample_id]
                    for sample_id in value.split.sample_ids.astype(str)
                ]
            ).astype(np.float32)
            shared = np.concatenate((shared, embedding_values), axis=1)
        result[split_name] = {}
        for candidate_index, candidate_name in enumerate(candidate_names):
            base = candidate_features(
                value,
                shared,
                candidate_name,
                include_session_context=False,
            )
            identity = np.zeros((len(base), len(candidate_names)), dtype=np.float32)
            identity[:, candidate_index] = 1.0
            result[split_name][candidate_name] = np.concatenate(
                (base, identity), axis=1
            ).astype(np.float32)
    return result, candidate_names


def stack_instances(
    split_names: list[str],
    data: dict[str, CandidateSplit],
    features: dict[str, dict[str, np.ndarray]],
    candidate_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrices = []
    gains = []
    disagreements = []
    users = []
    sample_keys = []
    candidate_keys = []
    offset = 0
    for split_name in split_names:
        value = data[split_name]
        rows = len(value.split.labels)
        for candidate_index, candidate_name in enumerate(candidate_names):
            gain, disagreement = gain_and_disagreement(value, candidate_name)
            matrices.append(features[split_name][candidate_name])
            gains.append(gain)
            disagreements.append(disagreement)
            users.append(value.split.users)
            sample_keys.append(np.arange(rows, dtype=np.int64) + offset)
            candidate_keys.append(np.full(rows, candidate_index, dtype=np.int64))
        offset += rows
    return (
        np.concatenate(matrices),
        np.concatenate(gains),
        np.concatenate(disagreements),
        np.concatenate(users),
        np.concatenate(sample_keys),
        np.concatenate(candidate_keys),
    )


def score_to_row_choice(
    scores: np.ndarray,
    disagreement: np.ndarray,
    sample_keys: np.ndarray,
    candidate_keys: np.ndarray,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    best_score = np.full(rows, -np.inf, dtype=np.float64)
    best_candidate = np.full(rows, -1, dtype=np.int64)
    for index in range(len(scores)):
        if not disagreement[index]:
            continue
        row = int(sample_keys[index])
        if scores[index] > best_score[row]:
            best_score[row] = float(scores[index])
            best_candidate[row] = int(candidate_keys[index])
    return best_score, best_candidate


def apply_choice(
    data_names: list[str],
    data: dict[str, CandidateSplit],
    candidate_names: list[str],
    best_score: np.ndarray,
    best_candidate: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.concatenate([data[name].split.labels for name in data_names])
    users = np.concatenate([data[name].split.users for name in data_names])
    safe = np.concatenate([data[name].split.safe_prediction for name in data_names])
    candidate_predictions = {
        candidate_name: np.concatenate(
            [data[name].candidates[candidate_name].argmax(axis=1) for name in data_names]
        )
        for candidate_name in candidate_names
    }
    output = safe.copy()
    route = (best_candidate >= 0) & (best_score >= threshold)
    for candidate_index, candidate_name in enumerate(candidate_names):
        selected = route & (best_candidate == candidate_index)
        output[selected] = candidate_predictions[candidate_name][selected]
    return labels, users, safe, output


def route_report(
    labels: np.ndarray,
    users: np.ndarray,
    safe: np.ndarray,
    output: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    changed = output != safe
    rescue = int(np.sum((output == labels) & (safe != labels)))
    harm = int(np.sum((output != labels) & (safe == labels)))
    per_user = {
        user: int(
            np.sum(output[users.astype(str) == user] == labels[users.astype(str) == user])
            - np.sum(safe[users.astype(str) == user] == labels[users.astype(str) == user])
        )
        for user in sorted(set(users.astype(str).tolist()))
    }
    return {
        "threshold": float(threshold),
        "correct": int(np.sum(output == labels)),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "changed": int(changed.sum()),
        "minimum_user_gain": int(min(per_user.values())),
        "positive_users": int(sum(value > 0 for value in per_user.values())),
        "per_user_gain": per_user,
    }


def nested_row_scores(
    x: np.ndarray,
    gain: np.ndarray,
    disagreement: np.ndarray,
    users: np.ndarray,
    sample_keys: np.ndarray,
    candidate_keys: np.ndarray,
    weighting: str,
) -> tuple[np.ndarray, np.ndarray]:
    instance_scores = np.zeros(len(x), dtype=np.float64)
    for user in sorted(set(users.astype(str).tolist())):
        held = users.astype(str) == user
        instance_scores[held] = router_fit_score(
            x[~held], gain[~held], x[held], users[~held], weighting
        )
    rows = int(sample_keys.max()) + 1
    return score_to_row_choice(
        instance_scores, disagreement, sample_keys, candidate_keys, rows
    )


def select_threshold(
    data_names: list[str],
    data: dict[str, CandidateSplit],
    candidate_names: list[str],
    best_score: np.ndarray,
    best_candidate: np.ndarray,
    policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    finite = best_score[np.isfinite(best_score)]
    thresholds = np.unique(
        np.concatenate(
            (
                np.arange(0.30, 0.901, 0.025),
                np.quantile(finite, np.linspace(0.25, 0.95, 15)),
            )
        )
    )
    reports = []
    for threshold in thresholds:
        values = apply_choice(
            data_names,
            data,
            candidate_names,
            best_score,
            best_candidate,
            float(threshold),
        )
        reports.append(route_report(*values, float(threshold)))
    eligible = [row for row in reports if row["net"] > 0 and row["changed"] >= 5]
    if policy == "stable":
        eligible.sort(
            key=lambda row: (
                row["minimum_user_gain"] >= 0,
                row["net"],
                row["positive_users"],
                -row["harm"],
                -row["changed"],
            ),
            reverse=True,
        )
    elif policy == "max_net":
        eligible.sort(
            key=lambda row: (
                row["net"],
                -row["harm"],
                row["positive_users"],
                row["minimum_user_gain"],
                -row["changed"],
            ),
            reverse=True,
        )
    else:
        raise ValueError(f"unknown threshold policy: {policy}")
    if eligible:
        return eligible[0], reports
    values = apply_choice(
        data_names, data, candidate_names, best_score, best_candidate, 1.1
    )
    return route_report(*values, 1.1), reports


def evaluate_outer(
    held_name: str,
    train_names: list[str],
    data: dict[str, CandidateSplit],
    features: dict[str, dict[str, np.ndarray]],
    candidate_names: list[str],
    threshold_policy: str,
    threshold_names: list[str] | None = None,
    router_weighting: str = "default",
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train = stack_instances(train_names, data, features, candidate_names)
    train_x, train_gain, train_disagreement, train_users, train_sample, train_candidate = train
    threshold_names = train_names if threshold_names is None else threshold_names
    if threshold_names == train_names:
        nested_score, nested_candidate = nested_row_scores(*train, router_weighting)
    else:
        selection = stack_instances(
            threshold_names, data, features, candidate_names
        )
        (
            selection_x,
            _,
            selection_disagreement,
            selection_users,
            selection_sample,
            selection_candidate,
        ) = selection
        instance_score = np.zeros(len(selection_x), dtype=np.float64)
        for user in sorted(set(selection_users.astype(str).tolist())):
            fit_rows = train_users.astype(str) != user
            held_rows = selection_users.astype(str) == user
            instance_score[held_rows] = router_fit_score(
                train_x[fit_rows],
                train_gain[fit_rows],
                selection_x[held_rows],
                train_users[fit_rows],
                router_weighting,
            )
        nested_score, nested_candidate = score_to_row_choice(
            instance_score,
            selection_disagreement,
            selection_sample,
            selection_candidate,
            int(selection_sample.max()) + 1,
        )
    selected, threshold_grid = select_threshold(
        threshold_names,
        data,
        candidate_names,
        nested_score,
        nested_candidate,
        threshold_policy,
    )

    held = stack_instances([held_name], data, features, candidate_names)
    held_x, _, held_disagreement, _, held_sample, held_candidate_key = held
    held_instance_score = router_fit_score(
        train_x, train_gain, held_x, train_users, router_weighting
    )
    held_score, held_candidate = score_to_row_choice(
        held_instance_score,
        held_disagreement,
        held_sample,
        held_candidate_key,
        len(data[held_name].split.labels),
    )
    values = apply_choice(
        [held_name],
        data,
        candidate_names,
        held_score,
        held_candidate,
        float(selected["threshold"]),
    )
    labels, users, safe, output = values
    report = {
        "held_split": held_name,
        "train_splits": train_names,
        "selected_threshold": selected,
        "held_route": route_report(*values, float(selected["threshold"])),
        "safe_metrics": classification_metrics(labels, safe),
        "router_metrics": classification_metrics(labels, output),
        "top_nested_thresholds": sorted(
            threshold_grid,
            key=lambda row: (
                row["minimum_user_gain"] >= 0,
                row["net"],
                row["positive_users"],
            ),
            reverse=True,
        )[:10],
    }
    nested_payload = {
        "sample_ids": np.concatenate(
            [data[name].split.sample_ids for name in threshold_names]
        ),
        "labels": np.concatenate([data[name].split.labels for name in threshold_names]),
        "users": np.concatenate([data[name].split.users for name in threshold_names]),
        "safe_prediction": np.concatenate(
            [data[name].split.safe_prediction for name in threshold_names]
        ),
        "route_score": nested_score,
        "candidate_index": nested_candidate,
    }
    return report, output, held_score, held_candidate, nested_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--structured-bank", action="store_true")
    parser.add_argument("--legacy-visual-bank", action="store_true")
    parser.add_argument("--hand-object-bank", action="store_true")
    parser.add_argument("--vjepa-dense-bank", action="store_true")
    parser.add_argument("--nonvisual-bank", action="store_true")
    parser.add_argument("--include-embargo-source", action="store_true")
    parser.add_argument("--hierarchical-bank", action="store_true")
    parser.add_argument("--epic-bank", action="store_true")
    parser.add_argument("--egovlp-bank", action="store_true")
    parser.add_argument("--embargo-train-only", action="store_true")
    parser.add_argument("--object-router-features", action="store_true")
    parser.add_argument("--clip-router-features", action="store_true")
    parser.add_argument("--nonvisual-router-features", action="store_true")
    parser.add_argument("--hierarchical-router-features", action="store_true")
    parser.add_argument("--frozen-embedding-router-features", action="store_true")
    parser.add_argument(
        "--threshold-policy", choices=("stable", "max_net"), default="stable"
    )
    parser.add_argument(
        "--router-weighting",
        choices=("default", "user_target_balanced"),
        default="default",
    )
    args = parser.parse_args()
    data = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=args.structured_bank,
        legacy_visual_bank=args.legacy_visual_bank,
        hand_object_bank=args.hand_object_bank,
        vjepa_dense_bank=args.vjepa_dense_bank,
        nonvisual_bank=args.nonvisual_bank,
        embargo_source=args.include_embargo_source,
        hierarchical_bank=args.hierarchical_bank,
        epic_bank=args.epic_bank,
        egovlp_bank=args.egovlp_bank,
    )
    features, candidate_names = feature_bank(
        data,
        include_object_features=args.object_router_features,
        include_clip_features=args.clip_router_features,
        include_nonvisual_features=args.nonvisual_router_features,
        include_hierarchical_features=args.hierarchical_router_features,
        include_frozen_embedding_features=args.frozen_embedding_router_features,
    )
    recipes = {
        "H1_selection": ["H2_confirmation", "H3_independent_fold0"],
        "H2_confirmation": ["H1_selection", "H3_independent_fold0"],
        "H3_independent_fold0": ["H1_selection", "H2_confirmation"],
    }
    threshold_recipes = {
        key: list(value) for key, value in recipes.items()
    }
    if args.include_embargo_source:
        for source_names in recipes.values():
            source_names.append("E0_p87_sequence_source")
    reports = {}
    payload: dict[str, np.ndarray] = {}
    safe_correct = router_correct = rows = 0
    for held_name, train_names in recipes.items():
        report, prediction, score, candidate, nested = evaluate_outer(
            held_name,
            train_names,
            data,
            features,
            candidate_names,
            args.threshold_policy,
            threshold_names=(
                threshold_recipes[held_name]
                if args.include_embargo_source and args.embargo_train_only
                else train_names
            ),
            router_weighting=args.router_weighting,
        )
        reports[held_name] = report
        split = data[held_name].split
        safe_correct += report["safe_metrics"]["correct"]
        router_correct += report["router_metrics"]["correct"]
        rows += len(split.labels)
        payload[f"{held_name}_sample_ids"] = split.sample_ids
        payload[f"{held_name}_labels"] = split.labels
        payload[f"{held_name}_safe_prediction"] = split.safe_prediction
        payload[f"{held_name}_router_prediction"] = prediction
        payload[f"{held_name}_route_score"] = score
        payload[f"{held_name}_candidate_index"] = candidate
        for key, value in nested.items():
            payload[f"{held_name}_source_{key}"] = value
    summary = {
        "stage": "P118_shared_candidate_conditioned_router_v1",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "candidate_names": candidate_names,
            "candidate_count": len(candidate_names),
            "structured_bank": args.structured_bank,
            "legacy_visual_bank": args.legacy_visual_bank,
            "hand_object_bank": args.hand_object_bank,
            "vjepa_dense_bank": args.vjepa_dense_bank,
            "nonvisual_bank": args.nonvisual_bank,
            "include_embargo_source": args.include_embargo_source,
            "hierarchical_bank": args.hierarchical_bank,
            "epic_bank": args.epic_bank,
            "egovlp_bank": args.egovlp_bank,
            "embargo_train_only": args.embargo_train_only,
            "object_router_features": args.object_router_features,
            "object_class_ids": list(OBJECT_CLASS_IDS) if args.object_router_features else [],
            "clip_router_features": args.clip_router_features,
            "nonvisual_router_features": args.nonvisual_router_features,
            "hierarchical_router_features": args.hierarchical_router_features,
            "frozen_embedding_router_features": args.frozen_embedding_router_features,
            "frozen_embedding_projection_is_label_free": True,
            "threshold_policy": args.threshold_policy,
            "router_weighting": args.router_weighting,
            "shared_router": True,
            "outer_held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": rows,
            "safe_correct": safe_correct,
            "safe_accuracy": safe_correct / rows,
            "router_correct": router_correct,
            "router_accuracy": router_correct / rows,
            "net": router_correct - safe_correct,
            "target_0.91_correct": int(np.ceil(0.91 * rows)),
            "gap_to_0.91_correct": int(np.ceil(0.91 * rows) - router_correct),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "predictions.npz", **payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
