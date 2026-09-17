"""Strict outer-cross-fit router over P89 safe, A18, and P90 visual teachers.

Only source OOF labels train/calibrate a held cohort.  Features may use the full
unlabeled held batch, including session-level uniqueness/conflict statistics, but
never user identity.  The script is an OOF audit and deliberately has no Test or
submission path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import classification_metrics
from p90_crossuser_visual_router import (
    CANDIDATE_NAME,
    SplitData,
    aligned_quality,
    load_splits,
    one_hot,
    probability_entropy,
    probability_margin,
)
from p90_visual_teacher_safe_fusion_audit import load_visual_candidates


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTPUT = HERE / "runs/p117_transductive_multicandidate_router_v1"
A18 = PROJECT / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
P88_SEQUENCE = HERE / "runs/p88_sequence_repeat_v1/oof_predictions.npz"
P88_LATENT = HERE / "runs/p88_latent_prefix_nested_v1/oof_predictions.npz"
P85_VISUAL = HERE / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
P86_MECHANISMS = HERE / "runs/p86_teacher_mechanism_audit_v1/fixed_model_predictions.npz"
P122_RELATIONS = HERE / "runs/p122_hand_object_relation_teacher_v1/oof_predictions.npz"
P123_DENSE = HERE / "runs/p123_vjepa2_dense24_full_oof_v1/oof_predictions.npz"
P12_MULTIMODAL = HERE / "runs/p12_complete_oof/complete_oof.npz"
P90_IMU_DEEP = PROJECT / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz"
P90_MOTIONBERT = PROJECT / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front-side-top_linear_oof.npz"
P87_SEQUENCE = HERE / "runs/p87_sequence_decoder_v1/oof_predictions.npz"
P87_STRUCTURED = HERE / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
P128_HIERARCHICAL = HERE / "runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz"
P130_EPIC = HERE / "runs/p130_epic_slowfast_teacher_v1/oof_predictions.npz"
P131_EGOVLP = HERE / "runs/p131_egovlp_teacher_v1/oof_predictions.npz"
P89_SKELETON_INVARIANT = HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz"
P16_LOCAL_DEPTH = HERE / "runs/p16_local_depth_oracle_oof/oof_logits.npz"
P90_MOTIONBERT_FRONT = PROJECT / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz"
P142_TOKEN = HERE / "runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz"
P144_HAND_TOKEN = HERE / "runs/p144_vjepa_hand_interaction_transformer_three_seed_v1/oof_predictions.npz"
P146_WORKSPACE_TOKEN = HERE / "runs/p146_vjepa_workspace_transformer_three_seed_v1/oof_predictions.npz"
P158_LAVILA_FRAME_TOKEN = HERE / "runs/p158_lavila_frame_token_transformer_single_seed_v1/oof_predictions.npz"
EPSILON = 1e-8


@dataclass(frozen=True)
class CandidateSplit:
    split: SplitData
    candidates: dict[str, np.ndarray]


def embargo_split(
    existing: dict[str, SplitData], a18: np.lib.npyio.NpzFile
) -> SplitData:
    used = {
        sample_id
        for split in existing.values()
        for sample_id in split.sample_ids.astype(str).tolist()
    }
    all_ids = a18["sample_ids"].astype(str)
    positions = np.flatnonzero(~np.isin(all_ids, list(used)))
    ids = all_ids[positions]
    if len(ids) != 444:
        raise RuntimeError(f"E0 embargo universe changed: {len(ids)}")
    sequence = np.load(P87_SEQUENCE)
    structured = np.load(P87_STRUCTURED)
    safe_prediction = align_probability(
        ids, sequence["sample_ids"], sequence["sequence_predictions"]
    ).astype(np.int64)
    safe_probability = align_probability(
        ids, structured["sample_ids"], structured["emission_probability"]
    )
    safe_probability = np.clip(safe_probability, EPSILON, None)
    safe_probability /= safe_probability.sum(axis=1, keepdims=True)
    visual_ids, visual = load_visual_candidates()
    quality, quality_names = aligned_quality(ids)
    users = a18["users"][positions].astype(str)
    if set(users.tolist()) != {"user1", "user2", "user21"}:
        raise RuntimeError(f"E0 embargo users changed: {sorted(set(users.tolist()))}")
    return SplitData(
        name="E0_p87_sequence_source",
        sample_ids=ids,
        labels=a18["labels"][positions].astype(np.int64),
        users=users,
        safe_probability=safe_probability.astype(np.float64),
        safe_prediction=safe_prediction,
        p87_prediction=safe_prediction.copy(),
        sessions=[],
        visual_probability={
            key: align_probability(ids, visual_ids, probability)
            for key, probability in visual.items()
        },
        quality=quality,
        quality_names=quality_names,
    )


def align_probability(
    target_ids: np.ndarray, source_ids: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    positions = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(probability[positions], dtype=np.float64)


def smoothed_hard_probability(prediction: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    output = np.full(
        (len(prediction), 40),
        (1.0 - confidence) / 39.0,
        dtype=np.float64,
    )
    output[np.arange(len(prediction)), np.asarray(prediction, dtype=np.int64)] = confidence
    return output


def logits_to_probability(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    probability = np.exp(values)
    return probability / probability.sum(axis=1, keepdims=True)


def load_candidate_splits(
    full_visual_bank: bool = False,
    structured_bank: bool = False,
    legacy_visual_bank: bool = False,
    hand_object_bank: bool = False,
    vjepa_dense_bank: bool = False,
    nonvisual_bank: bool = False,
    embargo_source: bool = False,
    hierarchical_bank: bool = False,
    epic_bank: bool = False,
    egovlp_bank: bool = False,
    expanded_bank: bool = False,
) -> dict[str, CandidateSplit]:
    splits = dict(load_splits())
    a18 = np.load(A18)
    if embargo_source:
        splits["E0_p87_sequence_source"] = embargo_split(splits, a18)
    a18_ids = a18["sample_ids"].astype(str)
    p88 = np.load(P88_SEQUENCE) if structured_bank else None
    latent = np.load(P88_LATENT) if structured_bank else None
    p85_visual = np.load(P85_VISUAL) if legacy_visual_bank else None
    p86_mechanisms = np.load(P86_MECHANISMS) if legacy_visual_bank else None
    p122_relations = np.load(P122_RELATIONS) if hand_object_bank else None
    p123_dense = np.load(P123_DENSE) if vjepa_dense_bank else None
    p12_multimodal = np.load(P12_MULTIMODAL) if nonvisual_bank else None
    p90_imu_deep = np.load(P90_IMU_DEEP) if nonvisual_bank else None
    p90_motionbert = np.load(P90_MOTIONBERT) if nonvisual_bank else None
    p128_hierarchical = np.load(P128_HIERARCHICAL) if hierarchical_bank else None
    p130_epic = np.load(P130_EPIC) if epic_bank else None
    p131_egovlp = np.load(P131_EGOVLP) if egovlp_bank else None
    expanded_p12 = np.load(P12_MULTIMODAL) if expanded_bank else None
    expanded_motion_front = np.load(P90_MOTIONBERT_FRONT) if expanded_bank else None
    expanded_skeleton = np.load(P89_SKELETON_INVARIANT) if expanded_bank else None
    expanded_relations = np.load(P122_RELATIONS) if expanded_bank else None
    expanded_local_depth = np.load(P16_LOCAL_DEPTH) if expanded_bank else None
    expanded_egovlp = np.load(P131_EGOVLP) if expanded_bank else None
    expanded_p142 = np.load(P142_TOKEN) if expanded_bank else None
    expanded_p144 = np.load(P144_HAND_TOKEN) if expanded_bank else None
    expanded_p146 = np.load(P146_WORKSPACE_TOKEN) if expanded_bank else None
    expanded_p158 = np.load(P158_LAVILA_FRAME_TOKEN) if expanded_bank else None
    result: dict[str, CandidateSplit] = {}
    for split_name, split in splits.items():
        candidates = {
            "a18_best_session": align_probability(
                split.sample_ids, a18_ids, a18["best_session_probability"]
            ),
            "p90_visual_equal": np.asarray(
                split.visual_probability[CANDIDATE_NAME], dtype=np.float64
            ),
        }
        if full_visual_bank:
            for visual_name, probability in split.visual_probability.items():
                if visual_name == CANDIDATE_NAME:
                    continue
                candidates[f"p90_{visual_name}"] = np.asarray(
                    probability, dtype=np.float64
                )
        if structured_bank:
            assert p88 is not None and latent is not None
            for candidate_name, source, key in (
                ("p87_sequence", p88, "p87_predictions"),
                ("p88_repeat", p88, "p88_predictions"),
                ("p88_latent_prefix", latent, "latent_prefix_predictions"),
            ):
                lookup = {
                    value: index
                    for index, value in enumerate(source["sample_ids"].astype(str))
                }
                positions = np.asarray(
                    [lookup[value] for value in split.sample_ids.astype(str)],
                    dtype=np.int64,
                )
                candidates[candidate_name] = smoothed_hard_probability(
                    source[key][positions]
                )
        if legacy_visual_bank:
            assert p85_visual is not None and p86_mechanisms is not None
            for candidate_name, source, key in (
                ("p85_window_mean", p85_visual, "window_mean_logits"),
                ("p85_early", p85_visual, "early_logits"),
                ("p86_drop_person", p86_mechanisms, "drop_person_logits"),
            ):
                lookup = {
                    value: index
                    for index, value in enumerate(source["sample_ids"].astype(str))
                }
                positions = np.asarray(
                    [lookup[value] for value in split.sample_ids.astype(str)],
                    dtype=np.int64,
                )
                candidates[candidate_name] = logits_to_probability(source[key][positions])
        if hand_object_bank:
            assert p122_relations is not None
            lookup = {
                value: index
                for index, value in enumerate(p122_relations["sample_ids"].astype(str))
            }
            positions = np.asarray(
                [lookup[value] for value in split.sample_ids.astype(str)], dtype=np.int64
            )
            candidates["p122_hand_object_all"] = np.asarray(
                p122_relations["all_probability"][positions], dtype=np.float64
            )
        if vjepa_dense_bank:
            assert p123_dense is not None
            lookup = {
                value: index
                for index, value in enumerate(p123_dense["sample_ids"].astype(str))
            }
            positions = np.asarray(
                [lookup[value] for value in split.sample_ids.astype(str)], dtype=np.int64
            )
            for dense_name in ("dense24_group8", "dense24_group8_ssv2"):
                candidates[f"p123_{dense_name}"] = np.asarray(
                    p123_dense[f"{dense_name}_probability"][positions], dtype=np.float64
                )
        if nonvisual_bank:
            assert (
                p12_multimodal is not None
                and p90_imu_deep is not None
                and p90_motionbert is not None
            )
            for candidate_name, source, key, is_probability in (
                (
                    "p12_thermal_candidate",
                    p12_multimodal,
                    "thermal_candidate_logits",
                    False,
                ),
                ("p90_deep_imu", p90_imu_deep, "probabilities", True),
                ("p90_motionbert_3view", p90_motionbert, "probabilities", True),
            ):
                lookup = {
                    value: index
                    for index, value in enumerate(source["sample_ids"].astype(str))
                }
                positions = np.asarray(
                    [lookup[value] for value in split.sample_ids.astype(str)],
                    dtype=np.int64,
                )
                values = np.asarray(source[key][positions], dtype=np.float64)
                candidates[candidate_name] = (
                    values if is_probability else logits_to_probability(values)
                )
        if hierarchical_bank:
            assert p128_hierarchical is not None
            lookup = {
                value: index
                for index, value in enumerate(
                    p128_hierarchical["sample_ids"].astype(str)
                )
            }
            positions = np.asarray(
                [lookup[value] for value in split.sample_ids.astype(str)],
                dtype=np.int64,
            )
            candidates["p128_hierarchical_multimodal"] = np.asarray(
                p128_hierarchical["probabilities"][positions], dtype=np.float64
            )
        if epic_bank:
            assert p130_epic is not None
            lookup = {
                value: index
                for index, value in enumerate(p130_epic["sample_ids"].astype(str))
            }
            positions = np.asarray(
                [lookup[value] for value in split.sample_ids.astype(str)], dtype=np.int64
            )
            candidates["p130_epic_slowfast"] = np.asarray(
                p130_epic["all_views_epic_logits_probability"][positions],
                dtype=np.float64,
            )
        if egovlp_bank:
            assert p131_egovlp is not None
            lookup = {
                value: index
                for index, value in enumerate(p131_egovlp["sample_ids"].astype(str))
            }
            positions = np.asarray(
                [lookup[value] for value in split.sample_ids.astype(str)], dtype=np.int64
            )
            candidates["p131_egovlp"] = np.asarray(
                p131_egovlp["all_raw_projected_probability"][positions],
                dtype=np.float64,
            )
        if expanded_bank:
            assert (
                expanded_p12 is not None
                and expanded_motion_front is not None
                and expanded_skeleton is not None
                and expanded_relations is not None
                and expanded_local_depth is not None
                and expanded_egovlp is not None
                and expanded_p142 is not None
                and expanded_p144 is not None
                and expanded_p146 is not None
                and expanded_p158 is not None
            )
            for candidate_name, source, key, is_probability in (
                ("expanded_thermal", expanded_p12, "thermal_logits", False),
                ("expanded_p12_imu", expanded_p12, "imu_logits", False),
                (
                    "expanded_motionbert_front",
                    expanded_motion_front,
                    "probabilities",
                    True,
                ),
                (
                    "expanded_skeleton_invariant",
                    expanded_skeleton,
                    "skeleton_logits",
                    False,
                ),
                (
                    "expanded_p122_pose",
                    expanded_relations,
                    "pose_only_probability",
                    True,
                ),
                (
                    "expanded_p122_object",
                    expanded_relations,
                    "object_only_probability",
                    True,
                ),
                (
                    "expanded_p122_relation",
                    expanded_relations,
                    "relations_only_probability",
                    True,
                ),
                (
                    "expanded_local_depth",
                    expanded_local_depth,
                    "logits",
                    False,
                ),
                (
                    "expanded_egovlp",
                    expanded_egovlp,
                    "all_raw_projected_probability",
                    True,
                ),
                (
                    "expanded_p142_token",
                    expanded_p142,
                    "probability",
                    True,
                ),
                (
                    "expanded_p144_hand_token",
                    expanded_p144,
                    "probability",
                    True,
                ),
                (
                    "expanded_p146_workspace_token",
                    expanded_p146,
                    "probability",
                    True,
                ),
                (
                    "expanded_p158_lavila_frame_token",
                    expanded_p158,
                    "probability",
                    True,
                ),
            ):
                source_lookup = {
                    value: index
                    for index, value in enumerate(source["sample_ids"].astype(str))
                }
                source_positions = np.asarray(
                    [source_lookup[value] for value in split.sample_ids.astype(str)],
                    dtype=np.int64,
                )
                raw = np.asarray(source[key][source_positions], dtype=np.float64)
                candidates[candidate_name] = (
                    raw if is_probability else logits_to_probability(raw)
                )
            if p123_dense is not None:
                dense_lookup = {
                    value: index
                    for index, value in enumerate(p123_dense["sample_ids"].astype(str))
                }
                dense_positions = np.asarray(
                    [dense_lookup[value] for value in split.sample_ids.astype(str)],
                    dtype=np.int64,
                )
                candidates["expanded_p123_old_ir_dense"] = np.asarray(
                    p123_dense[
                        "old_ir_plus_dense24_group8_ssv2_probability"
                    ][dense_positions],
                    dtype=np.float64,
                )
        result[split_name] = CandidateSplit(
            split=split,
            candidates=candidates,
        )
    if set(result) != set(splits):
        raise RuntimeError(f"candidate split keys changed: {sorted(result)}")
    return result


def probability_scalars(
    probability: np.ndarray,
    safe_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> np.ndarray:
    ordered = np.sort(probability, axis=1)[:, ::-1]
    rows = np.arange(len(probability))
    return np.column_stack(
        (
            probability.max(axis=1),
            probability_margin(probability),
            probability_entropy(probability),
            ordered[:, :5],
            probability[rows, safe_prediction],
            probability[rows, candidate_prediction],
        )
    ).astype(np.float32)


def session_context(
    sessions: list[np.ndarray],
    safe_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> np.ndarray:
    safe_prediction = safe_probability.argmax(axis=1)
    candidate_prediction = candidate_probability.argmax(axis=1)
    output = np.zeros((len(safe_prediction), 14), dtype=np.float32)
    for raw_session in sessions:
        indices = np.asarray(raw_session, dtype=np.int64)
        if not len(indices):
            continue
        safe_values = safe_prediction[indices]
        candidate_values = candidate_prediction[indices]
        for position, row in enumerate(indices):
            safe_class = int(safe_prediction[row])
            candidate_class = int(candidate_prediction[row])
            other = indices[indices != row]
            safe_duplicate = int(np.sum(safe_values == safe_class))
            candidate_duplicate = int(np.sum(candidate_values == candidate_class))
            switched_safe_duplicate = int(np.sum(safe_values == candidate_class))
            if len(other):
                other_safe_candidate_support = float(safe_probability[other, candidate_class].max())
                other_candidate_support = float(candidate_probability[other, candidate_class].max())
                safe_rank = float(
                    1 + np.sum(safe_probability[other, safe_class] > safe_probability[row, safe_class])
                )
                candidate_rank = float(
                    1
                    + np.sum(
                        candidate_probability[other, candidate_class]
                        > candidate_probability[row, candidate_class]
                    )
                )
            else:
                other_safe_candidate_support = 0.0
                other_candidate_support = 0.0
                safe_rank = candidate_rank = 1.0
            output[row] = (
                len(indices),
                position / max(len(indices) - 1, 1),
                safe_duplicate,
                candidate_duplicate,
                switched_safe_duplicate,
                safe_duplicate - switched_safe_duplicate,
                safe_rank / len(indices),
                candidate_rank / len(indices),
                other_safe_candidate_support,
                other_candidate_support,
                float(position > 0 and safe_values[position - 1] == safe_class),
                float(position + 1 < len(indices) and safe_values[position + 1] == safe_class),
                float(position > 0 and candidate_values[position - 1] == candidate_class),
                float(
                    position + 1 < len(indices)
                    and candidate_values[position + 1] == candidate_class
                ),
            )
    return output


def shared_features(data: CandidateSplit) -> np.ndarray:
    split = data.split
    probabilities = {
        "safe": split.safe_probability,
        **data.candidates,
    }
    safe_prediction = split.safe_prediction
    candidate_predictions = {
        name: probability.argmax(axis=1) for name, probability in data.candidates.items()
    }
    matrices: list[np.ndarray] = []
    for probability in probabilities.values():
        matrices.append(np.log(np.clip(probability, EPSILON, 1.0)).astype(np.float32))
        matrices.append(probability_scalars(probability, safe_prediction, probability.argmax(axis=1)))
    matrices.append(one_hot(safe_prediction))
    matrices.append(one_hot(split.safe_probability.argmax(axis=1)))
    matrices.append(one_hot(split.p87_prediction))
    for prediction in candidate_predictions.values():
        matrices.append(one_hot(prediction))
    stacked = np.stack(list(candidate_predictions.values()), axis=1)
    matrices.append(
        np.column_stack(
            (
                np.mean(stacked == safe_prediction[:, None], axis=1),
                candidate_predictions["a18_best_session"]
                == candidate_predictions["p90_visual_equal"],
                split.quality,
            )
        ).astype(np.float32)
    )
    output = np.concatenate(matrices, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("non-finite shared features")
    return output


def candidate_features(
    data: CandidateSplit,
    shared: np.ndarray,
    candidate_name: str,
    include_session_context: bool = True,
) -> np.ndarray:
    split = data.split
    candidate = data.candidates[candidate_name]
    safe = split.safe_probability
    safe_prediction = split.safe_prediction
    candidate_prediction = candidate.argmax(axis=1)
    rows = np.arange(len(safe))
    other_predictions = np.stack(
        [
            probability.argmax(axis=1)
            for name, probability in data.candidates.items()
            if name != candidate_name
        ],
        axis=1,
    )
    pair = np.column_stack(
        (
            candidate_prediction != safe_prediction,
            np.mean(other_predictions == candidate_prediction[:, None], axis=1),
            np.mean(other_predictions == safe_prediction[:, None], axis=1),
            candidate[rows, candidate_prediction] - safe[rows, safe_prediction],
            candidate[rows, candidate_prediction] - candidate[rows, safe_prediction],
            safe[rows, safe_prediction] - safe[rows, candidate_prediction],
            candidate[rows, safe_prediction],
            safe[rows, candidate_prediction],
            one_hot(candidate_prediction),
        )
    ).astype(np.float32)
    context = session_context(split.sessions, safe, candidate)
    if not include_session_context:
        context = np.zeros_like(context)
    marker = np.full((len(safe), 1), float(candidate_name == "p90_visual_equal"), dtype=np.float32)
    output = np.concatenate((shared, pair, context, marker), axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f"{candidate_name}: non-finite candidate features")
    return output


def gain_and_disagreement(
    data: CandidateSplit, candidate_name: str
) -> tuple[np.ndarray, np.ndarray]:
    prediction = data.candidates[candidate_name].argmax(axis=1)
    safe_correct = data.split.safe_prediction == data.split.labels
    candidate_correct = prediction == data.split.labels
    return (
        candidate_correct.astype(np.int8) - safe_correct.astype(np.int8),
        prediction != data.split.safe_prediction,
    )


def fit_score(train_x: np.ndarray, gain: np.ndarray, predict_x: np.ndarray) -> np.ndarray:
    decisive = gain != 0
    target = (gain[decisive] > 0).astype(np.int64)
    members = []
    models = (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, max_iter=1200, solver="liblinear"),
        ),
        ExtraTreesClassifier(
            n_estimators=240,
            max_depth=7,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced",
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
    )
    for model in models:
        model.fit(train_x[decisive], target)
        members.append(model.predict_proba(predict_x)[:, 1])
    return np.mean(np.stack(members, axis=1), axis=1)


def loso_scores(
    x: np.ndarray, gain: np.ndarray, users: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(x), dtype=np.float64)
    for user in sorted(set(users.astype(str).tolist())):
        held = users.astype(str) == user
        result[held] = fit_score(x[~held], gain[~held], x[held])
    return result


def threshold_metrics(
    score: np.ndarray,
    gain: np.ndarray,
    disagreement: np.ndarray,
    users: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    selected = disagreement & (score >= threshold)
    rescue = int(np.sum(selected & (gain > 0)))
    harm = int(np.sum(selected & (gain < 0)))
    per_user = {
        user: int(np.sum(gain[selected & (users.astype(str) == user)]))
        for user in sorted(set(users.astype(str).tolist()))
    }
    return {
        "threshold": float(threshold),
        "selected": int(selected.sum()),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "minimum_user_gain": int(min(per_user.values())),
        "positive_users": int(sum(value > 0 for value in per_user.values())),
        "per_user_gain": per_user,
    }


def select_threshold(
    score: np.ndarray, gain: np.ndarray, disagreement: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    values = np.unique(
        np.concatenate(
            (
                np.arange(0.30, 0.901, 0.025),
                np.quantile(score[disagreement], np.linspace(0.35, 0.95, 13)),
            )
        )
    )
    reports = [threshold_metrics(score, gain, disagreement, users, value) for value in values]
    eligible = [row for row in reports if row["net"] > 0 and row["selected"] >= 5]
    eligible.sort(
        key=lambda row: (
            row["minimum_user_gain"] >= 0,
            row["net"],
            row["positive_users"],
            -row["harm"],
            -row["selected"],
        ),
        reverse=True,
    )
    return eligible[0] if eligible else threshold_metrics(score, gain, disagreement, users, 1.1)


def concatenate(
    names: list[str],
    data: dict[str, CandidateSplit],
    features: dict[str, dict[str, np.ndarray]],
    candidate_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([features[name][candidate_name] for name in names])
    gain = np.concatenate([gain_and_disagreement(data[name], candidate_name)[0] for name in names])
    disagreement = np.concatenate(
        [gain_and_disagreement(data[name], candidate_name)[1] for name in names]
    )
    users = np.concatenate([data[name].split.users for name in names])
    return x, gain, disagreement, users


def evaluate_outer(
    held_name: str,
    train_names: list[str],
    data: dict[str, CandidateSplit],
    features: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    held = data[held_name]
    route_score: dict[str, np.ndarray] = {}
    thresholds: dict[str, dict[str, Any]] = {}
    routes: dict[str, np.ndarray] = {}
    for candidate_name in held.candidates:
        train_x, gain, disagreement, users = concatenate(
            train_names, data, features, candidate_name
        )
        nested_score = loso_scores(train_x, gain, users)
        selected = select_threshold(nested_score, gain, disagreement, users)
        thresholds[candidate_name] = selected
        route_score[candidate_name] = fit_score(
            train_x, gain, features[held_name][candidate_name]
        )
        candidate_prediction = held.candidates[candidate_name].argmax(axis=1)
        routes[candidate_name] = (
            candidate_prediction != held.split.safe_prediction
        ) & (route_score[candidate_name] >= float(selected["threshold"]))

    prediction = held.split.safe_prediction.copy()
    candidate_names = list(held.candidates)
    conflicts = 0
    for row in range(len(prediction)):
        active = [name for name in candidate_names if routes[name][row]]
        if not active:
            continue
        if len(active) > 1:
            labels = {int(held.candidates[name][row].argmax()) for name in active}
            conflicts += int(len(labels) > 1)
        winner = max(
            active,
            key=lambda name: route_score[name][row] - float(thresholds[name]["threshold"]),
        )
        prediction[row] = int(held.candidates[winner][row].argmax())

    labels = held.split.labels
    safe = held.split.safe_prediction
    rescue = int(np.sum((prediction == labels) & (safe != labels)))
    harm = int(np.sum((prediction != labels) & (safe == labels)))
    per_user = {}
    for user in sorted(set(held.split.users.astype(str).tolist())):
        selected = held.split.users.astype(str) == user
        per_user[user] = int(
            np.sum(prediction[selected] == labels[selected])
            - np.sum(safe[selected] == labels[selected])
        )
    oracle = np.zeros(len(labels), dtype=bool)
    for values in [safe, *[value.argmax(axis=1) for value in held.candidates.values()]]:
        oracle |= values == labels
    return (
        {
            "held_split": held_name,
            "train_splits": train_names,
            "safe_metrics": classification_metrics(labels, safe),
            "router_metrics": classification_metrics(labels, prediction),
            "oracle_correct": int(oracle.sum()),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "changed": int(np.sum(prediction != safe)),
            "route_conflicts": conflicts,
            "per_user_gain": per_user,
            "minimum_user_gain": int(min(per_user.values())),
            "thresholds": thresholds,
            "route_counts": {name: int(mask.sum()) for name, mask in routes.items()},
        },
        prediction,
        route_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--disable-session-context", action="store_true")
    parser.add_argument("--full-visual-bank", action="store_true")
    parser.add_argument("--structured-bank", action="store_true")
    parser.add_argument("--legacy-visual-bank", action="store_true")
    parser.add_argument("--hand-object-bank", action="store_true")
    parser.add_argument("--vjepa-dense-bank", action="store_true")
    parser.add_argument("--nonvisual-bank", action="store_true")
    parser.add_argument("--embargo-source", action="store_true")
    parser.add_argument("--hierarchical-bank", action="store_true")
    parser.add_argument("--epic-bank", action="store_true")
    args = parser.parse_args()
    data = load_candidate_splits(
        full_visual_bank=args.full_visual_bank,
        structured_bank=args.structured_bank,
        legacy_visual_bank=args.legacy_visual_bank,
        hand_object_bank=args.hand_object_bank,
        vjepa_dense_bank=args.vjepa_dense_bank,
        nonvisual_bank=args.nonvisual_bank,
        embargo_source=args.embargo_source,
        hierarchical_bank=args.hierarchical_bank,
        epic_bank=args.epic_bank,
    )
    features: dict[str, dict[str, np.ndarray]] = {}
    for name, value in data.items():
        shared = shared_features(value)
        features[name] = {
            candidate: candidate_features(
                value,
                shared,
                candidate,
                include_session_context=not args.disable_session_context,
            )
            for candidate in value.candidates
        }

    recipes = {
        "H1_selection": ["H2_confirmation", "H3_independent_fold0"],
        "H2_confirmation": ["H1_selection", "H3_independent_fold0"],
        "H3_independent_fold0": ["H1_selection", "H2_confirmation"],
    }
    reports = {}
    payload: dict[str, np.ndarray] = {}
    total_safe = total_router = total_oracle = rows = 0
    for held_name, train_names in recipes.items():
        report, prediction, scores = evaluate_outer(
            held_name, train_names, data, features
        )
        reports[held_name] = report
        split = data[held_name].split
        total_safe += report["safe_metrics"]["correct"]
        total_router += report["router_metrics"]["correct"]
        total_oracle += report["oracle_correct"]
        rows += len(split.labels)
        payload[f"{held_name}_sample_ids"] = split.sample_ids
        payload[f"{held_name}_labels"] = split.labels
        payload[f"{held_name}_safe_prediction"] = split.safe_prediction
        payload[f"{held_name}_router_prediction"] = prediction
        for candidate, score in scores.items():
            payload[f"{held_name}_{candidate}_route_score"] = score

    report = {
        "stage": "P117_transductive_multicandidate_router_v1",
        "status": "COMPLETE_STRICT_OUTER_CROSSFIT",
        "protocol": {
            "candidates": ["P89 safe", "A18 best+Session", "P90 visual equal blend"],
            "outer_selection": "held cohort labels excluded from model and threshold",
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "user_id_used_as_feature": False,
            "session_context_enabled": not args.disable_session_context,
            "full_visual_bank": args.full_visual_bank,
            "structured_bank": args.structured_bank,
            "legacy_visual_bank": args.legacy_visual_bank,
            "hand_object_bank": args.hand_object_bank,
            "vjepa_dense_bank": args.vjepa_dense_bank,
            "nonvisual_bank": args.nonvisual_bank,
            "embargo_source": args.embargo_source,
            "hierarchical_bank": args.hierarchical_bank,
            "epic_bank": args.epic_bank,
            "unlabeled_batch_features": (
                "session length/position, uniqueness conflicts, within-session class support/rank, "
                "teacher agreement, probabilities, margins, entropy, and sensor quality"
            ),
            "submission_generated": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": rows,
            "safe_correct": total_safe,
            "safe_accuracy": total_safe / rows,
            "router_correct": total_router,
            "router_accuracy": total_router / rows,
            "net": total_router - total_safe,
            "three_teacher_oracle_correct": total_oracle,
            "three_teacher_oracle_accuracy": total_oracle / rows,
            "target_0.91_correct": int(np.ceil(0.91 * rows)),
            "gap_to_0.91_correct": int(np.ceil(0.91 * rows) - total_router),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
