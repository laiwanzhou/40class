"""Frozen P87 session-context capability probe inside each P91 H2 Top-5.

This is a reporting-only audit.  It does not train a router, reranker, pair
specialist, or any other model.  The dynamic candidate set is always the
per-sample P91 Top-5.  P87's saved P85 clip emission and the corresponding
fixed session posterior are restricted to those candidates so that their
difference isolates the contribution of session context.

H2 is evaluated with the configuration selected by P87's nested protocol.
Only H1 plus the embargo users are used to fit the transition table.  H3 users
are explicitly excluded and no H3 predictions are read.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

from audit_p87_sequence_decoder import (
    DEFAULT_TRAIN_METADATA,
    align_metadata,
    build_sessions,
    decode_unique_beam_posterior,
    fit_transition_model,
)
from audit_p91_confidence095_hard_pool import (
    IMU,
    PROJECT,
    SKELETON,
    align,
    load_names,
    load_npz,
    load_p96_pairs,
    pair_id,
    reconstruct_p91_h2,
    write_csv,
)


HERE = Path(__file__).resolve().parent
P85_TEACHER = HERE / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
P96_VISUAL = PROJECT / "runs/p96_vjepa2_dense24_teacher_h1h2_v1/h2_predictions.npz"
OUTPUT = PROJECT / "runs/p91_p87_session_candidate_probe_v1"

THRESHOLD = 0.95
TOP_K = 5
NUM_CLASSES = 40
H1_USERS = ("user6", "user8", "user17", "user23")
EMBARGO_USERS = ("user1", "user2", "user21")
H2_USERS = ("user5", "user7", "user16", "user18", "user19")
H3_USERS = ("user3", "user4", "user9", "user20", "user22", "user24")

# Established P87 values are frozen before this probe and are never searched on
# P91 H2.  Only the decoder implementation is reused; no P87/H3 result file is
# opened to select or adjust them.
SESSION_GAP_SECONDS = 30.0
TRANSITION_WEIGHT = 0.25
TRIGRAM_BACKOFF = 5.0
BEAM_WIDTH = 50

VARIANT_ORDER = (
    "p91_baseline",
    "p87_emission",
    "p87_session",
    "p91_session",
    "visual",
    "visual_session",
    "session_skeleton_imu",
    "all_modal",
)
VARIANT_LABEL = {
    "p91_baseline": "P91 baseline",
    "p87_emission": "P87 clip emission",
    "p87_session": "P87 session posterior",
    "p91_session": "P91 + session (fixed 1/2)",
    "visual": "P96 Visual",
    "visual_session": "Visual + session (fixed 1/2)",
    "session_skeleton_imu": "Session + Skeleton + IMU (fixed 1/3)",
    "all_modal": "Visual + Session + Skeleton + IMU (fixed 1/4)",
}


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = array.sum(axis=1, keepdims=True)
    zero = total[:, 0] <= 1e-15
    result = np.divide(array, np.maximum(total, 1e-15))
    if np.any(zero):
        result[zero] = 1.0 / array.shape[1]
    return result


def restricted_probability(
    full_probability: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    return normalize_rows(np.take_along_axis(full_probability, candidates, axis=1))


def restricted_logits(full_logits: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return softmax(np.take_along_axis(full_logits, candidates, axis=1), axis=1)


def candidate_prediction(candidates: np.ndarray, scores: np.ndarray) -> np.ndarray:
    position = np.argmax(scores, axis=1)
    return np.take_along_axis(candidates, position[:, None], axis=1).reshape(-1)


def candidate_rank(
    candidates: np.ndarray, scores: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranked = np.take_along_axis(candidates, order, axis=1)
    contained = np.any(ranked == labels[:, None], axis=1)
    ranks = np.full(len(labels), -1, dtype=np.int64)
    ranks[contained] = (
        np.argmax(ranked[contained] == labels[contained, None], axis=1) + 1
    )
    return ranks


def direction_text(
    indices: np.ndarray,
    labels: np.ndarray,
    base_prediction: np.ndarray,
    names: dict[int, str],
) -> str:
    counts = Counter((int(labels[row]), int(base_prediction[row])) for row in indices)
    return ";".join(
        f"{truth}:{names[truth]}->{pred}:{names[pred]}={count}"
        for (truth, pred), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    )


def load_frozen_p87_config() -> dict[str, Any]:
    return {
        "source_interface": str(HERE / "audit_p87_sequence_decoder.py"),
        "frozen_before_this_probe": True,
        "gap_seconds": SESSION_GAP_SECONDS,
        "transition_weight": TRANSITION_WEIGHT,
        "trigram_backoff": TRIGRAM_BACKOFF,
        "beam_width": BEAM_WIDTH,
    }


def construct_p87_session_probability(
    target_sample_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    teacher = load_npz(P85_TEACHER)
    teacher_ids = teacher["oof_sample_ids"].astype(str)
    teacher_labels = teacher["oof_labels"].astype(np.int64)
    emission_probability = teacher["oof_teacher_probability"].astype(np.float64)
    emission_log_probability = teacher["oof_teacher_log_probability"].astype(
        np.float64
    )
    metadata = align_metadata(DEFAULT_TRAIN_METADATA, teacher_ids)

    h3_mask = np.isin(metadata.users, H3_USERS)
    source_mask = np.isin(metadata.users, H1_USERS + EMBARGO_USERS)
    target_lookup = {sample_id: row for row, sample_id in enumerate(teacher_ids)}
    try:
        target_indices = np.asarray(
            [target_lookup[str(sample_id)] for sample_id in target_sample_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError(f"P85 teacher lacks P91 sample {error.args[0]}") from error
    if np.any(h3_mask[target_indices]) or np.any(h3_mask & source_mask):
        raise RuntimeError("H3 user leaked into P87 session construction")
    if set(metadata.users[target_indices].tolist()) != set(H2_USERS):
        raise RuntimeError("P91 target ids do not match the frozen H2 user set")

    source_sessions = build_sessions(
        np.flatnonzero(source_mask),
        metadata,
        gap_seconds=SESSION_GAP_SECONDS,
        grouping="known_user",
    )
    transition_model = fit_transition_model(
        teacher_labels,
        source_sessions,
        num_classes=NUM_CLASSES,
        trigram_backoff=TRIGRAM_BACKOFF,
    )
    target_sessions = build_sessions(
        target_indices,
        metadata,
        gap_seconds=SESSION_GAP_SECONDS,
        grouping="anonymous_date",
    )

    # Missing metadata rows keep the clip emission.  This is the same fallback
    # used by the original P87 decoder outside constructed sessions.
    posterior = emission_probability.copy()
    session_index = np.full(len(teacher_ids), -1, dtype=np.int64)
    session_position = np.full(len(teacher_ids), -1, dtype=np.int64)
    session_length = np.zeros(len(teacher_ids), dtype=np.int64)
    for sequence_id, session in enumerate(target_sessions):
        result = decode_unique_beam_posterior(
            emission_log_probability[session],
            transition_model,
            transition_weight=TRANSITION_WEIGHT,
            beam_width=BEAM_WIDTH,
        )
        posterior[session] = result.marginals
        session_index[session] = sequence_id
        session_position[session] = np.arange(len(session), dtype=np.int64)
        session_length[session] = len(session)

    available = session_index[target_indices] >= 0
    source_users_observed = sorted(set(metadata.users[source_mask].tolist()))
    audit = {
        "teacher_rows": int(len(teacher_ids)),
        "source_users": source_users_observed,
        "source_rows": int(source_mask.sum()),
        "source_sessions": int(len(source_sessions)),
        "target_users": sorted(set(metadata.users[target_indices].tolist())),
        "target_rows": int(len(target_indices)),
        "target_sessions": int(len(target_sessions)),
        "session_available_rows": int(available.sum()),
        "emission_fallback_rows": int((~available).sum()),
        "h3_users_excluded": list(H3_USERS),
        "h3_rows_used": 0,
    }
    return (
        emission_probability[target_indices],
        posterior[target_indices],
        np.column_stack(
            (
                session_index[target_indices],
                session_position[target_indices],
                session_length[target_indices],
            )
        ),
        audit,
    )


def variant_summary(
    key: str,
    prediction: np.ndarray,
    labels: np.ndarray,
    primary_target: np.ndarray,
    candidate_eligible: np.ndarray,
    protected_correct: np.ndarray,
    base_correct: int,
) -> dict[str, Any]:
    correct = prediction == labels
    rescue = int(np.sum(primary_target & correct))
    harm = int(np.sum(protected_correct & ~correct))
    eligible_correct = int(np.sum(candidate_eligible & correct))
    total = len(labels)
    return {
        "variant": key,
        "primary_target_samples": int(primary_target.sum()),
        "primary_target_correct": rescue,
        "oracle_candidate_accuracy": safe_rate(rescue, int(primary_target.sum())),
        "excess_over_uniform_random_pp": float(
            100.0 * (rescue / int(primary_target.sum()) - 1.0 / TOP_K)
        ),
        "rescue": rescue,
        "candidate_eligible_samples": int(candidate_eligible.sum()),
        "candidate_eligible_correct": eligible_correct,
        "candidate_eligible_accuracy": safe_rate(
            eligible_correct, int(candidate_eligible.sum())
        ),
        "protected_correct_samples": int(protected_correct.sum()),
        "harm": harm,
        "harm_rate": safe_rate(harm, int(protected_correct.sum())),
        "net_change_if_all_low_are_reranked": rescue - harm,
        "oracle_error_gate_h2_correct": base_correct + rescue,
        "oracle_error_gate_h2_accuracy": float((base_correct + rescue) / total),
        "oracle_error_gate_uplift_pp": float(100.0 * rescue / total),
        "all_low_gate_h2_correct": base_correct + rescue - harm,
        "all_low_gate_h2_accuracy": float((base_correct + rescue - harm) / total),
        "all_low_gate_change_pp": float(100.0 * (rescue - harm) / total),
    }


def subset_pair_summary(
    mask: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    count = int(mask.sum())
    result: dict[str, Any] = {
        "target_errors": count,
        "users": int(len(set(users[mask].tolist()))) if count else 0,
        "user_ids": sorted(set(users[mask].tolist())) if count else [],
    }
    for key in VARIANT_ORDER:
        rescue = int(np.sum(mask & (predictions[key] == labels)))
        result[f"{key}_rescue"] = rescue
        result[f"{key}_accuracy"] = safe_rate(rescue, count)
    return result


def main() -> None:
    names = load_names()
    p96_pairs = load_p96_pairs()
    frozen_config = load_frozen_p87_config()
    p91 = reconstruct_p91_h2()
    sample_ids = np.asarray(p91["sample_ids"]).astype(str)
    labels = np.asarray(p91["labels"]).astype(np.int64)
    users = np.asarray(p91["users"]).astype(str)
    p91_probability = np.asarray(p91["probability"]).astype(np.float64)
    p91_prediction = np.asarray(p91["prediction"]).astype(np.int64)
    confidence = np.asarray(p91["confidence"]).astype(np.float64)

    candidates = np.argsort(-p91_probability, axis=1)[:, :TOP_K]
    p91_candidate = restricted_probability(p91_probability, candidates)
    p91_true_rank = (
        np.argmax(np.argsort(-p91_probability, axis=1) == labels[:, None], axis=1)
        + 1
    )
    true_in_candidate = np.any(candidates == labels[:, None], axis=1)
    low = confidence < THRESHOLD
    p91_correct = p91_prediction == labels
    low_error = low & ~p91_correct
    primary_target = low_error & true_in_candidate
    candidate_eligible = low & true_in_candidate
    protected_correct = low & p91_correct
    unreachable_error = low_error & ~true_in_candidate

    emission_full, session_full, session_metadata, session_source_audit = (
        construct_p87_session_probability(sample_ids)
    )
    p87_emission = restricted_probability(emission_full, candidates)
    p87_session = restricted_probability(session_full, candidates)

    visual = load_npz(P96_VISUAL)
    visual_logits = align(
        visual["sample_ids"].astype(str),
        visual["direct_logits"].astype(np.float64),
        sample_ids,
    )
    visual_labels = align(
        visual["sample_ids"].astype(str),
        visual["labels"].astype(np.int64),
        sample_ids,
    )
    visual_candidate = restricted_logits(visual_logits, candidates)

    skeleton = load_npz(SKELETON)
    skeleton_logits = align(
        skeleton["sample_ids"].astype(str),
        skeleton["skeleton_logits"].astype(np.float64),
        sample_ids,
    )
    skeleton_labels = align(
        skeleton["sample_ids"].astype(str),
        skeleton["labels"].astype(np.int64),
        sample_ids,
    )
    skeleton_candidate = restricted_logits(skeleton_logits, candidates)

    imu = load_npz(IMU)
    imu_probability = align(
        imu["sample_ids"].astype(str),
        imu["probabilities"].astype(np.float64),
        sample_ids,
    )
    imu_labels = align(
        imu["sample_ids"].astype(str),
        imu["labels"].astype(np.int64),
        sample_ids,
    )
    imu_candidate = restricted_probability(imu_probability, candidates)

    if not np.array_equal(visual_labels, labels):
        raise RuntimeError("aligned P96 Visual labels differ from P91 H2 labels")
    if not np.array_equal(skeleton_labels, labels):
        raise RuntimeError("aligned Skeleton labels differ from P91 H2 labels")
    if not np.array_equal(imu_labels, labels):
        raise RuntimeError("aligned IMU labels differ from P91 H2 labels")

    score_by_variant = {
        "p91_baseline": p91_candidate,
        "p87_emission": p87_emission,
        "p87_session": p87_session,
        "p91_session": 0.5 * p91_candidate + 0.5 * p87_session,
        "visual": visual_candidate,
        "visual_session": 0.5 * visual_candidate + 0.5 * p87_session,
        "session_skeleton_imu": (
            p87_session + skeleton_candidate + imu_candidate
        )
        / 3.0,
        "all_modal": (
            visual_candidate + p87_session + skeleton_candidate + imu_candidate
        )
        / 4.0,
    }
    prediction_by_variant = {
        key: candidate_prediction(candidates, score_by_variant[key])
        for key in VARIANT_ORDER
    }
    rank_by_variant = {
        key: candidate_rank(candidates, score_by_variant[key], labels)
        for key in VARIANT_ORDER
    }

    base_correct = int(p91_correct.sum())
    metrics_by_variant = {
        key: variant_summary(
            key,
            prediction_by_variant[key],
            labels,
            primary_target,
            candidate_eligible,
            protected_correct,
            base_correct,
        )
        for key in VARIANT_ORDER
    }
    session_available = session_metadata[:, 0] >= 0
    session_source_audit.update(
        {
            "primary_target_session_available": int(
                np.sum(primary_target & session_available)
            ),
            "primary_target_emission_fallback": int(
                np.sum(primary_target & ~session_available)
            ),
            "protected_session_available": int(
                np.sum(protected_correct & session_available)
            ),
            "protected_emission_fallback": int(
                np.sum(protected_correct & ~session_available)
            ),
        }
    )

    sample_rows: list[dict[str, Any]] = []
    for row in np.flatnonzero(low):
        sequence_id, sequence_position, sequence_length = session_metadata[row]
        item: dict[str, Any] = {
            "sample_id": sample_ids[row],
            "user": users[row],
            "true_class_id": int(labels[row]),
            "true_class_name": names[int(labels[row])],
            "p91_prediction_id": int(p91_prediction[row]),
            "p91_prediction_name": names[int(p91_prediction[row])],
            "p91_confidence": float(confidence[row]),
            "p91_correct": int(p91_correct[row]),
            "p91_true_rank": int(p91_true_rank[row]),
            "truth_in_candidate": int(true_in_candidate[row]),
            "primary_error_target": int(primary_target[row]),
            "protected_correct": int(protected_correct[row]),
            "unreachable_error": int(unreachable_error[row]),
            "session_available": int(sequence_id >= 0),
            "session_id": int(sequence_id) if sequence_id >= 0 else None,
            "session_position": (
                int(sequence_position) if sequence_position >= 0 else None
            ),
            "session_length": int(sequence_length) if sequence_length > 0 else None,
        }
        for position in range(TOP_K):
            class_id = int(candidates[row, position])
            item[f"candidate_{position + 1}_id"] = class_id
            item[f"candidate_{position + 1}_name"] = names[class_id]
            for key in VARIANT_ORDER:
                item[f"candidate_{position + 1}_{key}_score"] = float(
                    score_by_variant[key][row, position]
                )
        for key in VARIANT_ORDER:
            prediction = int(prediction_by_variant[key][row])
            is_correct = prediction == int(labels[row])
            item[f"{key}_prediction_id"] = prediction
            item[f"{key}_prediction_name"] = names[prediction]
            item[f"{key}_true_rank_in_candidate"] = (
                int(rank_by_variant[key][row]) if true_in_candidate[row] else None
            )
            item[f"{key}_correct"] = int(is_correct)
            item[f"{key}_rescue"] = int(primary_target[row] and is_correct)
            item[f"{key}_harm"] = int(protected_correct[row] and not is_correct)
        sample_rows.append(item)

    user_rows: list[dict[str, Any]] = []
    for user in sorted(np.unique(users)):
        selected = users == user
        item: dict[str, Any] = {
            "user": user,
            "low_samples": int(np.sum(selected & low)),
            "primary_target_samples": int(np.sum(selected & primary_target)),
            "candidate_eligible_samples": int(np.sum(selected & candidate_eligible)),
            "protected_correct_samples": int(np.sum(selected & protected_correct)),
            "unreachable_errors": int(np.sum(selected & unreachable_error)),
            "session_available_low": int(
                np.sum(selected & low & (session_metadata[:, 0] >= 0))
            ),
        }
        for key in VARIANT_ORDER:
            correct = prediction_by_variant[key] == labels
            rescue = int(np.sum(selected & primary_target & correct))
            harm = int(np.sum(selected & protected_correct & ~correct))
            eligible_correct = int(np.sum(selected & candidate_eligible & correct))
            item[f"{key}_rescue"] = rescue
            item[f"{key}_target_accuracy"] = safe_rate(
                rescue, int(item["primary_target_samples"])
            )
            item[f"{key}_harm"] = harm
            item[f"{key}_harm_rate"] = safe_rate(
                harm, int(item["protected_correct_samples"])
            )
            item[f"{key}_candidate_accuracy"] = safe_rate(
                eligible_correct, int(item["candidate_eligible_samples"])
            )
            item[f"{key}_net_if_all_low"] = rescue - harm
        user_rows.append(item)

    grouped_pairs: dict[str, list[int]] = defaultdict(list)
    for row in np.flatnonzero(primary_target):
        grouped_pairs[pair_id(int(labels[row]), int(p91_prediction[row]))].append(int(row))
    pair_rows: list[dict[str, Any]] = []
    for key, indices in grouped_pairs.items():
        rows = np.asarray(indices, dtype=np.int64)
        class_a, class_b = (int(value) for value in key.split("_")[1:])
        item: dict[str, Any] = {
            "pair_id": key,
            "class_a_id": class_a,
            "class_a_name": names[class_a],
            "class_b_id": class_b,
            "class_b_name": names[class_b],
            "target_errors": len(rows),
            "users": len(set(users[rows].tolist())),
            "user_ids": ";".join(sorted(set(users[rows].tolist()))),
            "directions": direction_text(rows, labels, p91_prediction, names),
            "truth_rank_2": int(np.sum(p91_true_rank[rows] == 2)),
            "truth_rank_3": int(np.sum(p91_true_rank[rows] == 3)),
            "truth_rank_4": int(np.sum(p91_true_rank[rows] == 4)),
            "truth_rank_5": int(np.sum(p91_true_rank[rows] == 5)),
        }
        true_position = np.argmax(candidates[rows] == labels[rows, None], axis=1)
        base_position = np.argmax(
            candidates[rows] == p91_prediction[rows, None], axis=1
        )
        for variant in VARIANT_ORDER:
            scores = score_by_variant[variant]
            prediction = prediction_by_variant[variant]
            rescue = int(np.sum(prediction[rows] == labels[rows]))
            truth_preferred = int(
                np.sum(scores[rows, true_position] > scores[rows, base_position])
            )
            item[f"{variant}_rescue"] = rescue
            item[f"{variant}_candidate_accuracy"] = safe_rate(rescue, len(rows))
            item[f"{variant}_truth_over_p91_top1"] = truth_preferred
            item[f"{variant}_truth_over_p91_top1_rate"] = safe_rate(
                truth_preferred, len(rows)
            )
        p96_pair = p96_pairs.get(key)
        item["p96_source_samples"] = (
            int(p96_pair["source_samples"]) if p96_pair else 0
        )
        item["p96_source_users"] = int(p96_pair["source_users"]) if p96_pair else 0
        item["p96_pool_tier"] = (
            p96_pair["pool_tier"] if p96_pair else "unseen_in_p96_source"
        )
        pair_rows.append(item)
    pair_rows.sort(key=lambda row: (-row["target_errors"], -row["users"], row["pair_id"]))

    emission_rescue = primary_target & (
        prediction_by_variant["p87_emission"] == labels
    )
    session_rescue = primary_target & (
        prediction_by_variant["p87_session"] == labels
    )
    emission_harm = protected_correct & (
        prediction_by_variant["p87_emission"] != labels
    )
    session_harm = protected_correct & (
        prediction_by_variant["p87_session"] != labels
    )
    context_increment = {
        "emission_rescues": int(emission_rescue.sum()),
        "session_rescues": int(session_rescue.sum()),
        "shared_rescues": int(np.sum(emission_rescue & session_rescue)),
        "session_unique_rescues": int(np.sum(session_rescue & ~emission_rescue)),
        "emission_rescues_lost_after_session": int(
            np.sum(emission_rescue & ~session_rescue)
        ),
        "rescue_delta": int(session_rescue.sum() - emission_rescue.sum()),
        "emission_harms": int(emission_harm.sum()),
        "session_harms": int(session_harm.sum()),
        "harm_delta": int(session_harm.sum() - emission_harm.sum()),
    }

    p91_session_user_net = {
        str(row["user"]): int(row["p91_session_net_if_all_low"])
        for row in user_rows
    }
    positive_user_nets = [value for value in p91_session_user_net.values() if value > 0]
    negative_user_nets = [value for value in p91_session_user_net.values() if value < 0]
    stability = {
        "p91_session_user_net": p91_session_user_net,
        "users_with_positive_net": len(positive_user_nets),
        "users_with_zero_net": sum(value == 0 for value in p91_session_user_net.values()),
        "users_with_negative_net": len(negative_user_nets),
        "largest_user_rescue_share": float(
            max(int(row["p91_session_rescue"]) for row in user_rows)
            / max(metrics_by_variant["p91_session"]["rescue"], 1)
        ),
    }

    exact_focus = {
        "Read_documents__Turn_pages": (21, 22),
        "Use_a_mobile_phone__Play_games": (24, 26),
        "Drink_water__Take_medicine": (6, 37),
    }
    special_pairs: dict[str, Any] = {}
    for title, (left, right) in exact_focus.items():
        mask = primary_target & (
            np.minimum(labels, p91_prediction) == min(left, right)
        ) & (np.maximum(labels, p91_prediction) == max(left, right))
        special_pairs[title] = subset_pair_summary(
            mask, users, labels, prediction_by_variant
        )
    tableware_ids = np.asarray([4, 8, 14, 15], dtype=np.int64)
    tableware_mask = primary_target & (
        np.isin(labels, tableware_ids) | np.isin(p91_prediction, tableware_ids)
    )
    special_pairs["tableware_related"] = subset_pair_summary(
        tableware_mask, users, labels, prediction_by_variant
    )

    perfect_candidate_rescue = int(primary_target.sum())
    perfect_candidate_correct = base_correct + perfect_candidate_rescue
    summary = {
        "experiment_id": "p91_p87_session_candidate_probe_v1",
        "status": "complete_frozen_oracle_candidate_conditioned_h2_probe",
        "protocol": {
            "evaluated_split": "P91 frozen H2 subject-disjoint confirmation",
            "threshold": THRESHOLD,
            "candidate_source": "per-sample P91 Top-5 probability ranking",
            "candidate_set_is_dynamic_per_sample": True,
            "candidate_size": TOP_K,
            "primary_target": (
                "P91 error with confidence < 0.95 and truth inside its Top-5"
            ),
            "p87_representation_interface": (
                "P85 clip emission plus P87 anonymous timestamp-session transition/no-repeat beam posterior marginals"
            ),
            "p87_emission_source": str(P85_TEACHER),
            "p87_decoder_interface": str(HERE / "audit_p87_sequence_decoder.py"),
            "p87_config": frozen_config,
            "transition_fit_users": list(H1_USERS + EMBARGO_USERS),
            "h2_users": list(H2_USERS),
            "h3_users_excluded": list(H3_USERS),
            "score_combinations": "fixed equal averages; no weights selected on H2",
            "training_performed": False,
            "router_or_specialist_created": False,
            "p91_pipeline_modified": False,
            "h3_read": False,
            "H1_limitation": (
                "Comparable P91 champion H1 probabilities are not saved, so no learned candidate probe was fitted; H2 is a parameter-free frozen capability audit."
            ),
        },
        "session_source_audit": session_source_audit,
        "sets": {
            "h2_samples": len(labels),
            "p91_h2_correct": base_correct,
            "p91_h2_accuracy": float(base_correct / len(labels)),
            "low_confidence_samples": int(low.sum()),
            "low_confidence_correct_protected": int(protected_correct.sum()),
            "low_confidence_errors": int(low_error.sum()),
            "primary_target_errors_truth_in_top5": int(primary_target.sum()),
            "unreachable_errors_truth_not_in_top5": int(unreachable_error.sum()),
            "all_low_truth_in_top5": int(candidate_eligible.sum()),
        },
        "baselines_and_ceiling": {
            "uniform_random_candidate_accuracy": 1.0 / TOP_K,
            "p91_primary_target_accuracy_by_definition": 0.0,
            "p91_accuracy_on_all_candidate_eligible_low_samples": safe_rate(
                int(np.sum(candidate_eligible & p91_correct)),
                int(candidate_eligible.sum()),
            ),
            "perfect_candidate_selector_rescue": perfect_candidate_rescue,
            "perfect_candidate_selector_h2_correct": perfect_candidate_correct,
            "perfect_candidate_selector_h2_accuracy": float(
                perfect_candidate_correct / len(labels)
            ),
            "perfect_candidate_selector_uplift_pp": float(
                100.0 * perfect_candidate_rescue / len(labels)
            ),
        },
        "variants": metrics_by_variant,
        "session_context_increment_over_its_clip_emission": context_increment,
        "cross_user_stability": stability,
        "special_focus": special_pairs,
        "pair_summary": {
            "unique_target_pairs": len(pair_rows),
            "pairs_with_at_least_2_target_errors": sum(
                int(row["target_errors"]) >= 2 for row in pair_rows
            ),
            "cross_user_pairs": sum(int(row["users"]) >= 2 for row in pair_rows),
        },
        "decision": {
            "session_adds_incremental_context_signal": True,
            "session_signal_is_broad_and_cross_user_stable": False,
            "ready_for_pipeline_integration": False,
            "reason": (
                "P87 session posterior beats its own clip emission and uniform random only slightly, while fixed P91+session gains are concentrated in one user and are negative for two users."
            ),
        },
    }

    if len(labels) != 834 or base_correct != 750:
        raise RuntimeError("P91 H2 identity check failed")
    if int(low.sum()) != 351 or int(low_error.sum()) != 79:
        raise RuntimeError("P91 low-confidence set no longer matches fixed audit")
    if int(primary_target.sum()) != 63 or int(unreachable_error.sum()) != 16:
        raise RuntimeError("unexpected P91 Top-5 target partition")
    if int(candidate_eligible.sum()) != 335 or int(protected_correct.sum()) != 272:
        raise RuntimeError("unexpected candidate-eligible/protected partition")
    if not np.all(candidates[:, 0] == p91_prediction):
        raise RuntimeError("P91 Top-1 differs from first candidate")
    if not np.all(np.asarray([len(np.unique(row)) for row in candidates]) == TOP_K):
        raise RuntimeError("duplicate class found in P91 Top-5")
    if session_source_audit["source_sessions"] != 284:
        raise RuntimeError("unexpected P87 source session count")
    if session_source_audit["target_sessions"] != 221:
        raise RuntimeError("unexpected P87 H2 session count")
    if session_source_audit["session_available_rows"] != 808:
        raise RuntimeError("unexpected P87 H2 session coverage")
    for key, scores in score_by_variant.items():
        if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-10):
            raise RuntimeError(f"{key} candidate scores do not sum to one")
    if sum(int(row["target_errors"]) for row in pair_rows) != int(primary_target.sum()):
        raise RuntimeError("pair table does not cover every primary target")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "candidate_session_samples_h2.csv", sample_rows)
    write_csv(OUTPUT / "candidate_session_users_h2.csv", user_rows)
    write_csv(OUTPUT / "candidate_session_pairs_h2.csv", pair_rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    frequent_pairs = [row for row in pair_rows if int(row["target_errors"]) >= 2]
    lines = [
        "# P91 Top-5 内的 P87 session context 能力审计",
        "",
        "## 结论",
        "",
        (
            "P87 session context **包含少量真实的增量信息，但不足以成为稳定的共享重排依据**。"
            f"在 63 个主要目标上，P87 clip emission 只选对 {context_increment['emission_rescues']} 个"
            f"（{fmt_pct(metrics_by_variant['p87_emission']['oracle_candidate_accuracy'])}），"
            f"加入固定 session 结构后选对 {context_increment['session_rescues']} 个"
            f"（{fmt_pct(metrics_by_variant['p87_session']['oracle_candidate_accuracy'])}）："
            f"净增加 {context_increment['rescue_delta']} 个 rescue，同时 protected harm 从 "
            f"{context_increment['emission_harms']} 降至 {context_increment['session_harms']}。"
        ),
        (
            f"但 session-only 只比均匀随机 5 选 1 的 20.00% 高 "
            f"{metrics_by_variant['p87_session']['excess_over_uniform_random_pp']:.2f}pp。"
            f"固定 P91+session 只救回 {metrics_by_variant['p91_session']['rescue']}/63，"
            f"并伤害 {metrics_by_variant['p91_session']['harm']}/272 个原本正确的低置信样本；"
            f"若无 oracle 地覆盖全部低置信样本，净变化为 "
            f"{metrics_by_variant['p91_session']['net_change_if_all_low_are_reranked']:+d} 个。"
            "这点小幅正净值主要由 user7 提供，不能视作跨用户稳定确认。"
        ),
        (
            f"作为能力参照，冻结 P96 Visual 在主要目标上可选对 "
            f"{metrics_by_variant['visual']['rescue']}/63"
            f"（{fmt_pct(metrics_by_variant['visual']['oracle_candidate_accuracy'])}），"
            f"但盲目作用于全部低置信样本会造成 {metrics_by_variant['visual']['harm']} 个 harm。"
            "这说明动态 Top-5 并非本身不可分，当前更强的 clip-level visual representation "
            "比 P87 session 上下文携带更多局部判别信息；它仍只是 oracle 能力证据，不是安全集成方案。"
        ),
        "",
        "## 协议与边界",
        "",
        "- 主要目标：P91 `confidence < 0.95`、当前 Top-1 错误、真实类位于该样本动态 Top-5 的 63 个样本。",
        "- P87 并没有单独保存普通 session embedding；其可用接口是 P85 clip emission 经时间 session、转移表、no-repeat beam 后的 40 维 posterior marginal。",
        "- 用同一 P85 emission 与 P87 session posterior 对比，二者都只在 P91 的 Top-5 内归一化，差值用于隔离 session context。没有评估 40 分类准确率。",
        f"- 固定使用进入本审计前已经确定的 P87 配置：gap={SESSION_GAP_SECONDS:g}s、transition weight={TRANSITION_WEIGHT:g}、trigram backoff={TRIGRAM_BACKOFF:g}、beam={BEAM_WIDTH}；脚本不读取 P87/H3 结果来重选配置。",
        "- 转移表只用 H1 用户 user6/user8/user17/user23 与 embargo 用户 user1/user2/user21 的标签；H2 不选权重，H3 六个用户明确排除且没有读取 H3 预测。",
        "- 所有跨模态组合均为预先固定等权平均；没有训练 probe、router、specialist 或最终模型，也没有修改 P91。",
        "- 26/834 个样本缺少可构造的时间 session，按 P87 原接口回退到 P85 clip emission；主要目标中有 6 个这种样本。",
        "",
        "## 测试集合",
        "",
        "| 集合 | 样本数 |",
        "|---|---:|",
        f"| P91 H2 | {len(labels)} |",
        f"| confidence < 0.95 | {int(low.sum())} |",
        f"| 低置信错误 | {int(low_error.sum())} |",
        f"| 主要目标：错误且 truth in Top-5 | {int(primary_target.sum())} |",
        f"| 错误但 truth not in Top-5 | {int(unreachable_error.sum())} |",
        f"| protected：低置信但 P91 正确 | {int(protected_correct.sum())} |",
        "",
        "## Candidate-conditioned 结果",
        "",
        "主要 accuracy 只在 63 个当前错误、truth in Top-5 样本上计算。harm 是 272 个低置信但 P91 原本正确的样本中被改错的数量。",
        "",
        "| 变体 | 主要目标 accuracy | rescue | 相对随机 | protected harm | 全低置信净变化 | 全低置信后 H2 accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in VARIANT_ORDER:
        metrics = metrics_by_variant[key]
        lines.append(
            f"| {VARIANT_LABEL[key]} | {fmt_pct(metrics['oracle_candidate_accuracy'])} | "
            f"{metrics['rescue']}/{metrics['primary_target_samples']} | "
            f"{metrics['excess_over_uniform_random_pp']:+.2f}pp | "
            f"{metrics['harm']}/{metrics['protected_correct_samples']} "
            f"({fmt_pct(metrics['harm_rate'])}) | "
            f"{metrics['net_change_if_all_low_are_reranked']:+d} | "
            f"{fmt_pct(metrics['all_low_gate_h2_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            (
                "跨模态等权组合没有形成可信协同：Session+Skeleton+IMU 相比 session-only "
                f"只多救回 {metrics_by_variant['session_skeleton_imu']['rescue'] - metrics_by_variant['p87_session']['rescue']} 个，"
                f"却多出 {metrics_by_variant['session_skeleton_imu']['harm'] - metrics_by_variant['p87_session']['harm']} 个 harm；"
                "全模态相比 session-only 只多 2 个 rescue、增加 11 个 harm。"
                f"Visual+session 反而从 Visual 的 {metrics_by_variant['visual']['rescue']} 个 rescue "
                f"降到 {metrics_by_variant['visual_session']['rescue']} 个。"
            ),
            "",
            (
                f"完美 Top-5 selector 的理论上限是救回全部 {perfect_candidate_rescue} 个主要目标，"
                f"把 P91 H2 从 {fmt_pct(base_correct / len(labels))} 提升到 "
                f"{fmt_pct(perfect_candidate_correct / len(labels))}"
                f"（+{100.0 * perfect_candidate_rescue / len(labels):.2f}pp）。"
                "表中的 oracle error gate 只用于能力上限解释，不能用于部署。"
            ),
            "",
            "## 用户稳定性",
            "",
            "| 用户 | 目标错误 | session rescue | P91+session rescue | P91+session harm | P91+session 净变化 | Visual rescue |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in user_rows:
        lines.append(
            f"| {row['user']} | {row['primary_target_samples']} | "
            f"{row['p87_session_rescue']} | {row['p91_session_rescue']} | "
            f"{row['p91_session_harm']} | {row['p91_session_net_if_all_low']:+d} | "
            f"{row['visual_rescue']} |"
        )
    lines.extend(
        [
            "",
            (
                "P91+session 的逐用户净变化为："
                + "、".join(
                    f"{user} {value:+d}"
                    for user, value in p91_session_user_net.items()
                )
                + "。12 个 rescue 中有 7 个来自 user7；两个用户出现负净值，因此当前收益不满足跨用户稳定。"
            ),
            "",
            "## 高频 confusion pair",
            "",
            "下表仍使用共享 scorer，只做分层统计；没有为任何 pair 建模。",
            "",
            "| pair | 目标数 | 用户数 | emission | session | P91+session | Visual | Session+Skel+IMU | All |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frequent_pairs:
        lines.append(
            f"| {row['class_a_name']} ↔ {row['class_b_name']} | "
            f"{row['target_errors']} | {row['users']} | "
            f"{row['p87_emission_rescue']} | {row['p87_session_rescue']} | "
            f"{row['p91_session_rescue']} | {row['visual_rescue']} | "
            f"{row['session_skeleton_imu_rescue']} | {row['all_modal_rescue']} |"
        )
    phone_headphones = next(
        (row for row in pair_rows if row["pair_id"] == pair_id(19, 23)), None
    )
    sweep_mop = next(
        (row for row in pair_rows if row["pair_id"] == pair_id(12, 13)), None
    )
    lines.extend(
        [
            "",
            "重点关系：",
            "",
            "- `Read_documents ↔ Turn_pages`：主要 63 样本中没有出现，当前 H2 子集不能估计 session 能力。",
            f"- `Use_a_mobile_phone ↔ Play_games`：session 0/{special_pairs['Use_a_mobile_phone__Play_games']['target_errors']}，没有支持证据。",
            f"- `Drink_water ↔ Take_medicine`：session 0/{special_pairs['Drink_water__Take_medicine']['target_errors']}，Visual 4/{special_pairs['Drink_water__Take_medicine']['target_errors']}；缺失信息更像 clip-level visual，而非 session。",
            f"- Tableware 相关：session {special_pairs['tableware_related']['p87_session_rescue']}/{special_pairs['tableware_related']['target_errors']}，来自 {special_pairs['tableware_related']['users']} 个用户，未见稳定优势。",
            (
                f"- session 最清楚的局部增量是 `Make_a_phone_call ↔ Listen_to_music_with_headphones`："
                f"emission {phone_headphones['p87_emission_rescue']}/{phone_headphones['target_errors']}，"
                f"session {phone_headphones['p87_session_rescue']}/{phone_headphones['target_errors']}，"
                f"覆盖 {phone_headphones['users']} 个用户。"
                if phone_headphones
                else "- Phone-call/headphones 在本子集中未出现。"
            ),
            (
                f"- `Sweep_the_floor ↔ Mop_the_floor` 由 session 救回 "
                f"{sweep_mop['p87_session_rescue']}/{sweep_mop['target_errors']}，覆盖 "
                f"{sweep_mop['users']} 个用户，但样本量只有 {sweep_mop['target_errors']}。"
                if sweep_mop
                else "- Sweep/Mop 在本子集中未出现。"
            ),
            "",
            "## 决策",
            "",
            "当前回答是：**session 级上下文确实补充了少量 clip 缺失信息，但强度与跨用户稳定性都不够，暂不值得集成进主 pipeline。**",
            "",
            "下一步不应直接设计 router。若继续该方向，应先在 H1 内构造可训练/可交叉验证的共享 context representation，并冻结后只在 H2 做一次确认；同时优先增强 teacher/visual representation，因为 Visual 的 oracle candidate 能力明显更高。只有共享 session probe 在 H1 多用户上稳定超过 emission 与随机、且 protected harm 可控时，才值得讨论最终 candidate reranker。",
            "",
            "## 产物",
            "",
            "- `summary.json`：协议、总表、增量与决策。",
            "- `candidate_session_samples_h2.csv`：351 个低置信样本的动态候选、各源分数、session 元数据、rescue/harm。",
            "- `candidate_session_users_h2.csv`：逐用户结果。",
            "- `candidate_session_pairs_h2.csv`：逐 confusion pair 结果。",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
