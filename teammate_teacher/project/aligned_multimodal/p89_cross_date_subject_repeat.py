from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    build_sessions,
    classification_metrics,
    decode_unique_beam,
)
from p88_aligned_repeat_holdout import align_probabilities
from p88_train_depth_residual import rescue_harm
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_probability_blend import evaluate as imu_evaluate
from p89_imu_rescue_gate import aligned_imu


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_cross_date_subject_repeat_v1"


@dataclass(frozen=True)
class CrossDateConfig:
    minimum_probability_similarity: float
    minimum_path_overlap: float
    minimum_length_ratio: float
    evidence_weight: float


def prepare(protocol_value):
    imu = np.load(
        PROJECT_DIR
        / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    )
    imu_probability, present = aligned_imu(
        imu["sample_ids"].astype(str),
        np.asarray(imu["imu_logits"], dtype=np.float64),
        protocol_value[0],
        3.0,
    )
    probability = protocol_value[2].copy()
    probability[present] = (
        0.95 * probability[present] + 0.05 * imu_probability[present]
    )
    probability /= probability.sum(axis=1, keepdims=True)
    grouping = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping_config = GlobalRepeatConfig(
        **grouping["H1_selected"]["configuration"]
    )
    safe = imu_evaluate(
        protocol_value,
        imu_probability,
        present,
        0.05,
        "joint",
        grouping_config,
    )[1]
    return probability, safe


def pair_cache(protocol_value, probability: np.ndarray, prediction: np.ndarray):
    metadata = protocol_value[4]
    sessions = build_sessions(
        protocol_value[5], metadata, protocol_value[8].gap_seconds, "known_user"
    )
    by_user: dict[str, list[np.ndarray]] = {}
    for session in sessions:
        by_user.setdefault(str(metadata.users[int(session[0])]), []).append(session)
    pairs = []
    for user, user_sessions in by_user.items():
        for first, second in itertools.combinations(range(len(user_sessions)), 2):
            left, right = user_sessions[first], user_sessions[second]
            left_date = str(metadata.dates[int(left[0])])
            right_date = str(metadata.dates[int(right[0])])
            if left_date == right_date:
                continue
            length_ratio = min(len(left), len(right)) / max(len(left), len(right))
            length_difference = abs(len(left) - len(right))
            aligned, similarity = align_probabilities(
                probability[left], probability[right], 0.2
            )
            left_set = set(map(int, prediction[left]))
            right_set = set(map(int, prediction[right]))
            overlap = len(left_set & right_set) / max(len(left_set | right_set), 1)
            pairs.append(
                {
                    "user": user,
                    "first": first,
                    "second": second,
                    "left": left,
                    "right": right,
                    "left_date": left_date,
                    "right_date": right_date,
                    "length_ratio": float(length_ratio),
                    "length_difference": int(length_difference),
                    "similarity": float(similarity),
                    "overlap": float(overlap),
                    "score": float(similarity + overlap + 0.08 * length_ratio),
                    "aligned": aligned,
                }
            )
    return pairs


def select_pairs(pairs, config: CrossDateConfig):
    candidates = [
        pair
        for pair in pairs
        if pair["similarity"] >= config.minimum_probability_similarity
        and pair["overlap"] >= config.minimum_path_overlap
        and pair["length_ratio"] >= config.minimum_length_ratio
        and pair["length_difference"] <= 3
    ]
    candidates.sort(key=lambda pair: pair["score"], reverse=True)
    selected = []
    used: set[tuple[str, int]] = set()
    for pair in candidates:
        left_key = (pair["user"], pair["first"])
        right_key = (pair["user"], pair["second"])
        if left_key in used or right_key in used:
            continue
        used.update((left_key, right_key))
        selected.append(pair)
    return selected


def decode(protocol_value, probability, safe, pairs, config: CrossDateConfig):
    selected = select_pairs(pairs, config)
    prediction = safe.copy()
    log_probability = np.log(np.maximum(probability, 1e-12))
    aligned_rows = 0
    for pair in selected:
        left, right = pair["left"], pair["right"]
        reference, other = (left, right) if len(left) >= len(right) else (right, left)
        aligned, _ = align_probabilities(
            probability[reference], probability[other], 0.2
        )
        columns: list[list[int]] = [[int(index)] for index in reference]
        for reference_position, other_position in aligned:
            columns[reference_position].append(int(other[other_position]))
        aggregate = []
        for rows in columns:
            if len(rows) == 1:
                aggregate.append(log_probability[rows[0]])
            else:
                other_logp = log_probability[np.asarray(rows[1:])].mean(axis=0)
                aggregate.append(
                    (
                        log_probability[rows[0]]
                        + config.evidence_weight * other_logp
                    )
                    / (1.0 + config.evidence_weight)
                )
        shared_path = decode_unique_beam(
            np.asarray(aggregate),
            protocol_value[7],
            transition_weight=protocol_value[8].transition_weight,
            beam_width=protocol_value[8].beam_width,
        )
        for position, rows in enumerate(columns):
            prediction[np.asarray(rows, dtype=np.int64)] = int(shared_path[position])
            aligned_rows += len(rows)
    return prediction, {
        "selected_cross_date_pairs": int(len(selected)),
        "selected_sessions": int(2 * len(selected)),
        "selected_rows": int(
            sum(len(pair["left"]) + len(pair["right"]) for pair in selected)
        ),
        "aligned_rows_with_duplicates": int(aligned_rows),
    }


def evaluate(protocol_value, probability, safe, pairs, config: CrossDateConfig):
    prediction, grouping = decode(protocol_value, probability, safe, pairs, config)
    users = protocol_value[4].users.astype(str)
    per_user = {}
    gains = []
    for user in sorted(set(users.tolist())):
        rows = users == user
        gain = int(
            np.sum(prediction[rows] == protocol_value[1][rows])
            - np.sum(safe[rows] == protocol_value[1][rows])
        )
        gains.append(gain)
        per_user[user] = gain
    return (
        {
            "configuration": config.__dict__,
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_safe": rescue_harm(
                protocol_value[1], safe, prediction
            ),
            "minimum_user_gain": int(min(gains)),
            "positive_users": int(sum(gain > 0 for gain in gains)),
            "per_user_gain": per_user,
            "grouping": grouping,
        },
        prediction,
    )


def main() -> None:
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_probability, h1_safe = prepare(h1)
    h2_probability, h2_safe = prepare(h2)
    h1_pairs = pair_cache(h1, h1_probability, h1_safe)
    h2_pairs = pair_cache(h2, h2_probability, h2_safe)
    configs = [
        CrossDateConfig(similarity, overlap, length, evidence)
        for similarity in (0.50, 0.60, 0.70, 0.78, 0.84, 0.90)
        for overlap in (0.0, 0.10, 0.20, 0.40)
        for length in (0.40, 0.55, 0.70, 0.80)
        for evidence in (0.10, 0.25, 0.50, 1.0)
    ]
    candidates = []
    predictions = []
    for index, config in enumerate(configs, start=1):
        item, prediction = evaluate(
            h1, h1_probability, h1_safe, h1_pairs, config
        )
        candidates.append(item)
        predictions.append(prediction)
        if index % 100 == 0:
            print(f"evaluated {index}/{len(configs)}", flush=True)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["positive_users"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_safe"]["net"],
            -candidates[index]["rescue_harm_vs_safe"]["harm"],
            -candidates[index]["grouping"]["selected_rows"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    configuration = CrossDateConfig(**selected["configuration"])
    confirmation, h2_prediction = evaluate(
        h2, h2_probability, h2_safe, h2_pairs, configuration
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
        h1_safe=h1_safe,
        h2_safe=h2_safe,
    )
    report = {
        "stage": "P89_cross_date_same_subject_repeat_v1",
        "protocol": (
            "Starting from the frozen 0.85572 validation recipe, align only "
            "disjoint repeated sessions belonging to the same known subject but "
            "recorded on different dates. Select thresholds on H1 and transfer "
            "unchanged to H2."
        ),
        "pair_cache": {"H1": len(h1_pairs), "H2": len(h2_pairs)},
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "all_H1_candidates": [candidates[index] for index in order],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "all_H1_candidates"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
