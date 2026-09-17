from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    align_metadata,
    classification_metrics,
    decode_unique_beam,
)
from p88_aligned_repeat_holdout import align_probabilities
from p88_train_depth_residual import rescue_harm
from p89_global_repeat_decoder import date_session_lists


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_supervised_trial_pairing_v1"
TRIAL_PATTERN = re.compile(r"__(\d+)-(\d+)-(\d+)$")
FEATURE_NAMES = (
    "rank_distance",
    "log_start_gap",
    "length_ratio",
    "length_difference",
    "probability_similarity",
    "path_jaccard",
    "alignment_coverage",
    "aligned_top1_agreement",
    "mean_top1_confidence",
    "confidence_difference",
    "entropy_difference",
)


@dataclass(frozen=True)
class PairCandidate:
    first: int
    second: int
    features: np.ndarray
    truth: int
    users: tuple[str, ...]


def entropy(probability: np.ndarray) -> float:
    value = np.asarray(probability, dtype=np.float64)
    return float(np.mean(-np.sum(value * np.log(np.maximum(value, 1e-12)), axis=1)))


def trial_signature(sample_ids: np.ndarray, session: np.ndarray):
    values = []
    for index in session:
        match = TRIAL_PATTERN.search(str(sample_ids[int(index)]))
        if match is None:
            return None
        values.append(tuple(map(int, match.groups())))
    unique = set(values)
    if len(unique) != 1:
        return None
    return values[0]


def pair_features(
    first: np.ndarray,
    second: np.ndarray,
    rank_distance: int,
    probability: np.ndarray,
    decoded: np.ndarray,
    metadata,
) -> np.ndarray:
    start_gap = abs(
        float(np.nanmin(metadata.starts[second]))
        - float(np.nanmin(metadata.starts[first]))
    )
    pairs, similarity = align_probabilities(
        probability[first], probability[second], gap_penalty=0.2
    )
    first_set = set(map(int, decoded[first]))
    second_set = set(map(int, decoded[second]))
    jaccard = len(first_set & second_set) / max(len(first_set | second_set), 1)
    if pairs:
        pair_array = np.asarray(pairs, dtype=np.int64)
        agreement = float(
            np.mean(
                decoded[first[pair_array[:, 0]]]
                == decoded[second[pair_array[:, 1]]]
            )
        )
    else:
        agreement = 0.0
    first_confidence = float(np.mean(np.max(probability[first], axis=1)))
    second_confidence = float(np.mean(np.max(probability[second], axis=1)))
    return np.asarray(
        (
            float(rank_distance),
            float(np.log1p(start_gap)),
            min(len(first), len(second)) / max(len(first), len(second)),
            float(abs(len(first) - len(second))),
            similarity,
            jaccard,
            len(pairs) / max(len(first), len(second)),
            agreement,
            0.5 * (first_confidence + second_confidence),
            abs(first_confidence - second_confidence),
            abs(entropy(probability[first]) - entropy(probability[second])),
        ),
        dtype=np.float64,
    )


def candidates_for_protocol(
    sample_ids: np.ndarray,
    probability: np.ndarray,
    metadata,
    short_gap: float,
    include_truth: bool,
) -> tuple[list[list[np.ndarray]], list[list[PairCandidate]]]:
    decoded = np.argmax(probability, axis=1).astype(np.int64)
    date_lists = date_session_lists(
        np.arange(len(sample_ids), dtype=np.int64), metadata, short_gap
    )
    all_candidates: list[list[PairCandidate]] = []
    for sessions in date_lists:
        date_candidates = []
        for first_index in range(len(sessions)):
            for second_index in range(
                first_index + 1, min(first_index + 9, len(sessions))
            ):
                first, second = sessions[first_index], sessions[second_index]
                start_gap = abs(
                    float(np.nanmin(metadata.starts[second]))
                    - float(np.nanmin(metadata.starts[first]))
                )
                if start_gap > 600.0:
                    continue
                first_trial = trial_signature(sample_ids, first) if include_truth else None
                second_trial = trial_signature(sample_ids, second) if include_truth else None
                truth = int(
                    first_trial is not None
                    and second_trial is not None
                    and first_trial[:2] == second_trial[:2]
                    and first_trial[2] != second_trial[2]
                    and len(set(metadata.users[first].tolist())) == 1
                    and np.array_equal(metadata.users[first[:1]], metadata.users[second[:1]])
                )
                users = tuple(
                    sorted(
                        set(metadata.users[first].astype(str).tolist())
                        | set(metadata.users[second].astype(str).tolist())
                    )
                )
                date_candidates.append(
                    PairCandidate(
                        first=first_index,
                        second=second_index,
                        features=pair_features(
                            first,
                            second,
                            second_index - first_index,
                            probability,
                            decoded,
                            metadata,
                        ),
                        truth=truth,
                        users=users,
                    )
                )
        all_candidates.append(date_candidates)
    return date_lists, all_candidates


def fit_model(name: str, x: np.ndarray, y: np.ndarray):
    if name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000),
        )
    elif name == "hist":
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=7,
            min_samples_leaf=16,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=20260816,
        )
    elif name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=5,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=20260816,
        )
    else:
        raise ValueError(name)
    model.fit(x, y)
    return model


def training_matrix(
    candidates: list[list[PairCandidate]], excluded_users: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        pair
        for date_pairs in candidates
        for pair in date_pairs
        if not (set(pair.users) & excluded_users)
    ]
    return (
        np.stack([pair.features for pair in selected]),
        np.asarray([pair.truth for pair in selected], dtype=np.int64),
    )


def predicted_groups(
    sessions: list[list[np.ndarray]],
    candidates: list[list[PairCandidate]],
    model,
    threshold: float,
) -> tuple[list[list[np.ndarray]], dict[str, int | float]]:
    groups: list[list[np.ndarray]] = []
    selected_pairs = candidate_pairs = 0
    scores = []
    for date_sessions, date_candidates in zip(sessions, candidates, strict=True):
        if not date_candidates:
            continue
        features = np.stack([pair.features for pair in date_candidates])
        probabilities = model.predict_proba(features)[:, 1]
        candidate_pairs += len(date_candidates)
        scores.extend(probabilities.tolist())
        ranked = sorted(
            zip(probabilities, date_candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
        local_groups: list[list[int]] = []
        membership: dict[int, int] = {}
        for score, pair in ranked:
            if score < threshold:
                break
            first, second = pair.first, pair.second
            left, right = membership.get(first), membership.get(second)
            if left is None and right is None:
                membership[first] = membership[second] = len(local_groups)
                local_groups.append([first, second])
                selected_pairs += 1
            elif left is not None and right is None and len(local_groups[left]) < 3:
                local_groups[left].append(second)
                membership[second] = left
                selected_pairs += 1
            elif left is None and right is not None and len(local_groups[right]) < 3:
                local_groups[right].append(first)
                membership[first] = right
                selected_pairs += 1
            elif left is not None and right is not None and left != right:
                if len(local_groups[left]) + len(local_groups[right]) <= 3:
                    merged = local_groups[left] + local_groups[right]
                    local_groups[left], local_groups[right] = merged, []
                    for index in merged:
                        membership[index] = left
                    selected_pairs += 1
        groups.extend(
            [[date_sessions[index] for index in group] for group in local_groups if group]
        )
    return groups, {
        "candidate_pairs": candidate_pairs,
        "selected_pairs": selected_pairs,
        "groups": len(groups),
        "sessions": int(sum(map(len, groups))),
        "rows": int(sum(len(session) for group in groups for session in group)),
        "maximum_score": float(max(scores, default=0.0)),
    }


def joint_decode_groups(
    probability: np.ndarray,
    base_prediction: np.ndarray,
    groups: list[list[np.ndarray]],
    transition,
    decoder,
    evidence_weight: float,
    transition_scale: float = 1.0,
) -> np.ndarray:
    prediction = np.asarray(base_prediction, dtype=np.int64).copy()
    log_probability = np.log(np.maximum(probability, 1e-12))
    for group in groups:
        reference = max(group, key=len)
        columns: list[list[int]] = [[int(index)] for index in reference]
        for session in group:
            if session is reference:
                continue
            pairs, _ = align_probabilities(
                probability[reference], probability[session], gap_penalty=0.2
            )
            for reference_position, other_position in pairs:
                columns[reference_position].append(int(session[other_position]))
        aggregate = np.empty((len(columns), probability.shape[1]), dtype=np.float64)
        for position, rows in enumerate(columns):
            reference_logp = log_probability[rows[0]]
            if len(rows) == 1:
                aggregate[position] = reference_logp
            else:
                other_logp = log_probability[np.asarray(rows[1:])].mean(axis=0)
                aggregate[position] = (
                    reference_logp + evidence_weight * other_logp
                ) / (1.0 + evidence_weight)
        shared_path = decode_unique_beam(
            aggregate,
            transition,
            transition_weight=decoder.transition_weight * transition_scale,
            beam_width=decoder.beam_width,
        )
        for position, rows in enumerate(columns):
            prediction[np.asarray(rows, dtype=np.int64)] = int(shared_path[position])
    return prediction


def align_probability(
    source_ids: np.ndarray,
    source_probability: np.ndarray,
    target_ids: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids.astype(str))}
    values = []
    for index, sample_id in enumerate(target_ids.astype(str)):
        source_index = lookup.get(sample_id)
        if source_index is None:
            if fallback is None:
                raise RuntimeError(f"missing probability for {sample_id}")
            values.append(fallback[index])
        else:
            values.append(source_probability[source_index])
    return np.asarray(values, dtype=np.float64)


def evaluate_target(
    protocol_value,
    pairing_probability: np.ndarray,
    all_train_candidates: list[list[PairCandidate]],
    excluded_users: set[str],
    model_name: str,
    threshold: float,
    evidence_weight: float,
):
    x_train, y_train = training_matrix(all_train_candidates, excluded_users)
    model = fit_model(model_name, x_train, y_train)
    sessions, target_candidates = candidates_for_protocol(
        protocol_value[0],
        pairing_probability,
        protocol_value[4],
        protocol_value[8].gap_seconds,
        include_truth=True,
    )
    return evaluate_prepared(
        protocol_value,
        sessions,
        target_candidates,
        model,
        model_name=model_name,
        threshold=threshold,
        evidence_weight=evidence_weight,
        training_labels=y_train,
    )


def evaluate_prepared(
    protocol_value,
    sessions: list[list[np.ndarray]],
    target_candidates: list[list[PairCandidate]],
    model,
    model_name: str,
    threshold: float,
    evidence_weight: float,
    training_labels: np.ndarray,
):
    groups, grouping = predicted_groups(
        sessions, target_candidates, model, threshold=threshold
    )
    prediction = joint_decode_groups(
        protocol_value[2],
        protocol_value[3],
        groups,
        protocol_value[7],
        protocol_value[8],
        evidence_weight=evidence_weight,
    )
    pair_truth = np.asarray(
        [pair.truth for date_pairs in target_candidates for pair in date_pairs],
        dtype=np.int64,
    )
    pair_score = (
        model.predict_proba(
            np.stack(
                [pair.features for date_pairs in target_candidates for pair in date_pairs]
            )
        )[:, 1]
        if len(pair_truth)
        else np.empty(0)
    )
    return {
        "configuration": {
            "model": model_name,
            "threshold": threshold,
            "evidence_weight": evidence_weight,
        },
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "grouping": grouping,
        "pair_training": {
            "rows": int(len(training_labels)),
            "positives": int(training_labels.sum()),
        },
        "pair_target": {
            "rows": int(len(pair_truth)),
            "positives": int(pair_truth.sum()),
            "selected_true": int(np.sum((pair_score >= threshold) & (pair_truth == 1))),
            "selected_false": int(np.sum((pair_score >= threshold) & (pair_truth == 0))),
        },
    }, prediction


def main() -> None:
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    ) as teacher:
        all_ids = teacher["oof_sample_ids"].astype(str)
        all_probability = np.asarray(teacher["oof_teacher_probability"], dtype=np.float64)
    all_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    _, all_candidates = candidates_for_protocol(
        all_ids,
        all_probability,
        all_metadata,
        h1[8].gap_seconds,
        include_truth=True,
    )
    h1_pairing_probability = align_probability(
        all_ids, all_probability, h1[0]
    )
    h2_pairing_probability = align_probability(
        all_ids, all_probability, h2[0]
    )

    candidates = []
    predictions = []
    h1_sessions, h1_target_candidates = candidates_for_protocol(
        h1[0],
        h1_pairing_probability,
        h1[4],
        h1[8].gap_seconds,
        include_truth=True,
    )
    for model_name in ("logistic", "hist", "extra_trees"):
        x_train, y_train = training_matrix(
            all_candidates, excluded_users=set(full40.H1_USERS)
        )
        model = fit_model(model_name, x_train, y_train)
        for threshold in (0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95):
            for evidence_weight in (0.05, 0.10, 0.25, 0.50):
                item, prediction = evaluate_prepared(
                    h1,
                    h1_sessions,
                    h1_target_candidates,
                    model,
                    model_name=model_name,
                    threshold=threshold,
                    evidence_weight=evidence_weight,
                    training_labels=y_train,
                )
                candidates.append(item)
                predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["metrics"]["correct"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_p87"]["net"],
            -candidates[index]["rescue_harm_vs_p87"]["harm"],
            -candidates[index]["grouping"]["rows"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    frozen = selected["configuration"]
    confirmation, h2_prediction = evaluate_target(
        h2,
        h2_pairing_probability,
        all_candidates,
        excluded_users=set(full40.H2_USERS),
        model_name=str(frozen["model"]),
        threshold=float(frozen["threshold"]),
        evidence_weight=float(frozen["evidence_weight"]),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_subject_disjoint_supervised_trial_pairing_v1",
        "protocol": (
            "Use train-only trial suffixes as pair-supervision. Pair models exclude "
            "the evaluated subjects; H1 selects model/threshold/evidence and H2 "
            "re-fits without H2 subjects then confirms the frozen hyperparameters."
        ),
        "feature_names": FEATURE_NAMES,
        "H1_selected": selected,
        "H2_confirmation": confirmation,
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
