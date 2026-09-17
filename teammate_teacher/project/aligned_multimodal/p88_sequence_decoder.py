from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DEFAULT_TEACHER,
    DEFAULT_TRAIN_METADATA,
    DecoderConfig,
    RecordingMetadata,
    align_metadata,
    build_sessions,
    choose_config,
    classification_metrics,
    decode_sessions,
    fit_transition_model,
    strict_nested_oof,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_sequence_repeat_v1"


@dataclass(frozen=True)
class RepeatConfig:
    medium_gap_seconds: float
    consensus_weight: float
    probability_similarity: float
    path_overlap: float
    maximum_group_size: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "P88 label-free repetition consensus layered on the frozen P87 decoder. "
            "All repetition hyperparameters are selected inside each outer subject fold."
        )
    )
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gap-seconds", type=float, nargs="+", default=(20.0, 30.0, 45.0))
    parser.add_argument("--transition-weights", type=float, nargs="+", default=(0.25, 0.30, 0.35))
    parser.add_argument("--trigram-backoffs", type=float, nargs="+", default=(1.0, 2.0, 5.0))
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--medium-gaps", type=float, nargs="+", default=(120.0, 180.0, 300.0))
    parser.add_argument("--consensus-weights", type=float, nargs="+", default=(0.25, 0.50, 0.75, 1.0))
    parser.add_argument("--probability-similarities", type=float, nargs="+", default=(0.70, 0.80, 0.88))
    parser.add_argument("--path-overlaps", type=float, nargs="+", default=(0.40, 0.60, 0.80))
    parser.add_argument("--maximum-group-size", type=int, default=3)
    return parser.parse_args()


def _session_start(session: np.ndarray, metadata: RecordingMetadata) -> float:
    return float(np.nanmin(metadata.starts[np.asarray(session, dtype=np.int64)]))


def _contained_short_sessions(
    indices: np.ndarray,
    metadata: RecordingMetadata,
    short_gap_seconds: float,
    medium_gap_seconds: float,
) -> list[list[np.ndarray]]:
    short_sessions = build_sessions(
        indices, metadata, short_gap_seconds, grouping="anonymous_date"
    )
    medium_blocks = build_sessions(
        indices, metadata, medium_gap_seconds, grouping="anonymous_date"
    )
    short_sessions.sort(key=lambda value: _session_start(value, metadata))
    result: list[list[np.ndarray]] = []
    for block in medium_blocks:
        members = set(map(int, block))
        contained = [
            session for session in short_sessions if all(int(value) in members for value in session)
        ]
        contained.sort(key=lambda value: _session_start(value, metadata))
        if contained:
            result.append(contained)
    return result


def _probability(log_probability: np.ndarray) -> np.ndarray:
    values = np.asarray(log_probability, dtype=np.float64)
    values = values - np.max(values, axis=1, keepdims=True)
    result = np.exp(values)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def _pair_score(
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    first_path: np.ndarray,
    second_path: np.ndarray,
) -> tuple[float, float]:
    if first_probability.shape != second_probability.shape:
        return 0.0, 0.0
    coefficient = np.sqrt(first_probability * second_probability).sum(axis=1)
    probability_similarity = float(np.mean(coefficient))
    first_set = set(map(int, first_path))
    second_set = set(map(int, second_path))
    path_overlap = len(first_set & second_set) / max(len(first_set | second_set), 1)
    return probability_similarity, float(path_overlap)


def _repeat_groups(
    sessions: list[np.ndarray],
    log_probability: np.ndarray,
    base_prediction: np.ndarray,
    config: RepeatConfig,
) -> list[list[np.ndarray]]:
    if len(sessions) < 2:
        return []
    probabilities = [_probability(log_probability[session]) for session in sessions]
    candidates: list[tuple[float, int, int]] = []
    for first in range(len(sessions)):
        # Repeated takes are locally adjacent. Limiting the ordinal distance prevents
        # equal-length but unrelated scripts in one long recording block from merging.
        for second in range(first + 1, min(first + 4, len(sessions))):
            if len(sessions[first]) != len(sessions[second]):
                continue
            similarity, overlap = _pair_score(
                probabilities[first],
                probabilities[second],
                base_prediction[sessions[first]],
                base_prediction[sessions[second]],
            )
            if (
                similarity >= config.probability_similarity
                and overlap >= config.path_overlap
            ):
                candidates.append((similarity + overlap, first, second))
    candidates.sort(reverse=True)
    groups: list[list[int]] = []
    membership: dict[int, int] = {}
    for _, first, second in candidates:
        first_group = membership.get(first)
        second_group = membership.get(second)
        if first_group is None and second_group is None:
            group_index = len(groups)
            groups.append([first, second])
            membership[first] = group_index
            membership[second] = group_index
        elif first_group is not None and second_group is None:
            if len(groups[first_group]) < config.maximum_group_size:
                groups[first_group].append(second)
                membership[second] = first_group
        elif first_group is None and second_group is not None:
            if len(groups[second_group]) < config.maximum_group_size:
                groups[second_group].append(first)
                membership[first] = second_group
        elif first_group != second_group:
            merged = groups[first_group] + groups[second_group]
            if len(merged) <= config.maximum_group_size:
                groups[first_group] = merged
                groups[second_group] = []
                for value in merged:
                    membership[value] = first_group
    return [[sessions[index] for index in group] for group in groups if len(group) >= 2]


def decode_repeat_consensus(
    log_probability: np.ndarray,
    indices: np.ndarray,
    metadata: RecordingMetadata,
    transition_model,
    decoder_config: DecoderConfig,
    repeat_config: RepeatConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    prediction = decode_sessions(
        log_probability,
        build_sessions(
            indices,
            metadata,
            decoder_config.gap_seconds,
            grouping="anonymous_date",
        ),
        transition_model,
        decoder_config,
    )
    adjusted = np.asarray(log_probability, dtype=np.float64).copy()
    grouped_sessions = 0
    grouped_rows = 0
    repeat_blocks = _contained_short_sessions(
        indices,
        metadata,
        decoder_config.gap_seconds,
        repeat_config.medium_gap_seconds,
    )
    for block_sessions in repeat_blocks:
        for group in _repeat_groups(
            block_sessions, log_probability, prediction, repeat_config
        ):
            stacked = np.stack([_probability(log_probability[session]) for session in group])
            consensus = stacked.mean(axis=0)
            for session, original in zip(group, stacked, strict=True):
                mixed = (
                    (1.0 - repeat_config.consensus_weight) * original
                    + repeat_config.consensus_weight * consensus
                )
                mixed /= np.maximum(mixed.sum(axis=1, keepdims=True), 1e-12)
                adjusted[session] = np.log(np.maximum(mixed, 1e-12))
            grouped_sessions += len(group)
            grouped_rows += sum(map(len, group))
    prediction = decode_sessions(
        adjusted,
        build_sessions(
            indices,
            metadata,
            decoder_config.gap_seconds,
            grouping="anonymous_date",
        ),
        transition_model,
        decoder_config,
    )
    return prediction, {
        "medium_blocks": len(repeat_blocks),
        "grouped_sessions": grouped_sessions,
        "grouped_rows": grouped_rows,
    }


def repeat_grid(args: argparse.Namespace) -> list[RepeatConfig]:
    return [
        RepeatConfig(
            medium_gap_seconds=float(medium_gap),
            consensus_weight=float(weight),
            probability_similarity=float(similarity),
            path_overlap=float(overlap),
            maximum_group_size=int(args.maximum_group_size),
        )
        for medium_gap in args.medium_gaps
        for weight in args.consensus_weights
        for similarity in args.probability_similarities
        for overlap in args.path_overlaps
    ]


def choose_repeat_config(
    labels: np.ndarray,
    folds: np.ndarray,
    log_probability: np.ndarray,
    metadata: RecordingMetadata,
    outer_fold: int,
    decoder_config: DecoderConfig,
    candidates: list[RepeatConfig],
) -> tuple[RepeatConfig | None, dict[str, float]]:
    outer_train = np.flatnonzero(folds != outer_fold)
    correct = np.zeros(len(candidates), dtype=np.int64)
    total = 0
    for inner_user in sorted(set(metadata.users[outer_train])):
        validation = outer_train[metadata.users[outer_train] == inner_user]
        fit = outer_train[metadata.users[outer_train] != inner_user]
        fit_sessions = build_sessions(
            fit,
            metadata,
            decoder_config.gap_seconds,
            grouping="known_user",
        )
        transition = fit_transition_model(
            labels,
            fit_sessions,
            num_classes=log_probability.shape[1],
            trigram_backoff=decoder_config.trigram_backoff,
        )
        for index, candidate in enumerate(candidates):
            prediction, _ = decode_repeat_consensus(
                log_probability,
                validation,
                metadata,
                transition,
                decoder_config,
                candidate,
            )
            correct[index] += int(np.sum(prediction[validation] == labels[validation]))
        total += len(validation)
    accuracy = correct.astype(np.float64) / max(total, 1)
    best_index = int(np.argmax(accuracy))
    best = candidates[best_index]
    score_map = {
        json.dumps(asdict(candidate), sort_keys=True): float(score)
        for candidate, score in zip(candidates, accuracy, strict=True)
    }
    return best, score_map


def strict_nested_repeat_oof(
    labels: np.ndarray,
    folds: np.ndarray,
    log_probability: np.ndarray,
    metadata: RecordingMetadata,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = log_probability.argmax(axis=1).astype(np.int64)
    results: list[dict[str, object]] = []
    candidates = repeat_grid(args)
    for outer_fold in sorted(map(int, np.unique(folds))):
        decoder_config, decoder_grid = choose_config(
            labels,
            folds,
            log_probability,
            metadata,
            outer_fold,
            tuple(map(float, args.gap_seconds)),
            tuple(map(float, args.transition_weights)),
            tuple(map(float, args.trigram_backoffs)),
            int(args.beam_width),
        )
        repeat_config, repeat_scores = choose_repeat_config(
            labels,
            folds,
            log_probability,
            metadata,
            outer_fold,
            decoder_config,
            candidates,
        )
        train_indices = np.flatnonzero(folds != outer_fold)
        validation_indices = np.flatnonzero(folds == outer_fold)
        fit_sessions = build_sessions(
            train_indices,
            metadata,
            decoder_config.gap_seconds,
            grouping="known_user",
        )
        transition = fit_transition_model(
            labels,
            fit_sessions,
            num_classes=log_probability.shape[1],
            trigram_backoff=decoder_config.trigram_backoff,
        )
        fold_prediction, grouping = decode_repeat_consensus(
            log_probability,
            validation_indices,
            metadata,
            transition,
            decoder_config,
            repeat_config,
        )
        predictions[validation_indices] = fold_prediction[validation_indices]
        results.append(
            {
                "fold": outer_fold,
                "users": sorted(set(metadata.users[validation_indices])),
                "decoder_config": asdict(decoder_config),
                "repeat_config": asdict(repeat_config),
                "outer_metrics": classification_metrics(
                    labels[validation_indices], predictions[validation_indices]
                ),
                "grouping": grouping,
                "best_inner_repeat_accuracy": max(repeat_scores.values()),
                "decoder_grid": decoder_grid,
                "repeat_grid": repeat_scores,
            }
        )
    return predictions, results


def main() -> None:
    args = parse_args()
    if args.maximum_group_size < 2:
        raise ValueError("maximum-group-size must be at least two")
    with np.load(args.teacher_targets.resolve(), allow_pickle=False) as teacher:
        sample_ids = np.asarray(teacher["oof_sample_ids"]).astype(str)
        labels = np.asarray(teacher["oof_labels"], dtype=np.int64)
        folds = np.asarray(teacher["oof_folds"], dtype=np.int64)
        log_probability = np.asarray(
            teacher["oof_teacher_log_probability"], dtype=np.float64
        )
    metadata = align_metadata(args.train_metadata, sample_ids)
    baseline = log_probability.argmax(axis=1).astype(np.int64)
    p87_prediction, p87_folds = strict_nested_oof(
        labels,
        folds,
        log_probability,
        metadata,
        gap_seconds_candidates=tuple(map(float, args.gap_seconds)),
        transition_weights=tuple(map(float, args.transition_weights)),
        trigram_backoffs=tuple(map(float, args.trigram_backoffs)),
        beam_width=int(args.beam_width),
    )
    p88_prediction, p88_folds = strict_nested_repeat_oof(
        labels, folds, log_probability, metadata, args
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P88_strict_nested_repeat_decoder",
        "status": "complete",
        "protocol": (
            "P87 remains frozen. Outer subject folds are untouched. P87 decoder and "
            "all repetition grouping/consensus hyperparameters are selected only by "
            "inner LOSO. Repetition grouping uses anonymous date/time and model "
            "posteriors, never labels or user IDs."
        ),
        "baseline": classification_metrics(labels, baseline),
        "p87": classification_metrics(labels, p87_prediction),
        "p88": classification_metrics(labels, p88_prediction),
        "delta_correct_vs_p87": int(
            np.sum(p88_prediction == labels) - np.sum(p87_prediction == labels)
        ),
        "p87_folds": p87_folds,
        "p88_folds": p88_folds,
        "grid_size": len(repeat_grid(args)),
        "config": vars(args) | {
            "teacher_targets": str(args.teacher_targets.resolve()),
            "train_metadata": str(args.train_metadata.resolve()),
            "output_dir": str(output),
        },
    }
    np.savez_compressed(
        output / "oof_predictions.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        baseline_predictions=baseline,
        p87_predictions=p87_prediction,
        p88_predictions=p88_prediction,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
