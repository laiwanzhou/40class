from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER = (
    PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
)
DEFAULT_FUSION = PROJECT_DIR / "runs/p85_multiexpert_submission_v1/fusion_logits.npz"
DEFAULT_TRAIN_METADATA = (
    PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
)
DEFAULT_TEST_METADATA = (
    PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87_sequence_decoder_v1"


@dataclass(frozen=True)
class RecordingMetadata:
    sample_ids: np.ndarray
    users: np.ndarray
    dates: np.ndarray
    starts: np.ndarray


@dataclass(frozen=True)
class TransitionModel:
    start_log_probability: np.ndarray
    end_log_probability: np.ndarray
    bigram_log_probability: np.ndarray
    trigram_log_probability: np.ndarray


@dataclass(frozen=True)
class DecoderConfig:
    gap_seconds: float
    transition_weight: float
    trigram_backoff: float
    beam_width: int


@dataclass(frozen=True)
class BeamPosterior:
    """Approximate structured posterior retained by the finite beam."""

    paths: np.ndarray
    path_scores: np.ndarray
    path_probability: np.ndarray
    marginals: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict nested subject-disjoint audit of P87 anonymous recording-sequence "
            "decoding. The decoder uses timestamp sessions, a no-repeat constraint, "
            "and backed-off second-order action transitions."
        )
    )
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--fusion-logits", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--test-metadata", type=Path, default=DEFAULT_TEST_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gap-seconds", type=float, nargs="+", default=(20.0, 30.0, 45.0)
    )
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument(
        "--transition-weights",
        type=float,
        nargs="+",
        default=(0.25, 0.30, 0.35),
    )
    parser.add_argument(
        "--trigram-backoffs", type=float, nargs="+", default=(1.0, 2.0, 5.0)
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only run OOF auditing; do not emit the direct-teacher Test diagnostic.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def align_metadata(path: Path, sample_ids: np.ndarray) -> RecordingMetadata:
    rows = {row["sample_id"]: row for row in read_csv(path)}
    missing = sorted(set(map(str, sample_ids)) - set(rows))
    if missing:
        raise ValueError(f"Metadata is missing {len(missing)} sample ids, e.g. {missing[:3]}")
    ordered = [rows[str(sample_id)] for sample_id in sample_ids]
    return RecordingMetadata(
        sample_ids=np.asarray(sample_ids).astype(str),
        users=np.asarray([row.get("user_id", "") for row in ordered]).astype(str),
        dates=np.asarray([row.get("recording_date", "") for row in ordered]).astype(str),
        starts=np.asarray(
            [
                float(row["start_seconds"])
                if row.get("start_seconds", "")
                else np.nan
                for row in ordered
            ],
            dtype=np.float64,
        ),
    )


def build_sessions(
    indices: Iterable[int] | np.ndarray,
    metadata: RecordingMetadata,
    gap_seconds: float,
    grouping: str,
) -> list[np.ndarray]:
    if grouping not in {"known_user", "anonymous_date"}:
        raise ValueError(f"Unsupported grouping: {grouping}")
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for value in indices:
        index = int(value)
        if not np.isfinite(metadata.starts[index]) or not metadata.dates[index]:
            continue
        if grouping == "known_user":
            key = (metadata.users[index], metadata.dates[index])
        else:
            key = (metadata.dates[index],)
        grouped[key].append(index)

    sessions: list[np.ndarray] = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda index: metadata.starts[index])
        current: list[int] = []
        previous_start: float | None = None
        for index in ordered:
            start = float(metadata.starts[index])
            if (
                current
                and previous_start is not None
                and start - previous_start > float(gap_seconds)
            ):
                sessions.append(np.asarray(current, dtype=np.int64))
                current = []
            current.append(index)
            previous_start = start
        if current:
            sessions.append(np.asarray(current, dtype=np.int64))
    return sessions


def fit_transition_model(
    labels: np.ndarray,
    sessions: list[np.ndarray],
    num_classes: int,
    trigram_backoff: float,
    alpha: float = 0.25,
) -> TransitionModel:
    if trigram_backoff <= 0:
        raise ValueError("trigram_backoff must be positive")
    start_counts = np.full(num_classes, alpha, dtype=np.float64)
    end_counts = np.full(num_classes, alpha, dtype=np.float64)
    bigram_counts = np.full((num_classes, num_classes), alpha, dtype=np.float64)
    trigram_counts = np.zeros(
        (num_classes, num_classes, num_classes), dtype=np.float64
    )
    for session in sessions:
        if len(session) == 0:
            continue
        sequence = labels[session].astype(np.int64, copy=False)
        start_counts[sequence[0]] += 1.0
        end_counts[sequence[-1]] += 1.0
        if len(sequence) >= 2:
            np.add.at(bigram_counts, (sequence[:-1], sequence[1:]), 1.0)
        if len(sequence) >= 3:
            np.add.at(
                trigram_counts,
                (sequence[:-2], sequence[1:-1], sequence[2:]),
                1.0,
            )

    bigram_probability = bigram_counts / bigram_counts.sum(axis=1, keepdims=True)
    trigram_total = trigram_counts.sum(axis=2, keepdims=True)
    trigram_mle = np.divide(
        trigram_counts,
        trigram_total,
        out=np.zeros_like(trigram_counts),
        where=trigram_total > 0,
    )
    interpolation = trigram_total / (trigram_total + float(trigram_backoff))
    trigram_probability = (
        interpolation * trigram_mle
        + (1.0 - interpolation) * bigram_probability[None, :, :]
    )
    return TransitionModel(
        start_log_probability=np.log(start_counts / start_counts.sum()),
        end_log_probability=np.log(end_counts / end_counts.sum()),
        bigram_log_probability=np.log(np.maximum(bigram_probability, 1e-300)),
        trigram_log_probability=np.log(np.maximum(trigram_probability, 1e-300)),
    )


def decode_unique_beam_paths(
    emission_log_probability: np.ndarray,
    model: TransitionModel,
    transition_weight: float,
    beam_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    emission = np.asarray(emission_log_probability, dtype=np.float64)
    if emission.ndim != 2:
        raise ValueError("emission_log_probability must have shape [time, class]")
    length, num_classes = emission.shape
    if length == 0:
        return (
            np.empty((1, 0), dtype=np.int64),
            np.zeros(1, dtype=np.float64),
        )
    if num_classes > 63:
        raise ValueError("The uint64 no-repeat mask supports at most 63 classes")
    if length > num_classes:
        raise ValueError("A no-repeat sequence cannot be longer than num_classes")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    initial_score = emission[0] + transition_weight * model.start_log_probability
    initial_count = min(int(beam_width), num_classes)
    initial_classes = np.argsort(initial_score, kind="stable")[-initial_count:]
    scores = initial_score[initial_classes]
    paths = initial_classes[:, None].astype(np.int64)
    used = np.left_shift(np.uint64(1), initial_classes.astype(np.uint64))
    previous = np.full(initial_count, -1, dtype=np.int64)
    last = initial_classes.astype(np.int64)
    class_bits = np.left_shift(
        np.uint64(1), np.arange(num_classes, dtype=np.uint64)
    )

    for time_index in range(1, length):
        if time_index == 1:
            transition = model.bigram_log_probability[last]
        else:
            transition = model.trigram_log_probability[previous, last]
        candidates = (
            scores[:, None]
            + emission[time_index][None, :]
            + transition_weight * transition
        )
        candidates[(used[:, None] & class_bits[None, :]) != 0] = -np.inf
        flat = candidates.ravel()
        finite_count = int(np.isfinite(flat).sum())
        keep_count = min(int(beam_width), finite_count)
        if keep_count == 0:
            raise RuntimeError("No valid unique-label beam remains")
        kept = np.argsort(flat, kind="stable")[-keep_count:]
        parent = kept // num_classes
        new_class = (kept % num_classes).astype(np.int64)
        scores = flat[kept]
        paths = np.concatenate((paths[parent], new_class[:, None]), axis=1)
        used = used[parent] | class_bits[new_class]
        previous = last[parent]
        last = new_class

    scores = scores + transition_weight * model.end_log_probability[last]
    order = np.argsort(scores, kind="stable")[::-1]
    return paths[order], scores[order]


def decode_unique_beam_posterior(
    emission_log_probability: np.ndarray,
    model: TransitionModel,
    transition_weight: float,
    beam_width: int,
    posterior_temperature: float = 1.0,
) -> BeamPosterior:
    """Marginalize the retained structured paths into uncertainty-aware targets.

    This is an approximation to the full sequence posterior because only the final
    beam is retained. Unlike mixing a MAP one-hot target with emissions at a fixed
    ratio, it preserves ambiguity when several legal sequence paths remain plausible.
    """

    if posterior_temperature <= 0:
        raise ValueError("posterior_temperature must be positive")
    paths, path_scores = decode_unique_beam_paths(
        emission_log_probability,
        model,
        transition_weight=transition_weight,
        beam_width=beam_width,
    )
    scaled = (path_scores - np.max(path_scores)) / float(posterior_temperature)
    path_probability = np.exp(scaled)
    path_probability /= path_probability.sum()
    length, num_classes = np.asarray(emission_log_probability).shape
    marginals = np.zeros((length, num_classes), dtype=np.float64)
    for time_index in range(length):
        np.add.at(marginals[time_index], paths[:, time_index], path_probability)
    return BeamPosterior(
        paths=paths,
        path_scores=path_scores,
        path_probability=path_probability,
        marginals=marginals,
    )


def decode_unique_beam(
    emission_log_probability: np.ndarray,
    model: TransitionModel,
    transition_weight: float,
    beam_width: int,
) -> np.ndarray:
    paths, _ = decode_unique_beam_paths(
        emission_log_probability,
        model,
        transition_weight=transition_weight,
        beam_width=beam_width,
    )
    return paths[0]


def decode_sessions(
    base_log_probability: np.ndarray,
    sessions: list[np.ndarray],
    model: TransitionModel,
    config: DecoderConfig,
) -> np.ndarray:
    predictions = np.asarray(base_log_probability).argmax(axis=1).astype(np.int64)
    for session in sessions:
        predictions[session] = decode_unique_beam(
            base_log_probability[session],
            model,
            transition_weight=config.transition_weight,
            beam_width=config.beam_width,
        )
    return predictions


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    classes = np.unique(labels)
    recalls: list[float] = []
    f1s: list[float] = []
    for class_id in classes:
        true_positive = int(np.sum((labels == class_id) & (predictions == class_id)))
        false_negative = int(np.sum((labels == class_id) & (predictions != class_id)))
        false_positive = int(np.sum((labels != class_id) & (predictions == class_id)))
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "correct": int(np.sum(labels == predictions)),
        "total": int(len(labels)),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def choose_config(
    labels: np.ndarray,
    folds: np.ndarray,
    log_probability: np.ndarray,
    metadata: RecordingMetadata,
    outer_fold: int,
    gap_seconds_candidates: tuple[float, ...],
    transition_weights: tuple[float, ...],
    trigram_backoffs: tuple[float, ...],
    beam_width: int,
) -> tuple[DecoderConfig, dict[str, float]]:
    outer_train = np.flatnonzero(folds != outer_fold)
    inner_users = sorted(set(metadata.users[outer_train]))
    scores: dict[tuple[float, float, float], list[int]] = {
        (gap_seconds, backoff, weight): []
        for gap_seconds in gap_seconds_candidates
        for backoff in trigram_backoffs
        for weight in transition_weights
    }
    totals: dict[tuple[float, float, float], int] = {key: 0 for key in scores}

    for inner_user in inner_users:
        inner_validation = outer_train[metadata.users[outer_train] == inner_user]
        inner_fit = outer_train[metadata.users[outer_train] != inner_user]
        for gap_seconds in gap_seconds_candidates:
            fit_sessions = build_sessions(
                inner_fit, metadata, gap_seconds, grouping="known_user"
            )
            validation_sessions = build_sessions(
                inner_validation, metadata, gap_seconds, grouping="anonymous_date"
            )
            for backoff in trigram_backoffs:
                model = fit_transition_model(
                    labels,
                    fit_sessions,
                    num_classes=log_probability.shape[1],
                    trigram_backoff=backoff,
                )
                for weight in transition_weights:
                    config = DecoderConfig(
                        gap_seconds=gap_seconds,
                        transition_weight=weight,
                        trigram_backoff=backoff,
                        beam_width=beam_width,
                    )
                    prediction = decode_sessions(
                        log_probability, validation_sessions, model, config
                    )
                    key = (gap_seconds, backoff, weight)
                    scores[key].append(
                        int(
                            np.sum(
                                prediction[inner_validation]
                                == labels[inner_validation]
                            )
                        )
                    )
                    totals[key] += int(len(inner_validation))

    accuracy = {
        key: sum(correct_by_user) / totals[key]
        for key, correct_by_user in scores.items()
    }
    # Accuracy is primary. The remaining keys make ties deterministic and prefer the
    # pre-registered centre of the small grid rather than an extreme value.
    best_key = max(
        accuracy,
        key=lambda key: (
            accuracy[key],
            -abs(key[0] - 30.0),
            -abs(key[2] - 0.30),
            -abs(key[1] - 2.0),
        ),
    )
    selected = DecoderConfig(
        gap_seconds=best_key[0],
        transition_weight=best_key[2],
        trigram_backoff=best_key[1],
        beam_width=beam_width,
    )
    return selected, {
        f"gap={key[0]:g},backoff={key[1]:g},weight={key[2]:g}": float(value)
        for key, value in sorted(accuracy.items())
    }


def strict_nested_oof(
    labels: np.ndarray,
    folds: np.ndarray,
    log_probability: np.ndarray,
    metadata: RecordingMetadata,
    gap_seconds_candidates: tuple[float, ...],
    transition_weights: tuple[float, ...],
    trigram_backoffs: tuple[float, ...],
    beam_width: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = log_probability.argmax(axis=1).astype(np.int64)
    fold_results: list[dict[str, object]] = []
    for outer_fold in sorted(map(int, np.unique(folds))):
        config, inner_grid = choose_config(
            labels,
            folds,
            log_probability,
            metadata,
            outer_fold,
            gap_seconds_candidates,
            transition_weights,
            trigram_backoffs,
            beam_width,
        )
        train_indices = np.flatnonzero(folds != outer_fold)
        validation_indices = np.flatnonzero(folds == outer_fold)
        fit_sessions = build_sessions(
            train_indices, metadata, config.gap_seconds, grouping="known_user"
        )
        validation_sessions = build_sessions(
            validation_indices,
            metadata,
            config.gap_seconds,
            grouping="anonymous_date",
        )
        model = fit_transition_model(
            labels,
            fit_sessions,
            num_classes=log_probability.shape[1],
            trigram_backoff=config.trigram_backoff,
        )
        fold_prediction = decode_sessions(
            log_probability, validation_sessions, model, config
        )
        predictions[validation_indices] = fold_prediction[validation_indices]
        fold_results.append(
            {
                "fold": outer_fold,
                "users": sorted(set(metadata.users[validation_indices])),
                "selected": {
                    "gap_seconds": config.gap_seconds,
                    "transition_weight": config.transition_weight,
                    "trigram_backoff": config.trigram_backoff,
                    "beam_width": config.beam_width,
                },
                "inner_selected_accuracy": inner_grid[
                    f"gap={config.gap_seconds:g},backoff={config.trigram_backoff:g},weight={config.transition_weight:g}"
                ],
                "outer": classification_metrics(
                    labels[validation_indices], predictions[validation_indices]
                ),
                "anonymous_sessions": len(validation_sessions),
                "inner_grid": inner_grid,
            }
        )
    return predictions, fold_results


def consensus_test_config(
    fold_results: list[dict[str, object]], beam_width: int
) -> DecoderConfig:
    gaps = sorted(
        float(result["selected"]["gap_seconds"])  # type: ignore[index]
        for result in fold_results
    )
    weights = sorted(
        float(result["selected"]["transition_weight"])  # type: ignore[index]
        for result in fold_results
    )
    backoffs = sorted(
        float(result["selected"]["trigram_backoff"])  # type: ignore[index]
        for result in fold_results
    )
    return DecoderConfig(
        gap_seconds=gaps[len(gaps) // 2],
        transition_weight=weights[len(weights) // 2],
        trigram_backoff=backoffs[len(backoffs) // 2],
        beam_width=beam_width,
    )


def write_submission(
    path: Path,
    metadata_rows: list[dict[str, str]],
    predictions_by_id: dict[str, int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "prediction"])
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(
                {
                    "path": row["official_path"],
                    "prediction": int(predictions_by_id[row["sample_id"]]),
                }
            )


def main() -> None:
    args = parse_args()
    teacher = np.load(args.teacher_targets.resolve())
    sample_ids = teacher["oof_sample_ids"].astype(str)
    labels = teacher["oof_labels"].astype(np.int64)
    folds = teacher["oof_folds"].astype(np.int64)
    log_probability = teacher["oof_teacher_log_probability"].astype(np.float64)
    metadata = align_metadata(args.train_metadata, sample_ids)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    baseline_prediction = log_probability.argmax(axis=1)
    nested_prediction, fold_results = strict_nested_oof(
        labels,
        folds,
        log_probability,
        metadata,
        gap_seconds_candidates=tuple(map(float, args.gap_seconds)),
        transition_weights=tuple(map(float, args.transition_weights)),
        trigram_backoffs=tuple(map(float, args.trigram_backoffs)),
        beam_width=int(args.beam_width),
    )
    selected_test_config = consensus_test_config(fold_results, int(args.beam_width))
    summary: dict[str, object] = {
        "protocol": (
            "Strict outer subject folds; session gap, transition weight, and trigram "
            "backoff are selected by inner LOSO using only the other outer-fold "
            "users. Outer validation sessions use anonymous date/time grouping, "
            "never user ids."
        ),
        "deployment_note": (
            "The sequence decoder is only a few transition tables and is compatible "
            "with a sub-100 MB student. This run still uses P85 Large-teacher OOF/Test "
            "probabilities, so its direct Test CSV is diagnostic rather than the final "
            "compliant student submission."
        ),
        "session_gap_selection": (
            "The 20/30/45 second session boundary is selected inside each outer fold "
            "by the same inner LOSO protocol as the transition hyperparameters."
        ),
        "baseline": classification_metrics(labels, baseline_prediction),
        "strict_nested": classification_metrics(labels, nested_prediction),
        "delta_correct": int(
            np.sum(nested_prediction == labels) - np.sum(baseline_prediction == labels)
        ),
        "folds": fold_results,
        "test_config_from_outer_medians": {
            "gap_seconds": selected_test_config.gap_seconds,
            "transition_weight": selected_test_config.transition_weight,
            "trigram_backoff": selected_test_config.trigram_backoff,
            "beam_width": selected_test_config.beam_width,
        },
    }
    np.savez_compressed(
        output / "oof_predictions.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        baseline_predictions=baseline_prediction,
        sequence_predictions=nested_prediction,
    )

    if not args.skip_test:
        fusion = np.load(args.fusion_logits.resolve())
        all_test_ids = fusion["test_all_sample_ids"].astype(str)
        predictions_by_id = {
            str(sample_id): int(prediction)
            for sample_id, prediction in zip(
                all_test_ids, fusion["test_base_predictions"].astype(np.int64)
            )
        }
        teacher_test_ids = teacher["test_sample_ids"].astype(str)
        teacher_test_log_probability = teacher["test_teacher_log_probability"].astype(
            np.float64
        )
        test_metadata = align_metadata(args.test_metadata, teacher_test_ids)
        train_sessions = build_sessions(
            np.arange(len(labels)),
            metadata,
            selected_test_config.gap_seconds,
            grouping="known_user",
        )
        model = fit_transition_model(
            labels,
            train_sessions,
            num_classes=log_probability.shape[1],
            trigram_backoff=selected_test_config.trigram_backoff,
        )
        test_sessions = build_sessions(
            np.arange(len(teacher_test_ids)),
            test_metadata,
            selected_test_config.gap_seconds,
            grouping="anonymous_date",
        )
        test_prediction = decode_sessions(
            teacher_test_log_probability, test_sessions, model, selected_test_config
        )
        teacher_test_probability = teacher["test_teacher_probability"].astype(
            np.float64
        )
        # The sharpened distribution is a training-only target for the legal Test
        # pseudo-label/self-training stage. It is deliberately saved as data, not as
        # an inference dependency of the final small student.
        sharpened_teacher_probability = 0.25 * teacher_test_probability
        sharpened_teacher_probability[
            np.arange(len(test_prediction)), test_prediction
        ] += 0.75
        for sample_id, prediction in zip(teacher_test_ids, test_prediction):
            predictions_by_id[str(sample_id)] = int(prediction)
        test_rows = read_csv(args.test_metadata)
        submission_path = output / "submission_p87_sequence_teacher_diagnostic.csv"
        write_submission(submission_path, test_rows, predictions_by_id)
        np.savez_compressed(
            output / "test_predictions.npz",
            sample_ids=all_test_ids,
            predictions=np.asarray(
                [predictions_by_id[sample_id] for sample_id in all_test_ids],
                dtype=np.int64,
            ),
            teacher_sample_ids=teacher_test_ids,
            teacher_sequence_predictions=test_prediction,
            teacher_sequence_distillation_probability=sharpened_teacher_probability.astype(
                np.float32
            ),
        )
        summary["test_diagnostic"] = {
            "rows": len(all_test_ids),
            "teacher_rows": len(teacher_test_ids),
            "anonymous_sessions": len(test_sessions),
            "changed_teacher_rows": int(
                np.sum(
                    test_prediction
                    != teacher_test_log_probability.argmax(axis=1).astype(np.int64)
                )
            ),
            "submission": str(submission_path),
        }

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
