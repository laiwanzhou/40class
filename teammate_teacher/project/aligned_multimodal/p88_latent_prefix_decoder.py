from __future__ import annotations

import argparse
import csv
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
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_latent_prefix_nested_v1"


@dataclass(frozen=True)
class PrefixConfig:
    block_gap_seconds: float
    prior_weight: float
    posterior_temperature: float
    length_weight: float
    alpha: float = 1.0


@dataclass(frozen=True)
class PrefixModel:
    names: np.ndarray
    class_probability: np.ndarray
    global_probability: np.ndarray
    prefix_probability: np.ndarray
    length_mean: np.ndarray
    length_std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "P88 latent trial-prefix prior. Prefix IDs are used only on outer-train "
            "to fit class bags; validation blocks infer the prefix from anonymous posteriors."
        )
    )
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gap-seconds", type=float, nargs="+", default=(20.0, 30.0, 45.0))
    parser.add_argument("--transition-weights", type=float, nargs="+", default=(0.25, 0.30, 0.35))
    parser.add_argument("--trigram-backoffs", type=float, nargs="+", default=(1.0, 2.0, 5.0))
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--block-gaps", type=float, nargs="+", default=(45.0, 60.0, 75.0))
    parser.add_argument("--prior-weights", type=float, nargs="+", default=(0.10, 0.25, 0.50, 0.75))
    parser.add_argument("--posterior-temperatures", type=float, nargs="+", default=(0.50, 1.0, 2.0))
    parser.add_argument("--length-weights", type=float, nargs="+", default=(0.0, 0.25, 0.50))
    return parser.parse_args()


def trial_prefixes(manifest: Path, sample_ids: np.ndarray) -> np.ndarray:
    with manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows]
    if missing:
        raise RuntimeError(f"P88 manifest misses sample IDs: {missing[:3]}")
    return np.asarray(
        ["-".join(rows[sample_id]["trial_id"].split("-")[:2]) for sample_id in sample_ids]
    )


def fit_prefix_model(
    labels: np.ndarray,
    prefixes: np.ndarray,
    fit_indices: np.ndarray,
    metadata: RecordingMetadata,
    block_gap_seconds: float,
    alpha: float,
) -> PrefixModel:
    names = np.asarray(sorted(set(prefixes[fit_indices])))
    lookup = {name: index for index, name in enumerate(names)}
    counts = np.full((len(names), 40), float(alpha), dtype=np.float64)
    prefix_counts = np.full(len(names), float(alpha), dtype=np.float64)
    global_counts = np.full(40, float(alpha), dtype=np.float64)
    for index in fit_indices:
        prefix_index = lookup[prefixes[index]]
        counts[prefix_index, labels[index]] += 1.0
        prefix_counts[prefix_index] += 1.0
        global_counts[labels[index]] += 1.0
    lengths: list[list[int]] = [[] for _ in names]
    for block in build_sessions(
        fit_indices, metadata, block_gap_seconds, grouping="known_user"
    ):
        block_prefixes = set(prefixes[block])
        if len(block_prefixes) == 1:
            name = next(iter(block_prefixes))
            lengths[lookup[name]].append(len(block))
    length_mean = np.asarray(
        [np.mean(values) if values else 1.0 for values in lengths], dtype=np.float64
    )
    length_std = np.asarray(
        [max(float(np.std(values)), 1.5) if values else 5.0 for values in lengths],
        dtype=np.float64,
    )
    return PrefixModel(
        names=names,
        class_probability=counts / counts.sum(axis=1, keepdims=True),
        global_probability=global_counts / global_counts.sum(),
        prefix_probability=prefix_counts / prefix_counts.sum(),
        length_mean=length_mean,
        length_std=length_std,
    )


def infer_prefix_posterior(
    log_probability: np.ndarray,
    block: np.ndarray,
    model: PrefixModel,
    config: PrefixConfig,
) -> np.ndarray:
    probability = np.exp(log_probability[block])
    ratio = model.class_probability / np.maximum(model.global_probability[None, :], 1e-12)
    evidence = np.log(np.maximum(probability @ ratio.T, 1e-12)).sum(axis=0)
    length_z = (len(block) - model.length_mean) / model.length_std
    evidence -= float(config.length_weight) * 0.5 * np.square(length_z)
    evidence += np.log(np.maximum(model.prefix_probability, 1e-12))
    evidence = evidence / float(config.posterior_temperature)
    evidence -= np.max(evidence)
    posterior = np.exp(evidence)
    return posterior / posterior.sum()


def apply_prefix_prior(
    log_probability: np.ndarray,
    indices: np.ndarray,
    metadata: RecordingMetadata,
    model: PrefixModel,
    config: PrefixConfig,
    true_prefixes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    adjusted = np.asarray(log_probability, dtype=np.float64).copy()
    blocks = build_sessions(
        indices, metadata, config.block_gap_seconds, grouping="anonymous_date"
    )
    inferred_confidence: list[float] = []
    oracle_available = 0
    name_lookup = {name: index for index, name in enumerate(model.names)}
    for block in blocks:
        if true_prefixes is None:
            posterior = infer_prefix_posterior(adjusted, block, model, config)
        else:
            block_names = set(true_prefixes[block])
            if len(block_names) != 1 or next(iter(block_names)) not in name_lookup:
                continue
            posterior = np.zeros(len(model.names), dtype=np.float64)
            posterior[name_lookup[next(iter(block_names))]] = 1.0
            oracle_available += len(block)
        inferred_confidence.append(float(posterior.max()))
        prior = posterior @ model.class_probability
        ratio = prior / np.maximum(model.global_probability, 1e-12)
        adjusted[block] += float(config.prior_weight) * np.log(np.maximum(ratio, 1e-12))
        adjusted[block] -= np.logaddexp.reduce(adjusted[block], axis=1, keepdims=True)
    return adjusted, {
        "blocks": len(blocks),
        "mean_prefix_confidence": float(np.mean(inferred_confidence)) if inferred_confidence else 0.0,
        "oracle_available_rows": oracle_available,
    }


def config_grid(args: argparse.Namespace) -> list[PrefixConfig]:
    return [
        PrefixConfig(float(gap), float(weight), float(temperature), float(length_weight))
        for gap in args.block_gaps
        for weight in args.prior_weights
        for temperature in args.posterior_temperatures
        for length_weight in args.length_weights
    ]


def choose_prefix_config(
    labels: np.ndarray,
    prefixes: np.ndarray,
    folds: np.ndarray,
    log_probability: np.ndarray,
    metadata: RecordingMetadata,
    outer_fold: int,
    decoder_config: DecoderConfig,
    candidates: list[PrefixConfig],
) -> tuple[PrefixConfig, dict[str, float]]:
    outer_train = np.flatnonzero(folds != outer_fold)
    correct = np.zeros(len(candidates), dtype=np.int64)
    total = 0
    for inner_user in sorted(set(metadata.users[outer_train])):
        validation = outer_train[metadata.users[outer_train] == inner_user]
        fit = outer_train[metadata.users[outer_train] != inner_user]
        transition = fit_transition_model(
            labels,
            build_sessions(fit, metadata, decoder_config.gap_seconds, "known_user"),
            40,
            decoder_config.trigram_backoff,
        )
        models = {
            gap: fit_prefix_model(labels, prefixes, fit, metadata, gap, 1.0)
            for gap in sorted(set(candidate.block_gap_seconds for candidate in candidates))
        }
        sessions = build_sessions(
            validation, metadata, decoder_config.gap_seconds, "anonymous_date"
        )
        for candidate_index, candidate in enumerate(candidates):
            adjusted, _ = apply_prefix_prior(
                log_probability,
                validation,
                metadata,
                models[candidate.block_gap_seconds],
                candidate,
            )
            prediction = decode_sessions(adjusted, sessions, transition, decoder_config)
            correct[candidate_index] += int(np.sum(prediction[validation] == labels[validation]))
        total += len(validation)
    accuracy = correct / max(total, 1)
    best_index = int(np.argmax(accuracy))
    return candidates[best_index], {
        json.dumps(asdict(candidate), sort_keys=True): float(score)
        for candidate, score in zip(candidates, accuracy, strict=True)
    }


def strict_nested_prefix_oof(
    labels: np.ndarray,
    prefixes: np.ndarray,
    folds: np.ndarray,
    log_probability: np.ndarray,
    metadata: RecordingMetadata,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    latent_prediction = log_probability.argmax(axis=1)
    oracle_prediction = latent_prediction.copy()
    results: list[dict[str, object]] = []
    candidates = config_grid(args)
    for outer_fold in sorted(map(int, np.unique(folds))):
        decoder_config, _ = choose_config(
            labels,
            folds,
            log_probability,
            metadata,
            outer_fold,
            tuple(map(float, args.gap_seconds)),
            tuple(map(float, args.transition_weights)),
            tuple(map(float, args.trigram_backoffs)),
            args.beam_width,
        )
        prefix_config, inner_grid = choose_prefix_config(
            labels,
            prefixes,
            folds,
            log_probability,
            metadata,
            outer_fold,
            decoder_config,
            candidates,
        )
        fit = np.flatnonzero(folds != outer_fold)
        held = np.flatnonzero(folds == outer_fold)
        transition = fit_transition_model(
            labels,
            build_sessions(fit, metadata, decoder_config.gap_seconds, "known_user"),
            40,
            decoder_config.trigram_backoff,
        )
        model = fit_prefix_model(
            labels,
            prefixes,
            fit,
            metadata,
            prefix_config.block_gap_seconds,
            prefix_config.alpha,
        )
        sessions = build_sessions(
            held, metadata, decoder_config.gap_seconds, "anonymous_date"
        )
        latent_logp, latent_audit = apply_prefix_prior(
            log_probability, held, metadata, model, prefix_config
        )
        oracle_logp, oracle_audit = apply_prefix_prior(
            log_probability, held, metadata, model, prefix_config, true_prefixes=prefixes
        )
        latent_all = decode_sessions(latent_logp, sessions, transition, decoder_config)
        oracle_all = decode_sessions(oracle_logp, sessions, transition, decoder_config)
        latent_prediction[held] = latent_all[held]
        oracle_prediction[held] = oracle_all[held]
        results.append(
            {
                "fold": outer_fold,
                "decoder_config": asdict(decoder_config),
                "prefix_config": asdict(prefix_config),
                "latent_metrics": classification_metrics(labels[held], latent_prediction[held]),
                "oracle_metrics": classification_metrics(labels[held], oracle_prediction[held]),
                "latent_audit": latent_audit,
                "oracle_audit": oracle_audit,
                "best_inner_accuracy": max(inner_grid.values()),
                "inner_grid": inner_grid,
            }
        )
    return latent_prediction, oracle_prediction, results


def main() -> None:
    args = parse_args()
    with np.load(args.teacher_targets.resolve(), allow_pickle=False) as teacher:
        sample_ids = np.asarray(teacher["oof_sample_ids"]).astype(str)
        labels = np.asarray(teacher["oof_labels"], dtype=np.int64)
        folds = np.asarray(teacher["oof_folds"], dtype=np.int64)
        log_probability = np.asarray(teacher["oof_teacher_log_probability"], dtype=np.float64)
    metadata = align_metadata(args.train_metadata, sample_ids)
    prefixes = trial_prefixes(args.manifest, sample_ids)
    p87_prediction, _ = strict_nested_oof(
        labels,
        folds,
        log_probability,
        metadata,
        tuple(map(float, args.gap_seconds)),
        tuple(map(float, args.transition_weights)),
        tuple(map(float, args.trigram_backoffs)),
        args.beam_width,
    )
    latent, oracle, fold_results = strict_nested_prefix_oof(
        labels, prefixes, folds, log_probability, metadata, args
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P88_strict_nested_latent_prefix_decoder",
        "status": "complete",
        "protocol": (
            "Trial prefixes and labels fit class/length bags only on inner/outer train. "
            "Validation uses anonymous date/time blocks and infers a soft prefix posterior. "
            "The separately reported true-prefix result is an illegal oracle ceiling only."
        ),
        "p87": classification_metrics(labels, p87_prediction),
        "latent_prefix": classification_metrics(labels, latent),
        "true_prefix_oracle": classification_metrics(labels, oracle),
        "latent_delta_correct_vs_p87": int(np.sum(latent == labels) - np.sum(p87_prediction == labels)),
        "oracle_delta_correct_vs_p87": int(np.sum(oracle == labels) - np.sum(p87_prediction == labels)),
        "folds": fold_results,
        "grid_size": len(config_grid(args)),
    }
    np.savez_compressed(
        output / "oof_predictions.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        p87_predictions=p87_prediction,
        latent_prefix_predictions=latent,
        true_prefix_oracle_predictions=oracle,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
