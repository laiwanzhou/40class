from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DecoderConfig,
    TransitionModel,
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
    decode_unique_beam,
)
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_oof_candidate_ensemble import load_protocol
from p88_train_depth_residual import rescue_harm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode P88 sessions with cross-subject full action-script templates."
    )
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=Path("runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"))
    parser.add_argument("--train-metadata", type=Path, default=Path("data/p85_recording_metadata/train_recording_metadata.csv"))
    parser.add_argument("--repeat-config-summary", type=Path, default=Path("runs/p88_aligned_repeat_h1_v1/summary.json"))
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument("--prior-weights", type=float, nargs="+", default=(0.0, 0.25, 0.5))
    parser.add_argument("--posterior-temperatures", type=float, nargs="+", default=(0.25, 0.5, 1.0))
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.10, 0.25, 0.50, 0.75))
    parser.add_argument("--score-loss-gates", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--maximum-length", type=int, default=10)
    return parser.parse_args()


def path_score(
    logp: np.ndarray, path: np.ndarray, model: TransitionModel, weight: float
) -> float:
    length = len(path)
    score = float(logp[np.arange(length), path].sum())
    score += weight * float(model.start_log_probability[path[0]])
    score += weight * float(model.end_log_probability[path[-1]])
    if length >= 2:
        score += weight * float(model.bigram_log_probability[path[0], path[1]])
    if length >= 3:
        score += weight * float(
            model.trigram_log_probability[path[:-2], path[1:-1], path[2:]].sum()
        )
    return score


def fit_templates(
    labels: np.ndarray, sessions: list[np.ndarray], maximum_length: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    counts: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    for session in sessions:
        path = tuple(map(int, labels[session]))
        if len(path) <= maximum_length and len(set(path)) == len(path):
            counts[len(path)][path] += 1
    result = {}
    for length, values in counts.items():
        paths = np.asarray(list(values), dtype=np.int64)
        frequencies = np.asarray([values[tuple(map(int, row))] for row in paths], dtype=np.float64)
        result[length] = paths, frequencies
    return result


def apply_template_posterior(
    logp: np.ndarray,
    sessions: list[np.ndarray],
    templates: dict[int, tuple[np.ndarray, np.ndarray]],
    transition: TransitionModel,
    decoder: DecoderConfig,
    configuration: dict[str, float],
) -> tuple[np.ndarray, dict[str, float | int]]:
    probability = np.exp(logp)
    adjusted = probability.copy()
    applied = 0
    covered_rows = 0
    losses = []
    for session in sessions:
        source = templates.get(len(session))
        if source is None:
            continue
        paths, frequencies = source
        emission = logp[session]
        scores = np.asarray([
            path_score(emission, path, transition, decoder.transition_weight)
            for path in paths
        ])
        baseline_path = decode_unique_beam(
            emission, transition, decoder.transition_weight, decoder.beam_width
        )
        baseline_score = path_score(
            emission, baseline_path, transition, decoder.transition_weight
        )
        best_loss = float(baseline_score - np.max(scores))
        if best_loss > float(configuration["score_loss_gate"]):
            continue
        posterior_scores = (
            scores + float(configuration["prior_weight"]) * np.log(frequencies)
        ) / float(configuration["posterior_temperature"])
        posterior_scores -= posterior_scores.max()
        posterior = np.exp(posterior_scores); posterior /= posterior.sum()
        marginal = np.zeros_like(emission)
        for time_index in range(len(session)):
            np.add.at(marginal[time_index], paths[:, time_index], posterior)
        strength = float(configuration["strength"])
        adjusted[session] = (1.0 - strength) * probability[session] + strength * marginal
        adjusted[session] /= adjusted[session].sum(axis=1, keepdims=True)
        applied += 1; covered_rows += len(session); losses.append(best_loss)
    return np.log(np.maximum(adjusted, 1e-12)), {
        "applied_sessions": applied,
        "covered_rows": covered_rows,
        "mean_best_template_score_loss": float(np.mean(losses)) if losses else 0.0,
    }


def main() -> None:
    args = parse_args()
    (
        sample_ids, labels, base_probability, base_decoded, metadata, indices,
        sessions, transition, decoder, repeat,
    ) = load_protocol(args)
    with np.load(args.teacher_targets.resolve(), allow_pickle=False) as teacher:
        all_ids = teacher["oof_sample_ids"].astype(str)
        all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(args.train_metadata.resolve(), all_ids)
    fit_indices = np.flatnonzero(~np.isin(all_metadata.users, args.holdout_users))
    fit_sessions = build_sessions(
        fit_indices, all_metadata, decoder.gap_seconds, grouping="known_user"
    )
    templates = fit_templates(all_labels, fit_sessions, args.maximum_length)
    base_logp = np.log(np.maximum(base_probability, 1e-12))
    if args.fixed_summary:
        source = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8"))
        configurations = [source["best"]["configuration"]]
        selected_here = False
    else:
        configurations = [
            {
                "prior_weight": float(prior),
                "posterior_temperature": float(temperature),
                "strength": float(strength),
                "score_loss_gate": float(gate),
            }
            for prior in args.prior_weights
            for temperature in args.posterior_temperatures
            for strength in args.strengths
            for gate in args.score_loss_gates
        ]
        selected_here = True
    results = []
    for configuration in configurations:
        adjusted, template_audit = apply_template_posterior(
            base_logp, sessions, templates, transition, decoder, configuration
        )
        decoded = decode_sessions(adjusted, sessions, transition, decoder)
        aligned, grouping = decode_aligned_repeat(
            adjusted, indices, metadata, transition, decoder, repeat
        )
        results.append({
            "configuration": configuration,
            "decoded": classification_metrics(labels, decoded),
            "aligned": classification_metrics(labels, aligned),
            "aligned_rescue_harm_vs_base": rescue_harm(labels, base_decoded, aligned),
            "template_audit": template_audit,
            "grouping": grouping,
        })
    results.sort(key=lambda result: (
        result["aligned"]["correct"], result["aligned"]["balanced_accuracy"],
        result["decoded"]["correct"], -result["configuration"]["strength"],
    ), reverse=True)
    summary = {
        "stage": "P88_cross_subject_session_template_decoder", "status": "complete",
        "selected_on_current_holdout": selected_here,
        "holdout_users": sorted(args.holdout_users),
        "base_decoded": classification_metrics(labels, base_decoded),
        "template_counts_by_length": {str(k): len(v[0]) for k, v in templates.items()},
        "best": results[0], "repeat_configuration": asdict(repeat),
        "decoder_configuration": asdict(decoder), "grid_size": len(results),
        "all_candidates": results,
    }
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "all_candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
