from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DEFAULT_TEACHER,
    DEFAULT_TRAIN_METADATA,
    classification_metrics,
    decode_unique_beam,
)
from p88_aligned_repeat_holdout import (
    align_probabilities,
    group_sessions,
    short_sessions_in_blocks,
)
from p88_oof_candidate_ensemble import DEFAULT_H1_REPEAT, load_protocol
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly decode dynamically aligned repeated takes with one shared "
            "latent action path."
        )
    )
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--repeat-config-summary", type=Path, default=DEFAULT_H1_REPEAT)
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument(
        "--evidence-weights", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0, 4.0)
    )
    parser.add_argument(
        "--transition-scales", type=float, nargs="+", default=(0.5, 1.0, 1.5)
    )
    return parser.parse_args()


def joint_repeat_decode(
    probability: np.ndarray,
    base_prediction: np.ndarray,
    indices: np.ndarray,
    metadata,
    transition,
    decoder,
    repeat,
    evidence_weight: float,
    transition_scale: float,
) -> tuple[np.ndarray, dict[str, int]]:
    prediction = np.asarray(base_prediction, dtype=np.int64).copy()
    log_probability = np.log(np.maximum(probability, 1e-12))
    blocks = short_sessions_in_blocks(
        indices, metadata, decoder.gap_seconds, repeat.medium_gap_seconds
    )
    grouped_sessions = grouped_rows = aligned_pairs = assigned_rows = 0
    for sessions in blocks:
        for group in group_sessions(sessions, probability, base_prediction, repeat):
            reference = max(group, key=len)
            columns: list[list[int]] = [[int(index)] for index in reference]
            for session in group:
                if session is reference:
                    continue
                pairs, _ = align_probabilities(
                    probability[reference],
                    probability[session],
                    repeat.alignment_gap_penalty,
                )
                for reference_position, other_position in pairs:
                    columns[reference_position].append(int(session[other_position]))
                aligned_pairs += len(pairs)
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
                assigned_rows += len(rows)
            grouped_sessions += len(group)
            grouped_rows += sum(map(len, group))
    return prediction, {
        "multi_session_blocks": len(blocks),
        "grouped_sessions": grouped_sessions,
        "grouped_rows": grouped_rows,
        "aligned_pairs": aligned_pairs,
        "assigned_rows_with_duplicates": assigned_rows,
    }


def main() -> None:
    args = parse_args()
    (
        _sample_ids,
        labels,
        base_probability,
        base_decoded,
        metadata,
        indices,
        _sessions,
        transition,
        decoder,
        repeat,
    ) = load_protocol(args)
    selected_here = args.fixed_summary is None
    if args.fixed_summary:
        source = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8"))
        configurations = [source["best"]["configuration"]]
    else:
        configurations = [
            {
                "evidence_weight": float(evidence_weight),
                "transition_scale": float(transition_scale),
            }
            for evidence_weight in args.evidence_weights
            for transition_scale in args.transition_scales
        ]
    results = []
    for configuration in configurations:
        prediction, grouping = joint_repeat_decode(
            base_probability,
            base_decoded,
            indices,
            metadata,
            transition,
            decoder,
            repeat,
            float(configuration["evidence_weight"]),
            float(configuration["transition_scale"]),
        )
        results.append(
            {
                "configuration": configuration,
                "metrics": classification_metrics(labels, prediction),
                "rescue_harm_vs_base": rescue_harm(labels, base_decoded, prediction),
                "grouping": grouping,
            }
        )
    results.sort(
        key=lambda result: (
            result["metrics"]["correct"],
            result["metrics"]["balanced_accuracy"],
            -abs(result["configuration"]["evidence_weight"] - 1.0),
            -abs(result["configuration"]["transition_scale"] - 1.0),
        ),
        reverse=True,
    )
    summary = {
        "stage": "P88_joint_repeated_take_decoder",
        "status": "complete",
        "selected_on_current_holdout": selected_here,
        "holdout_users": sorted(args.holdout_users),
        "base_decoded": classification_metrics(labels, base_decoded),
        "best": results[0],
        "repeat_configuration": asdict(repeat),
        "decoder_configuration": asdict(decoder),
        "grid_size": len(configurations),
        "all_candidates": results,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
