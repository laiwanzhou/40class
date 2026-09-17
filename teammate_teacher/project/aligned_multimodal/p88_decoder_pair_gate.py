from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DEFAULT_TEACHER,
    DEFAULT_TRAIN_METADATA,
    align_metadata,
    classification_metrics,
)
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_oof_candidate_ensemble import DEFAULT_H1_REPEAT, load_protocol
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SEQUENCE_OOF = PROJECT_DIR / "runs/p87_sequence_decoder_v1/oof_predictions.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate P87 sequence changes with cross-subject raw-to-decoded pair reliability."
    )
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--repeat-config-summary", type=Path, default=DEFAULT_H1_REPEAT)
    parser.add_argument("--sequence-oof", type=Path, default=DEFAULT_SEQUENCE_OOF)
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument("--prior-strengths", type=float, nargs="+", default=(0.0, 2.0, 5.0, 10.0))
    parser.add_argument("--revert-margins", type=float, nargs="+", default=(0.0, 0.05, 0.10, 0.20))
    parser.add_argument("--minimum-informative-counts", type=int, nargs="+", default=(1, 2, 3, 5))
    return parser.parse_args()


def pair_statistics(
    raw: np.ndarray,
    decoded: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[dict[tuple[int, int], tuple[int, int]], float]:
    raw_win = (raw == labels) & (decoded != labels) & fit_mask
    decoded_win = (decoded == labels) & (raw != labels) & fit_mask
    informative = raw_win | decoded_win
    global_advantage = float(
        (np.sum(decoded_win) - np.sum(raw_win)) / max(np.sum(informative), 1)
    )
    statistics: dict[tuple[int, int], tuple[int, int]] = {}
    for source in range(40):
        for target in range(40):
            if source == target:
                continue
            pair = fit_mask & (raw == source) & (decoded == target)
            decoded_wins = int(np.sum(pair & decoded_win))
            raw_wins = int(np.sum(pair & raw_win))
            if decoded_wins or raw_wins:
                statistics[(source, target)] = (decoded_wins, raw_wins)
    return statistics, global_advantage


def gate_prediction(
    raw: np.ndarray,
    decoded: np.ndarray,
    statistics: dict[tuple[int, int], tuple[int, int]],
    global_advantage: float,
    prior_strength: float,
    revert_margin: float,
    minimum_informative_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    result = np.asarray(decoded, dtype=np.int64).copy()
    reverted_pairs: dict[str, dict[str, float | int]] = {}
    reverted_rows = 0
    for (source, target), (decoded_wins, raw_wins) in statistics.items():
        informative = decoded_wins + raw_wins
        posterior = (
            decoded_wins
            - raw_wins
            + prior_strength * global_advantage
        ) / max(informative + prior_strength, 1e-12)
        if informative >= minimum_informative_count and posterior < -revert_margin:
            mask = (raw == source) & (decoded == target)
            result[mask] = raw[mask]
            rows = int(np.sum(mask))
            reverted_rows += rows
            reverted_pairs[f"{source}->{target}"] = {
                "decoded_wins": decoded_wins,
                "raw_wins": raw_wins,
                "posterior_advantage": float(posterior),
                "applied_rows": rows,
            }
    return result, {
        "global_decoded_advantage": global_advantage,
        "reverted_rows": reverted_rows,
        "reverted_pairs": reverted_pairs,
    }


def main() -> None:
    args = parse_args()
    (
        sample_ids,
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
    raw_prediction = base_probability.argmax(axis=1)
    aligned_prediction, grouping = decode_aligned_repeat(
        np.log(np.maximum(base_probability, 1e-12)),
        indices,
        metadata,
        transition,
        decoder,
        repeat,
    )
    with np.load(args.sequence_oof.resolve(), allow_pickle=False) as source:
        oof_ids = source["sample_ids"].astype(str)
        oof_labels = source["labels"].astype(np.int64)
        oof_raw = source["baseline_predictions"].astype(np.int64)
        oof_decoded = source["sequence_predictions"].astype(np.int64)
    oof_metadata = align_metadata(args.train_metadata.resolve(), oof_ids)
    fit_mask = ~np.isin(oof_metadata.users, args.holdout_users)
    statistics, global_advantage = pair_statistics(
        oof_raw, oof_decoded, oof_labels, fit_mask
    )
    selected_here = args.fixed_summary is None
    if args.fixed_summary:
        fixed = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8"))
        configurations = [fixed["best"]["configuration"]]
    else:
        configurations = [
            {
                "candidate": candidate,
                "prior_strength": float(prior),
                "revert_margin": float(margin),
                "minimum_informative_count": int(minimum),
            }
            for candidate in ("sequence", "aligned_repeat")
            for prior in args.prior_strengths
            for margin in args.revert_margins
            for minimum in args.minimum_informative_counts
        ]
    results = []
    for configuration in configurations:
        candidate_prediction = (
            base_decoded
            if configuration["candidate"] == "sequence"
            else aligned_prediction
        )
        prediction, gate_audit = gate_prediction(
            raw_prediction,
            candidate_prediction,
            statistics,
            global_advantage,
            float(configuration["prior_strength"]),
            float(configuration["revert_margin"]),
            int(configuration["minimum_informative_count"]),
        )
        results.append(
            {
                "configuration": configuration,
                "metrics": classification_metrics(labels, prediction),
                "rescue_harm_vs_sequence_base": rescue_harm(
                    labels, base_decoded, prediction
                ),
                "gate": gate_audit,
            }
        )
    results.sort(
        key=lambda result: (
            result["metrics"]["correct"],
            result["metrics"]["balanced_accuracy"],
            -result["gate"]["reverted_rows"],
            -result["configuration"]["prior_strength"],
        ),
        reverse=True,
    )
    summary = {
        "stage": "P88_cross_subject_decoder_pair_gate",
        "status": "complete",
        "selected_on_current_holdout": selected_here,
        "holdout_users": sorted(args.holdout_users),
        "raw": classification_metrics(labels, raw_prediction),
        "sequence_base": classification_metrics(labels, base_decoded),
        "aligned_repeat_base": classification_metrics(labels, aligned_prediction),
        "best": results[0],
        "repeat_configuration": asdict(repeat),
        "decoder_configuration": asdict(decoder),
        "repeat_grouping": grouping,
        "fit_rows": int(fit_mask.sum()),
        "fit_global_decoded_advantage": global_advantage,
        "grid_size": len(configurations),
        "all_candidates": results,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "all_candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
