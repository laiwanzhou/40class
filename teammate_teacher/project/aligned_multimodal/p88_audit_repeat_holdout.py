from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DEFAULT_TEACHER,
    DEFAULT_TRAIN_METADATA,
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    fit_transition_model,
)
from p88_sequence_decoder import (
    RepeatConfig,
    decode_repeat_consensus,
)
from p88_train_depth_residual import log_softmax_numpy, read_rows, rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "runs/p87s_fusion_holdout1_c7_structured12_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_repeat_student_h1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply P88 anonymous repetition consensus to one frozen P87-S holdout. "
            "H1 may search; H2 must receive --fixed-config-summary."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--fixed-config-summary", type=Path)
    parser.add_argument("--medium-gaps", type=float, nargs="+", default=(90.0, 120.0, 180.0, 300.0))
    parser.add_argument("--consensus-weights", type=float, nargs="+", default=(0.10, 0.25, 0.50, 0.75, 1.0))
    parser.add_argument("--probability-similarities", type=float, nargs="+", default=(0.70, 0.80, 0.88, 0.94))
    parser.add_argument("--path-overlaps", type=float, nargs="+", default=(0.40, 0.60, 0.80))
    return parser.parse_args()


def repeat_grid(args: argparse.Namespace) -> list[RepeatConfig]:
    return [
        RepeatConfig(
            medium_gap_seconds=float(gap),
            consensus_weight=float(weight),
            probability_similarity=float(similarity),
            path_overlap=float(overlap),
            maximum_group_size=3,
        )
        for gap in args.medium_gaps
        for weight in args.consensus_weights
        for similarity in args.probability_similarities
        for overlap in args.path_overlaps
    ]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    prediction_rows = read_rows(run_dir / "subject_holdout_predictions.csv")
    sample_ids = np.asarray([row["sample_id"] for row in prediction_rows])
    labels = np.asarray([int(row["label"]) for row in prediction_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in prediction_rows])
    if set(users) != set(map(str, args.holdout_users)):
        raise RuntimeError("P88 holdout users differ from frozen P87 evidence")
    logits = np.asarray(
        np.load(run_dir / "subject_holdout_logits.npy", allow_pickle=False),
        dtype=np.float64,
    )
    audit = json.loads((run_dir / "decoder_audit.json").read_text(encoding="utf-8"))
    frozen = audit["selected_decoder_config"]
    decoder_config = DecoderConfig(
        gap_seconds=float(frozen["gap_seconds"]),
        transition_weight=float(frozen["transition_weight"]),
        trigram_backoff=float(frozen["trigram_backoff"]),
        beam_width=int(frozen["beam_width"]),
    )
    with np.load(args.teacher_targets.resolve(), allow_pickle=False) as teacher:
        all_ids = np.asarray(teacher["oof_sample_ids"]).astype(str)
        all_labels = np.asarray(teacher["oof_labels"], dtype=np.int64)
    all_metadata = align_metadata(args.train_metadata, all_ids)
    fit_indices = np.flatnonzero(~np.isin(all_metadata.users, list(args.holdout_users)))
    fit_sessions = build_sessions(
        fit_indices,
        all_metadata,
        decoder_config.gap_seconds,
        grouping="known_user",
    )
    transition = fit_transition_model(
        all_labels,
        fit_sessions,
        num_classes=40,
        trigram_backoff=decoder_config.trigram_backoff,
    )
    metadata = align_metadata(args.train_metadata, sample_ids)
    held_indices = np.arange(len(sample_ids), dtype=np.int64)
    baseline, baseline_grouping = decode_repeat_consensus(
        log_softmax_numpy(logits),
        held_indices,
        metadata,
        transition,
        decoder_config,
        RepeatConfig(90.0, 0.0, 1.1, 1.1, 3),
    )
    expected = int(audit["raw_vs_decoder"]["decoded"]["correct"])
    if int(np.sum(baseline == labels)) != expected:
        raise RuntimeError("P88 failed to reproduce frozen P87 decoded evidence")

    selected_on_current_holdout = args.fixed_config_summary is None
    if args.fixed_config_summary:
        source = json.loads(
            args.fixed_config_summary.resolve().read_text(encoding="utf-8")
        )
        candidates = [RepeatConfig(**source["best"]["config"])]
    else:
        candidates = repeat_grid(args)
    scored: list[dict[str, object]] = []
    best_key: tuple[int, int, float, float, float] | None = None
    best: dict[str, object] | None = None
    for config in candidates:
        prediction, grouping = decode_repeat_consensus(
            log_softmax_numpy(logits),
            held_indices,
            metadata,
            transition,
            decoder_config,
            config,
        )
        metrics = classification_metrics(labels, prediction)
        changes = rescue_harm(labels, baseline, prediction)
        result = {
            "config": asdict(config),
            "metrics": metrics,
            "rescue_harm": changes,
            "grouping": grouping,
        }
        scored.append(result)
        key = (
            int(metrics["correct"]),
            int(changes["rescue"] - changes["harm"]),
            -float(config.consensus_weight),
            -abs(float(config.medium_gap_seconds) - 120.0),
            float(config.probability_similarity),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = result
    assert best is not None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P88_repeat_consensus_on_frozen_P87_student",
        "status": "complete",
        "protocol": (
            "P87 logits, transition fitting labels and base decoder remain frozen. "
            + (
                "Repetition parameters are selected on this declared H1 development holdout."
                if selected_on_current_holdout
                else "Repetition parameters are loaded unchanged from H1; this is an independent H2 confirmation."
            )
        ),
        "run_dir": str(run_dir),
        "holdout_users": sorted(map(str, args.holdout_users)),
        "selected_on_current_holdout": selected_on_current_holdout,
        "baseline": classification_metrics(labels, baseline),
        "baseline_grouping": baseline_grouping,
        "best": best,
        "grid_size": len(candidates),
        "all_candidates": scored,
        "decoder_config": asdict(decoder_config),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
