from __future__ import annotations

import argparse
import csv
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
    decode_sessions,
    fit_transition_model,
)
from p88_aligned_repeat_holdout import AlignedRepeatConfig, decode_aligned_repeat
from p88_train_depth_residual import log_softmax_numpy, read_rows, rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_H1_REPEAT = PROJECT_DIR / "runs/p88_aligned_repeat_h1_v1/summary.json"

CANDIDATE_SOURCES = {
    "p85_teacher": (
        "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz",
        "oof_sample_ids",
        "oof_teacher_log_probability",
    ),
    **{
        f"p85_head_{key}": (
            "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz",
            "sample_ids",
            key,
        )
        for key in (
            "early_logits",
            "late_logits",
            "window_mean_logits",
            "early_late_logits",
            "temporal_delta_logits",
            "kinetics_logits",
        )
    },
    **{
        f"p86_mechanism_{key}": (
            "runs/p86_teacher_mechanism_audit_v1/fixed_model_predictions.npz",
            "sample_ids",
            key,
        )
        for key in (
            "baseline_logits",
            "drop_scene_logits",
            "drop_person_logits",
            "drop_workspace_logits",
            "drop_early_logits",
            "drop_late_logits",
            "swap_early_late_logits",
            "collapse_early_late_logits",
            "swap_person_workspace_logits",
            "collapse_view_identity_logits",
        )
    },
    "p12_skeleton": (
        "runs/p12_complete_oof/complete_oof.npz",
        "sample_ids",
        "skeleton_logits",
    ),
    "p12_thermal": (
        "runs/p12_complete_oof/complete_oof.npz",
        "sample_ids",
        "thermal_logits",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P88 H1-selected, H2-confirmed OOF expert probability ensemble."
    )
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--repeat-config-summary", type=Path, default=DEFAULT_H1_REPEAT)
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument(
        "--candidate-temperatures", type=float, nargs="+", default=(0.75, 1.0, 1.5)
    )
    parser.add_argument(
        "--candidate-weights", type=float, nargs="+", default=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    )
    parser.add_argument("--aligned-shortlist", type=int, default=40)
    return parser.parse_args()


def softmax_temperature(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logp = log_softmax_numpy(np.asarray(values, dtype=np.float64) / temperature)
    return np.exp(logp)


def load_candidate(name: str, sample_ids: np.ndarray) -> np.ndarray:
    relative_path, id_key, value_key = CANDIDATE_SOURCES[name]
    with np.load(PROJECT_DIR / relative_path, allow_pickle=False) as source:
        source_ids = source[id_key].astype(str)
        values = np.asarray(source[value_key], dtype=np.float64)
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise RuntimeError(f"Candidate {name} misses {len(missing)} holdout rows")
    return values[np.asarray([lookup[sample_id] for sample_id in sample_ids])]


def load_protocol(args: argparse.Namespace):
    run_dir = args.base_run.resolve()
    rows = read_rows(run_dir / "subject_holdout_predictions.csv")
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in rows])
    if set(users.tolist()) != set(args.holdout_users):
        raise RuntimeError("holdout user mismatch")
    logits = np.asarray(np.load(run_dir / "subject_holdout_logits.npy"), dtype=np.float64)
    base_probability = np.exp(log_softmax_numpy(logits))
    audit = json.loads((run_dir / "decoder_audit.json").read_text(encoding="utf-8"))
    frozen = audit["selected_decoder_config"]
    decoder = DecoderConfig(
        float(frozen["gap_seconds"]),
        float(frozen["transition_weight"]),
        float(frozen["trigram_backoff"]),
        int(frozen["beam_width"]),
    )
    with np.load(args.teacher_targets.resolve(), allow_pickle=False) as teacher:
        all_ids = teacher["oof_sample_ids"].astype(str)
        all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(args.train_metadata.resolve(), all_ids)
    fit_indices = np.flatnonzero(~np.isin(all_metadata.users, args.holdout_users))
    transition = fit_transition_model(
        all_labels,
        build_sessions(
            fit_indices,
            all_metadata,
            decoder.gap_seconds,
            grouping="known_user",
        ),
        40,
        decoder.trigram_backoff,
    )
    metadata = align_metadata(args.train_metadata.resolve(), sample_ids)
    indices = np.arange(len(sample_ids), dtype=np.int64)
    sessions = build_sessions(
        indices, metadata, decoder.gap_seconds, grouping="anonymous_date"
    )
    base_decoded = decode_sessions(
        np.log(np.maximum(base_probability, 1e-12)), sessions, transition, decoder
    )
    expected = int(audit["raw_vs_decoder"]["decoded"]["correct"])
    if int(np.sum(base_decoded == labels)) != expected:
        raise RuntimeError("failed to reproduce frozen P87 decoder")
    repeat_source = json.loads(
        args.repeat_config_summary.resolve().read_text(encoding="utf-8")
    )
    repeat = AlignedRepeatConfig(**repeat_source["best"]["config"])
    return (
        sample_ids,
        labels,
        base_probability,
        base_decoded,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        repeat,
    )


def main() -> None:
    args = parse_args()
    (
        sample_ids,
        labels,
        base_probability,
        base_decoded,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        repeat,
    ) = load_protocol(args)

    selected_here = args.fixed_summary is None
    conventional_results: list[dict] = []
    if args.fixed_summary:
        fixed = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8"))
        configurations = [fixed["best"]["configuration"]]
    else:
        configurations = [
            {"candidate": candidate, "temperature": float(temperature), "weight": float(weight)}
            for candidate in CANDIDATE_SOURCES
            for temperature in args.candidate_temperatures
            for weight in args.candidate_weights
        ]

    candidate_cache: dict[tuple[str, float], np.ndarray] = {}
    probability_cache: dict[tuple[str, float, float], np.ndarray] = {}
    for configuration in configurations:
        candidate = str(configuration["candidate"])
        temperature = float(configuration["temperature"])
        weight = float(configuration["weight"])
        cache_key = (candidate, temperature)
        if cache_key not in candidate_cache:
            candidate_cache[cache_key] = softmax_temperature(
                load_candidate(candidate, sample_ids), temperature
            )
        blended = (
            (1.0 - weight) * base_probability
            + weight * candidate_cache[cache_key]
        )
        blended /= blended.sum(axis=1, keepdims=True)
        probability_cache[(candidate, temperature, weight)] = blended
        raw = blended.argmax(axis=1)
        decoded = decode_sessions(
            np.log(np.maximum(blended, 1e-12)), sessions, transition, decoder
        )
        conventional_results.append(
            {
                "configuration": configuration,
                "raw": classification_metrics(labels, raw),
                "decoded": classification_metrics(labels, decoded),
                "decoded_rescue_harm_vs_base": rescue_harm(
                    labels, base_decoded, decoded
                ),
            }
        )

    conventional_results.sort(
        key=lambda result: (
            result["decoded"]["correct"],
            result["raw"]["correct"],
            -result["configuration"]["weight"],
        ),
        reverse=True,
    )
    shortlisted = conventional_results[: max(1, int(args.aligned_shortlist))]
    aligned_results: list[dict] = []
    for result in shortlisted:
        configuration = result["configuration"]
        key = (
            str(configuration["candidate"]),
            float(configuration["temperature"]),
            float(configuration["weight"]),
        )
        blended = probability_cache[key]
        aligned, grouping = decode_aligned_repeat(
            np.log(np.maximum(blended, 1e-12)),
            indices,
            metadata,
            transition,
            decoder,
            repeat,
        )
        aligned_results.append(
            {
                **result,
                "aligned": classification_metrics(labels, aligned),
                "aligned_rescue_harm_vs_base": rescue_harm(
                    labels, base_decoded, aligned
                ),
                "grouping": grouping,
            }
        )
    aligned_results.sort(
        key=lambda result: (
            result["aligned"]["correct"],
            result["aligned"]["balanced_accuracy"],
            result["decoded"]["correct"],
            -result["configuration"]["weight"],
        ),
        reverse=True,
    )
    best = aligned_results[0]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P88_OOF_candidate_probability_ensemble",
        "status": "complete",
        "selected_on_current_holdout": selected_here,
        "holdout_users": sorted(args.holdout_users),
        "base_decoded": classification_metrics(labels, base_decoded),
        "best": best,
        "repeat_configuration": asdict(repeat),
        "decoder_configuration": asdict(decoder),
        "grid_size": len(configurations),
        "aligned_shortlist_size": len(shortlisted),
        "aligned_candidates": aligned_results,
        "top_conventional_candidates": conventional_results[:50],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"aligned_candidates", "top_conventional_candidates"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
