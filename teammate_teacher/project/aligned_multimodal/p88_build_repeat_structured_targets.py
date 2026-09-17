from __future__ import annotations

import argparse
import csv
import hashlib
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
    decode_sessions,
    fit_transition_model,
)
from p88_aligned_repeat_holdout import (
    AlignedRepeatConfig,
    align_probabilities,
    group_sessions,
    probability,
    short_sessions_in_blocks,
)
from p88_train_depth_residual import log_softmax_numpy


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build label-free P88 structured targets by aligning repeated anonymous "
            "takes with a frozen P87 configuration."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--structured-targets", type=Path, required=True)
    parser.add_argument("--repeat-config-summary", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction_ids(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray([row["sample_id"] for row in csv.DictReader(handle)])


def normalized_confidence(probabilities: np.ndarray) -> np.ndarray:
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
    )
    return 1.0 - entropy / np.log(probabilities.shape[1])


def repeat_consensus_targets(
    source_target: np.ndarray,
    grouping_probability: np.ndarray,
    base_prediction: np.ndarray,
    indices: np.ndarray,
    metadata,
    decoder: DecoderConfig,
    config: AlignedRepeatConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    adjusted = np.asarray(source_target, dtype=np.float64).copy()
    blocks = short_sessions_in_blocks(
        indices, metadata, decoder.gap_seconds, config.medium_gap_seconds
    )
    grouped_sessions = grouped_rows = aligned_pairs = 0
    changed_rows: set[int] = set()
    for sessions in blocks:
        for group in group_sessions(
            sessions, grouping_probability, base_prediction, config
        ):
            reference = max(group, key=len)
            all_rows = np.concatenate(group)
            accumulators = {
                int(index): [source_target[int(index)]] for index in all_rows
            }
            for session in group:
                if session is reference:
                    continue
                pairs, _ = align_probabilities(
                    grouping_probability[reference],
                    grouping_probability[session],
                    config.alignment_gap_penalty,
                )
                for reference_position, other_position in pairs:
                    reference_index = int(reference[reference_position])
                    other_index = int(session[other_position])
                    accumulators[reference_index].append(source_target[other_index])
                    accumulators[other_index].append(source_target[reference_index])
                aligned_pairs += len(pairs)
            for index, values in accumulators.items():
                consensus = np.mean(values, axis=0)
                adjusted[index] = (
                    (1.0 - config.consensus_weight) * source_target[index]
                    + config.consensus_weight * consensus
                )
                adjusted[index] /= adjusted[index].sum()
                if not np.allclose(adjusted[index], source_target[index]):
                    changed_rows.add(index)
            grouped_sessions += len(group)
            grouped_rows += len(all_rows)
    return adjusted, {
        "multi_session_blocks": len(blocks),
        "grouped_sessions": grouped_sessions,
        "grouped_rows": grouped_rows,
        "aligned_pairs": aligned_pairs,
        "changed_target_rows": len(changed_rows),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    source_path = args.structured_targets.resolve()
    source = np.load(source_path, allow_pickle=False)
    sample_ids = source["sample_ids"].astype(str)
    target_mask = source["target_mask"].astype(bool)
    selected_ids = sample_ids[target_mask]
    run_ids = read_prediction_ids(run_dir / "subject_holdout_predictions.csv")
    if not np.array_equal(run_ids, selected_ids):
        raise RuntimeError("P87 run and structured-target row order differs")

    logits = np.asarray(
        np.load(run_dir / "subject_holdout_logits.npy"), dtype=np.float64
    )
    grouping_probability = probability(log_softmax_numpy(logits))
    metadata = align_metadata(args.train_metadata.resolve(), selected_ids)
    holdout_users = tuple(sorted(set(metadata.users.tolist())))

    audit = json.loads(
        (run_dir / "decoder_audit.json").read_text(encoding="utf-8")
    )
    frozen = audit["selected_decoder_config"]
    decoder = DecoderConfig(
        float(frozen["gap_seconds"]),
        float(frozen["transition_weight"]),
        float(frozen["trigram_backoff"]),
        int(frozen["beam_width"]),
    )
    teacher = np.load(args.teacher_targets.resolve(), allow_pickle=False)
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(args.train_metadata.resolve(), all_ids)
    train_indices = np.flatnonzero(~np.isin(all_metadata.users, holdout_users))
    transition = fit_transition_model(
        all_labels,
        build_sessions(
            train_indices,
            all_metadata,
            decoder.gap_seconds,
            grouping="known_user",
        ),
        num_classes=grouping_probability.shape[1],
        trigram_backoff=decoder.trigram_backoff,
    )
    local_indices = np.arange(len(selected_ids), dtype=np.int64)
    short_sessions = build_sessions(
        local_indices,
        metadata,
        decoder.gap_seconds,
        grouping="anonymous_date",
    )
    base_prediction = decode_sessions(
        np.log(np.maximum(grouping_probability, 1e-12)),
        short_sessions,
        transition,
        decoder,
    )

    repeat_summary = json.loads(
        args.repeat_config_summary.resolve().read_text(encoding="utf-8")
    )
    config = AlignedRepeatConfig(**repeat_summary["best"]["config"])
    original_full = source["structured_distillation_probability"].astype(np.float64)
    original = original_full[target_mask]
    adjusted, grouping = repeat_consensus_targets(
        original,
        grouping_probability,
        base_prediction,
        local_indices,
        metadata,
        decoder,
        config,
    )
    adjusted_full = original_full.copy()
    adjusted_full[target_mask] = adjusted

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "structured_targets.npz"
    payload = {key: source[key] for key in source.files}
    payload["structured_distillation_probability"] = adjusted_full.astype(np.float32)
    payload["structured_distillation_prediction"] = adjusted_full.argmax(axis=1).astype(
        np.int64
    )
    confidence = source["structured_confidence"].astype(np.float32).copy()
    confidence[target_mask] = normalized_confidence(adjusted).astype(np.float32)
    payload["structured_confidence"] = confidence
    payload["p88_repeat_target_mask"] = target_mask
    np.savez_compressed(target_path, **payload)

    original_map = original.argmax(axis=1)
    adjusted_map = adjusted.argmax(axis=1)
    summary = {
        "stage": "P88_repeat_aligned_structured_targets",
        "status": "complete",
        "protocol": (
            "Pseudo-Test labels are never loaded. Repeated takes are grouped from "
            "anonymous timestamps and frozen P87 probabilities; the H1-selected "
            "alignment configuration is reused unchanged."
        ),
        "holdout_users": list(holdout_users),
        "target_rows": int(target_mask.sum()),
        "configuration": asdict(config),
        "decoder_configuration": asdict(decoder),
        "grouping": grouping,
        "hard_target_changes": int(np.sum(original_map != adjusted_map)),
        "mean_probability_l1": float(np.mean(np.abs(original - adjusted).sum(axis=1))),
        "source_targets": str(source_path),
        "source_targets_sha256": sha256(source_path),
        "p87_run": str(run_dir),
        "repeat_config_summary": str(args.repeat_config_summary.resolve()),
        "output": str(target_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
