from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audit_p87_sequence_decoder import (
    classification_metrics,
    build_sessions,
    decode_sessions,
)
from p88_aligned_repeat_holdout import align_probabilities
from p88_oof_candidate_ensemble import load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class GlobalRepeatConfig:
    maximum_session_rank_distance: int
    maximum_start_gap_seconds: float
    minimum_probability_similarity: float
    minimum_path_overlap: float
    minimum_length_ratio: float
    consensus_weight: float
    alignment_gap_penalty: float = 0.2
    maximum_group_size: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P89 date-global repeated-session consensus decoder.")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz")
    parser.add_argument("--train-metadata", type=Path, default=PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv")
    parser.add_argument("--repeat-config-summary", type=Path, default=PROJECT_DIR / "runs/p88_aligned_repeat_h1_v1/summary.json")
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument("--class-bias", type=Path)
    return parser.parse_args()


def probabilities(logp: np.ndarray) -> np.ndarray:
    value = np.exp(logp - np.logaddexp.reduce(logp, axis=1, keepdims=True))
    return value / value.sum(axis=1, keepdims=True)


def date_session_lists(indices: np.ndarray, metadata, short_gap: float) -> list[list[np.ndarray]]:
    sessions = build_sessions(indices, metadata, short_gap, "anonymous_date")
    result: list[list[np.ndarray]] = []
    for date in sorted(set(metadata.dates[indices].tolist())):
        selected = [session for session in sessions if metadata.dates[int(session[0])] == date]
        selected.sort(key=lambda session: float(np.nanmin(metadata.starts[session])))
        if len(selected) >= 2:
            result.append(selected)
    return result


def cluster_sessions(
    sessions: list[np.ndarray],
    probability: np.ndarray,
    decoded: np.ndarray,
    metadata,
    config: GlobalRepeatConfig,
) -> list[list[np.ndarray]]:
    candidates: list[tuple[float, int, int]] = []
    for first in range(len(sessions)):
        stop = min(len(sessions), first + config.maximum_session_rank_distance + 1)
        for second in range(first + 1, stop):
            a, b = sessions[first], sessions[second]
            start_gap = abs(float(np.nanmin(metadata.starts[b])) - float(np.nanmin(metadata.starts[a])))
            if start_gap > config.maximum_start_gap_seconds:
                continue
            length_ratio = min(len(a), len(b)) / max(len(a), len(b))
            if length_ratio < config.minimum_length_ratio or abs(len(a) - len(b)) > 3:
                continue
            _, similarity = align_probabilities(probability[a], probability[b], config.alignment_gap_penalty)
            first_set, second_set = set(map(int, decoded[a])), set(map(int, decoded[b]))
            overlap = len(first_set & second_set) / max(len(first_set | second_set), 1)
            if similarity < config.minimum_probability_similarity or overlap < config.minimum_path_overlap:
                continue
            # A small rank-distance prior resolves ties without imposing a hard
            # 90-second block; interrupted recordings can still reunite.
            score = similarity + overlap + 0.08 * length_ratio - 0.01 * (second - first)
            candidates.append((score, first, second))
    candidates.sort(reverse=True)
    groups: list[list[int]] = []
    membership: dict[int, int] = {}
    for _, first, second in candidates:
        left, right = membership.get(first), membership.get(second)
        if left is None and right is None:
            membership[first] = membership[second] = len(groups)
            groups.append([first, second])
        elif left is not None and right is None and len(groups[left]) < config.maximum_group_size:
            groups[left].append(second)
            membership[second] = left
        elif left is None and right is not None and len(groups[right]) < config.maximum_group_size:
            groups[right].append(first)
            membership[first] = right
        elif left is not None and right is not None and left != right:
            if len(groups[left]) + len(groups[right]) <= config.maximum_group_size:
                merged = groups[left] + groups[right]
                groups[left], groups[right] = merged, []
                for value in merged:
                    membership[value] = left
    return [[sessions[index] for index in group] for group in groups if len(group) >= 2]


def decode_global_repeat(logp, indices, metadata, transition, decoder, config):
    short_sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    decoded = decode_sessions(logp, short_sessions, transition, decoder)
    adjusted = probabilities(logp)
    date_lists = date_session_lists(indices, metadata, decoder.gap_seconds)
    groups = grouped_sessions = grouped_rows = aligned_pairs = 0
    for sessions in date_lists:
        for group in cluster_sessions(sessions, adjusted, decoded, metadata, config):
            reference = max(group, key=len)
            accumulators = {int(index): [adjusted[int(index)]] for index in np.concatenate(group)}
            for session in group:
                if session is reference:
                    continue
                pairs, _ = align_probabilities(adjusted[reference], adjusted[session], config.alignment_gap_penalty)
                for ref_position, other_position in pairs:
                    ref_index, other_index = int(reference[ref_position]), int(session[other_position])
                    accumulators[ref_index].append(adjusted[other_index])
                    accumulators[other_index].append(adjusted[ref_index])
                aligned_pairs += len(pairs)
            for index, values in accumulators.items():
                consensus = np.mean(values, axis=0)
                adjusted[index] = (1.0 - config.consensus_weight) * adjusted[index] + config.consensus_weight * consensus
                adjusted[index] /= adjusted[index].sum()
            groups += 1
            grouped_sessions += len(group)
            grouped_rows += sum(map(len, group))
    prediction = decode_sessions(np.log(np.maximum(adjusted, 1e-12)), short_sessions, transition, decoder)
    return prediction, {
        "date_lists": len(date_lists), "groups": groups, "grouped_sessions": grouped_sessions,
        "grouped_rows": grouped_rows, "aligned_pairs": aligned_pairs,
    }


def configs(fixed: dict[str, Any] | None) -> list[GlobalRepeatConfig]:
    if fixed is not None:
        return [GlobalRepeatConfig(**fixed)]
    return [
        GlobalRepeatConfig(rank, gap, similarity, overlap, length, weight)
        for rank in (3, 5, 8, 12)
        for gap in (180.0, 300.0, 600.0, 1200.0)
        for similarity in (0.78, 0.84, 0.90)
        for overlap in (0.20, 0.40, 0.60)
        for length in (0.65, 0.80)
        for weight in (0.25, 0.50, 0.75)
    ]


def main() -> None:
    args = parse_args()
    (
        sample_ids, labels, base_probability, base_decoded, metadata, indices,
        sessions, transition, decoder, repeat,
    ) = load_protocol(args)
    if args.class_bias:
        logits = np.asarray(np.load(args.base_run.resolve() / "subject_holdout_logits.npy"), dtype=np.float64)
        bias = np.asarray(np.load(args.class_bias.resolve()), dtype=np.float64)
        if bias.shape != (logits.shape[1],):
            raise ValueError("class bias shape mismatch")
        logp = log_softmax_numpy(logits + bias)
    else:
        logp = np.log(np.maximum(base_probability, 1e-12))
    fixed_source = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8")) if args.fixed_summary else None
    candidates = configs(fixed_source["selected_config"] if fixed_source else None)
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for index, config in enumerate(candidates, start=1):
        prediction, grouping = decode_global_repeat(logp, indices, metadata, transition, decoder, config)
        item = {
            "config": asdict(config),
            "metrics": classification_metrics(labels, prediction),
            "rescue_harm": rescue_harm(labels, base_decoded, prediction),
            "grouping": grouping,
        }
        results.append(item)
        key = (item["metrics"]["correct"], item["metrics"]["balanced_accuracy"], item["rescue_harm"]["net"], -item["rescue_harm"]["harm"], -grouping["grouped_rows"])
        if best_key is None or key > best_key:
            best_key, best = key, item
        if index % 250 == 0:
            print(f"evaluated {index}/{len(candidates)}", flush=True)
    assert best is not None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P89_date_global_repeat_decoder_v1",
        "status": "complete",
        "selected_on_current_holdout": args.fixed_summary is None,
        "holdout_users": sorted(args.holdout_users),
        "base": classification_metrics(labels, base_decoded),
        "selected_config": best["config"],
        "best": best,
        "grid_size": len(results),
        "all_candidates": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "all_candidates"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
