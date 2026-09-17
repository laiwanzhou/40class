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
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
    fit_transition_model,
)
from p88_train_depth_residual import log_softmax_numpy, read_rows, rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "runs/p87s_fusion_holdout1_c7_structured12_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_aligned_repeat_h1_v1"


@dataclass(frozen=True)
class AlignedRepeatConfig:
    medium_gap_seconds: float
    consensus_weight: float
    probability_similarity: float
    path_overlap: float
    alignment_gap_penalty: float
    maximum_group_size: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P88 variable-length repeated-take consensus on frozen P87-S.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--fixed-config-summary", type=Path)
    parser.add_argument("--medium-gaps", type=float, nargs="+", default=(60.0, 75.0, 90.0))
    parser.add_argument("--consensus-weights", type=float, nargs="+", default=(0.25, 0.50, 0.75, 1.0))
    parser.add_argument("--probability-similarities", type=float, nargs="+", default=(0.70, 0.80, 0.88))
    parser.add_argument("--path-overlaps", type=float, nargs="+", default=(0.30, 0.50, 0.70))
    parser.add_argument("--alignment-gap-penalties", type=float, nargs="+", default=(0.20, 0.40, 0.60))
    return parser.parse_args()


def probability(log_probability: np.ndarray) -> np.ndarray:
    return np.exp(log_probability - np.logaddexp.reduce(log_probability, axis=1, keepdims=True))


def align_probabilities(
    first: np.ndarray, second: np.ndarray, gap_penalty: float
) -> tuple[list[tuple[int, int]], float]:
    similarity = np.sqrt(first[:, None, :] * second[None, :, :]).sum(axis=2)
    rows, columns = similarity.shape
    score = np.full((rows + 1, columns + 1), -np.inf, dtype=np.float64)
    trace = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    score[0, 0] = 0.0
    for i in range(1, rows + 1):
        score[i, 0] = score[i - 1, 0] - gap_penalty; trace[i, 0] = 1
    for j in range(1, columns + 1):
        score[0, j] = score[0, j - 1] - gap_penalty; trace[0, j] = 2
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            values = (
                score[i - 1, j - 1] + similarity[i - 1, j - 1],
                score[i - 1, j] - gap_penalty,
                score[i, j - 1] - gap_penalty,
            )
            trace[i, j] = int(np.argmax(values)); score[i, j] = max(values)
    pairs: list[tuple[int, int]] = []
    i, j = rows, columns
    while i or j:
        action = int(trace[i, j])
        if i and j and action == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and (not j or action == 1):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    mean_similarity = float(np.mean([similarity[i, j] for i, j in pairs])) if pairs else 0.0
    return pairs, mean_similarity


def short_sessions_in_blocks(
    indices: np.ndarray, metadata, short_gap: float, medium_gap: float
) -> list[list[np.ndarray]]:
    short = build_sessions(indices, metadata, short_gap, "anonymous_date")
    blocks = build_sessions(indices, metadata, medium_gap, "anonymous_date")
    result = []
    for block in blocks:
        members = set(map(int, block))
        contained = [session for session in short if all(int(x) in members for x in session)]
        contained.sort(key=lambda session: float(np.nanmin(metadata.starts[session])))
        if len(contained) >= 2:
            result.append(contained)
    return result


def group_sessions(
    sessions: list[np.ndarray],
    probabilities: np.ndarray,
    base_prediction: np.ndarray,
    config: AlignedRepeatConfig,
) -> list[list[np.ndarray]]:
    candidates: list[tuple[float, int, int]] = []
    for first in range(len(sessions)):
        for second in range(first + 1, min(first + 4, len(sessions))):
            length_ratio = min(len(sessions[first]), len(sessions[second])) / max(len(sessions[first]), len(sessions[second]))
            if length_ratio < 0.65 or abs(len(sessions[first]) - len(sessions[second])) > 3:
                continue
            _, similarity = align_probabilities(
                probabilities[sessions[first]], probabilities[sessions[second]], config.alignment_gap_penalty
            )
            first_set = set(map(int, base_prediction[sessions[first]])); second_set = set(map(int, base_prediction[sessions[second]]))
            overlap = len(first_set & second_set) / max(len(first_set | second_set), 1)
            if similarity >= config.probability_similarity and overlap >= config.path_overlap:
                candidates.append((similarity + overlap, first, second))
    candidates.sort(reverse=True)
    groups: list[list[int]] = []; membership: dict[int, int] = {}
    for _, first, second in candidates:
        a = membership.get(first); b = membership.get(second)
        if a is None and b is None:
            membership[first] = membership[second] = len(groups); groups.append([first, second])
        elif a is not None and b is None and len(groups[a]) < config.maximum_group_size:
            groups[a].append(second); membership[second] = a
        elif a is None and b is not None and len(groups[b]) < config.maximum_group_size:
            groups[b].append(first); membership[first] = b
        elif a is not None and b is not None and a != b and len(groups[a]) + len(groups[b]) <= config.maximum_group_size:
            merged = groups[a] + groups[b]; groups[a] = merged; groups[b] = []
            for value in merged: membership[value] = a
    return [[sessions[index] for index in group] for group in groups if len(group) >= 2]


def decode_aligned_repeat(
    log_probability: np.ndarray,
    indices: np.ndarray,
    metadata,
    transition,
    decoder_config: DecoderConfig,
    config: AlignedRepeatConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    short_sessions = build_sessions(indices, metadata, decoder_config.gap_seconds, "anonymous_date")
    base = decode_sessions(log_probability, short_sessions, transition, decoder_config)
    adjusted_probability = probability(log_probability)
    blocks = short_sessions_in_blocks(indices, metadata, decoder_config.gap_seconds, config.medium_gap_seconds)
    grouped_sessions = grouped_rows = aligned_pairs = 0
    for sessions in blocks:
        for group in group_sessions(sessions, adjusted_probability, base, config):
            reference = max(group, key=len)
            accumulators = {int(index): [adjusted_probability[int(index)]] for index in np.concatenate(group)}
            for session in group:
                if session is reference:
                    continue
                pairs, _ = align_probabilities(
                    adjusted_probability[reference], adjusted_probability[session], config.alignment_gap_penalty
                )
                for ref_position, other_position in pairs:
                    ref_index = int(reference[ref_position]); other_index = int(session[other_position])
                    accumulators[ref_index].append(adjusted_probability[other_index])
                    accumulators[other_index].append(adjusted_probability[ref_index])
                aligned_pairs += len(pairs)
            for index, values in accumulators.items():
                consensus = np.mean(values, axis=0)
                adjusted_probability[index] = (
                    (1.0 - config.consensus_weight) * adjusted_probability[index]
                    + config.consensus_weight * consensus
                )
                adjusted_probability[index] /= adjusted_probability[index].sum()
            grouped_sessions += len(group); grouped_rows += sum(map(len, group))
    adjusted_logp = np.log(np.maximum(adjusted_probability, 1e-12))
    prediction = decode_sessions(adjusted_logp, short_sessions, transition, decoder_config)
    return prediction, {"multi_session_blocks": len(blocks), "grouped_sessions": grouped_sessions, "grouped_rows": grouped_rows, "aligned_pairs": aligned_pairs}


def grid(args: argparse.Namespace) -> list[AlignedRepeatConfig]:
    return [AlignedRepeatConfig(float(g), float(w), float(s), float(o), float(p)) for g in args.medium_gaps for w in args.consensus_weights for s in args.probability_similarities for o in args.path_overlaps for p in args.alignment_gap_penalties]


def main() -> None:
    args = parse_args(); run_dir = args.run_dir.resolve()
    rows = read_rows(run_dir / "subject_holdout_predictions.csv")
    sample_ids = np.asarray([row["sample_id"] for row in rows]); labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64); users = np.asarray([row["user_id"] for row in rows])
    if set(users) != set(args.holdout_users): raise RuntimeError("holdout user mismatch")
    logits = np.asarray(np.load(run_dir / "subject_holdout_logits.npy"), dtype=np.float64); logp = log_softmax_numpy(logits)
    audit = json.loads((run_dir / "decoder_audit.json").read_text(encoding="utf-8")); frozen = audit["selected_decoder_config"]
    decoder = DecoderConfig(float(frozen["gap_seconds"]), float(frozen["transition_weight"]), float(frozen["trigram_backoff"]), int(frozen["beam_width"]))
    with np.load(args.teacher_targets.resolve(), allow_pickle=False) as teacher:
        all_ids = teacher["oof_sample_ids"].astype(str); all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(args.train_metadata, all_ids); fit = np.flatnonzero(~np.isin(all_metadata.users, args.holdout_users))
    transition = fit_transition_model(all_labels, build_sessions(fit, all_metadata, decoder.gap_seconds, "known_user"), 40, decoder.trigram_backoff)
    metadata = align_metadata(args.train_metadata, sample_ids); indices = np.arange(len(labels))
    baseline = decode_sessions(logp, build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date"), transition, decoder)
    expected = int(audit["raw_vs_decoder"]["decoded"]["correct"])
    if int(np.sum(baseline == labels)) != expected: raise RuntimeError("failed to reproduce P87")
    selected_here = args.fixed_config_summary is None
    if args.fixed_config_summary:
        source = json.loads(args.fixed_config_summary.resolve().read_text(encoding="utf-8")); candidates = [AlignedRepeatConfig(**source["best"]["config"])]
    else: candidates = grid(args)
    best = None; best_key = None; results = []
    for config in candidates:
        prediction, grouping = decode_aligned_repeat(logp, indices, metadata, transition, decoder, config)
        metrics = classification_metrics(labels, prediction); changes = rescue_harm(labels, baseline, prediction)
        result = {"config": asdict(config), "metrics": metrics, "rescue_harm": changes, "grouping": grouping}; results.append(result)
        key = (metrics["correct"], changes["net"], -config.consensus_weight, config.probability_similarity)
        if best_key is None or key > best_key: best_key = key; best = result
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = {"stage":"P88_variable_length_aligned_repeat_holdout","status":"complete","selected_on_current_holdout":selected_here,"holdout_users":sorted(args.holdout_users),"baseline":classification_metrics(labels,baseline),"best":best,"grid_size":len(candidates),"all_candidates":results,"decoder_config":asdict(decoder)}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
