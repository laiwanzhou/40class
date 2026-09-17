"""H2-selected sequence decoding audit for the P91 hierarchical teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import log_softmax

from audit_p87_sequence_decoder import (
    DEFAULT_TRAIN_METADATA,
    DecoderConfig,
    align_metadata,
    build_sessions,
    decode_sessions,
    fit_transition_model,
)
from p91_hierarchical_multimodal_teacher import audit, build_data, load_npz


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_INPUT = PROJECT / "runs/p91_hierarchical_multimodal_h3_v3"
DEFAULT_OUTPUT = PROJECT / "runs/p91_hierarchical_sequence_h3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--beam-width", type=int, default=50)
    return parser.parse_args()


def blended_log_probability(
    logits: np.ndarray, teacher_prediction: np.ndarray, weight: float
) -> np.ndarray:
    neural_logp = log_softmax(np.asarray(logits, dtype=np.float64), axis=1)
    teacher = np.full((len(logits), 40), 0.06 / 39.0, dtype=np.float64)
    teacher[np.arange(len(logits)), teacher_prediction.astype(np.int64)] = 0.94
    values = weight * neural_logp + (1.0 - weight) * np.log(teacher)
    return log_softmax(values, axis=1)


def user_stability(
    labels: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    users: np.ndarray,
) -> tuple[int, int, dict[str, int]]:
    nets = {
        str(user): int(np.sum(candidate[users == user] == labels[users == user]))
        - int(np.sum(base[users == user] == labels[users == user]))
        for user in np.unique(users)
    }
    return int(np.sum(np.asarray(list(nets.values())) < 0)), min(nets.values(), default=0), nets


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inner = load_npz(args.input / "inner_predictions.npz")
    target = load_npz(args.input / "predictions.npz")
    data = build_data()
    metadata = align_metadata(DEFAULT_TRAIN_METADATA, data.sample_ids)
    id_to_row = {str(sample_id): index for index, sample_id in enumerate(data.sample_ids)}
    h2 = np.asarray([id_to_row[str(value)] for value in inner["sample_ids"]], dtype=np.int64)
    h3 = np.asarray([id_to_row[str(value)] for value in target["sample_ids"]], dtype=np.int64)
    source_train = np.concatenate(
        (
            data.boundaries["H1_selection"],
            data.boundaries["E0_p87_sequence_source"],
        )
    )
    final_train = np.concatenate(
        (
            data.boundaries["H1_selection"],
            data.boundaries["H2_confirmation"],
            data.boundaries["E0_p87_sequence_source"],
        )
    )
    inner_weight = float(np.asarray(inner["selected_constant_weight"]).item())
    target_weight = float(np.asarray(target["selected_blend_weight"]).item())
    inner_logp = blended_log_probability(
        inner["direct_logits"], inner["teacher_prediction"], inner_weight
    )
    target_logp = blended_log_probability(
        target["direct_logits"], target["base_prediction"], target_weight
    )
    inner_base = inner_logp.argmax(axis=1)
    target_base = target_logp.argmax(axis=1)
    inner_full_logp = np.zeros((len(data.labels), 40), dtype=np.float64)
    inner_full_logp[h2] = inner_logp
    target_full_logp = np.zeros((len(data.labels), 40), dtype=np.float64)
    target_full_logp[h3] = target_logp

    rows: list[dict[str, Any]] = []
    for gap in (20.0, 30.0, 45.0):
        fit_sessions = build_sessions(source_train, metadata, gap, grouping="known_user")
        validation_sessions = build_sessions(h2, metadata, gap, grouping="anonymous_date")
        for backoff in (1.0, 2.0, 5.0):
            transition = fit_transition_model(
                data.labels, fit_sessions, num_classes=40, trigram_backoff=backoff
            )
            for weight in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
                config = DecoderConfig(gap, weight, backoff, args.beam_width)
                decoded_full = decode_sessions(
                    inner_full_logp, validation_sessions, transition, config
                )
                prediction = decoded_full[h2]
                negative_users, worst_user, user_nets = user_stability(
                    data.labels[h2], inner_base, prediction, data.users[h2]
                )
                rows.append(
                    {
                        "gap_seconds": gap,
                        "transition_weight": weight,
                        "trigram_backoff": backoff,
                        "negative_users": negative_users,
                        "worst_user_net": worst_user,
                        "user_nets": user_nets,
                        **audit(data.labels[h2], inner_base, prediction),
                    }
                )
    base_correct = int(np.sum(inner_base == data.labels[h2]))
    eligible = [
        row
        for row in rows
        if row["correct"] > base_correct
        and row["negative_users"] <= 1
        and row["worst_user_net"] >= -1
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["correct"],
                -row["harm"],
                -row["negative_users"],
                row["worst_user_net"],
                -abs(row["transition_weight"] - 0.25),
            ),
        )
        config = DecoderConfig(
            selected["gap_seconds"],
            selected["transition_weight"],
            selected["trigram_backoff"],
            args.beam_width,
        )
        fit_sessions = build_sessions(
            final_train, metadata, config.gap_seconds, grouping="known_user"
        )
        target_sessions = build_sessions(
            h3, metadata, config.gap_seconds, grouping="anonymous_date"
        )
        transition = fit_transition_model(
            data.labels,
            fit_sessions,
            num_classes=40,
            trigram_backoff=config.trigram_backoff,
        )
        decoded_full = decode_sessions(target_full_logp, target_sessions, transition, config)
        target_prediction = decoded_full[h3]
        selection_note = "positive stable H2 sequence configuration applied to H3"
    else:
        selected = None
        target_prediction = target_base
        selection_note = "no sequence configuration improved H2 stability gate; kept fusion"
    report = {
        "protocol": (
            "Transition model fit on H1+embargo for H2 selection, then H1+H2+embargo "
            "for one untouched H3 decode."
        ),
        "inner_fusion": audit(
            data.labels[h2], inner["teacher_prediction"], inner_base
        ),
        "selected": selected,
        "selection_note": selection_note,
        "target_fusion": audit(data.labels[h3], target["base_prediction"], target_base),
        "target_sequence": audit(
            data.labels[h3], target["base_prediction"], target_prediction
        ),
        "target_sequence_vs_fusion": audit(
            data.labels[h3], target_base, target_prediction
        ),
        "top_h2_configs": sorted(
            rows,
            key=lambda row: (
                row["correct"], -row["harm"], -row["negative_users"], row["worst_user_net"]
            ),
            reverse=True,
        )[:20],
    }
    np.savez_compressed(
        output / "predictions.npz",
        sample_ids=data.sample_ids[h3],
        labels=data.labels[h3],
        p90_prediction=target["base_prediction"],
        fusion_prediction=target_base,
        sequence_prediction=target_prediction,
        fusion_log_probability=target_logp.astype(np.float32),
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
