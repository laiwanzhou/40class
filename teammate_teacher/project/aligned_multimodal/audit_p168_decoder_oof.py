"""Leakage-safe decoder audit for the frozen P168 cross-fit predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
    fit_transition_model,
)
from deploy_p168_historical_micro_union import frozen_oof
from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TEACHER = RUNS / "p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
OUTPUT = RUNS / "p168_decoder_oof_audit_v1"
CASES = (
    (
        "H1_selection",
        RUNS / "p87s_holdout1_structured_targets_v1/summary.json",
    ),
    (
        "H2_confirmation",
        RUNS / "p87s_confirm2_structured_targets_v1/summary.json",
    ),
    (
        "H3_independent_fold0",
        RUNS / "p90_p87s_h3_structured_targets_v1/summary.json",
    ),
)


def main() -> None:
    labels, base, prediction, gains = frozen_oof()
    if len(prediction) != 2470:
        raise RuntimeError("P168 OOF universe changed")
    splits = load_splits()
    with np.load(TEACHER, allow_pickle=False) as saved:
        all_ids = saved["oof_sample_ids"].astype(str)
        all_labels = saved["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(METADATA, all_ids)
    offset = 0
    report = {
        "stage": "P168_leakage_safe_decoder_OOF",
        "protocol": (
            "Decode the frozen P168 cross-fit hard targets using the exact 0.9805/0.0005 "
            "distillation distribution. Each held cohort transition is fit only on "
            "non-held Train subjects; no held label selects a decoder setting."
        ),
        "cohorts": {},
    }
    raw_correct = decoded_correct = changed_total = 0
    for name, config_path in CASES:
        split = splits[name]
        count = len(split.labels)
        held_prediction = prediction[offset : offset + count]
        held_labels = labels[offset : offset + count]
        if not np.array_equal(held_labels, split.labels):
            raise RuntimeError(f"{name}: P168/load_splits label order differs")
        offset += count
        selected = json.loads(config_path.read_text(encoding="utf-8"))["selected_config"]
        config = DecoderConfig(
            gap_seconds=float(selected["gap_seconds"]),
            transition_weight=float(selected["transition_weight"]),
            trigram_backoff=float(selected["trigram_backoff"]),
            beam_width=int(selected["beam_width"]),
        )
        held_users = sorted(set(split.users.astype(str).tolist()))
        source = ~np.isin(all_metadata.users.astype(str), held_users)
        source_ids = all_ids[source]
        source_labels = all_labels[source]
        transition = fit_transition_model(
            source_labels,
            build_sessions(
                np.arange(len(source_ids), dtype=np.int64),
                align_metadata(METADATA, source_ids),
                config.gap_seconds,
                grouping="known_user",
            ),
            40,
            config.trigram_backoff,
        )
        probability = np.full((count, 40), 0.0005, dtype=np.float64)
        probability[np.arange(count), held_prediction] = 0.9805
        sessions = build_sessions(
            np.arange(count, dtype=np.int64),
            align_metadata(METADATA, split.sample_ids.astype(str)),
            config.gap_seconds,
            grouping="anonymous_date",
        )
        decoded = decode_sessions(
            np.log(np.maximum(probability, 1e-12)), sessions, transition, config
        )
        raw_ok = held_prediction == held_labels
        decoded_ok = decoded == held_labels
        report["cohorts"][name] = {
            "rows": count,
            "raw": classification_metrics(held_labels, held_prediction),
            "decoded": classification_metrics(held_labels, decoded),
            "changed": int(np.sum(decoded != held_prediction)),
            "rescue": int(np.sum(~raw_ok & decoded_ok)),
            "harm": int(np.sum(raw_ok & ~decoded_ok)),
            "net": int(decoded_ok.sum() - raw_ok.sum()),
        }
        raw_correct += int(raw_ok.sum())
        decoded_correct += int(decoded_ok.sum())
        changed_total += int(np.sum(decoded != held_prediction))
    report["aggregate"] = {
        "rows": len(prediction),
        "raw_correct": raw_correct,
        "raw_accuracy": raw_correct / len(prediction),
        "decoded_correct": decoded_correct,
        "decoded_accuracy": decoded_correct / len(prediction),
        "changed": changed_total,
        "decoder_net": decoded_correct - raw_correct,
        "preferred": "decoded" if decoded_correct > raw_correct else "raw",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
