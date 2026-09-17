"""Leakage-safe frozen-decoder audit for the three P162 Student cohorts."""

from __future__ import annotations

import csv
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


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TEACHER = RUNS / "p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
OUTPUT = RUNS / "p163_p162_decoder_oof_audit_v1"
CASES = (
    (
        "H1",
        RUNS / "p162_p150_student_h1_e40_v1",
        RUNS / "p87s_holdout1_structured_targets_v1/summary.json",
    ),
    (
        "H2",
        RUNS / "p162_p150_student_h2_e40_v1",
        RUNS / "p87s_confirm2_structured_targets_v1/summary.json",
    ),
    (
        "H3",
        RUNS / "p162_p150_student_h3_e40_v1",
        RUNS / "p90_p87s_h3_structured_targets_v1/summary.json",
    ),
)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    value = logits.astype(np.float64)
    value -= np.logaddexp.reduce(value, axis=1, keepdims=True)
    return value


def main() -> None:
    teacher = np.load(TEACHER, allow_pickle=False)
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    metadata = align_metadata(METADATA, all_ids)
    payload = {"stage": "P163_P162_leakage_safe_decoder_OOF", "cohorts": {}}
    raw_correct = decoded_correct = rows_total = 0
    for name, run, target_summary_path in CASES:
        logits = np.load(run / "subject_holdout_logits.npy").astype(np.float64)
        with (run / "subject_holdout_predictions.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        sample_ids = np.asarray([row["sample_id"] for row in rows]).astype(str)
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        users = np.asarray([row["user_id"] for row in rows]).astype(str)
        holdout = np.isin(metadata.users, sorted(set(users.tolist())))
        held_indices = np.flatnonzero(holdout)
        source_indices = np.flatnonzero(~holdout)
        if set(all_ids[held_indices].tolist()) != set(sample_ids.tolist()):
            raise RuntimeError(f"{name} held universe differs")
        summary = json.loads(target_summary_path.read_text(encoding="utf-8"))
        selected = summary["selected_config"]
        config = DecoderConfig(
            gap_seconds=float(selected["gap_seconds"]),
            transition_weight=float(selected["transition_weight"]),
            trigram_backoff=float(selected["trigram_backoff"]),
            beam_width=int(selected["beam_width"]),
        )
        transition = fit_transition_model(
            all_labels[source_indices],
            build_sessions(
                np.arange(len(source_indices)),
                align_metadata(METADATA, all_ids[source_indices]),
                config.gap_seconds,
                grouping="known_user",
            ),
            num_classes=40,
            trigram_backoff=config.trigram_backoff,
        )
        held_metadata = align_metadata(METADATA, sample_ids)
        sessions = build_sessions(
            np.arange(len(sample_ids)), held_metadata, config.gap_seconds, grouping="anonymous_date"
        )
        raw = logits.argmax(axis=1).astype(np.int64)
        decoded = decode_sessions(log_softmax(logits), sessions, transition, config)
        raw_ok = raw == labels
        decoded_ok = decoded == labels
        payload["cohorts"][name] = {
            "rows": len(labels),
            "raw": classification_metrics(labels, raw),
            "decoded": classification_metrics(labels, decoded),
            "changed": int(np.sum(raw != decoded)),
            "rescue": int(np.sum(~raw_ok & decoded_ok)),
            "harm": int(np.sum(raw_ok & ~decoded_ok)),
            "net": int(decoded_ok.sum() - raw_ok.sum()),
        }
        raw_correct += int(raw_ok.sum())
        decoded_correct += int(decoded_ok.sum())
        rows_total += len(labels)
    payload["aggregate"] = {
        "rows": rows_total,
        "raw_correct": raw_correct,
        "raw_accuracy": raw_correct / rows_total,
        "decoded_correct": decoded_correct,
        "decoded_accuracy": decoded_correct / rows_total,
        "decoder_net": decoded_correct - raw_correct,
        "preferred": "decoded" if decoded_correct > raw_correct else "raw",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
