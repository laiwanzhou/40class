from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_p87_sequence_decoder import (
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
    fit_transition_model,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER = PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
DEFAULT_TARGETS = (
    PROJECT_DIR / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
)
DEFAULT_TARGET_SUMMARY = (
    PROJECT_DIR / "runs/p87s_holdout1_structured_targets_v1/summary.json"
)
DEFAULT_METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-config raw versus +P87 decoder audit for one P87-S Student run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--structured-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--target-summary", type=Path, default=DEFAULT_TARGET_SUMMARY)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def calibration_metrics(
    labels: np.ndarray, probability: np.ndarray, bins: int = 10
) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = prediction == labels
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    nll = -np.log(np.maximum(probability[np.arange(len(labels)), labels], 1e-12)).mean()
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[labels]
    brier = np.square(probability - one_hot).sum(axis=1).mean()
    return {"nll": float(nll), "brier": float(brier), "ece_10": float(ece)}


def comparison(
    labels: np.ndarray, raw: np.ndarray, decoded: np.ndarray
) -> dict[str, Any]:
    raw_correct = raw == labels
    decoded_correct = decoded == labels
    return {
        "raw": classification_metrics(labels, raw),
        "decoded": classification_metrics(labels, decoded),
        "delta_correct": int(decoded_correct.sum() - raw_correct.sum()),
        "delta_accuracy_pp": float(
            100.0 * (decoded_correct.mean() - raw_correct.mean())
        ),
        "rescue": int(np.sum(~raw_correct & decoded_correct)),
        "harm": int(np.sum(raw_correct & ~decoded_correct)),
    }


def teacher_copy_audit(
    labels: np.ndarray, student: np.ndarray, teacher: np.ndarray
) -> dict[str, Any]:
    teacher_correct = teacher == labels
    student_correct = student == labels
    return {
        "agreement": float(np.mean(student == teacher)),
        "teacher_errors_copied": int(np.sum(~teacher_correct & (student == teacher))),
        "student_correct_when_teacher_wrong": int(np.sum(~teacher_correct & student_correct)),
        "student_exceeds_teacher_hard_correct": bool(
            student_correct.sum() > teacher_correct.sum()
        ),
        "student_correct": int(student_correct.sum()),
        "teacher_correct": int(teacher_correct.sum()),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    logits = np.load(run_dir / "subject_holdout_logits.npy").astype(np.float64)
    with (run_dir / "subject_holdout_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = np.asarray([row["sample_id"] for row in rows]).astype(str)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in rows]).astype(str)
    if len(logits) != len(sample_ids):
        raise ValueError("Student logits and prediction rows differ")

    target = np.load(args.structured_targets.resolve(), allow_pickle=False)
    target_ids = target["sample_ids"].astype(str)
    target_mask = target["target_mask"].astype(bool)
    expected_ids = target_ids[target_mask]
    if sample_ids.tolist() != expected_ids.tolist():
        raise ValueError("Student row order differs from the frozen pseudo-Test target order")
    target_lookup = {value: index for index, value in enumerate(target_ids)}
    target_index = np.asarray([target_lookup[value] for value in sample_ids], dtype=np.int64)
    structured_teacher = target["structured_map_prediction"].astype(np.int64)[
        target_index
    ]
    emission_teacher = target["emission_prediction"].astype(np.int64)[target_index]

    teacher = np.load(args.teacher_targets.resolve(), allow_pickle=False)
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64).copy()
    all_metadata = align_metadata(args.train_metadata.resolve(), all_ids)
    holdout_users = sorted(set(users.tolist()))
    all_holdout = np.isin(all_metadata.users, holdout_users)
    if int(all_holdout.sum()) != len(sample_ids):
        raise ValueError("Decoder holdout subject scope differs from Student evaluation")
    train_indices = np.flatnonzero(~all_holdout)
    holdout_indices = np.flatnonzero(all_holdout)
    # Guard the fixed decoder fit against accidental pseudo-Test label use.
    all_labels[all_holdout] = -10_000
    target_summary = json.loads(
        args.target_summary.resolve().read_text(encoding="utf-8")
    )
    selected = target_summary["selected_config"]
    config = DecoderConfig(
        gap_seconds=float(selected["gap_seconds"]),
        transition_weight=float(selected["transition_weight"]),
        trigram_backoff=float(selected["trigram_backoff"]),
        beam_width=int(selected["beam_width"]),
    )
    train_sessions = build_sessions(
        train_indices, all_metadata, config.gap_seconds, grouping="known_user"
    )
    model = fit_transition_model(
        all_labels,
        train_sessions,
        num_classes=logits.shape[1],
        trigram_backoff=config.trigram_backoff,
    )
    holdout_sessions_global = build_sessions(
        holdout_indices,
        all_metadata,
        config.gap_seconds,
        grouping="anonymous_date",
    )
    local_lookup = {global_index: local for local, global_index in enumerate(holdout_indices)}
    holdout_sessions = [
        np.asarray([local_lookup[int(index)] for index in session], dtype=np.int64)
        for session in holdout_sessions_global
    ]
    probability = softmax(logits)
    log_probability = np.log(np.maximum(probability, 1e-300))
    raw_prediction = probability.argmax(axis=1).astype(np.int64)
    decoded_prediction = decode_sessions(log_probability, holdout_sessions, model, config)

    result: dict[str, Any] = {
        "stage": "P87-S Student raw and frozen-decoder audit",
        "run_dir": str(run_dir),
        "protocol": (
            "Decoder config was frozen by inner-LOSO on the other 14 subjects. "
            "Pseudo-Test labels are overwritten by -10000 before transition fitting."
        ),
        "selected_decoder_config": selected,
        "rows": len(labels),
        "subjects": holdout_users,
        "anonymous_sessions": len(holdout_sessions),
        "raw_vs_decoder": comparison(labels, raw_prediction, decoded_prediction),
        "raw_calibration": calibration_metrics(labels, probability),
        "raw_vs_emission_teacher": teacher_copy_audit(
            labels, raw_prediction, emission_teacher
        ),
        "raw_vs_structured_teacher": teacher_copy_audit(
            labels, raw_prediction, structured_teacher
        ),
        "decoded_vs_structured_teacher": teacher_copy_audit(
            labels, decoded_prediction, structured_teacher
        ),
        "by_subject": {},
        "by_class": {},
    }
    by_subject: dict[str, Any] = {}
    for user in holdout_users:
        selected_rows = users == user
        by_subject[user] = comparison(
            labels[selected_rows],
            raw_prediction[selected_rows],
            decoded_prediction[selected_rows],
        )
    result["by_subject"] = by_subject
    by_class: dict[str, Any] = {}
    for class_id in sorted(map(int, np.unique(labels))):
        selected_rows = labels == class_id
        raw_correct = int(np.sum(raw_prediction[selected_rows] == class_id))
        decoded_correct = int(np.sum(decoded_prediction[selected_rows] == class_id))
        by_class[str(class_id)] = {
            "support": int(selected_rows.sum()),
            "raw_correct": raw_correct,
            "decoded_correct": decoded_correct,
            "delta_correct": decoded_correct - raw_correct,
            "rescue": int(
                np.sum(
                    (raw_prediction[selected_rows] != class_id)
                    & (decoded_prediction[selected_rows] == class_id)
                )
            ),
            "harm": int(
                np.sum(
                    (raw_prediction[selected_rows] == class_id)
                    & (decoded_prediction[selected_rows] != class_id)
                )
            ),
        }
    result["by_class"] = by_class

    (run_dir / "decoder_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_dir / "decoded_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "user_id",
                "label",
                "raw_prediction",
                "decoded_prediction",
                "emission_teacher_prediction",
                "structured_teacher_prediction",
            ],
        )
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "user_id": users[index],
                    "label": int(labels[index]),
                    "raw_prediction": int(raw_prediction[index]),
                    "decoded_prediction": int(decoded_prediction[index]),
                    "emission_teacher_prediction": int(emission_teacher[index]),
                    "structured_teacher_prediction": int(structured_teacher[index]),
                }
            )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
