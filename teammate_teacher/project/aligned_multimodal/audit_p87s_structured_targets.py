from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER = PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
DEFAULT_TARGETS = PROJECT_DIR / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87s_holdout1_structured_targets_v1/target_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate already-built P87-S targets; never used during generation."
    )
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--structured-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def aligned_labels(teacher: np.lib.npyio.NpzFile, sample_ids: np.ndarray) -> np.ndarray:
    lookup = {
        str(sample_id): int(label)
        for sample_id, label in zip(
            teacher["oof_sample_ids"].astype(str),
            teacher["oof_labels"].astype(np.int64),
        )
    }
    missing = sorted(set(sample_ids.tolist()) - set(lookup))
    if missing:
        raise ValueError(f"Teacher labels missing {len(missing)} target ids")
    return np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)


def comparison(labels: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    base_correct = base == labels
    candidate_correct = candidate == labels
    return {
        "base": classification_metrics(labels, base),
        "candidate": classification_metrics(labels, candidate),
        "delta_correct": int(candidate_correct.sum() - base_correct.sum()),
        "delta_accuracy_pp": float(100.0 * (candidate_correct.mean() - base_correct.mean())),
        "rescue": int(np.sum(~base_correct & candidate_correct)),
        "harm": int(np.sum(base_correct & ~candidate_correct)),
        "both_wrong_same_prediction": int(
            np.sum(~base_correct & ~candidate_correct & (base == candidate))
        ),
        "both_wrong_different_prediction": int(
            np.sum(~base_correct & ~candidate_correct & (base != candidate))
        ),
        "candidate_exceeds_base": bool(candidate_correct.sum() > base_correct.sum()),
    }


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    true_probability = probability[np.arange(len(labels)), labels]
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[labels]
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 10, endpoint=False):
        upper = lower + 0.1
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float((prediction[selected] == labels[selected]).mean())
                - float(confidence[selected].mean())
            )
    return {
        "nll": float(-np.log(np.maximum(true_probability, 1e-12)).mean()),
        "brier": float(np.square(probability - one_hot).sum(axis=1).mean()),
        "ece_10": float(ece),
        "zero_true_support": int(np.sum(true_probability <= 1e-8)),
        "mean_true_probability": float(true_probability.mean()),
    }


def main() -> None:
    args = parse_args()
    targets = np.load(args.structured_targets.resolve(), allow_pickle=False)
    teacher = np.load(args.teacher_targets.resolve(), allow_pickle=False)
    sample_ids = targets["sample_ids"].astype(str)
    users = targets["users"].astype(str)
    mask = targets["target_mask"].astype(bool)
    labels = aligned_labels(teacher, sample_ids)
    emission = targets["emission_prediction"].astype(np.int64)
    sequence_map = targets["structured_map_prediction"].astype(np.int64)
    marginal_map = targets["structured_marginal_prediction"].astype(np.int64)
    distillation_map = targets["structured_distillation_prediction"].astype(np.int64)
    emission_probability = targets["emission_probability"].astype(np.float64)
    structured_probability = targets["structured_probability"].astype(np.float64)
    distillation_probability = targets[
        "structured_distillation_probability"
    ].astype(np.float64)

    result: dict[str, object] = {
        "protocol": (
            "Post-generation label audit. This process reads pseudo-Test labels, but "
            "its output is never consumed by the structured target builder."
        ),
        "rows": int(mask.sum()),
        "users": sorted(set(users[mask].tolist())),
        "sequence_map_vs_emission": comparison(
            labels[mask], emission[mask], sequence_map[mask]
        ),
        "marginal_map_vs_emission": comparison(
            labels[mask], emission[mask], marginal_map[mask]
        ),
        "distillation_map_vs_emission": comparison(
            labels[mask], emission[mask], distillation_map[mask]
        ),
        "probability_quality": {
            "emission": probability_metrics(
                labels[mask], emission_probability[mask]
            ),
            "raw_structured_posterior": probability_metrics(
                labels[mask], structured_probability[mask]
            ),
            "backed_off_structured_distillation": probability_metrics(
                labels[mask], distillation_probability[mask]
            ),
        },
        "by_subject": {},
        "by_class": {},
    }
    by_subject: dict[str, object] = {}
    for user in sorted(set(users[mask].tolist())):
        selected = mask & (users == user)
        by_subject[user] = comparison(
            labels[selected], emission[selected], sequence_map[selected]
        )
    result["by_subject"] = by_subject

    by_class: dict[str, object] = {}
    for class_id in sorted(map(int, np.unique(labels[mask]))):
        selected = mask & (labels == class_id)
        emission_correct = int(np.sum(emission[selected] == labels[selected]))
        structured_correct = int(np.sum(sequence_map[selected] == labels[selected]))
        by_class[str(class_id)] = {
            "support": int(selected.sum()),
            "emission_correct": emission_correct,
            "structured_correct": structured_correct,
            "delta_correct": structured_correct - emission_correct,
            "rescue": int(
                np.sum(
                    (emission[selected] != labels[selected])
                    & (sequence_map[selected] == labels[selected])
                )
            ),
            "harm": int(
                np.sum(
                    (emission[selected] == labels[selected])
                    & (sequence_map[selected] != labels[selected])
                )
            ),
        }
    result["by_class"] = by_class
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
