from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two aligned OOF logit files.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logit-key", default="logits")
    parser.add_argument("--reference-logit-key", default=None)
    parser.add_argument("--candidate-logit-key", default=None)
    return parser.parse_args()


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    args = parse_args()
    reference_path = args.reference.resolve()
    candidate_path = args.candidate.resolve()
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    reference_key = args.reference_logit_key or args.logit_key
    candidate_key = args.candidate_logit_key or args.logit_key
    reference_ids = reference["sample_ids"].astype(str)
    candidate_ids = candidate["sample_ids"].astype(str)
    if not np.array_equal(reference_ids, candidate_ids):
        raise ValueError("Sample IDs differ or are not in the same order.")
    reference_labels = reference["labels"].astype(np.int64)
    candidate_labels = candidate["labels"].astype(np.int64)
    if not np.array_equal(reference_labels, candidate_labels):
        raise ValueError("Labels differ.")
    reference_logits = reference[reference_key].astype(np.float64)
    candidate_logits = candidate[candidate_key].astype(np.float64)
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("Logit shapes differ.")
    difference = candidate_logits - reference_logits
    reference_predictions = reference_logits.argmax(axis=1)
    candidate_predictions = candidate_logits.argmax(axis=1)
    summary = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_logit_key": reference_key,
        "candidate_logit_key": candidate_key,
        "samples": len(reference_ids),
        "mean_abs_logit_difference": float(np.abs(difference).mean()),
        "max_abs_logit_difference": float(np.abs(difference).max()),
        "changed_predictions": int(
            np.count_nonzero(reference_predictions != candidate_predictions)
        ),
        "changed_prediction_rate": float(
            np.mean(reference_predictions != candidate_predictions)
        ),
        "reference_metrics": metrics(reference_labels, reference_logits),
        "candidate_metrics": metrics(candidate_labels, candidate_logits),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
