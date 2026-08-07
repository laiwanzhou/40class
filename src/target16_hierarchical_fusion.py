from __future__ import annotations

from typing import Any

import numpy as np

from src.engine import classification_metrics


def hierarchical_predictions(
    b2_probabilities: np.ndarray,
    e2_probabilities: np.ndarray,
    target_class_ids: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if b2_probabilities.ndim != 2 or b2_probabilities.shape[1] != 40:
        raise ValueError("B2 probabilities must have shape [N, 40].")
    if e2_probabilities.shape != (len(b2_probabilities), len(target_class_ids)):
        raise ValueError("E2 probabilities do not align with B2 samples and target classes.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    targets = np.asarray(target_class_ids, dtype=np.int64)
    if len(targets) != 16 or len(np.unique(targets)) != 16:
        raise ValueError("Exactly 16 unique target class IDs are required.")

    b2_predictions = b2_probabilities.argmax(axis=1)
    gate = np.isin(b2_predictions, targets)
    fused = b2_predictions.copy()
    if gate.any():
        epsilon = np.finfo(np.float64).tiny
        conditional = b2_probabilities[gate][:, targets]
        conditional /= conditional.sum(axis=1, keepdims=True)
        scores = (1.0 - alpha) * np.log(np.clip(conditional, epsilon, None))
        scores += alpha * np.log(np.clip(e2_probabilities[gate], epsilon, None))
        fused[gate] = targets[scores.argmax(axis=1)]
    return fused, gate


def fusion_metrics(
    labels: np.ndarray,
    b2_predictions: np.ndarray,
    fused_predictions: np.ndarray,
    gate: np.ndarray,
    target_class_ids: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    targets = np.asarray(target_class_ids, dtype=np.int64)
    target_truth = np.isin(labels, targets)
    base_correct = b2_predictions == labels
    fused_correct = fused_predictions == labels
    eligible_error = target_truth & gate & ~base_correct
    protected_correct = target_truth & gate & base_correct
    rescued = int(np.sum(eligible_error & fused_correct))
    harmed = int(np.sum(protected_correct & ~fused_correct))
    metrics = classification_metrics(labels, fused_predictions, 40)
    per_f1 = np.asarray(metrics["per_class_f1"], dtype=np.float64)
    metrics.update(
        {
            "target16_macro_f1": float(per_f1[targets].mean()),
            "rescued": rescued,
            "harmed": harmed,
            "net_rescue": rescued - harmed,
            "gated_samples": int(gate.sum()),
            "gated_true_target_samples": int(np.sum(gate & target_truth)),
            "gated_external_samples": int(np.sum(gate & ~target_truth)),
        }
    )
    correct_delta = int(fused_correct.sum() - base_correct.sum())
    if correct_delta != rescued - harmed:
        raise AssertionError(f"Correct-count delta {correct_delta} != rescued-harmed {rescued - harmed}.")
    if not np.array_equal(fused_predictions[~gate], b2_predictions[~gate]):
        raise AssertionError("Predictions outside the B2 target gate changed.")
    return metrics
