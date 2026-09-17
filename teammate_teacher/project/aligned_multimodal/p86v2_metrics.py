from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


CLASS_IDS = np.arange(40, dtype=np.int64)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index + 1 == bins:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return result


def emission_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    users: Iterable[str],
) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    users = np.asarray(list(map(str, users)))
    if logits.shape != (len(labels), 40) or users.shape != (len(labels),):
        raise ValueError("logits, labels and users have incompatible shapes")
    probabilities = softmax(logits)
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(40, dtype=np.float64)[labels]
    true_probability = probabilities[np.arange(len(labels)), labels]
    per_subject = {
        user: float(accuracy_score(labels[users == user], prediction[users == user]))
        for user in sorted(set(users.tolist()))
    }
    per_class = {}
    for class_id in CLASS_IDS:
        mask = labels == class_id
        per_class[str(int(class_id))] = {
            "support": int(mask.sum()),
            "accuracy": float((prediction[mask] == labels[mask]).mean()) if mask.any() else None,
        }
    return {
        "correct": int((prediction == labels).sum()),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(
            f1_score(labels, prediction, labels=CLASS_IDS, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            recall_score(labels, prediction, labels=CLASS_IDS, average="macro", zero_division=0)
        ),
        "worst_subject_accuracy": min(per_subject.values()),
        "negative_log_likelihood": float(-np.log(true_probability.clip(1e-12)).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15": expected_calibration_error(probabilities, labels, bins=15),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "mean_entropy": float(
            -(probabilities * np.log(probabilities.clip(1e-12))).sum(axis=1).mean()
        ),
        "per_subject_accuracy": per_subject,
        "per_class": per_class,
    }


def rescue_harm(
    candidate_logits: np.ndarray,
    baseline_logits: np.ndarray,
    labels: np.ndarray,
    users: Iterable[str],
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    users = np.asarray(list(map(str, users)))
    candidate = np.asarray(candidate_logits).argmax(axis=1)
    baseline = np.asarray(baseline_logits).argmax(axis=1)
    rescued = (baseline != labels) & (candidate == labels)
    harmed = (baseline == labels) & (candidate != labels)
    changed = candidate != baseline
    per_subject = {}
    for user in sorted(set(users.tolist())):
        mask = users == user
        per_subject[user] = {
            "rescued": int((rescued & mask).sum()),
            "harmed": int((harmed & mask).sum()),
            "net": int((rescued & mask).sum() - (harmed & mask).sum()),
            "changed": int((changed & mask).sum()),
        }
    per_class = {}
    for class_id in CLASS_IDS:
        mask = labels == class_id
        per_class[str(int(class_id))] = {
            "rescued": int((rescued & mask).sum()),
            "harmed": int((harmed & mask).sum()),
            "net": int((rescued & mask).sum() - (harmed & mask).sum()),
            "support": int(mask.sum()),
        }
    return {
        "rescued": int(rescued.sum()),
        "harmed": int(harmed.sum()),
        "net": int(rescued.sum() - harmed.sum()),
        "changed": int(changed.sum()),
        "per_subject": per_subject,
        "per_class": per_class,
    }
