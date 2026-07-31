from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(40),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(40), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=np.arange(40), average="weighted", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_support": support.tolist(),
    }
