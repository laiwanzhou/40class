from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    row_totals = matrix.sum(axis=1)
    recall = np.divide(
        np.diag(matrix),
        row_totals,
        out=np.zeros(40, dtype=np.float64),
        where=row_totals != 0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(40), average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class_recall": recall.tolist(),
    }
