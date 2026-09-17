from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from probe_p27r3_incremental_information import metric_bundle, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
BASE = (
    PROJECT_DIR
    / "runs"
    / "p27_strong_inner"
    / "skeleton_imu"
    / "calibration"
    / "cross_fitted_logits.npz"
)
IR = (
    PROJECT_DIR
    / "runs"
    / "p27_strong_inner"
    / "ir_skeleton_imu"
    / "calibration"
    / "cross_fitted_logits.npz"
)
OUTPUT = PROJECT_DIR / "runs" / "p27_strong_inner" / "ir_confidence_gate"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def confidence_features(
    base_logits: np.ndarray, ir_logits: np.ndarray
) -> np.ndarray:
    base_probability = softmax(base_logits.astype(np.float64))
    ir_probability = softmax(ir_logits.astype(np.float64))
    base_order = np.argsort(base_probability, axis=1)
    ir_order = np.argsort(ir_probability, axis=1)
    base_top1 = base_order[:, -1]
    ir_top1 = ir_order[:, -1]
    rows = np.arange(len(base_logits))
    base_sorted = np.take_along_axis(base_probability, base_order, axis=1)
    ir_sorted = np.take_along_axis(ir_probability, ir_order, axis=1)
    epsilon = 1e-8
    base_entropy = -np.sum(
        base_probability * np.log(base_probability + epsilon), axis=1
    )
    ir_entropy = -np.sum(
        ir_probability * np.log(ir_probability + epsilon), axis=1
    )
    symmetric_kl = 0.5 * np.sum(
        base_probability
        * np.log((base_probability + epsilon) / (ir_probability + epsilon))
        + ir_probability
        * np.log((ir_probability + epsilon) / (base_probability + epsilon)),
        axis=1,
    )
    probability_cosine = np.sum(
        base_probability * ir_probability, axis=1
    ) / (
        np.linalg.norm(base_probability, axis=1)
        * np.linalg.norm(ir_probability, axis=1)
        + epsilon
    )
    top2_overlap = np.asarray(
        [
            len(set(base_order[index, -2:]) & set(ir_order[index, -2:]))
            for index in range(len(rows))
        ],
        dtype=np.float64,
    )
    return np.column_stack(
        [
            base_sorted[:, -1],
            base_sorted[:, -1] - base_sorted[:, -2],
            base_entropy,
            ir_sorted[:, -1],
            ir_sorted[:, -1] - ir_sorted[:, -2],
            ir_entropy,
            (base_top1 == ir_top1).astype(np.float64),
            top2_overlap,
            symmetric_kl,
            probability_cosine,
            ir_probability[rows, base_top1],
            base_probability[rows, ir_top1],
            ir_probability[rows, ir_top1]
            - base_probability[rows, ir_top1],
            base_probability[rows, base_top1]
            - ir_probability[rows, base_top1],
        ]
    ).astype(np.float32)


def flatten(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def main() -> None:
    base = np.load(BASE, allow_pickle=False)
    ir = np.load(IR, allow_pickle=False)
    if bool(base["outer_held_predictions_generated"]) or bool(
        ir["outer_held_predictions_generated"]
    ):
        raise RuntimeError("Outer-held predictions are forbidden")
    sample_ids = base["sample_ids"].astype(str)
    labels = base["labels"].astype(np.int64)
    subjects = base["subjects"].astype(str)
    folds = base["inner_folds"].astype(np.int64)
    position = {
        str(sample_id): index
        for index, sample_id in enumerate(ir["sample_ids"].astype(str))
    }
    indices = np.asarray([position[sample_id] for sample_id in sample_ids])
    if not np.array_equal(ir["labels"][indices], labels):
        raise RuntimeError("Labels do not align")
    base_logits = base["logits"].astype(np.float32)
    ir_logits = ir["logits"][indices].astype(np.float32)
    features = confidence_features(base_logits, ir_logits)
    base_predictions = base_logits.argmax(axis=1)
    ir_predictions = ir_logits.argmax(axis=1)
    base_correct = base_predictions == labels
    ir_correct = ir_predictions == labels
    sensitive = base_correct != ir_correct

    routed_logits = base_logits.copy()
    route_probability = np.zeros(len(labels), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for target_fold in range(3):
        train = (folds != target_fold) & sensitive
        held = folds == target_fold
        train_targets = ir_correct[train].astype(np.int64)
        if len(np.unique(train_targets)) != 2:
            raise RuntimeError(f"Fold {target_fold} gate has one target class")
        gate = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=27090 + target_fold,
        )
        gate.fit(features[train], train_targets)
        probability = gate.predict_proba(features[held])[:, 1]
        route_probability[held] = probability
        route = held.copy()
        route[held] = (
            (probability >= 0.5)
            & (base_predictions[held] != ir_predictions[held])
        )
        routed_logits[route] = ir_logits[route]
        rows.append(
            {
                "inner_fold": target_fold,
                "gate_train_sensitive_samples": int(train.sum()),
                "gate_train_rescues": int(train_targets.sum()),
                "held_samples": int(held.sum()),
                "route_to_ir": int(route.sum()),
                "base_accuracy": float(base_correct[held].mean()),
                "ir_accuracy": float(ir_correct[held].mean()),
                "gated_accuracy": float(
                    np.mean(routed_logits[held].argmax(axis=1) == labels[held])
                ),
            }
        )

    gated_predictions = routed_logits.argmax(axis=1)
    methods = {
        "skeleton_rfimu": base_predictions,
        "ir_skeleton_rfimu": ir_predictions,
        "confidence_gated": gated_predictions,
    }
    metric_rows = [
        {"method": method, **flatten(metric_bundle(labels, predictions))}
        for method, predictions in methods.items()
    ]
    subject_rows: list[dict[str, Any]] = []
    for subject in sorted(set(subjects.tolist())):
        selected = subjects == subject
        subject_rows.append(
            {
                "subject": subject,
                "samples": int(selected.sum()),
                **{
                    f"{method}_accuracy": float(
                        np.mean(predictions[selected] == labels[selected])
                    )
                    for method, predictions in methods.items()
                },
            }
        )
    gated_correct = gated_predictions == labels
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "fold_metrics.csv", rows)
    write_csv(OUTPUT / "metrics.csv", metric_rows)
    write_csv(OUTPUT / "per_subject.csv", subject_rows)
    np.savez_compressed(
        OUTPUT / "cross_fitted_logits.npz",
        protocol=np.asarray("p27-ir-scalar-confidence-gate-inner-v1"),
        sample_ids=sample_ids,
        labels=labels,
        subjects=subjects,
        inner_folds=folds,
        logits=routed_logits.astype(np.float32),
        route_probability=route_probability,
        outer_held_predictions_generated=np.asarray(False),
    )
    summary = {
        "protocol": "leave-one-inner-fold-out logistic gate; scalar confidence/disagreement features only; no predicted-class one-hot, subject or sample identifier",
        "outer_held_predictions_generated": False,
        "feature_count": int(features.shape[1]),
        "routed_samples": int(
            np.sum(gated_predictions != base_predictions)
        ),
        "rescues_vs_base": int(((~base_correct) & gated_correct).sum()),
        "new_errors_vs_base": int((base_correct & (~gated_correct)).sum()),
        "metrics": {row["method"]: row for row in metric_rows},
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
