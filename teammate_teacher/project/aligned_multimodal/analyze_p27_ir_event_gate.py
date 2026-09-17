from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_p27_ir_confidence_gate import (
    BASE,
    IR,
    confidence_features,
    flatten,
)
from p27r2_event_data import load_event_cache
from probe_p27r3_incremental_information import (
    explicit_event_sequence,
    metric_bundle,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
EVENT_CACHE = (
    PROJECT_DIR / "runs" / "p27_r2_event_audit" / "event_cache_v2.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_strong_inner" / "ir_event_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit a class-agnostic event-quality gate for the IR increment"
    )
    parser.add_argument(
        "--event-feature-mode",
        choices=("sequence_summary", "weak_targets"),
        default="sequence_summary",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def summarize_events(sequence: np.ndarray, presence: np.ndarray) -> np.ndarray:
    difference = np.diff(sequence, axis=1)
    thirds = np.array_split(sequence, 3, axis=1)
    return np.concatenate(
        [
            sequence.mean(axis=1),
            sequence.std(axis=1),
            sequence.max(axis=1),
            sequence.min(axis=1),
            np.abs(difference).mean(axis=1),
            thirds[0].mean(axis=1),
            thirds[1].mean(axis=1),
            thirds[2].mean(axis=1),
            presence.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
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
    ir_position = {
        str(sample_id): index
        for index, sample_id in enumerate(ir["sample_ids"].astype(str))
    }
    ir_indices = np.asarray([ir_position[sample_id] for sample_id in sample_ids])
    if not np.array_equal(ir["labels"][ir_indices], labels):
        raise RuntimeError("IR labels do not align")
    base_logits = base["logits"].astype(np.float32)
    ir_logits = ir["logits"][ir_indices].astype(np.float32)

    event_cache = load_event_cache(EVENT_CACHE)
    event_position = {
        str(sample_id): index
        for index, sample_id in enumerate(event_cache.sample_ids.astype(str))
    }
    event_indices = np.asarray(
        [event_position[sample_id] for sample_id in sample_ids]
    )
    if np.any(event_cache.outer_folds[event_indices] == 0):
        raise RuntimeError("Event gate attempted to touch fold-0 outer-held")
    if args.event_feature_mode == "sequence_summary":
        sequence = explicit_event_sequence(event_cache, event_indices)
        event_features = summarize_events(
            sequence, event_cache.modality_mask[event_indices]
        )
    else:
        event_features = np.concatenate(
            [
                event_cache.event_targets[event_indices],
                event_cache.event_quality[event_indices],
                event_cache.modality_mask[event_indices].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    features = np.concatenate(
        [confidence_features(base_logits, ir_logits), event_features], axis=1
    )
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
        gate = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=27100 + target_fold,
            ),
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
        "event_quality_gated": gated_predictions,
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
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "fold_metrics.csv", rows)
    write_csv(output / "metrics.csv", metric_rows)
    write_csv(output / "per_subject.csv", subject_rows)
    np.savez_compressed(
        output / "cross_fitted_logits.npz",
        protocol=np.asarray("p27-ir-event-quality-gate-inner-v1"),
        sample_ids=sample_ids,
        labels=labels,
        subjects=subjects,
        inner_folds=folds,
        logits=routed_logits.astype(np.float32),
        route_probability=route_probability,
        outer_held_predictions_generated=np.asarray(False),
    )
    summary = {
        "protocol": "leave-one-inner-fold-out L2-logistic gate; scalar logits confidence plus class-agnostic temporal-event summaries; no subject/sample identifier or predicted-class one-hot",
        "outer_held_predictions_generated": False,
        "confidence_feature_count": 14,
        "event_feature_mode": args.event_feature_mode,
        "event_feature_count": int(event_features.shape[1]),
        "total_feature_count": int(features.shape[1]),
        "routed_samples": int(
            np.sum(gated_predictions != base_predictions)
        ),
        "rescues_vs_base": int(((~base_correct) & gated_correct).sum()),
        "new_errors_vs_base": int((base_correct & (~gated_correct)).sum()),
        "metrics": {row["method"]: row for row in metric_rows},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
