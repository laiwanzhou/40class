from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probe_p27r3_incremental_information import metric_bundle, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_DIR / "runs" / "p27_strong_inner"
DEFAULT_OUTPUT = DEFAULT_ROOT / "low_capacity_stacking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Outer-train-only, leave-one-inner-fold-out probe of whether the "
            "existing P27 experts contain learnable complementary information."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1],
    )
    return parser.parse_args()


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if (
            "outer_held_predictions_generated" in archive.files
            and bool(archive["outer_held_predictions_generated"])
        ):
            raise RuntimeError(f"Outer-held predictions are forbidden: {path}")
        return {key: archive[key] for key in archive.files}


def align(
    reference_ids: np.ndarray,
    reference_labels: np.ndarray,
    source: dict[str, np.ndarray],
    key: str,
) -> np.ndarray:
    positions = {
        str(sample_id): index
        for index, sample_id in enumerate(source["sample_ids"].astype(str))
    }
    indices = np.asarray([positions[str(sample_id)] for sample_id in reference_ids])
    if not np.array_equal(source["labels"][indices], reference_labels):
        raise RuntimeError(f"Label mismatch while aligning {key}")
    return source[key][indices].astype(np.float64)


def log_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    shifted -= np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted


def fold_features(root: Path, fold: int) -> dict[str, np.ndarray]:
    base = load_archive(root / "ir_skeleton_imu" / f"fold_{fold}_logits.npz")
    event = load_archive(root / "event_expert" / f"fold_{fold}_logits.npz")
    imu_event = load_archive(root / "imu_event_forest" / f"fold_{fold}_logits.npz")
    sample_ids = base["sample_ids"].astype(str)
    labels = base["labels"].astype(np.int64)
    experts = {
        "ir_skeleton": base["joint_logits"].astype(np.float64),
        "rf_imu": base["imu_logits"].astype(np.float64),
        "event_sequence": align(sample_ids, labels, event, "logits"),
        "imu_shape": align(sample_ids, labels, imu_event, "logits"),
    }
    features = np.concatenate(
        [log_probabilities(experts[name]) for name in experts],
        axis=1,
    )
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "subjects": base["subjects"].astype(str),
        "features": features,
        "fixed_logits": base["fused_logits"].astype(np.float64),
        **{f"{name}_logits": value for name, value in experts.items()},
    }


def flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = [fold_features(root, fold) for fold in range(3)]
    rows: list[dict[str, Any]] = []
    predictions_by_c: dict[float, list[np.ndarray]] = {
        float(value): [] for value in args.c_values
    }
    labels_all: list[np.ndarray] = []
    sample_ids_all: list[np.ndarray] = []
    subjects_all: list[np.ndarray] = []
    inner_folds_all: list[np.ndarray] = []

    for target in range(3):
        train = [folds[index] for index in range(3) if index != target]
        train_x = np.concatenate([part["features"] for part in train])
        train_y = np.concatenate([part["labels"] for part in train])
        target_x = folds[target]["features"]
        target_y = folds[target]["labels"]
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        target_x = scaler.transform(target_x)

        fixed_predictions = folds[target]["fixed_logits"].argmax(axis=1)
        rows.append(
            {
                "inner_fold": target,
                "method": "fixed_ir_skeleton_rfimu",
                "c": "",
                **flatten_metrics(metric_bundle(target_y, fixed_predictions)),
            }
        )
        for c_value in args.c_values:
            model = LogisticRegression(
                C=float(c_value),
                solver="lbfgs",
                max_iter=600,
                class_weight=None,
                random_state=27083,
            )
            model.fit(train_x, train_y)
            predictions = model.predict(target_x).astype(np.int64)
            predictions_by_c[float(c_value)].append(predictions)
            rows.append(
                {
                    "inner_fold": target,
                    "method": "four_expert_linear_stacker",
                    "c": float(c_value),
                    **flatten_metrics(metric_bundle(target_y, predictions)),
                }
            )

        labels_all.append(target_y)
        sample_ids_all.append(folds[target]["sample_ids"])
        subjects_all.append(folds[target]["subjects"])
        inner_folds_all.append(
            np.full(len(target_y), target, dtype=np.int64)
        )

    write_csv(output / "fold_metrics.csv", rows)
    aggregate_rows: list[dict[str, Any]] = []
    fixed_rows = [
        row for row in rows if row["method"] == "fixed_ir_skeleton_rfimu"
    ]
    aggregate_rows.append(
        {
            "method": "fixed_ir_skeleton_rfimu",
            "c": "",
            **{
                key: float(np.mean([float(row[key]) for row in fixed_rows]))
                for key in fixed_rows[0]
                if key not in {"inner_fold", "method", "c"}
            },
        }
    )
    for c_value in args.c_values:
        selected = [
            row
            for row in rows
            if row["method"] == "four_expert_linear_stacker"
            and float(row["c"]) == float(c_value)
        ]
        aggregate_rows.append(
            {
                "method": "four_expert_linear_stacker",
                "c": float(c_value),
                **{
                    key: float(np.mean([float(row[key]) for row in selected]))
                    for key in selected[0]
                    if key not in {"inner_fold", "method", "c"}
                },
            }
        )
    write_csv(output / "mean_metrics.csv", aggregate_rows)

    labels = np.concatenate(labels_all)
    best = max(
        aggregate_rows[1:],
        key=lambda row: (
            float(row["overall_accuracy"]),
            float(row["overall_macro_f1"]),
        ),
    )
    selected_c = float(best["c"])
    np.savez_compressed(
        output / "cross_fitted_predictions.npz",
        protocol=np.asarray("p27-outer-train-inner-linear-stacking-probe-v1"),
        sample_ids=np.concatenate(sample_ids_all),
        labels=labels,
        subjects=np.concatenate(subjects_all),
        inner_folds=np.concatenate(inner_folds_all),
        predictions=np.concatenate(predictions_by_c[selected_c]),
        selected_c=np.asarray(selected_c),
        outer_held_predictions_generated=np.asarray(False),
    )
    summary = {
        "protocol": "p27-outer-train-inner-linear-stacking-probe-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "purpose": (
            "information-utilization probe only; not a frozen final configuration"
        ),
        "features": [
            "IR+Skeleton log-probabilities",
            "RF-IMU log-probabilities",
            "explicit event-sequence log-probabilities",
            "engineered IMU-shape log-probabilities",
        ],
        "regularization_grid": [float(value) for value in args.c_values],
        "selection_scope": (
            "all rows are from fold-0 outer-train subjects; each reported row is "
            "predicted by a stacker fitted on the other two inner folds"
        ),
        "best_probe": best,
        "mean_metrics": aggregate_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
