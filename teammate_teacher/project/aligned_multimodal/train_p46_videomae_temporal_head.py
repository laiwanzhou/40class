from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from p46_protocol import HARD_CLASS_IDS
from train_p46_videomae_head import (
    aligned_scores,
    fit_temperature,
    make_model,
    metrics,
    p12_prediction,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs/p46_videomae_temporal_v2/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_temporal_head_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a low-capacity head over position-preserving VideoMAE features "
            "using only grouped P46 training-user CV."
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def l2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def temporal_stats(values: np.ndarray) -> np.ndarray:
    """Preserve level, variation, direction, and ordered phase in compact blocks."""
    count = values.shape[-2]
    phase = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    phase /= np.square(phase).sum()
    mean = values.mean(axis=-2)
    standard_deviation = values.std(axis=-2)
    delta = values[..., -1, :] - values[..., 0, :]
    slope = np.einsum("t,...td->...d", phase, values)
    return np.concatenate((l2(mean), l2(standard_deviation), l2(delta), l2(slope)), axis=-1)


def feature_sets(
    pooled: np.ndarray,
    temporal: np.ndarray,
    quadrants: np.ndarray,
) -> dict[str, np.ndarray]:
    pooled = l2(pooled)
    temporal = l2(temporal)
    quadrants = l2(quadrants)
    mean_temporal = l2(temporal.mean(axis=1))
    person_workspace = l2(temporal[:, 1:3].mean(axis=1))
    relation = l2(temporal[:, 1] - temporal[:, 2])
    stats_all_views = temporal_stats(temporal).reshape(len(temporal), -1)
    stats_mean_views = temporal_stats(mean_temporal)
    stats_person_workspace = temporal_stats(person_workspace)
    stats_relation = temporal_stats(relation)
    spatial_time_mean = l2(quadrants.mean(axis=2)).reshape(len(quadrants), -1)
    workspace_quadrant_stats = temporal_stats(
        quadrants[:, 2].transpose(0, 2, 3, 1, 4)
    ).reshape(len(quadrants), -1)
    return {
        "pooled_mean_views": l2(pooled.mean(axis=1)),
        "temporal_mean_views": mean_temporal.reshape(len(temporal), -1),
        "temporal_person_workspace": person_workspace.reshape(len(temporal), -1),
        "temporal_workspace": temporal[:, 2].reshape(len(temporal), -1),
        "temporal_stats_mean_views": stats_mean_views,
        "temporal_stats_person_workspace": stats_person_workspace,
        "temporal_stats_all_views": stats_all_views,
        "temporal_relation_stats": stats_relation,
        "spatial_time_mean_all_views": spatial_time_mean,
        "workspace_quadrant_stats": workspace_quadrant_stats,
        "unified_position_preserving": np.concatenate(
            (
                stats_all_views,
                stats_relation,
                spatial_time_mean,
                workspace_quadrant_stats,
            ),
            axis=1,
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        source_ids = np.asarray(data["source_ids"]).astype(str)
        users = np.asarray(data["users"]).astype(str)
        class_labels = np.asarray(data["labels"], dtype=np.int64)
        views = tuple(np.asarray(data["view_names"]).astype(str))
        pooled = np.asarray(data["features"], dtype=np.float32)
        temporal = np.asarray(data["temporal_features"], dtype=np.float32)
        quadrants = np.asarray(data["quadrant_features"], dtype=np.float32)
    expected = ((1384, 3, 768), (1384, 3, 8, 768), (1384, 3, 8, 2, 2, 768))
    if (pooled.shape, temporal.shape, quadrants.shape) != expected:
        raise RuntimeError(
            f"temporal VideoMAE feature schema changed: "
            f"{pooled.shape}, {temporal.shape}, {quadrants.shape}"
        )
    if views != ("scene", "person", "workspace"):
        raise RuntimeError(f"VideoMAE view order changed: {views}")
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    labels = np.asarray([class_to_index[value] for value in class_labels], dtype=np.int64)
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["detail_selected"] == "1"
        }
    if set(sample_ids) != set(manifest):
        raise RuntimeError("temporal VideoMAE cache and P46 manifest do not align")
    split = np.asarray([manifest[value]["p46_split"] for value in sample_ids])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    if len(train_indices) != 1094 or len(val_indices) != 290:
        raise RuntimeError("Frozen P46 split counts changed")

    matrices = feature_sets(pooled, temporal, quadrants)
    folds = list(
        StratifiedGroupKFold(
            n_splits=args.cv_splits, shuffle=True, random_state=args.seed
        ).split(train_indices, labels[train_indices], groups=users[train_indices])
    )
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    for fold, (_, held_local) in enumerate(folds):
        fold_vector[held_local] = fold
    alphas = (100.0, 300.0, 1000.0, 3000.0, 10000.0)
    cv_rows: list[dict[str, Any]] = []
    oof_by_config: dict[tuple[str, float], np.ndarray] = {}
    for feature_name, matrix in matrices.items():
        for alpha in alphas:
            oof_scores = np.full((len(train_indices), 21), np.nan, dtype=np.float64)
            for fit_local, held_local in folds:
                model = make_model(alpha)
                model.fit(matrix[train_indices[fit_local]], labels[train_indices[fit_local]])
                oof_scores[held_local] = aligned_scores(
                    model, matrix[train_indices[held_local]]
                )
            result = metrics(labels[train_indices], oof_scores.argmax(axis=1))
            row = {
                "feature_set": feature_name,
                "alpha": alpha,
                "dimensions": int(matrix.shape[1]),
                **result,
            }
            cv_rows.append(row)
            oof_by_config[(feature_name, alpha)] = oof_scores
            print(
                f"CV {feature_name:30s} alpha={alpha:7g} "
                f"acc={100*float(result['accuracy']):.2f}% "
                f"macro={100*float(result['macro_f1']):.2f}%",
                flush=True,
            )
    selected = max(
        cv_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -int(row["dimensions"]),
            -float(row["alpha"]),
        ),
    )
    write_csv(output / "group_cv_results.csv", cv_rows)
    feature_name = str(selected["feature_set"])
    alpha = float(selected["alpha"])
    oof_scores = oof_by_config[(feature_name, alpha)]
    temperature = fit_temperature(oof_scores, labels[train_indices])
    oof_logits = oof_scores / temperature
    print(
        f"Selected on training users only: feature={feature_name} alpha={alpha:g} "
        f"temperature={temperature:.4f}",
        flush=True,
    )
    final_model = make_model(alpha)
    final_model.fit(matrices[feature_name][train_indices], labels[train_indices])
    val_logits = aligned_scores(final_model, matrices[feature_name][val_indices]) / temperature
    val_prediction = val_logits.argmax(axis=1)
    validation = metrics(labels[val_indices], val_prediction)
    p12 = p12_prediction(args.p12_oof, sample_ids[val_indices])
    oracle = np.logical_or(val_prediction == labels[val_indices], p12 == labels[val_indices])

    complete_indices = np.concatenate((train_indices, val_indices))
    complete_logits = np.concatenate((oof_logits, val_logits), axis=0)
    crossfit_path = output / "crossfit_logits.npz"
    temporary = crossfit_path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=sample_ids[complete_indices],
            source_ids=source_ids[complete_indices],
            labels=class_labels[complete_indices],
            users=users[complete_indices],
            folds=np.concatenate(
                (fold_vector, np.full(len(val_indices), args.cv_splits, dtype=np.int64))
            ),
            logits=complete_logits.astype(np.float32),
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[complete_logits.argmax(axis=1)],
            train_oof_mask=np.concatenate(
                (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
            ),
        )
    temporary.replace(crossfit_path)
    prediction_rows: list[dict[str, Any]] = []
    for local, index in enumerate(val_indices):
        true_class = int(class_labels[index])
        predicted_class = int(HARD_CLASS_IDS[val_prediction[local]])
        prediction_rows.append(
            {
                "sample_id": sample_ids[index],
                "source_id": source_ids[index],
                "user_id": users[index],
                "true_class_id": true_class,
                "predicted_class_id": predicted_class,
                "correct": int(true_class == predicted_class),
                "p12_correct": int(p12[local] == labels[index]),
            }
        )
    write_csv(output / "validation_predictions.csv", prediction_rows)
    joblib.dump(final_model, output / "final_head.joblib", compress=3)
    summary = {
        "protocol": (
            "Position-preserving VideoMAE features; config selected by four-fold "
            "grouped CV on the 14 P46 training users; target users evaluated once"
        ),
        "selected_cv": selected,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_temporal_oracle_diagnostic": {
            "correct": int(oracle.sum()),
            "total": int(len(oracle)),
            "accuracy": float(oracle.mean()),
        },
        "required_for_70_percent": 203,
        "crossfit_logits": str(crossfit_path),
        "final_head": str((output / "final_head.joblib").resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
