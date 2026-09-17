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
DEFAULT_IR = PROJECT_DIR / "runs/p46_videomae_foundation_v1/complete_features.npz"
DEFAULT_DEPTH = PROJECT_DIR / "runs/p46_videomae_depth_v1/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_ir_depth_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a unified IR+Depth VideoMAE head on P46 training-user CV."
    )
    parser.add_argument("--ir-features", type=Path, default=DEFAULT_IR)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def l2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def row_standardize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values - values.mean(axis=-1, keepdims=True)) / np.maximum(
        values.std(axis=-1, keepdims=True), 1e-6
    )


def matrices(
    ir: np.ndarray,
    depth: np.ndarray,
    ir_kinetics: np.ndarray,
    depth_kinetics: np.ndarray,
) -> dict[str, np.ndarray]:
    ir = l2(ir)
    depth = l2(depth)
    ir_mean = l2(ir.mean(axis=1))
    depth_mean = l2(depth.mean(axis=1))
    matched_mean = l2(0.5 * (ir + depth))
    difference = l2(ir_mean - depth_mean)
    product = l2(ir_mean * depth_mean)
    ir_k = row_standardize(ir_kinetics).mean(axis=1)
    depth_k = row_standardize(depth_kinetics).mean(axis=1)
    return {
        "ir_mean_baseline": ir_mean,
        "depth_mean": depth_mean,
        "cross_modal_mean": l2(ir_mean + depth_mean),
        "concat_modality_means": np.concatenate((ir_mean, depth_mean), axis=1),
        "matched_view_mean_concat": matched_mean.reshape(len(ir), -1),
        "concat_all_views": np.concatenate((ir, depth), axis=1).reshape(len(ir), -1),
        "mean_relation": np.concatenate(
            (ir_mean, depth_mean, difference, product), axis=1
        ),
        "workspace_pair": np.concatenate((ir[:, 2], depth[:, 2]), axis=1),
        "mean_kinetics_pair": np.concatenate((ir_k, depth_k), axis=1),
        "unified_compact": np.concatenate(
            (ir_mean, depth_mean, difference, product, ir_k, depth_k), axis=1
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.ir_features.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        source_ids = np.asarray(data["source_ids"]).astype(str)
        users = np.asarray(data["users"]).astype(str)
        class_labels = np.asarray(data["labels"], dtype=np.int64)
        ir = np.asarray(data["features"], dtype=np.float32)
        ir_kinetics = np.asarray(data["kinetics_logits"], dtype=np.float32)
    with np.load(args.depth_features.resolve(), allow_pickle=False) as data:
        depth_ids = np.asarray(data["sample_ids"]).astype(str)
        depth_labels = np.asarray(data["labels"], dtype=np.int64)
        depth = np.asarray(data["features"], dtype=np.float32)
        depth_kinetics = np.asarray(data["kinetics_logits"], dtype=np.float32)
    if not np.array_equal(sample_ids, depth_ids) or not np.array_equal(class_labels, depth_labels):
        raise RuntimeError("IR and Depth feature caches are not row-aligned")
    feature_sets = matrices(ir, depth, ir_kinetics, depth_kinetics)
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    labels = np.asarray([class_to_index[value] for value in class_labels], dtype=np.int64)
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["detail_selected"] == "1"
        }
    split = np.asarray([manifest[value]["p46_split"] for value in sample_ids])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    folds = list(
        StratifiedGroupKFold(
            n_splits=args.cv_splits, shuffle=True, random_state=args.seed
        ).split(train_indices, labels[train_indices], groups=users[train_indices])
    )
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    for fold, (_, held_local) in enumerate(folds):
        fold_vector[held_local] = fold
    alphas = (10.0, 100.0, 1000.0, 10000.0)
    cv_rows: list[dict[str, Any]] = []
    oof_by_config: dict[tuple[str, float], np.ndarray] = {}
    for name, values in feature_sets.items():
        for alpha in alphas:
            oof_scores = np.full((len(train_indices), 21), np.nan, dtype=np.float64)
            for fit_local, held_local in folds:
                model = make_model(alpha)
                model.fit(values[train_indices[fit_local]], labels[train_indices[fit_local]])
                oof_scores[held_local] = aligned_scores(
                    model, values[train_indices[held_local]]
                )
            result = metrics(labels[train_indices], oof_scores.argmax(axis=1))
            row = {
                "feature_set": name,
                "alpha": alpha,
                "dimensions": int(values.shape[1]),
                **result,
            }
            cv_rows.append(row)
            oof_by_config[(name, alpha)] = oof_scores
            print(
                f"CV {name:26s} alpha={alpha:7g} "
                f"acc={100*float(result['accuracy']):.2f}%",
                flush=True,
            )
    selected = max(
        cv_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -int(row["dimensions"]),
        ),
    )
    write_csv(output / "group_cv_results.csv", cv_rows)
    name = str(selected["feature_set"])
    alpha = float(selected["alpha"])
    oof_scores = oof_by_config[(name, alpha)]
    temperature = fit_temperature(oof_scores, labels[train_indices])
    oof_logits = oof_scores / temperature
    print(f"Selected on training users only: {selected}, T={temperature:.4f}", flush=True)
    final_model = make_model(alpha)
    final_model.fit(feature_sets[name][train_indices], labels[train_indices])
    val_logits = aligned_scores(final_model, feature_sets[name][val_indices]) / temperature
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
    joblib.dump(final_model, output / "final_head.joblib", compress=3)
    summary = {
        "protocol": (
            "Unified IR+Depth VideoMAE feature head selected by grouped CV on 14 "
            "training users; target four users evaluated after selection"
        ),
        "selected_cv": selected,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_ir_depth_oracle_diagnostic": {
            "correct": int(oracle.sum()),
            "total": int(len(oracle)),
            "accuracy": float(oracle.mean()),
        },
        "crossfit_logits": str(crossfit_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
