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
    l2_normalize,
    make_model,
    metrics,
    p12_prediction,
    write_csv,
)
from train_p46_videomae_large_weighted import sample_weights


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs/p46_dinov2_base_ir_v1/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_dinov2_base_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a frozen DINOv2 appearance head on training-user GroupCV only."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def feature_sets(features: np.ndarray) -> dict[str, np.ndarray]:
    # N x view x time x hidden; normalize every frozen image token before pooling.
    tokens = l2_normalize(features.astype(np.float32))
    view_mean = l2_normalize(tokens.mean(axis=2))
    time_mean = l2_normalize(tokens.mean(axis=1))
    view_std = tokens.std(axis=2)
    view_delta = tokens[:, :, -1] - tokens[:, :, 0]
    quarters = tokens.reshape(len(tokens), 3, 4, 2, 768).mean(axis=3)
    return {
        "global_mean": l2_normalize(tokens.mean(axis=(1, 2))),
        "view_mean": view_mean.reshape(len(tokens), -1),
        "time_mean": time_mean.reshape(len(tokens), -1),
        "view_mean_std": np.concatenate((view_mean, view_std), axis=-1).reshape(len(tokens), -1),
        "view_mean_delta": np.concatenate((view_mean, view_delta), axis=-1).reshape(len(tokens), -1),
        "view_temporal_stats": np.concatenate((view_mean, view_std, view_delta), axis=-1).reshape(len(tokens), -1),
        "view_quarters": quarters.reshape(len(tokens), -1),
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
        features = np.asarray(data["features"], dtype=np.float32)
    if features.shape != (1384, 3, 8, 768):
        raise RuntimeError(f"DINOv2 feature universe changed: {features.shape}")
    if views != ("scene", "person", "workspace"):
        raise RuntimeError(f"DINOv2 view order changed: {views}")
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    labels = np.asarray([class_to_index[value] for value in class_labels], dtype=np.int64)
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["detail_selected"] == "1"
        }
    if set(sample_ids) != set(manifest):
        raise RuntimeError("DINOv2 cache and frozen P46 manifest do not align")
    split = np.asarray([manifest[value]["p46_split"] for value in sample_ids])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    if len(train_indices) != 1094 or len(val_indices) != 290:
        raise RuntimeError("Frozen P46 split counts changed")
    matrices = feature_sets(features)
    folds = list(
        StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed).split(
            train_indices, labels[train_indices], groups=users[train_indices]
        )
    )
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    for fold, (_, held_local) in enumerate(folds):
        fold_vector[held_local] = fold
    rows: list[dict[str, Any]] = []
    oof_by_config: dict[tuple[str, float, float], np.ndarray] = {}
    for name, values in matrices.items():
        for power in (0.0, 0.5):
            for alpha in (100.0, 1000.0, 10000.0):
                oof = np.full((len(train_indices), 21), np.nan, dtype=np.float64)
                for fit_local, held_local in folds:
                    fit_indices = train_indices[fit_local]
                    held_indices = train_indices[held_local]
                    model = make_model(alpha)
                    model.fit(
                        values[fit_indices],
                        labels[fit_indices],
                        ridge__sample_weight=sample_weights(labels[fit_indices], power),
                    )
                    oof[held_local] = aligned_scores(model, values[held_indices])
                if not np.isfinite(oof).all():
                    raise RuntimeError(f"Incomplete DINOv2 OOF logits: {name}")
                result = metrics(labels[train_indices], oof.argmax(axis=1))
                row = {
                    "feature_set": name,
                    "class_weight_power": power,
                    "alpha": alpha,
                    "dimensions": values.shape[1],
                    **result,
                }
                rows.append(row)
                oof_by_config[(name, power, alpha)] = oof
                print(
                    f"CV {name:22s} dim={values.shape[1]:5d} power={power:.1f} "
                    f"alpha={alpha:7g} acc={100*float(result['accuracy']):.2f}% "
                    f"macro={100*float(result['macro_f1']):.2f}%",
                    flush=True,
                )
    selected = max(
        rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -int(row["dimensions"]),
            -float(row["class_weight_power"]),
        ),
    )
    write_csv(output / "group_cv_results.csv", rows)
    name = str(selected["feature_set"])
    power = float(selected["class_weight_power"])
    alpha = float(selected["alpha"])
    oof = oof_by_config[(name, power, alpha)]
    temperature = fit_temperature(oof, labels[train_indices])
    oof_logits = oof / temperature
    model = make_model(alpha)
    model.fit(
        matrices[name][train_indices],
        labels[train_indices],
        ridge__sample_weight=sample_weights(labels[train_indices], power),
    )
    val_logits = aligned_scores(model, matrices[name][val_indices]) / temperature
    val_prediction = val_logits.argmax(axis=1)
    validation = metrics(labels[val_indices], val_prediction)
    p12 = p12_prediction(args.p12_oof, sample_ids[val_indices])
    oracle = np.logical_or(val_prediction == labels[val_indices], p12 == labels[val_indices])
    complete_indices = np.concatenate((train_indices, val_indices))
    complete_logits = np.concatenate((oof_logits, val_logits), axis=0)
    path = output / "crossfit_logits.npz"
    temporary = path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=sample_ids[complete_indices],
            source_ids=source_ids[complete_indices],
            labels=class_labels[complete_indices],
            users=users[complete_indices],
            folds=np.concatenate((fold_vector, np.full(len(val_indices), 4, dtype=np.int64))),
            logits=complete_logits.astype(np.float32),
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[complete_logits.argmax(axis=1)],
            train_oof_mask=np.concatenate(
                (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
            ),
        )
    temporary.replace(path)
    joblib.dump(model, output / "final_head.joblib", compress=3)
    summary = {
        "protocol": "DINOv2 appearance head selected by training-user GroupCV only",
        "selected_cv": selected,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_dinov2_oracle_diagnostic": {
            "correct": int(oracle.sum()),
            "total": int(len(oracle)),
            "accuracy": float(oracle.mean()),
        },
        "crossfit_logits": str(path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
