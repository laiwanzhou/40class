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
    feature_sets,
    fit_temperature,
    make_model,
    metrics,
    p12_prediction,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = PROJECT_DIR / "runs/p46_videomae_foundation_v1/complete_features.npz"
DEFAULT_LARGE = PROJECT_DIR / "runs/p46_videomae_large_ir_v1/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_base_large_joint_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one unified Base+Large VideoMAE feature head using training-user GroupCV only."
    )
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--large-features", type=Path, default=DEFAULT_LARGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {
            "sample_ids": np.asarray(data["sample_ids"]).astype(str),
            "source_ids": np.asarray(data["source_ids"]).astype(str),
            "users": np.asarray(data["users"]).astype(str),
            "labels": np.asarray(data["labels"], dtype=np.int64),
            "features": np.asarray(data["features"], dtype=np.float32),
            "kinetics_logits": np.asarray(data["kinetics_logits"], dtype=np.float32),
        }


def align_cache(reference_ids: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    lookup = {value: index for index, value in enumerate(cache["sample_ids"])}
    if len(lookup) != len(reference_ids) or set(lookup) != set(reference_ids):
        raise RuntimeError("Base and Large VideoMAE caches do not contain the same samples")
    order = np.asarray([lookup[value] for value in reference_ids], dtype=np.int64)
    return {key: values[order] for key, values in cache.items()}


def build_matrices(
    base_features: np.ndarray,
    base_kinetics: np.ndarray,
    large_features: np.ndarray,
    large_kinetics: np.ndarray,
) -> dict[str, np.ndarray]:
    base = feature_sets(base_features, base_kinetics)
    large = feature_sets(large_features, large_kinetics)
    base_mean = base["mean_views"]
    large_mean = large["mean_views"]
    base_logits = base["kinetics_views"].reshape(len(base_features), 3, -1).mean(axis=1)
    large_logits = large["kinetics_views"].reshape(len(large_features), 3, -1).mean(axis=1)
    return {
        "large_concat": large["concat_views"],
        "base_large_mean": np.concatenate((base_mean, large_mean), axis=1),
        "base_large_concat": np.concatenate((base["concat_views"], large["concat_views"]), axis=1),
        "base_large_mean_logits": np.concatenate(
            (base_mean, large_mean, base_logits, large_logits), axis=1
        ),
        "base_large_all": np.concatenate(
            (
                base["concat_views"],
                large["concat_views"],
                base["kinetics_views"],
                large["kinetics_views"],
            ),
            axis=1,
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = load_cache(args.base_features)
    large = align_cache(base["sample_ids"], load_cache(args.large_features))
    for key in ("source_ids", "users", "labels"):
        if not np.array_equal(base[key], large[key]):
            raise RuntimeError(f"Base/Large metadata mismatch: {key}")
    sample_ids = base["sample_ids"]
    source_ids = base["source_ids"]
    users = base["users"]
    class_labels = base["labels"]
    if base["features"].shape != (1384, 3, 768):
        raise RuntimeError(f"Unexpected Base cache shape: {base['features'].shape}")
    if large["features"].shape != (1384, 3, 1024):
        raise RuntimeError(f"Unexpected Large cache shape: {large['features'].shape}")
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    labels = np.asarray([class_to_index[value] for value in class_labels], dtype=np.int64)
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["detail_selected"] == "1"
        }
    if set(sample_ids) != set(manifest):
        raise RuntimeError("VideoMAE caches and frozen P46 manifest do not align")
    split = np.asarray([manifest[value]["p46_split"] for value in sample_ids])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    if len(train_indices) != 1094 or len(val_indices) != 290:
        raise RuntimeError("Frozen P46 split counts changed")

    matrices = build_matrices(
        base["features"],
        base["kinetics_logits"],
        large["features"],
        large["kinetics_logits"],
    )
    folds = list(
        StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed).split(
            train_indices, labels[train_indices], groups=users[train_indices]
        )
    )
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    for fold, (_, held_local) in enumerate(folds):
        fold_vector[held_local] = fold
    if np.any(fold_vector < 0):
        raise RuntimeError("Grouped CV did not cover every P46 training sample")

    rows: list[dict[str, Any]] = []
    oof_by_config: dict[tuple[str, float], np.ndarray] = {}
    for name, values in matrices.items():
        for alpha in (100.0, 300.0, 1000.0, 3000.0, 10000.0):
            oof = np.full((len(train_indices), 21), np.nan, dtype=np.float64)
            for fit_local, held_local in folds:
                model = make_model(alpha)
                model.fit(values[train_indices[fit_local]], labels[train_indices[fit_local]])
                oof[held_local] = aligned_scores(model, values[train_indices[held_local]])
            if not np.isfinite(oof).all():
                raise RuntimeError(f"Incomplete OOF logits: {name}, alpha={alpha:g}")
            result = metrics(labels[train_indices], oof.argmax(axis=1))
            row = {"feature_set": name, "alpha": alpha, "dimensions": values.shape[1], **result}
            rows.append(row)
            oof_by_config[(name, alpha)] = oof
            print(
                f"CV {name:24s} dim={values.shape[1]:5d} alpha={alpha:7g} "
                f"acc={100*float(result['accuracy']):.2f}% macro={100*float(result['macro_f1']):.2f}%",
                flush=True,
            )
    selected = max(
        rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -int(row["dimensions"]),
            -float(row["alpha"]),
        ),
    )
    write_csv(output / "group_cv_results.csv", rows)
    name = str(selected["feature_set"])
    alpha = float(selected["alpha"])
    oof = oof_by_config[(name, alpha)]
    temperature = fit_temperature(oof, labels[train_indices])
    oof_logits = oof / temperature
    final_model = make_model(alpha)
    final_model.fit(matrices[name][train_indices], labels[train_indices])
    val_logits = aligned_scores(final_model, matrices[name][val_indices]) / temperature
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
    joblib.dump(final_model, output / "final_head.joblib", compress=3)
    summary = {
        "protocol": "Unified Base+Large feature head selected by training-user GroupCV only",
        "selected_cv": selected,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_joint_oracle_diagnostic": {
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
