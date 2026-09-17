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
    row_standardize,
    write_csv,
)
from train_p46_videomae_large_weighted import sample_weights


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FULL = PROJECT_DIR / "runs/p46_videomae_large_ir_v1/complete_features.npz"
DEFAULT_WINDOWS = PROJECT_DIR / "runs/p46_videomae_large_multiclip_v1/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_large_multiclip_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one full+early+late Large VideoMAE head on training-user GroupCV."
    )
    parser.add_argument("--full-features", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--window-features", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def align(reference: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = np.asarray(cache["sample_ids"]).astype(str)
    lookup = {value: index for index, value in enumerate(ids)}
    if len(lookup) != len(reference) or set(lookup) != set(reference):
        raise RuntimeError("Full and multi-clip caches contain different samples")
    order = np.asarray([lookup[value] for value in reference], dtype=np.int64)
    return {key: values[order] if len(values.shape) > 0 and values.shape[0] == len(ids) else values for key, values in cache.items()}


def matrices(
    full_features: np.ndarray,
    full_logits: np.ndarray,
    window_features: np.ndarray,
    window_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    full = l2_normalize(full_features.astype(np.float32))
    windows = l2_normalize(window_features.astype(np.float32))
    early = windows[:, 0]
    late = windows[:, 1]
    window_mean = l2_normalize(windows.mean(axis=1))
    difference = late - early
    kinetics = row_standardize(
        np.concatenate(
            (
                full_logits.astype(np.float32),
                window_logits[:, 0].astype(np.float32),
                window_logits[:, 1].astype(np.float32),
            ),
            axis=1,
        ).reshape(len(full), -1)
    )
    return {
        "full": full.reshape(len(full), -1),
        "early": early.reshape(len(full), -1),
        "late": late.reshape(len(full), -1),
        "window_mean": window_mean.reshape(len(full), -1),
        "early_late": windows.reshape(len(full), -1),
        "full_window_mean": np.concatenate((full, window_mean), axis=1).reshape(len(full), -1),
        "full_temporal_delta": np.concatenate((full, difference), axis=1).reshape(len(full), -1),
        "full_early_late": np.concatenate((full[:, None], windows), axis=1).reshape(len(full), -1),
        "three_clip_kinetics": kinetics,
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    full = load(args.full_features)
    sample_ids = np.asarray(full["sample_ids"]).astype(str)
    windows = align(sample_ids, load(args.window_features))
    source_ids = np.asarray(full["source_ids"]).astype(str)
    users = np.asarray(full["users"]).astype(str)
    class_labels = np.asarray(full["labels"], dtype=np.int64)
    for key in ("source_ids", "users", "labels"):
        if not np.array_equal(np.asarray(full[key]).astype(str), np.asarray(windows[key]).astype(str)):
            raise RuntimeError(f"Full/multi-clip metadata mismatch: {key}")
    full_features = np.asarray(full["features"], dtype=np.float32)
    window_features = np.asarray(windows["features"], dtype=np.float32)
    if full_features.shape != (1384, 3, 1024) or window_features.shape != (1384, 2, 3, 1024):
        raise RuntimeError(f"Unexpected multi-clip shapes: {full_features.shape}, {window_features.shape}")
    values_by_name = matrices(
        full_features,
        np.asarray(full["kinetics_logits"], dtype=np.float32),
        window_features,
        np.asarray(windows["kinetics_logits"], dtype=np.float32),
    )
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
    if len(train_indices) != 1094 or len(val_indices) != 290:
        raise RuntimeError("Frozen P46 split counts changed")
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
    for name, values in values_by_name.items():
        for power in (0.0, 0.5, 0.75):
            for alpha in (300.0, 1000.0, 3000.0, 10000.0):
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
                    f"CV {name:21s} dim={values.shape[1]:5d} power={power:.2f} "
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
        values_by_name[name][train_indices],
        labels[train_indices],
        ridge__sample_weight=sample_weights(labels[train_indices], power),
    )
    val_logits = aligned_scores(model, values_by_name[name][val_indices]) / temperature
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

    # Preserve the independently Group-CV-selected temporal views as separate
    # cross-fit experts.  A later stacker can then learn whether early, late, or
    # full-window evidence is complementary without ever fitting on validation labels.
    candidate_payload: dict[str, np.ndarray] = {
        "sample_ids": sample_ids[complete_indices],
        "source_ids": source_ids[complete_indices],
        "labels": class_labels[complete_indices],
        "users": users[complete_indices],
        "folds": np.concatenate((fold_vector, np.full(len(val_indices), 4, dtype=np.int64))),
        "train_oof_mask": np.concatenate(
            (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
        ),
    }
    candidate_summaries: dict[str, dict[str, Any]] = {}
    for candidate_name in values_by_name:
        candidate_row = max(
            (row for row in rows if row["feature_set"] == candidate_name),
            key=lambda row: (
                float(row["accuracy"]),
                float(row["balanced_accuracy"]),
                float(row["macro_f1"]),
                -float(row["class_weight_power"]),
            ),
        )
        candidate_power = float(candidate_row["class_weight_power"])
        candidate_alpha = float(candidate_row["alpha"])
        candidate_oof = oof_by_config[(candidate_name, candidate_power, candidate_alpha)]
        candidate_temperature = fit_temperature(candidate_oof, labels[train_indices])
        candidate_model = make_model(candidate_alpha)
        candidate_model.fit(
            values_by_name[candidate_name][train_indices],
            labels[train_indices],
            ridge__sample_weight=sample_weights(labels[train_indices], candidate_power),
        )
        candidate_val = aligned_scores(
            candidate_model, values_by_name[candidate_name][val_indices]
        ) / candidate_temperature
        candidate_logits = np.concatenate(
            (candidate_oof / candidate_temperature, candidate_val), axis=0
        )
        candidate_payload[f"{candidate_name}_logits"] = candidate_logits.astype(np.float32)
        candidate_summaries[candidate_name] = {
            "selected_cv": candidate_row,
            "temperature": candidate_temperature,
            "validation": metrics(labels[val_indices], candidate_val.argmax(axis=1)),
        }
    candidate_path = output / "candidate_crossfit_logits.npz"
    candidate_temporary = candidate_path.with_suffix(".npz.building")
    with candidate_temporary.open("wb") as handle:
        np.savez_compressed(handle, **candidate_payload)
    candidate_temporary.replace(candidate_path)
    summary = {
        "protocol": "Large VideoMAE full+early+late selected by training-user GroupCV only",
        "selected_cv": selected,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_multiclip_oracle_diagnostic": {
            "correct": int(oracle.sum()),
            "total": int(len(oracle)),
            "accuracy": float(oracle.mean()),
        },
        "crossfit_logits": str(path),
        "candidate_crossfit_logits": str(candidate_path),
        "candidate_diagnostics": candidate_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
