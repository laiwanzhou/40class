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
DEFAULT_FEATURES = PROJECT_DIR / "runs/p46_videomae_large_ir_v1/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_large_weighted_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Large VideoMAE class reweighting on training-user CV.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def sample_weights(labels: np.ndarray, power: float) -> np.ndarray:
    if power == 0.0:
        return np.ones(len(labels), dtype=np.float64)
    counts = np.bincount(labels, minlength=21).astype(np.float64)
    present = counts > 0
    reference = counts[present].mean()
    class_weights = np.zeros(21, dtype=np.float64)
    class_weights[present] = np.power(reference / counts[present], power)
    weights = class_weights[labels]
    return weights / weights.mean()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        source_ids = np.asarray(data["source_ids"]).astype(str)
        users = np.asarray(data["users"]).astype(str)
        class_labels = np.asarray(data["labels"], dtype=np.int64)
        features = np.asarray(data["features"], dtype=np.float32)
        kinetics = np.asarray(data["kinetics_logits"], dtype=np.float32)
    values = feature_sets(features, kinetics)["concat_views"]
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
        StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed).split(
            train_indices, labels[train_indices], groups=users[train_indices]
        )
    )
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    for fold, (_, held_local) in enumerate(folds):
        fold_vector[held_local] = fold
    rows: list[dict[str, Any]] = []
    oof_by_config: dict[tuple[float, float], np.ndarray] = {}
    for power in (0.0, 0.25, 0.5, 0.75, 1.0):
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
            row = {"class_weight_power": power, "alpha": alpha, **result}
            rows.append(row)
            oof_by_config[(power, alpha)] = oof
            print(
                f"power={power:.2f} alpha={alpha:g} acc={100*float(result['accuracy']):.2f}% "
                f"macro={100*float(result['macro_f1']):.2f}%",
                flush=True,
            )
    selected = max(
        rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -float(row["class_weight_power"]),
        ),
    )
    write_csv(output / "group_cv_results.csv", rows)
    key = (float(selected["class_weight_power"]), float(selected["alpha"]))
    oof = oof_by_config[key]
    temperature = fit_temperature(oof, labels[train_indices])
    oof /= temperature
    model = make_model(float(selected["alpha"]))
    model.fit(
        values[train_indices],
        labels[train_indices],
        ridge__sample_weight=sample_weights(labels[train_indices], float(selected["class_weight_power"])),
    )
    val_logits = aligned_scores(model, values[val_indices]) / temperature
    val_prediction = val_logits.argmax(axis=1)
    validation = metrics(labels[val_indices], val_prediction)
    p12 = p12_prediction(args.p12_oof, sample_ids[val_indices])
    oracle = np.logical_or(val_prediction == labels[val_indices], p12 == labels[val_indices])
    complete_indices = np.concatenate((train_indices, val_indices))
    complete_logits = np.concatenate((oof, val_logits), axis=0)
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
        "protocol": "Large VideoMAE class weighting selected by training-user GroupCV only",
        "selected_cv": selected,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_weighted_oracle_diagnostic": {
            "correct": int(oracle.sum()),
            "total": int(len(oracle)),
            "accuracy": float(oracle.mean()),
        },
        "crossfit_logits": str(path),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
