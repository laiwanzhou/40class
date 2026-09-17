from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

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
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_large_bagging_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested held-user bagging for the selected Large VideoMAE Ridge head."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


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
    alpha = 1000.0
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
    outer_folds = list(
        StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed).split(
            train_indices, labels[train_indices], groups=users[train_indices]
        )
    )
    direct_oof = np.full((len(train_indices), 21), np.nan, dtype=np.float64)
    bagged_oof = np.full_like(direct_oof, np.nan)
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    outer_models: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    for outer_fold, (fit_local, held_local) in enumerate(outer_folds):
        fit_indices = train_indices[fit_local]
        held_indices = train_indices[held_local]
        direct_model = make_model(alpha)
        direct_model.fit(values[fit_indices], labels[fit_indices])
        direct_oof[held_local] = aligned_scores(direct_model, values[held_indices])
        outer_models.append(direct_model)
        inner_scores: list[np.ndarray] = []
        inner_models = []
        inner_splitter = GroupKFold(n_splits=3)
        for inner_fit_local, _ in inner_splitter.split(
            fit_indices, labels[fit_indices], groups=users[fit_indices]
        ):
            inner_fit = fit_indices[inner_fit_local]
            model = make_model(alpha)
            model.fit(values[inner_fit], labels[inner_fit])
            inner_scores.append(aligned_scores(model, values[held_indices]))
            inner_models.append(model)
        bagged_oof[held_local] = np.mean(inner_scores, axis=0)
        fold_vector[held_local] = outer_fold
        diagnostics.append(
            {
                "outer_fold": outer_fold,
                "fit_users": sorted(set(users[fit_indices])),
                "held_users": sorted(set(users[held_indices])),
                "direct_accuracy": float(
                    (direct_oof[held_local].argmax(axis=1) == labels[held_indices]).mean()
                ),
                "inner_bag_accuracy": float(
                    (bagged_oof[held_local].argmax(axis=1) == labels[held_indices]).mean()
                ),
            }
        )
    blend_rows: list[dict[str, Any]] = []
    blended_by_fraction: dict[float, np.ndarray] = {}
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = (1.0 - fraction) * direct_oof + fraction * bagged_oof
        result = metrics(labels[train_indices], scores.argmax(axis=1))
        blend_rows.append({"bagged_fraction": fraction, **result})
        blended_by_fraction[fraction] = scores
        print(
            f"bagged_fraction={fraction:.2f} OOF={100*float(result['accuracy']):.2f}%",
            flush=True,
        )
    selected = max(
        blend_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -float(row["bagged_fraction"]),
        ),
    )
    write_csv(output / "group_cv_blends.csv", blend_rows)
    fraction = float(selected["bagged_fraction"])
    oof_scores = blended_by_fraction[fraction]
    temperature = fit_temperature(oof_scores, labels[train_indices])
    oof_logits = oof_scores / temperature
    print(f"Selected on training users only: {selected}, T={temperature:.4f}", flush=True)

    final_model = make_model(alpha)
    final_model.fit(values[train_indices], labels[train_indices])
    direct_val = aligned_scores(final_model, values[val_indices])
    bagged_val = np.mean(
        [aligned_scores(model, values[val_indices]) for model in outer_models], axis=0
    )
    val_logits = ((1.0 - fraction) * direct_val + fraction * bagged_val) / temperature
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
            folds=np.concatenate((fold_vector, np.full(len(val_indices), 4, dtype=np.int64))),
            logits=complete_logits.astype(np.float32),
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[complete_logits.argmax(axis=1)],
            train_oof_mask=np.concatenate(
                (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
            ),
        )
    temporary.replace(crossfit_path)
    joblib.dump(
        {"final_model": final_model, "outer_models": outer_models, "bagged_fraction": fraction},
        output / "bagged_models.joblib",
        compress=3,
    )
    summary = {
        "protocol": (
            "Large VideoMAE Ridge group bagging; bagged fraction selected by nested "
            "training-user CV only; target users evaluated after selection"
        ),
        "feature_set": "concat_views",
        "alpha": alpha,
        "selected_cv": selected,
        "outer_fold_diagnostics": diagnostics,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_bagged_oracle_diagnostic": {
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
