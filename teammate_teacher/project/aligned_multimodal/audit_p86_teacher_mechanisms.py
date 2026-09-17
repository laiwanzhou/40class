from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from train_p46_videomae_head import l2_normalize, make_model
from train_p85_videomae_full40_head import aligned_scores_40, sample_weights


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86_teacher_mechanism_audit_v1"

WINDOW_NAMES = ("early", "late")
VIEW_NAMES = ("scene", "person", "workspace")
FROZEN_ALPHA = 3000.0
FROZEN_CLASS_WEIGHT_POWER = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit which P85 Large VideoMAE windows and spatial views carry "
            "subject-disjoint full-40 information before designing P86."
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metric_dict(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def feature_probe_sets(features: np.ndarray) -> dict[str, np.ndarray]:
    values = l2_normalize(np.asarray(features, dtype=np.float32))
    if values.ndim != 4 or values.shape[1:] != (2, 3, 1024):
        raise RuntimeError(f"Unexpected early/late feature shape: {values.shape}")
    early = values[:, 0]
    late = values[:, 1]
    window_mean = l2_normalize(values.mean(axis=1))
    return {
        "early_late_all_views": values.reshape(len(values), -1),
        "early_all_views": early.reshape(len(values), -1),
        "late_all_views": late.reshape(len(values), -1),
        "window_mean_all_views": window_mean.reshape(len(values), -1),
        "scene_early_late": values[:, :, 0].reshape(len(values), -1),
        "person_early_late": values[:, :, 1].reshape(len(values), -1),
        "workspace_early_late": values[:, :, 2].reshape(len(values), -1),
        "scene_person_early_late": values[:, :, (0, 1)].reshape(len(values), -1),
        "scene_workspace_early_late": values[:, :, (0, 2)].reshape(len(values), -1),
        "person_workspace_early_late": values[:, :, (1, 2)].reshape(len(values), -1),
    }


def mean_impute_blocks(
    values: np.ndarray,
    training_mean: np.ndarray,
    *,
    windows: tuple[int, ...] = (),
    views: tuple[int, ...] = (),
) -> np.ndarray:
    """Replace selected [window, view] blocks by outer-train means."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 4 or source.shape[1:3] != (2, 3):
        raise ValueError(f"Unexpected values shape: {source.shape}")
    mean = np.asarray(training_mean, dtype=np.float32)
    if mean.shape != source.shape[1:]:
        raise ValueError(f"Unexpected training mean shape: {mean.shape}")
    output = source.copy()
    selected_windows = windows or tuple(range(2))
    selected_views = views or tuple(range(3))
    for window in selected_windows:
        for view in selected_views:
            output[:, window, view] = mean[window, view]
    return output


def fixed_model_perturbations(
    values: np.ndarray, training_mean: np.ndarray
) -> dict[str, np.ndarray]:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 4 or source.shape[1:3] != (2, 3):
        raise ValueError(f"Unexpected values shape: {source.shape}")
    window_mean = source.mean(axis=1, keepdims=True)
    view_mean = source.mean(axis=2, keepdims=True)
    swapped_views = source.copy()
    swapped_views[:, :, [1, 2]] = swapped_views[:, :, [2, 1]]
    return {
        "baseline": source,
        "drop_scene": mean_impute_blocks(source, training_mean, views=(0,)),
        "drop_person": mean_impute_blocks(source, training_mean, views=(1,)),
        "drop_workspace": mean_impute_blocks(source, training_mean, views=(2,)),
        "drop_early": mean_impute_blocks(source, training_mean, windows=(0,)),
        "drop_late": mean_impute_blocks(source, training_mean, windows=(1,)),
        "swap_early_late": source[:, ::-1].copy(),
        "collapse_early_late": np.repeat(window_mean, 2, axis=1),
        "swap_person_workspace": swapped_views,
        "collapse_view_identity": np.repeat(view_mean, 3, axis=2),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_and_align(
    teacher: dict[str, np.ndarray], p12: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = np.asarray(teacher["sample_ids"]).astype(str)
    labels = np.asarray(teacher["labels"], dtype=np.int64)
    users = np.asarray(teacher["users"]).astype(str)
    features = np.asarray(teacher["features"], dtype=np.float32)
    if len(sample_ids) != 2914 or len(set(users.tolist())) != 18:
        raise RuntimeError("P86 frozen full-40 universe changed")
    if set(labels.tolist()) != set(range(40)):
        raise RuntimeError("P86 frozen class universe changed")
    if tuple(np.asarray(teacher["window_names"]).astype(str)) != WINDOW_NAMES:
        raise RuntimeError("Teacher window order changed")
    if tuple(np.asarray(teacher["view_names"]).astype(str)) != VIEW_NAMES:
        raise RuntimeError("Teacher view order changed")
    p12_ids = np.asarray(p12["sample_ids"]).astype(str)
    lookup = {sample_id: index for index, sample_id in enumerate(p12_ids)}
    if len(lookup) != len(p12_ids) or set(sample_ids) != set(lookup):
        raise RuntimeError("Teacher features and P12 fold source do not align")
    order = np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(p12["labels"], dtype=np.int64)[order]):
        raise RuntimeError("Teacher/P12 labels disagree")
    folds = np.asarray(p12["folds"], dtype=np.int64)[order]
    if set(folds.tolist()) != {0, 1, 2}:
        raise RuntimeError(f"Unexpected outer folds: {set(folds.tolist())}")
    for fold in (0, 1, 2):
        train_users = set(users[folds != fold].tolist())
        held_users = set(users[folds == fold].tolist())
        if train_users & held_users:
            raise RuntimeError(f"Subject leakage in fold {fold}")
    return sample_ids, labels, users, folds, features


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(args.features)
    p12 = load_npz(args.p12_oof)
    sample_ids, labels, users, folds, raw_features = validate_and_align(teacher, p12)
    normalized = l2_normalize(raw_features)

    probe_rows: list[dict[str, Any]] = []
    probe_predictions: dict[str, np.ndarray] = {}
    for name, values in feature_probe_sets(raw_features).items():
        oof = np.full((len(labels), 40), np.nan, dtype=np.float64)
        for held_fold in (0, 1, 2):
            fit_indices = np.flatnonzero(folds != held_fold)
            held_indices = np.flatnonzero(folds == held_fold)
            model = make_model(FROZEN_ALPHA)
            model.fit(
                values[fit_indices],
                labels[fit_indices],
                ridge__sample_weight=sample_weights(
                    labels[fit_indices], FROZEN_CLASS_WEIGHT_POWER
                ),
            )
            oof[held_indices] = aligned_scores_40(model, values[held_indices])
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Incomplete probe OOF: {name}")
        prediction = oof.argmax(axis=1)
        probe_predictions[name] = prediction
        probe_rows.append(
            {
                "condition": name,
                "dimensions": int(values.shape[1]),
                **metric_dict(labels, prediction),
            }
        )

    perturbation_logits = {
        name: np.full((len(labels), 40), np.nan, dtype=np.float64)
        for name in fixed_model_perturbations(normalized[:1], normalized.mean(axis=0))
    }
    baseline_values = normalized.reshape(len(normalized), -1)
    for held_fold in (0, 1, 2):
        fit_indices = np.flatnonzero(folds != held_fold)
        held_indices = np.flatnonzero(folds == held_fold)
        model = make_model(FROZEN_ALPHA)
        model.fit(
            baseline_values[fit_indices],
            labels[fit_indices],
            ridge__sample_weight=sample_weights(
                labels[fit_indices], FROZEN_CLASS_WEIGHT_POWER
            ),
        )
        train_mean = normalized[fit_indices].mean(axis=0)
        perturbations = fixed_model_perturbations(normalized[held_indices], train_mean)
        for name, values in perturbations.items():
            perturbation_logits[name][held_indices] = aligned_scores_40(
                model, values.reshape(len(values), -1)
            )

    baseline_prediction = perturbation_logits["baseline"].argmax(axis=1)
    baseline_metrics = metric_dict(labels, baseline_prediction)
    perturbation_rows: list[dict[str, Any]] = []
    per_user_rows: list[dict[str, Any]] = []
    prediction_payload: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "labels": labels,
        "users": users,
        "folds": folds,
    }
    for name, logits in perturbation_logits.items():
        if not np.isfinite(logits).all():
            raise RuntimeError(f"Incomplete perturbation OOF: {name}")
        prediction = logits.argmax(axis=1)
        result = metric_dict(labels, prediction)
        perturbation_rows.append(
            {
                "condition": name,
                **result,
                "accuracy_delta_pp": 100.0
                * (float(result["accuracy"]) - float(baseline_metrics["accuracy"])),
                "macro_f1_delta_pp": 100.0
                * (float(result["macro_f1"]) - float(baseline_metrics["macro_f1"])),
                "prediction_changes": int(np.sum(prediction != baseline_prediction)),
                "baseline_correct_to_wrong": int(
                    np.sum((baseline_prediction == labels) & (prediction != labels))
                ),
                "baseline_wrong_to_correct": int(
                    np.sum((baseline_prediction != labels) & (prediction == labels))
                ),
            }
        )
        prediction_payload[f"{name}_logits"] = logits.astype(np.float32)
        for user in sorted(set(users.tolist())):
            target = users == user
            user_result = metric_dict(labels[target], prediction[target])
            per_user_rows.append({"condition": name, "user": user, **user_result})

    write_csv(output / "information_probes.csv", probe_rows)
    write_csv(output / "fixed_model_perturbations.csv", perturbation_rows)
    write_csv(output / "fixed_model_per_user.csv", per_user_rows)
    prediction_path = output / "fixed_model_predictions.npz"
    building = prediction_path.with_suffix(".npz.building")
    with building.open("wb") as handle:
        np.savez_compressed(handle, **prediction_payload)
    building.replace(prediction_path)

    best_probe = max(
        probe_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
        ),
    )
    summary = {
        "stage": "P86_teacher_spatiotemporal_mechanism_audit",
        "protocol": (
            "Frozen P12 subject folds; frozen Ridge alpha/class weighting; no Test "
            "labels or development-only selection. Information probes retrain the "
            "same linear head per feature subset. Causal perturbations keep each "
            "outer-fold head fixed and alter only held-subject inputs."
        ),
        "samples": len(labels),
        "classes": 40,
        "users": len(set(users.tolist())),
        "folds": 3,
        "teacher_features": str(args.features.resolve()),
        "frozen_head": {
            "alpha": FROZEN_ALPHA,
            "class_weight_power": FROZEN_CLASS_WEIGHT_POWER,
        },
        "baseline": baseline_metrics,
        "best_information_probe": best_probe,
        "information_probes": probe_rows,
        "fixed_model_perturbations": perturbation_rows,
        "outputs": {
            "information_probes": str(output / "information_probes.csv"),
            "fixed_model_perturbations": str(output / "fixed_model_perturbations.csv"),
            "fixed_model_per_user": str(output / "fixed_model_per_user.csv"),
            "fixed_model_predictions": str(prediction_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
