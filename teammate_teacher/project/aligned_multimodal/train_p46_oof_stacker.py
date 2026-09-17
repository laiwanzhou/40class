from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "p46_single_split.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_70_oof_stacker_v1"


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    relative_path: str
    field: str
    family: str


# This list is fixed before looking at P46 validation labels.  It deliberately
# keeps raw unimodal experts as well as architecturally different fusion models;
# correlated duplicate exports are not included.
EXPERT_SPECS = (
    ExpertSpec("p12_skeleton", "runs/p12_complete_oof/complete_oof.npz", "skeleton_logits", "skeleton"),
    ExpertSpec("p12_depth", "runs/p12_complete_oof/complete_oof.npz", "depth_logits", "depth"),
    ExpertSpec("p12_imu", "runs/p12_complete_oof/complete_oof.npz", "imu_logits", "imu"),
    ExpertSpec("p12_thermal", "runs/p12_complete_oof/complete_oof.npz", "thermal_logits", "thermal"),
    ExpertSpec("p12_sd", "runs/p12_complete_oof/complete_oof.npz", "sd_logits", "multimodal"),
    ExpertSpec("p12_sd_imu", "runs/p12_complete_oof/complete_oof.npz", "sd_imu_logits", "multimodal"),
    ExpertSpec("p12_final", "runs/p12_complete_oof/complete_oof.npz", "final_logits", "multimodal"),
    ExpertSpec("local_depth", "runs/p12_local_depth_fusion_audit/local_fusion_oof.npz", "local_logits", "local_depth"),
    ExpertSpec("local_depth_fixed_fusion", "runs/p12_local_depth_fusion_audit/local_fusion_oof.npz", "fixed_low_weight_fused_logits", "multimodal"),
    ExpertSpec("skeleton_thermal", "runs/p11_final_package/oof_thermal_fusion/s_plus_thermal_logits.npz", "candidate_logits", "thermal_fusion"),
    ExpertSpec("sd_thermal", "runs/p11_final_package/oof_thermal_fusion/sd_plus_thermal_logits.npz", "candidate_logits", "thermal_fusion"),
    ExpertSpec("peak_depth", "runs/p18_peak_sampling_shared_oof/oof_logits.npz", "fused_logits", "local_depth"),
    ExpertSpec("depth_difference", "runs/p19_local_depth_difference_shared_oof/oof_logits.npz", "fused_logits", "local_depth"),
    ExpertSpec("tiny_imu_teacher", "runs/p20_tiny_imu_student_audit/complete_oof.npz", "teacher_fused_logits", "imu_fusion"),
    ExpertSpec("tiny_imu_student", "runs/p20_tiny_imu_student_audit/complete_oof.npz", "student_fused_logits", "imu_fusion"),
    ExpertSpec("thermal_fixed", "runs/p21_tiny_imu_thermal_integration_audit/complete_oof.npz", "fixed_thermal_logits", "thermal_fusion"),
    ExpertSpec("p22_feature", "runs/p22_joint_pooled_fusion/experiment/P22-F_oof_logits.npz", "logits", "joint_fusion"),
    ExpertSpec("p22_logit", "runs/p22_joint_pooled_fusion/experiment/P22-L_oof_logits.npz", "logits", "joint_fusion"),
    ExpertSpec("p22_bypass", "runs/p22_joint_pooled_fusion/experiment/P22-U_oof_logits.npz", "logits", "joint_fusion"),
    ExpertSpec("p25_depth_adapter", "runs/p25_subject_invariant_adapters/depth/P25-C/oof_outputs.npz", "logits", "subject_adapter"),
    ExpertSpec("p25_skeleton_adapter", "runs/p25_subject_invariant_adapters/skeleton/P25-C/oof_outputs.npz", "logits", "subject_adapter"),
    ExpertSpec("p25_imu_adapter", "runs/p25_subject_invariant_adapters/imu/P25-C/oof_outputs.npz", "logits", "subject_adapter"),
    ExpertSpec("p25_thermal_adapter", "runs/p25_subject_invariant_adapters/thermal/P25-C/oof_outputs.npz", "logits", "subject_adapter"),
    ExpertSpec("videomae_foundation", "runs/p46_videomae_head_v1/crossfit_logits.npz", "logits", "foundation_video"),
    ExpertSpec("videomae_depth", "runs/p46_videomae_depth_head_v1/crossfit_logits.npz", "logits", "foundation_depth"),
    ExpertSpec("videomae_thermal", "runs/p46_videomae_thermal_head_v1/crossfit_logits.npz", "logits", "foundation_thermal"),
    ExpertSpec("videomae_ssv2", "runs/p46_videomae_ssv2_head_v1/crossfit_logits.npz", "logits", "foundation_video"),
    ExpertSpec("videomae_relation", "runs/p46_videomae_relation_head_v1/crossfit_logits.npz", "logits", "foundation_video"),
    ExpertSpec("videomae_large", "runs/p46_videomae_large_head_v1/crossfit_logits.npz", "logits", "foundation_video_large"),
    ExpertSpec("videomae_large_weighted", "runs/p46_videomae_large_weighted_v1/crossfit_logits.npz", "logits", "foundation_video_large_weighted"),
    ExpertSpec("videomae_mc_early", "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz", "early_logits", "foundation_video_multiclip"),
    ExpertSpec("videomae_mc_late", "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz", "late_logits", "foundation_video_multiclip"),
    ExpertSpec("videomae_mc_window_mean", "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz", "window_mean_logits", "foundation_video_multiclip"),
    ExpertSpec("videomae_mc_early_late", "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz", "early_late_logits", "foundation_video_multiclip"),
    ExpertSpec("videomae_mc_full_window_mean", "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz", "full_window_mean_logits", "foundation_video_multiclip"),
    ExpertSpec("videomae_mc_temporal_delta", "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz", "full_temporal_delta_logits", "foundation_video_multiclip"),
    ExpertSpec("videomae_mc_full_early_late", "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz", "full_early_late_logits", "foundation_video_multiclip"),
    ExpertSpec("videomae_mc_kinetics", "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz", "three_clip_kinetics_logits", "foundation_video_multiclip"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a leakage-controlled Detail21 stacker from saved subject-OOF logits. "
            "All selection happens on P46-train users; P46-val is evaluated once."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV fields for empty output: {path}")
        fields = rows[0].keys()
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int((labels == predictions).sum()),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = [row for row in read_csv(path) if row["detail_selected"] == "1"]
    if len(rows) != 1384:
        raise RuntimeError(f"Expected 1384 frozen Detail21 rows, found {len(rows)}")
    if sum(row["p46_split"] == "train" for row in rows) != 1094:
        raise RuntimeError("Frozen P46 training count changed")
    if sum(row["p46_split"] == "val" for row in rows) != 290:
        raise RuntimeError("Frozen P46 validation count changed")
    return rows


def verify_fold_purity(sample_ids: np.ndarray, folds: np.ndarray, rows: list[dict[str, str]]) -> None:
    user_by_sample = {row["sample_id"]: row["user_id"] for row in rows}
    by_user: dict[str, set[int]] = {}
    for sample_id, fold in zip(sample_ids.astype(str), folds.astype(int)):
        user = user_by_sample.get(sample_id)
        if user is not None:
            by_user.setdefault(user, set()).add(int(fold))
    expected_users = {row["user_id"] for row in rows}
    if set(by_user) != expected_users:
        raise RuntimeError("OOF file does not cover every P46 user")
    impure = {user: values for user, values in by_user.items() if len(values) != 1}
    if impure:
        raise RuntimeError(f"OOF fold leakage: users cross folds: {impure}")
    if len({next(iter(values)) for values in by_user.values()}) < 2:
        raise RuntimeError("OOF file does not span multiple held-user folds")


def load_expert(spec: ExpertSpec, rows: list[dict[str, str]]) -> np.ndarray:
    path = PROJECT_DIR / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        fold_key = next(
            (key for key in ("folds", "held_fold", "held_folds") if key in data.files),
            None,
        )
        required = {"sample_ids", "labels", spec.field}
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"{spec.name}: missing {sorted(missing)} in {path}")
        if fold_key is None:
            raise RuntimeError(f"{spec.name}: missing OOF fold vector in {path}")
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        labels = np.asarray(data["labels"], dtype=np.int64)
        folds = np.asarray(data[fold_key], dtype=np.int64)
        logits = np.asarray(data[spec.field], dtype=np.float64)
        valid_mask = (
            np.asarray(data["valid_mask"], dtype=bool)
            if "valid_mask" in data.files
            else np.ones(len(logits), dtype=bool)
        )
    if logits.ndim != 2 or logits.shape[1] not in {21, 40}:
        raise RuntimeError(f"{spec.name}: unsupported logit shape {logits.shape}")
    if not (len(sample_ids) == len(labels) == len(folds) == len(logits)):
        raise RuntimeError(f"{spec.name}: inconsistent NPZ row counts")
    verify_fold_purity(sample_ids, folds, rows)
    index = {sample_id: number for number, sample_id in enumerate(sample_ids)}
    if len(index) != len(sample_ids):
        raise RuntimeError(f"{spec.name}: duplicate sample IDs")
    selected = np.asarray([index[row["sample_id"]] for row in rows], dtype=np.int64)
    expected_labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    if not np.array_equal(labels[selected], expected_labels):
        raise RuntimeError(f"{spec.name}: labels disagree with frozen manifest")
    aligned = logits[selected]
    aligned_valid = valid_mask[selected]
    if aligned.shape[1] == 40:
        aligned = aligned[:, np.asarray(HARD_CLASS_IDS, dtype=np.int64)]
    finite_rows = np.isfinite(aligned).all(axis=1)
    if np.any(~finite_rows & aligned_valid):
        raise RuntimeError(f"{spec.name}: non-finite logits on rows marked valid")
    # Missing-modality experts use NaN for absent trials.  A zero logit vector is
    # the neutral/uniform expert opinion and keeps absence from becoming a label.
    aligned[~finite_rows] = 0.0
    return aligned


def log_softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    return values - logsumexp(values, axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        log_probabilities = log_softmax(logits / temperature)
        return float(-log_probabilities[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(objective, bounds=(-2.302585, 2.302585), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature fit failed: {result.message}")
    return math.exp(float(result.x))


def calibrated_features(
    expert_logits: np.ndarray,
    fit_indices: np.ndarray,
    output_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    temperatures = np.asarray(
        [fit_temperature(values[fit_indices], LABELS[fit_indices]) for values in expert_logits],
        dtype=np.float64,
    )
    blocks = [
        log_softmax(values[output_indices] / temperature)
        for values, temperature in zip(expert_logits, temperatures)
    ]
    return np.concatenate(blocks, axis=1), temperatures


def calibrated_probabilities(
    expert_logits: np.ndarray,
    temperatures: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            np.exp(log_softmax(values[indices] / temperature))
            for values, temperature in zip(expert_logits, temperatures)
        ],
        axis=1,
    )


def fit_mixture_weights(
    probabilities: np.ndarray,
    labels: np.ndarray,
    regularization: float,
) -> np.ndarray:
    expert_count = probabilities.shape[1]

    def objective(raw_weights: np.ndarray) -> float:
        shifted = raw_weights - raw_weights.max()
        weights = np.exp(shifted)
        weights /= weights.sum()
        mixture = np.einsum("ne,ned->nd", weights[None, :].repeat(len(labels), axis=0), probabilities)
        nll = -np.log(np.clip(mixture[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
        penalty = regularization * expert_count * np.square(weights - 1.0 / expert_count).mean()
        return float(nll + penalty)

    result = minimize(
        objective,
        np.zeros(expert_count, dtype=np.float64),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Mixture fit failed: {result.message}")
    shifted = result.x - result.x.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def mixture_predictions(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("e,ned->nd", weights, probabilities).argmax(axis=1)


def cv_score(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    result = metrics(labels, predictions)
    result["log_loss"] = float("nan")
    return result


def choose_by_cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -float(row.get("complexity", 0.0)),
        ),
    )


def fit_selected_model(
    selected: dict[str, Any],
    expert_logits: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_features, temperatures = calibrated_features(expert_logits, train_indices, train_indices)
    val_features = np.concatenate(
        [
            log_softmax(values[val_indices] / temperature)
            for values, temperature in zip(expert_logits, temperatures)
        ],
        axis=1,
    )
    if selected["model"] == "mixture":
        train_probabilities = calibrated_probabilities(expert_logits, temperatures, train_indices)
        val_probabilities = calibrated_probabilities(expert_logits, temperatures, val_indices)
        weights = fit_mixture_weights(
            train_probabilities, LABELS[train_indices], float(selected["regularization"])
        )
        prediction = mixture_predictions(val_probabilities, weights)
        artifact = {
            "model": "constant_probability_mixture",
            "regularization": float(selected["regularization"]),
            "temperatures": temperatures.tolist(),
            "weights": weights.tolist(),
        }
        return prediction, artifact

    model = LogisticRegression(
        C=float(selected["C"]),
        class_weight=None if selected["class_weight"] == "none" else "balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=int(selected["seed"]),
    )
    model.fit(train_features, LABELS[train_indices])
    prediction = model.predict(val_features).astype(np.int64)
    artifact = {
        "model": "multinomial_logistic_stacker",
        "C": float(selected["C"]),
        "class_weight": selected["class_weight"],
        "temperatures": temperatures.tolist(),
        "classes": model.classes_.astype(int).tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
    return prediction, artifact


def main() -> None:
    global LABELS
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest.resolve())
    LABELS = np.asarray([int(row["detail_index"]) for row in rows], dtype=np.int64)
    groups = np.asarray([row["user_id"] for row in rows])
    train_indices = np.flatnonzero(np.asarray([row["p46_split"] == "train" for row in rows]))
    val_indices = np.flatnonzero(np.asarray([row["p46_split"] == "val" for row in rows]))

    print(f"Loading {len(EXPERT_SPECS)} independently OOF expert fields...", flush=True)
    expert_logits = np.stack([load_expert(spec, rows) for spec in EXPERT_SPECS], axis=0)

    expert_rows: list[dict[str, Any]] = []
    for expert_number, spec in enumerate(EXPERT_SPECS):
        train_prediction = expert_logits[expert_number, train_indices].argmax(axis=1)
        val_prediction = expert_logits[expert_number, val_indices].argmax(axis=1)
        train_metric = metrics(LABELS[train_indices], train_prediction)
        val_metric = metrics(LABELS[val_indices], val_prediction)
        expert_rows.append(
            {
                "expert": spec.name,
                "family": spec.family,
                "path": spec.relative_path,
                "field": spec.field,
                **{f"train_{key}": value for key, value in train_metric.items()},
                **{f"val_{key}": value for key, value in val_metric.items()},
            }
        )
    write_csv(output / "expert_metrics.csv", expert_rows)

    splitter = GroupKFold(n_splits=args.cv_splits)
    cv_folds = list(splitter.split(train_indices, LABELS[train_indices], groups[train_indices]))
    cv_rows: list[dict[str, Any]] = []

    mixture_regularizations = (0.0, 0.01, 0.1, 1.0, 10.0)
    for regularization in mixture_regularizations:
        predictions = np.full(len(train_indices), -1, dtype=np.int64)
        for fold, (fit_local, held_local) in enumerate(cv_folds):
            fit_indices = train_indices[fit_local]
            held_indices = train_indices[held_local]
            _, temperatures = calibrated_features(expert_logits, fit_indices, fit_indices)
            fit_probabilities = calibrated_probabilities(expert_logits, temperatures, fit_indices)
            held_probabilities = calibrated_probabilities(expert_logits, temperatures, held_indices)
            weights = fit_mixture_weights(
                fit_probabilities, LABELS[fit_indices], regularization
            )
            predictions[held_local] = mixture_predictions(held_probabilities, weights)
        result = cv_score(LABELS[train_indices], predictions)
        cv_rows.append(
            {
                "model": "mixture",
                "regularization": regularization,
                "C": "",
                "class_weight": "",
                "seed": args.seed,
                "complexity": 1 + (regularization == 0.0),
                **result,
            }
        )
        print(
            f"CV mixture reg={regularization:g}: acc={100*float(result['accuracy']):.2f}% "
            f"macro={100*float(result['macro_f1']):.2f}%",
            flush=True,
        )

    for class_weight in ("none", "balanced"):
        for c_value in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
            predictions = np.full(len(train_indices), -1, dtype=np.int64)
            losses: list[float] = []
            for fold, (fit_local, held_local) in enumerate(cv_folds):
                fit_indices = train_indices[fit_local]
                held_indices = train_indices[held_local]
                fit_features, temperatures = calibrated_features(
                    expert_logits, fit_indices, fit_indices
                )
                held_features = np.concatenate(
                    [
                        log_softmax(values[held_indices] / temperature)
                        for values, temperature in zip(expert_logits, temperatures)
                    ],
                    axis=1,
                )
                model = LogisticRegression(
                    C=c_value,
                    class_weight=None if class_weight == "none" else "balanced",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=args.seed + fold,
                )
                model.fit(fit_features, LABELS[fit_indices])
                predictions[held_local] = model.predict(held_features).astype(np.int64)
                probabilities = model.predict_proba(held_features)
                losses.append(
                    float(log_loss(LABELS[held_indices], probabilities, labels=np.arange(21)))
                )
            result = cv_score(LABELS[train_indices], predictions)
            result["log_loss"] = float(np.mean(losses))
            cv_rows.append(
                {
                    "model": "logistic",
                    "regularization": "",
                    "C": c_value,
                    "class_weight": class_weight,
                    "seed": args.seed,
                    "complexity": c_value * (2 if class_weight == "balanced" else 1),
                    **result,
                }
            )
            print(
                f"CV logistic weight={class_weight} C={c_value:g}: "
                f"acc={100*float(result['accuracy']):.2f}% "
                f"macro={100*float(result['macro_f1']):.2f}%",
                flush=True,
            )

    selected = choose_by_cv(cv_rows)
    write_csv(output / "selector_cv_results.csv", cv_rows)
    print(f"Selected from P46-train Group-CV only: {selected}", flush=True)
    val_prediction_index, model_artifact = fit_selected_model(
        selected, expert_logits, train_indices, val_indices
    )
    val_labels = LABELS[val_indices]
    val_metrics = metrics(val_labels, val_prediction_index)
    val_prediction_class = np.asarray(HARD_CLASS_IDS, dtype=np.int64)[val_prediction_index]

    prediction_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for local, global_index in enumerate(val_indices):
        row = rows[global_index]
        prediction_row = {
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "user_id": row["user_id"],
            "class_id": int(row["class_id"]),
            "class_name": row["class_name"],
            "prediction": int(val_prediction_class[local]),
            "correct": int(val_prediction_class[local] == int(row["class_id"])),
        }
        prediction_rows.append(prediction_row)
        if not prediction_row["correct"]:
            error_rows.append(prediction_row)
    write_csv(output / "validation_predictions.csv", prediction_rows)
    write_csv(
        output / "validation_errors.csv",
        error_rows,
        fields=("sample_id", "source_id", "user_id", "class_id", "class_name", "prediction", "correct"),
    )
    np.savez_compressed(
        output / "stacker_artifact.npz",
        expert_names=np.asarray([spec.name for spec in EXPERT_SPECS]),
        hard_class_ids=np.asarray(HARD_CLASS_IDS, dtype=np.int64),
        temperatures=np.asarray(model_artifact["temperatures"], dtype=np.float64),
        weights=np.asarray(model_artifact.get("weights", []), dtype=np.float64),
        coef=np.asarray(model_artifact.get("coef", []), dtype=np.float64),
        intercept=np.asarray(model_artifact.get("intercept", []), dtype=np.float64),
    )
    summary = {
        "protocol": "P46 Detail21 fixed 14-user train / 4-user development validation",
        "leakage_control": (
            "Every base logit is subject-OOF. Candidate selection, temperature fitting, "
            "stacker hyperparameter selection, and final fitting use only P46-train users. "
            "P46-val labels are used only for the final reported evaluation."
        ),
        "expert_count": len(EXPERT_SPECS),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(val_indices)),
        "train_users": sorted(set(groups[train_indices])),
        "validation_users": sorted(set(groups[val_indices])),
        "selected_cv": selected,
        "validation": val_metrics,
        "model_artifact": model_artifact,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"selected_cv": selected, "validation": val_metrics}, indent=2), flush=True)


LABELS = np.empty(0, dtype=np.int64)


if __name__ == "__main__":
    main()
