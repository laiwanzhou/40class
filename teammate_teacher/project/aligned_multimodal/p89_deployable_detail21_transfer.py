from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import classification_metrics
from p46_protocol import HARD_CLASS_IDS
from p88_oof_candidate_ensemble import load_candidate, load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm
from p89_detail21_multiexpert_transfer import blend_detail
from p89_global_repeat_decoder import GlobalRepeatConfig, decode_global_repeat


PROJECT_DIR = Path(__file__).resolve().parent
HARD_CLASSES = np.asarray(HARD_CLASS_IDS, dtype=np.int64)
P46_CANDIDATE_PATH = (
    PROJECT_DIR
    / "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz"
)
P46_KEYS = (
    "full_logits",
    "early_logits",
    "late_logits",
    "window_mean_logits",
    "early_late_logits",
    "full_window_mean_logits",
    "full_temporal_delta_logits",
    "full_early_late_logits",
    "three_clip_kinetics_logits",
)
P46_LARGE_RUNS = (
    "p46_videomae_base_large_joint_v1",
    "p46_videomae_large_bagging_v1",
    "p46_videomae_large_head_v1",
    "p46_videomae_large_weighted_v1",
    "p46_videomae_subject_svm_v1",
)
DEPLOYABLE_P85 = (
    "p85_teacher",
    "p85_head_early_logits",
    "p85_head_late_logits",
    "p85_head_window_mean_logits",
    "p85_head_early_late_logits",
    "p85_head_temporal_delta_logits",
    "p85_head_kinetics_logits",
    "p12_skeleton",
    "p12_thermal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a Detail21 stacker using only experts that have an exact "
            "full-refit Test counterpart. Hyperparameters are selected on H1 and "
            "confirmed once on H2."
        )
    )
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--selection-users", nargs="+", required=True)
    parser.add_argument("--confirmation-run", type=Path, required=True)
    parser.add_argument("--confirmation-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--global-repeat-summary",
        type=Path,
        default=PROJECT_DIR / "runs/p89_global_repeat_h1_v1/summary.json",
    )
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        default=PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz",
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
    )
    parser.add_argument(
        "--repeat-config-summary",
        type=Path,
        default=PROJECT_DIR / "runs/p88_aligned_repeat_h1_v1/summary.json",
    )
    return parser.parse_args()


def protocol(args: argparse.Namespace, run: Path, users: list[str]):
    return load_protocol(
        SimpleNamespace(
            base_run=run,
            holdout_users=users,
            teacher_targets=args.teacher_targets,
            train_metadata=args.train_metadata,
            repeat_config_summary=args.repeat_config_summary,
        )
    )


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(np.asarray(values, dtype=np.float64)))


def aligned_rows(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    missing = [value for value in target_ids.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"missing {len(missing)} rows while aligning deployable experts")
    return np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)


def feature_matrix(probabilities: list[np.ndarray]) -> np.ndarray:
    all_probability = np.stack(probabilities, axis=1)
    ordered = np.sort(all_probability, axis=2)
    entropy = -np.sum(
        all_probability * np.log(np.maximum(all_probability, 1e-12)), axis=2
    )
    diagnostics = np.stack(
        (entropy, ordered[:, :, -1], ordered[:, :, -1] - ordered[:, :, -2]),
        axis=2,
    )
    aggregates = np.concatenate(
        (
            all_probability.mean(axis=1),
            all_probability.max(axis=1),
            all_probability.std(axis=1),
            np.median(all_probability, axis=1),
        ),
        axis=1,
    )
    return np.concatenate(
        probabilities + [diagnostics.reshape(len(all_probability), -1), aggregates],
        axis=1,
    ).astype(np.float32)


def build_features(reference_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probabilities: list[np.ndarray] = []
    names: list[str] = []
    with np.load(P46_CANDIDATE_PATH, allow_pickle=False) as source:
        order = aligned_rows(source["sample_ids"], reference_ids)
        for key in P46_KEYS:
            probabilities.append(softmax(source[key][order]))
            names.append(f"p46_mc_{key.removesuffix('_logits')}")
    for run_name in P46_LARGE_RUNS:
        with np.load(
            PROJECT_DIR / "runs" / run_name / "crossfit_logits.npz",
            allow_pickle=False,
        ) as source:
            order = aligned_rows(source["sample_ids"], reference_ids)
            probabilities.append(softmax(source["logits"][order]))
            names.append(run_name)
    for name in DEPLOYABLE_P85:
        probability40 = softmax(load_candidate(name, reference_ids))
        probability21 = probability40[:, HARD_CLASSES]
        probability21 /= np.maximum(probability21.sum(axis=1, keepdims=True), 1e-12)
        probabilities.append(probability21)
        names.append(name)
    return feature_matrix(probabilities), names


def make_model(configuration: dict[str, Any]):
    if configuration["model"] == "ridge":
        return RidgeClassifier(
            alpha=float(configuration["regularization"]), class_weight="balanced"
        )
    return LogisticRegression(
        C=float(configuration["regularization"]),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        tol=2e-4,
    )


def model_probability(model, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        partial = np.asarray(model.predict_proba(values), dtype=np.float64)
    else:
        partial = softmax(np.asarray(model.decision_function(values), dtype=np.float64))
    result = np.full((len(values), len(HARD_CLASSES)), 1e-12, dtype=np.float64)
    class_to_column = {int(value): index for index, value in enumerate(HARD_CLASSES)}
    for source_column, class_id in enumerate(np.asarray(model.classes_, dtype=np.int64)):
        result[:, class_to_column[int(class_id)]] = partial[:, source_column]
    result /= result.sum(axis=1, keepdims=True)
    return result


def decode(probability, protocol_value, config):
    _, labels, _, _, metadata, indices, _, transition, decoder, _ = protocol_value
    prediction, grouping = decode_global_repeat(
        np.log(np.maximum(probability, 1e-12)),
        indices,
        metadata,
        transition,
        decoder,
        config,
    )
    return prediction, grouping, classification_metrics(labels, prediction)


def main() -> None:
    args = parse_args()
    selection = protocol(args, args.selection_run.resolve(), list(args.selection_users))
    confirmation = protocol(
        args, args.confirmation_run.resolve(), list(args.confirmation_users)
    )
    ids1, labels1, base1, *_ = selection
    ids2, labels2, base2, *_ = confirmation
    repeat_source = json.loads(
        args.global_repeat_summary.resolve().read_text(encoding="utf-8")
    )
    repeat = GlobalRepeatConfig(**repeat_source["selected_config"])
    base_prediction1, base_grouping1, base_metrics1 = decode(base1, selection, repeat)
    base_prediction2, base_grouping2, base_metrics2 = decode(base2, confirmation, repeat)

    with np.load(P46_CANDIDATE_PATH, allow_pickle=False) as reference:
        detail_ids = reference["sample_ids"].astype(str)
        detail_labels = reference["labels"].astype(np.int64)
        detail_users = reference["users"].astype(str)
    features, expert_names = build_features(detail_ids)
    fit_mask = ~np.isin(detail_users, list(args.confirmation_users))
    fit_indices = np.flatnonzero(fit_mask)
    fit_groups = detail_users[fit_mask]
    configurations = [
        *(
            {"model": "ridge", "regularization": value}
            for value in (10.0, 30.0, 100.0, 300.0, 1000.0)
        ),
        *(
            {"model": "logistic", "regularization": value}
            for value in (0.0003, 0.001, 0.003, 0.01)
        ),
    ]
    temperatures = (0.5, 0.75, 1.0, 1.5, 2.0)
    weights = (0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50)
    splitter = GroupKFold(n_splits=5)
    candidates: list[dict[str, Any]] = []
    best = None
    best_key = None
    best_selection_probability = None
    best_selection_prediction = None
    for configuration in configurations:
        crossfit = np.zeros((len(detail_ids), len(HARD_CLASSES)), dtype=np.float64)
        for train_relative, valid_relative in splitter.split(
            features[fit_mask], detail_labels[fit_mask], fit_groups
        ):
            train_rows = fit_indices[train_relative]
            valid_rows = fit_indices[valid_relative]
            scaler = StandardScaler()
            train_x = scaler.fit_transform(features[train_rows])
            valid_x = scaler.transform(features[valid_rows])
            model = make_model(configuration)
            model.fit(train_x, detail_labels[train_rows])
            crossfit[valid_rows] = model_probability(model, valid_x)
        for temperature in temperatures:
            for weight in weights:
                probability = blend_detail(
                    base1, ids1, detail_ids, crossfit, temperature, weight
                )
                prediction, grouping, metrics = decode(probability, selection, repeat)
                change = rescue_harm(labels1, base_prediction1, prediction)
                item = {
                    "configuration": configuration,
                    "temperature": temperature,
                    "weight": weight,
                    "metrics": metrics,
                    "rescue_harm": change,
                    "grouping": grouping,
                }
                candidates.append(item)
                key = (
                    metrics["correct"],
                    metrics["balanced_accuracy"],
                    change["net"],
                    -change["harm"],
                    -weight,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = item
                    best_selection_probability = probability.copy()
                    best_selection_prediction = prediction.copy()
        print(f"finished {configuration}", flush=True)
    assert best is not None

    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[fit_mask])
    confirmation_mask = np.isin(detail_users, list(args.confirmation_users))
    confirmation_x = scaler.transform(features[confirmation_mask])
    model = make_model(best["configuration"])
    model.fit(train_x, detail_labels[fit_mask])
    detail_probability2 = model_probability(model, confirmation_x)
    detail_ids2 = detail_ids[confirmation_mask]
    blended2 = blend_detail(
        base2,
        ids2,
        detail_ids2,
        detail_probability2,
        float(best["temperature"]),
        float(best["weight"]),
    )
    prediction2, grouping2, metrics2 = decode(blended2, confirmation, repeat)
    confirmation_result = {
        "metrics": metrics2,
        "rescue_harm": rescue_harm(labels2, base_prediction2, prediction2),
        "grouping": grouping2,
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P89_deployable_detail21_H1_select_H2_confirm_v1",
        "status": "complete",
        "protocol": (
            "Every input expert has a frozen subject-OOF training output and a "
            "matching full-refit Test path. H2 users are excluded from stacker fit; "
            "model and blend hyperparameters are selected on H1 only."
        ),
        "expert_names": expert_names,
        "feature_dim": int(features.shape[1]),
        "fit_rows": int(fit_mask.sum()),
        "confirmation_detail_rows": int(confirmation_mask.sum()),
        "selection_base": base_metrics1,
        "selection_base_grouping": base_grouping1,
        "selection_best": best,
        "confirmation_base": base_metrics2,
        "confirmation_base_grouping": base_grouping2,
        "confirmation": confirmation_result,
        "grid_size": len(candidates),
        "all_selection_candidates": candidates,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "confirmation_predictions.npz",
        selection_sample_ids=ids1,
        selection_labels=labels1,
        selection_base_probability=base1.astype(np.float32),
        selection_blended_probability=best_selection_probability.astype(np.float32),
        selection_base_prediction=base_prediction1,
        selection_prediction=best_selection_prediction,
        sample_ids=ids2,
        labels=labels2,
        base_probability=base2.astype(np.float32),
        blended_probability=blended2.astype(np.float32),
        prediction=prediction2,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "all_selection_candidates"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
