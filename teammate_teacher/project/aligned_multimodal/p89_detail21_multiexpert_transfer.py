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
from train_p46_oof_stacker import EXPERT_SPECS, load_expert, load_rows
from p88_oof_candidate_ensemble import CANDIDATE_SOURCES, load_candidate, load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm
from p89_global_repeat_decoder import GlobalRepeatConfig, decode_global_repeat


PROJECT_DIR = Path(__file__).resolve().parent
HARD_CLASSES = np.asarray(HARD_CLASS_IDS, dtype=np.int64)
P46_EXPERT_RUNS = (
    "p46_70_subject_calibrated_v2",
    "p46_dinov2_base_head_v1",
    "p46_validation70_final_v1",
    "p46_videomae_base_large_joint_v1",
    "p46_videomae_depth_head_v1",
    "p46_videomae_head_v1",
    "p46_videomae_ir_depth_head_v1",
    "p46_videomae_large_bagging_v1",
    "p46_videomae_large_head_v1",
    "p46_videomae_large_multiclip_head_v1",
    "p46_videomae_large_weighted_v1",
    "p46_videomae_relation_head_v1",
    "p46_videomae_ssv2_head_v1",
    "p46_videomae_subject_svm_v1",
    "p46_videomae_temporal_head_v2",
    "p46_videomae_thermal_head_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P89 detail-21 OOF multiexpert stacker, selected on H1 and confirmed on H2."
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
    parser.add_argument("--class-bias", type=Path)
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
    return np.exp(log_softmax_numpy(values))


def aligned_rows(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    missing = [value for value in target_ids.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"missing {len(missing)} rows while aligning detail experts")
    return np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)


def build_features(reference_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    p46_probabilities: list[np.ndarray] = []
    for run_name in P46_EXPERT_RUNS:
        with np.load(PROJECT_DIR / "runs" / run_name / "crossfit_logits.npz") as source:
            order = aligned_rows(source["sample_ids"], reference_ids)
            probability = softmax(np.asarray(source["logits"], dtype=np.float64)[order])
        p46_probabilities.append(probability)
        blocks.append(probability)
        names.append(run_name)

    # Recover the broad, subject-OOF expert bank built before P87.  P89's first
    # detail stacker used only the newest VideoMAE heads plus P85 candidates;
    # these older depth-difference, local-depth, thermal, IMU and adapter heads
    # contribute deliberately different error modes on the 21 object/detail
    # classes.  Duplicate visual families are harmless under the strong L2
    # regularization selected by grouped OOF.
    frozen_rows = load_rows(PROJECT_DIR / "data/p46_single_split.csv")
    frozen_ids = np.asarray([row["sample_id"] for row in frozen_rows])
    if not np.array_equal(frozen_ids.astype(str), reference_ids.astype(str)):
        raise RuntimeError("P46 frozen manifest order changed")
    for spec in EXPERT_SPECS:
        probability = softmax(load_expert(spec, frozen_rows))
        p46_probabilities.append(probability)
        blocks.append(probability)
        names.append(f"legacy_{spec.name}")

    candidate_probabilities: list[np.ndarray] = []
    for name in CANDIDATE_SOURCES:
        probability40 = softmax(load_candidate(name, reference_ids))
        probability = probability40[:, HARD_CLASSES]
        probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
        candidate_probabilities.append(probability)
        blocks.append(probability)
        names.append(name)

    all_probability = np.stack(p46_probabilities + candidate_probabilities, axis=1)
    ordered = np.sort(all_probability, axis=2)
    entropy = -np.sum(
        all_probability * np.log(np.maximum(all_probability, 1e-12)), axis=2
    )
    diagnostics = np.stack(
        (entropy, ordered[:, :, -1], ordered[:, :, -1] - ordered[:, :, -2]), axis=2
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
    blocks.extend((diagnostics.reshape(len(reference_ids), -1), aggregates))
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def make_model(config: dict[str, Any]):
    if config["model"] == "ridge":
        return RidgeClassifier(
            alpha=float(config["regularization"]), class_weight="balanced"
        )
    return LogisticRegression(
        C=float(config["regularization"]),
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


def decode(probability_value, protocol_value, config):
    _, labels, _, _, metadata, indices, _, transition, decoder, _ = protocol_value
    prediction, grouping = decode_global_repeat(
        np.log(np.maximum(probability_value, 1e-12)),
        indices,
        metadata,
        transition,
        decoder,
        config,
    )
    return prediction, grouping, classification_metrics(labels, prediction)


def blend_detail(
    base: np.ndarray,
    sample_ids: np.ndarray,
    detail_ids: np.ndarray,
    detail_probability: np.ndarray,
    temperature: float,
    weight: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    lookup = {value: index for index, value in enumerate(detail_ids.astype(str))}
    present_positions = [i for i, value in enumerate(sample_ids.astype(str)) if value in lookup]
    if not present_positions:
        return output
    detail_positions = np.asarray(
        [lookup[sample_ids[i]] for i in present_positions], dtype=np.int64
    )
    present_positions_array = np.asarray(present_positions, dtype=np.int64)
    tempered = softmax(
        np.log(np.maximum(detail_probability[detail_positions], 1e-12)) / temperature
    )
    expanded = np.zeros((len(present_positions), 40), dtype=np.float64)
    expanded[:, HARD_CLASSES] = tempered
    # Preserve the base mass assigned outside the hard subset. This makes the
    # specialist a residual within its family instead of a 21-class override.
    hard_mass = output[present_positions_array][:, HARD_CLASSES].sum(axis=1, keepdims=True)
    expanded *= hard_mass
    output[present_positions_array] = (
        (1.0 - weight) * output[present_positions_array] + weight * expanded
    )
    output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-12)
    return output


def main() -> None:
    args = parse_args()
    selection = protocol(args, args.selection_run.resolve(), list(args.selection_users))
    confirmation = protocol(args, args.confirmation_run.resolve(), list(args.confirmation_users))
    ids1, labels1, base1, _, _, *_ = selection
    ids2, labels2, base2, _, _, *_ = confirmation
    global_source = json.loads(args.global_repeat_summary.resolve().read_text(encoding="utf-8"))
    global_config = GlobalRepeatConfig(**global_source["selected_config"])
    base_prediction1, base_grouping1, base_metrics1 = decode(base1, selection, global_config)
    base_prediction2, base_grouping2, base_metrics2 = decode(base2, confirmation, global_config)

    reference_path = PROJECT_DIR / "runs/p46_validation70_final_v1/crossfit_logits.npz"
    with np.load(reference_path) as reference:
        detail_ids = reference["sample_ids"].astype(str)
        detail_labels = reference["labels"].astype(np.int64)
        detail_users = reference["users"].astype(str)
    features, expert_names = build_features(detail_ids)
    fit_mask = ~np.isin(detail_users, list(args.confirmation_users))
    fit_indices = np.flatnonzero(fit_mask)
    fit_groups = detail_users[fit_mask]

    model_configs = [
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
    best: dict[str, Any] | None = None
    best_key = None
    best_selection_probability: np.ndarray | None = None
    best_selection_prediction: np.ndarray | None = None
    for model_config in model_configs:
        crossfit_probability = np.zeros((len(detail_ids), len(HARD_CLASSES)), dtype=np.float64)
        for train_relative, valid_relative in splitter.split(
            features[fit_mask], detail_labels[fit_mask], fit_groups
        ):
            train_rows = fit_indices[train_relative]
            valid_rows = fit_indices[valid_relative]
            scaler = StandardScaler()
            train_x = scaler.fit_transform(features[train_rows])
            valid_x = scaler.transform(features[valid_rows])
            model = make_model(model_config)
            model.fit(train_x, detail_labels[train_rows])
            crossfit_probability[valid_rows] = model_probability(model, valid_x)
        for temperature in temperatures:
            for weight in weights:
                blended = blend_detail(
                    base1,
                    ids1,
                    detail_ids,
                    crossfit_probability,
                    temperature,
                    weight,
                )
                prediction, grouping, metrics = decode(blended, selection, global_config)
                change = rescue_harm(labels1, base_prediction1, prediction)
                item = {
                    "configuration": model_config,
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
                    best_key, best = key, item
                    best_selection_probability = blended.copy()
                    best_selection_prediction = prediction.copy()
        print(f"finished {model_config}", flush=True)
    assert best is not None
    assert best_selection_probability is not None
    assert best_selection_prediction is not None

    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[fit_mask])
    confirmation_rows = np.isin(detail_users, list(args.confirmation_users))
    confirmation_x = scaler.transform(features[confirmation_rows])
    final_model = make_model(best["configuration"])
    final_model.fit(train_x, detail_labels[fit_mask])
    confirmation_detail_probability = model_probability(final_model, confirmation_x)
    confirmation_detail_ids = detail_ids[confirmation_rows]
    blended2 = blend_detail(
        base2,
        ids2,
        confirmation_detail_ids,
        confirmation_detail_probability,
        float(best["temperature"]),
        float(best["weight"]),
    )
    prediction2, grouping2, metrics2 = decode(blended2, confirmation, global_config)
    confirmation_result = {
        "metrics": metrics2,
        "rescue_harm": rescue_harm(labels2, base_prediction2, prediction2),
        "grouping": grouping2,
    }

    biased_result = None
    biased_blended2 = None
    biased_prediction2 = None
    if args.class_bias:
        logits2 = np.asarray(
            np.load(args.confirmation_run.resolve() / "subject_holdout_logits.npy"),
            dtype=np.float64,
        )
        bias = np.asarray(np.load(args.class_bias.resolve()), dtype=np.float64)
        biased_base = softmax(logits2 + bias)
        biased_base_prediction, biased_grouping, biased_base_metrics = decode(
            biased_base, confirmation, global_config
        )
        biased_blended2 = blend_detail(
            biased_base,
            ids2,
            confirmation_detail_ids,
            confirmation_detail_probability,
            float(best["temperature"]),
            float(best["weight"]),
        )
        biased_prediction2, biased_grouping2, biased_metrics2 = decode(
            biased_blended2, confirmation, global_config
        )
        biased_result = {
            "base": biased_base_metrics,
            "metrics": biased_metrics2,
            "rescue_harm": rescue_harm(
                labels2, biased_base_prediction, biased_prediction2
            ),
            "grouping": biased_grouping2,
        }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P89_detail21_multiexpert_H1_select_H2_confirm_v1",
        "status": "complete",
        "protocol": (
            "P46 expert inputs are subject-OOF. The stacker excludes all H2 users; "
            "its model and blend hyperparameters are selected using grouped OOF and H1 only."
        ),
        "expert_names": expert_names,
        "feature_dim": int(features.shape[1]),
        "fit_rows": int(fit_mask.sum()),
        "confirmation_detail_rows": int(confirmation_rows.sum()),
        "selection_base": base_metrics1,
        "selection_base_grouping": base_grouping1,
        "selection_best": best,
        "confirmation_base": base_metrics2,
        "confirmation_base_grouping": base_grouping2,
        "confirmation": confirmation_result,
        "confirmation_with_H1_class_bias": biased_result,
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
        biased_blended_probability=(
            biased_blended2.astype(np.float32)
            if biased_blended2 is not None
            else np.empty((0, 40), dtype=np.float32)
        ),
        biased_prediction=(
            biased_prediction2
            if biased_prediction2 is not None
            else np.empty(0, dtype=np.int64)
        ),
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
