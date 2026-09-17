from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import classification_metrics
from p88_oof_candidate_ensemble import CANDIDATE_SOURCES, load_candidate, load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm
from p89_global_repeat_decoder import GlobalRepeatConfig, decode_global_repeat


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P89 H1 user-LOO expert router with untouched H2 transfer.")
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--selection-users", nargs="+", required=True)
    parser.add_argument("--confirmation-run", type=Path, required=True)
    parser.add_argument("--confirmation-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-repeat-summary", type=Path, default=PROJECT_DIR / "runs/p89_global_repeat_h1_v1/summary.json")
    parser.add_argument("--teacher-targets", type=Path, default=PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz")
    parser.add_argument("--train-metadata", type=Path, default=PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv")
    parser.add_argument("--repeat-config-summary", type=Path, default=PROJECT_DIR / "runs/p88_aligned_repeat_h1_v1/summary.json")
    parser.add_argument("--class-bias", type=Path)
    return parser.parse_args()


def protocol(args: argparse.Namespace, run: Path, users: list[str]):
    return load_protocol(SimpleNamespace(
        base_run=run, holdout_users=users, teacher_targets=args.teacher_targets,
        train_metadata=args.train_metadata, repeat_config_summary=args.repeat_config_summary,
    ))


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(values))


def router_features(sample_ids: np.ndarray, base_probability: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probabilities = [base_probability]
    names = ["p87"]
    for name in CANDIDATE_SOURCES:
        probabilities.append(softmax(load_candidate(name, sample_ids)))
        names.append(name)
    stacked = np.stack(probabilities, axis=1)
    ordered = np.sort(stacked, axis=2)
    entropy = -np.sum(stacked * np.log(np.maximum(stacked, 1e-12)), axis=2)
    diagnostics = np.stack((entropy, ordered[:, :, -1], ordered[:, :, -1] - ordered[:, :, -2]), axis=2)
    aggregates = np.concatenate((
        stacked.mean(axis=1), stacked.max(axis=1), stacked.std(axis=1), np.median(stacked, axis=1),
    ), axis=1)
    # Individual expert posteriors preserve pair-specific evidence; aggregate
    # statistics make the regularized linear router robust to redundant heads.
    features = np.concatenate((stacked.reshape(len(stacked), -1), diagnostics.reshape(len(stacked), -1), aggregates), axis=1)
    return features.astype(np.float32), names


def make_model(name: str, regularization: float):
    if name == "ridge":
        return RidgeClassifier(alpha=regularization, class_weight="balanced")
    return LogisticRegression(
        C=regularization, class_weight="balanced", solver="lbfgs", max_iter=600, tol=2e-4,
    )


def probability(model, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        partial = np.asarray(model.predict_proba(values), dtype=np.float64)
    else:
        partial = softmax(np.asarray(model.decision_function(values), dtype=np.float64))
    classes = np.asarray(model.classes_, dtype=np.int64)
    if len(classes) == 40 and np.array_equal(classes, np.arange(40)):
        return partial
    result = np.full((len(values), 40), 1e-12, dtype=np.float64)
    result[:, classes] = partial
    result /= result.sum(axis=1, keepdims=True)
    return result


def decode(probability_value, protocol_value, config):
    _, labels, _, _, metadata, indices, _, transition, decoder, _ = protocol_value
    prediction, grouping = decode_global_repeat(
        np.log(np.maximum(probability_value, 1e-12)), indices, metadata, transition, decoder, config,
    )
    return prediction, grouping, classification_metrics(labels, prediction)


def main() -> None:
    args = parse_args()
    selection = protocol(args, args.selection_run.resolve(), list(args.selection_users))
    confirmation = protocol(args, args.confirmation_run.resolve(), list(args.confirmation_users))
    ids1, labels1, base1, _, metadata1, *_ = selection
    ids2, labels2, base2, _, metadata2, *_ = confirmation
    x1, expert_names = router_features(ids1, base1)
    x2, names2 = router_features(ids2, base2)
    if names2 != expert_names:
        raise RuntimeError("expert mismatch")
    global_source = json.loads(args.global_repeat_summary.resolve().read_text(encoding="utf-8"))
    global_config = GlobalRepeatConfig(**global_source["selected_config"])
    base_prediction1, base_grouping1, base_metrics1 = decode(base1, selection, global_config)
    base_prediction2, base_grouping2, base_metrics2 = decode(base2, confirmation, global_config)

    model_configs = [
        *(('ridge', alpha) for alpha in (10.0, 30.0, 100.0, 300.0, 1000.0)),
        *(('logistic', c_value) for c_value in (0.0003, 0.001, 0.003, 0.01, 0.03)),
    ]
    temperatures = (0.5, 0.75, 1.0, 1.5, 2.0)
    weights = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)
    candidates: list[dict[str, Any]] = []
    best = None
    best_key = None
    best_selection_probability = None
    best_selection_prediction = None
    users1 = metadata1.users.astype(str)
    for name, regularization in model_configs:
        crossfit = np.zeros_like(base1)
        for user in sorted(set(users1.tolist())):
            valid = users1 == user
            fit = ~valid
            scaler = StandardScaler()
            train_x = scaler.fit_transform(x1[fit])
            valid_x = scaler.transform(x1[valid])
            model = make_model(name, regularization)
            model.fit(train_x, labels1[fit])
            crossfit[valid] = probability(model, valid_x)
        for temperature in temperatures:
            routed = softmax(np.log(np.maximum(crossfit, 1e-12)) / temperature)
            for weight in weights:
                blended = (1.0 - weight) * base1 + weight * routed
                blended /= blended.sum(axis=1, keepdims=True)
                prediction, grouping, metrics = decode(blended, selection, global_config)
                item = {
                    "configuration": {"model": name, "regularization": regularization, "temperature": temperature, "weight": weight},
                    "router_raw": classification_metrics(labels1, routed.argmax(axis=1)),
                    "metrics": metrics,
                    "rescue_harm": rescue_harm(labels1, base_prediction1, prediction),
                    "grouping": grouping,
                }
                candidates.append(item)
                key = (metrics["correct"], metrics["balanced_accuracy"], item["rescue_harm"]["net"], -item["rescue_harm"]["harm"], -weight)
                if best_key is None or key > best_key:
                    best_key, best = key, item
                    best_selection_probability = blended.copy()
                    best_selection_prediction = prediction.copy()
        print(f"finished user-LOO {name} {regularization:g}", flush=True)
    assert best is not None
    assert best_selection_probability is not None
    assert best_selection_prediction is not None

    chosen = best["configuration"]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(x1)
    confirmation_x = scaler.transform(x2)
    final_model = make_model(chosen["model"], float(chosen["regularization"]))
    final_model.fit(train_x, labels1)
    routed2 = probability(final_model, confirmation_x)
    routed2 = softmax(np.log(np.maximum(routed2, 1e-12)) / float(chosen["temperature"]))
    blended2 = (1.0 - float(chosen["weight"])) * base2 + float(chosen["weight"]) * routed2
    blended2 /= blended2.sum(axis=1, keepdims=True)
    prediction2, grouping2, metrics2 = decode(blended2, confirmation, global_config)
    confirmation_result = {
        "router_raw": classification_metrics(labels2, routed2.argmax(axis=1)),
        "metrics": metrics2, "rescue_harm": rescue_harm(labels2, base_prediction2, prediction2),
        "grouping": grouping2,
    }

    biased_result = None
    if args.class_bias:
        logits2 = np.asarray(np.load(args.confirmation_run.resolve() / "subject_holdout_logits.npy"), dtype=np.float64)
        bias = np.asarray(np.load(args.class_bias.resolve()), dtype=np.float64)
        biased_base = softmax(logits2 + bias)
        biased_base_prediction, _, biased_base_metrics = decode(biased_base, confirmation, global_config)
        biased_blend = (1.0 - float(chosen["weight"])) * biased_base + float(chosen["weight"]) * routed2
        biased_blend /= biased_blend.sum(axis=1, keepdims=True)
        biased_prediction, biased_grouping, biased_metrics = decode(biased_blend, confirmation, global_config)
        biased_result = {
            "base": biased_base_metrics, "metrics": biased_metrics,
            "rescue_harm": rescue_harm(labels2, biased_base_prediction, biased_prediction),
            "grouping": biased_grouping,
        }

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P89_supervised_expert_router_H1_LOO_to_H2_v1", "status": "complete",
        "protocol": "Router hyperparameters selected by leave-one-user-out inside H1, refit on all H1, transferred untouched to H2.",
        "expert_names": expert_names, "feature_dim": int(x1.shape[1]),
        "selection_base": base_metrics1, "selection_base_grouping": base_grouping1,
        "selected_config": chosen, "selection_best_user_loo": best,
        "confirmation_base": base_metrics2, "confirmation_base_grouping": base_grouping2,
        "confirmation": confirmation_result, "confirmation_with_H1_class_bias": biased_result,
        "grid_size": len(candidates), "all_selection_candidates": candidates,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "confirmation_predictions.npz",
        selection_sample_ids=ids1,
        selection_labels=labels1,
        selection_base_probability=base1.astype(np.float32),
        selection_blended_probability=best_selection_probability.astype(np.float32),
        selection_base_prediction=base_prediction1,
        selection_routed_prediction=best_selection_prediction,
        sample_ids=ids2,
        labels=labels2,
        base_probability=base2.astype(np.float32),
        routed_probability=routed2.astype(np.float32),
        blended_probability=blended2.astype(np.float32),
        base_prediction=base_prediction2,
        routed_prediction=prediction2,
        biased_base_prediction=(biased_base_prediction if biased_result is not None else np.empty(0, dtype=np.int64)),
        biased_routed_prediction=(biased_prediction if biased_result is not None else np.empty(0, dtype=np.int64)),
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "all_selection_candidates"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
