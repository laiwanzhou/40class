from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import DEFAULT_TRAIN_METADATA, classification_metrics
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_oof_candidate_ensemble import CANDIDATE_SOURCES, load_candidate, load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p89_teacher_stacker_h1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P89 subject-disjoint complementary-teacher stacker.")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--teacher-targets", type=Path, default=PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz")
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--repeat-config-summary", type=Path, default=PROJECT_DIR / "runs/p88_aligned_repeat_h1_v1/summary.json")
    parser.add_argument("--fixed-summary", type=Path)
    return parser.parse_args()


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(values))


def expert_features(sample_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for name in CANDIDATE_SOURCES:
        values = load_candidate(name, sample_ids)
        logp = log_softmax_numpy(values)
        blocks.append(logp.astype(np.float32))
        names.append(name)
    # The linear stacker gets every class score and can learn class-specific
    # teacher trust.  Entropy and top margins allow confidence-dependent trust.
    probabilities = [np.exp(block) for block in blocks]
    diagnostics = []
    for probability in probabilities:
        ordered = np.sort(probability, axis=1)
        entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
        diagnostics.append(np.stack((entropy, ordered[:, -1], ordered[:, -1] - ordered[:, -2]), axis=1))
    return np.concatenate([*blocks, *diagnostics], axis=1), names


def models(fixed: dict[str, Any] | None) -> list[tuple[str, Any]]:
    if fixed is not None:
        name = str(fixed["model"])
        parameter = float(fixed["regularization"])
        if name == "ridge":
            return [(name, make_pipeline(StandardScaler(), RidgeClassifier(alpha=parameter, class_weight="balanced")))]
        return [(name, make_pipeline(StandardScaler(), LogisticRegression(
            C=parameter, class_weight="balanced", solver="lbfgs", max_iter=500, tol=2e-4,
        )))]
    result: list[tuple[str, Any]] = []
    for alpha in (10.0, 30.0, 100.0, 300.0):
        result.append(("ridge", make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, class_weight="balanced"))))
    for c_value in (0.003, 0.01, 0.03):
        result.append(("logistic", make_pipeline(StandardScaler(), LogisticRegression(
            C=c_value, class_weight="balanced", solver="lbfgs", max_iter=500, tol=2e-4,
        ))))
    return result


def parameter(model: Any, name: str) -> float:
    estimator = model.steps[-1][1]
    return float(estimator.alpha if name == "ridge" else estimator.C)


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args)
    (
        eval_ids, labels, base_probability, base_decoded, metadata, indices,
        sessions, transition, decoder, repeat,
    ) = protocol
    with np.load(args.teacher_targets.resolve(), allow_pickle=False) as teacher:
        all_ids = teacher["oof_sample_ids"].astype(str)
        all_labels = teacher["oof_labels"].astype(np.int64)
    all_x, expert_names = expert_features(all_ids)
    lookup = {sample_id: index for index, sample_id in enumerate(all_ids)}
    evaluation = np.asarray([lookup[sample_id] for sample_id in eval_ids], dtype=np.int64)
    train = np.asarray([index for index, sample_id in enumerate(all_ids) if not any(f"__{user}__" in sample_id for user in args.holdout_users)], dtype=np.int64)
    if np.intersect1d(train, evaluation).size:
        raise RuntimeError("stacker train/evaluation overlap")
    if not np.array_equal(all_labels[evaluation], labels):
        raise RuntimeError("teacher/P87 alignment failed")

    fixed_source = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8")) if args.fixed_summary else None
    fixed_config = fixed_source["selected_config"] if fixed_source else None
    if fixed_config:
        temperatures = [float(fixed_config["temperature"])]
        weights = [float(fixed_config["weight"])]
    else:
        temperatures = [0.75, 1.0, 1.5, 2.0, 3.0]
        weights = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for name, model in models(fixed_config):
        model.fit(all_x[train], all_labels[train])
        decision = np.asarray(model.decision_function(all_x[evaluation]), dtype=np.float64)
        for temperature in temperatures:
            stacked_probability = softmax(decision / temperature)
            for weight in weights:
                blended = (1.0 - weight) * base_probability + weight * stacked_probability
                blended /= blended.sum(axis=1, keepdims=True)
                prediction, grouping = decode_aligned_repeat(
                    np.log(np.maximum(blended, 1e-12)), indices, metadata,
                    transition, decoder, repeat,
                )
                result = {
                    "configuration": {
                        "model": name,
                        "regularization": parameter(model, name),
                        "temperature": temperature,
                        "weight": weight,
                    },
                    "stacker_raw": classification_metrics(labels, stacked_probability.argmax(axis=1)),
                    "metrics": classification_metrics(labels, prediction),
                    "rescue_harm": rescue_harm(labels, base_decoded, prediction),
                    "grouping": grouping,
                }
                results.append(result)
                key = (
                    result["metrics"]["correct"], result["metrics"]["balanced_accuracy"],
                    result["rescue_harm"]["net"], -result["rescue_harm"]["harm"], -weight,
                )
                if best_key is None or key > best_key:
                    best_key, best = key, result
        print(f"finished {name} regularization={parameter(model, name):g}", flush=True)

    assert best is not None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P89_cross_subject_complementary_teacher_stacker_v1",
        "status": "complete",
        "selected_on_current_holdout": args.fixed_summary is None,
        "holdout_users": sorted(args.holdout_users),
        "train_samples": int(len(train)),
        "eval_samples": int(len(evaluation)),
        "feature_dim": int(all_x.shape[1]),
        "expert_names": expert_names,
        "base": classification_metrics(labels, base_decoded),
        "selected_config": best["configuration"],
        "best": best,
        "grid_size": len(results),
        "all_candidates": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "all_candidates"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
