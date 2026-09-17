from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import classification_metrics
from p88_oof_candidate_ensemble import load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm
from p89_global_repeat_decoder import GlobalRepeatConfig, decode_global_repeat


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs/p89_crossmodal_statistics_v1/crossmodal_statistics.npz"
DEFAULT_GLOBAL = PROJECT_DIR / "runs/p89_global_repeat_h1_v1/summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P89 Skeleton/IMU cross-modal statistics probe and P87 fusion.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-repeat-summary", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--teacher-targets", type=Path, default=PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz")
    parser.add_argument("--train-metadata", type=Path, default=PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv")
    parser.add_argument("--repeat-config-summary", type=Path, default=PROJECT_DIR / "runs/p88_aligned_repeat_h1_v1/summary.json")
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument("--class-bias", type=Path)
    return parser.parse_args()


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(values))


def configurations(fixed: dict[str, Any] | None) -> list[dict[str, Any]]:
    if fixed:
        return [fixed]
    return [
        *({"model": "ridge", "regularization": alpha} for alpha in (10.0, 30.0, 100.0, 300.0)),
        *({"model": "logistic", "regularization": c_value} for c_value in (0.01, 0.03, 0.10)),
        {"model": "lightgbm", "regularization": 3.0, "leaves": 7, "estimators": 160},
        {"model": "lightgbm", "regularization": 5.0, "leaves": 15, "estimators": 160},
    ]


def fit_predict(config: dict[str, Any], train_x, train_y, eval_x) -> np.ndarray:
    if config["model"] == "ridge":
        model = RidgeClassifier(alpha=float(config["regularization"]), class_weight="balanced")
        model.fit(train_x, train_y)
        return softmax(np.asarray(model.decision_function(eval_x), dtype=np.float64))
    if config["model"] == "logistic":
        model = LogisticRegression(
            C=float(config["regularization"]), class_weight="balanced", solver="lbfgs",
            max_iter=500, tol=2e-4,
        )
        model.fit(train_x, train_y)
        return np.asarray(model.predict_proba(eval_x), dtype=np.float64)
    model = LGBMClassifier(
        objective="multiclass", num_class=40, n_estimators=int(config["estimators"]),
        learning_rate=0.04, num_leaves=int(config["leaves"]), max_depth=5,
        min_child_samples=18, subsample=0.85, colsample_bytree=0.75,
        reg_alpha=1.0, reg_lambda=float(config["regularization"]),
        class_weight="balanced", random_state=20260816, n_jobs=-1, verbosity=-1,
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict_proba(eval_x), dtype=np.float64)


def main() -> None:
    args = parse_args()
    with np.load(args.features.resolve(), allow_pickle=False) as source:
        sample_ids = source["sample_ids"].astype(str)
        users = source["users"].astype(str)
        labels_all = source["labels"].astype(np.int64)
        features = source["features"].astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("cross-modal feature cache contains non-finite values")
    protocol = load_protocol(args)
    eval_ids, labels, base_probability, base_decoded, metadata, indices, sessions, transition, decoder, repeat = protocol
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    evaluation = np.asarray([lookup[sample_id] for sample_id in eval_ids], dtype=np.int64)
    train = np.flatnonzero(~np.isin(users, args.holdout_users))
    if np.intersect1d(train, evaluation).size or not np.array_equal(labels_all[evaluation], labels):
        raise RuntimeError("feature split/alignment failure")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(features[train])
    eval_scaled = scaler.transform(features[evaluation])
    components = min(384, len(train) - 40, features.shape[1])
    pca = PCA(n_components=components, whiten=True, svd_solver="randomized", iterated_power=3, random_state=20260816)
    train_x = pca.fit_transform(train_scaled)
    eval_x = pca.transform(eval_scaled)
    print(f"PCA components={components} explained={pca.explained_variance_ratio_.sum():.4f}", flush=True)

    if args.class_bias:
        logits = np.asarray(np.load(args.base_run.resolve() / "subject_holdout_logits.npy"), dtype=np.float64)
        bias = np.asarray(np.load(args.class_bias.resolve()), dtype=np.float64)
        base_probability = softmax(logits + bias)
    global_source = json.loads(args.global_repeat_summary.resolve().read_text(encoding="utf-8"))
    global_config = GlobalRepeatConfig(**global_source["selected_config"])
    base_prediction, base_grouping = decode_global_repeat(
        np.log(np.maximum(base_probability, 1e-12)), indices, metadata, transition, decoder, global_config,
    )

    fixed_source = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8")) if args.fixed_summary else None
    fixed = fixed_source["selected_config"] if fixed_source else None
    model_configs = configurations(fixed["model_config"] if fixed else None)
    temperatures = [float(fixed["temperature"])] if fixed else [0.5, 0.75, 1.0, 1.5, 2.0]
    weights = [float(fixed["weight"])] if fixed else [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    results: list[dict[str, Any]] = []
    best = None
    best_key = None
    for config in model_configs:
        candidate_probability = fit_predict(config, train_x, labels_all[train], eval_x)
        for temperature in temperatures:
            candidate = softmax(np.log(np.maximum(candidate_probability, 1e-12)) / temperature)
            for weight in weights:
                blended = (1.0 - weight) * base_probability + weight * candidate
                blended /= blended.sum(axis=1, keepdims=True)
                prediction, grouping = decode_global_repeat(
                    np.log(np.maximum(blended, 1e-12)), indices, metadata, transition, decoder, global_config,
                )
                item = {
                    "config": {"model_config": config, "temperature": temperature, "weight": weight},
                    "candidate_raw": classification_metrics(labels, candidate.argmax(axis=1)),
                    "metrics": classification_metrics(labels, prediction),
                    "rescue_harm": rescue_harm(labels, base_prediction, prediction),
                    "grouping": grouping,
                }
                results.append(item)
                key = (item["metrics"]["correct"], item["metrics"]["balanced_accuracy"], item["rescue_harm"]["net"], -item["rescue_harm"]["harm"], -weight)
                if best_key is None or key > best_key:
                    best_key, best = key, item
        print(f"finished {config}", flush=True)
    assert best is not None
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P89_crossmodal_statistics_P87_fusion_v1", "status": "complete",
        "selected_on_current_holdout": args.fixed_summary is None,
        "holdout_users": sorted(args.holdout_users), "train_samples": int(len(train)), "eval_samples": int(len(evaluation)),
        "raw_feature_dim": int(features.shape[1]), "pca_components": components,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "base": classification_metrics(labels, base_prediction), "base_grouping": base_grouping,
        "selected_config": best["config"], "best": best, "grid_size": len(results), "all_candidates": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "all_candidates"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
