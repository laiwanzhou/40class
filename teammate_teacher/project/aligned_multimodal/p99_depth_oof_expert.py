"""P99-D0 leakage-safe source-OOF Depth expert.

The frozen VideoMAEv2 Depth cache is label-free.  Every classifier, scaler and
temperature is fit outside the evaluated user.  H1 is available by default;
H2 requires an explicit frozen H1 summary.  This module has no H3 path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from train_p46_videomae_head import l2_normalize, make_model, row_standardize


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_depth_d0.json"
DEFAULT_DEPTH = PROJECT / "runs/p91_videomaev2_depth_fold0_v1/complete_features.npz"
DEFAULT_SPLITS = PROJECT / "runs/p90_crossuser_visual_router_v1/full_predictions.npz"
DEFAULT_E0_BASE = HERE / "runs/p87_sequence_decoder_v1/oof_predictions.npz"
NUM_CLASSES = 40


@dataclass(frozen=True)
class Cohort:
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    anchor_prediction: np.ndarray
    features: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99-D0 source-OOF Depth expert")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--e0-base", type=Path, default=DEFAULT_E0_BASE)
    parser.add_argument("--h1-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def feature_families(features: np.ndarray, action_logits: np.ndarray) -> dict[str, np.ndarray]:
    features = l2_normalize(np.asarray(features, dtype=np.float32))
    action = row_standardize(
        np.asarray(action_logits, dtype=np.float32).reshape(len(features), -1)
    )
    mean = l2_normalize(features.mean(axis=1))
    return {
        "roi_all": features.reshape(len(features), -1),
        "roi_mean": mean,
        "k710_logits": action,
        "roi_all_plus_k710": np.concatenate(
            (features.reshape(len(features), -1), action), axis=1
        ),
    }


def class_sample_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    reference = counts[counts > 0].mean()
    class_weight = np.zeros(NUM_CLASSES, dtype=np.float64)
    present = counts > 0
    class_weight[present] = np.power(reference / counts[present], power)
    weights = class_weight[labels]
    return weights / weights.mean()


def decision_scores(model: Any, values: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(values), dtype=np.float64)
    classes = np.asarray(model.named_steps["ridge"].classes_, dtype=np.int64)
    if scores.ndim != 2:
        raise RuntimeError(f"expected multiclass decision scores, got {scores.shape}")
    if not set(classes.tolist()).issubset(set(range(NUM_CLASSES))):
        raise RuntimeError("Depth head learned an invalid class id")
    if len(classes) == NUM_CLASSES and np.array_equal(classes, np.arange(NUM_CLASSES)):
        return scores
    spread = np.maximum(np.ptp(scores, axis=1, keepdims=True), 1.0)
    output = np.repeat(scores.min(axis=1, keepdims=True) - spread, NUM_CLASSES, axis=1)
    output[:, classes] = scores
    return output


def fit_head(values: np.ndarray, labels: np.ndarray, indices: np.ndarray, recipe: dict[str, Any]) -> Any:
    model = make_model(float(recipe["alpha"]))
    model.fit(
        values[indices],
        labels[indices],
        ridge__sample_weight=class_sample_weights(
            labels[indices], float(recipe["class_weight_power"])
        ),
    )
    return model


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    def objective(log_temperature: float) -> float:
        scaled = scores / float(np.exp(log_temperature))
        nll = logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]
        return float(nll.mean())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    return float(np.exp(result.x))


def calibrated_outer_prediction(
    values: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    recipe: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, Any]:
    inner_scores = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    train_users = users[train]
    for inner_user in sorted(set(train_users.tolist())):
        inner_eval_local = np.flatnonzero(train_users == inner_user)
        inner_fit_local = np.flatnonzero(train_users != inner_user)
        if not len(inner_eval_local) or not len(inner_fit_local):
            raise RuntimeError("invalid inner user split")
        model = fit_head(values, labels, train[inner_fit_local], recipe)
        inner_scores[inner_eval_local] = decision_scores(
            model, values[train[inner_eval_local]]
        )
    temperature = fit_temperature(inner_scores, labels[train])
    model = fit_head(values, labels, train, recipe)
    return (
        decision_scores(model, values[evaluation]) / temperature,
        inner_scores / temperature,
        temperature,
        model,
    )


def topk_accuracy(probability: np.ndarray, labels: np.ndarray, k: int) -> float:
    top = np.argpartition(-probability, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(top == labels[:, None], axis=1)))


def metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probability = softmax(logits)
    prediction = probability.argmax(axis=1)
    return {
        "correct": int(np.sum(prediction == labels)),
        "total": int(len(labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "top3": topk_accuracy(probability, labels, 3),
        "top5": topk_accuracy(probability, labels, 5),
        "top10": topk_accuracy(probability, labels, 10),
        "log_loss": float(
            np.mean(-np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1.0)))
        ),
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=np.arange(NUM_CLASSES)
        ).astype(int).tolist(),
    }


def change_audit(labels: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    base_correct = base == labels
    candidate_correct = candidate == labels
    return {
        "rescue": int(np.sum(~base_correct & candidate_correct)),
        "harm": int(np.sum(base_correct & ~candidate_correct)),
        "net": int(candidate_correct.sum() - base_correct.sum()),
        "changed": int(np.sum(base != candidate)),
        "oracle_union_correct": int(np.sum(base_correct | candidate_correct)),
    }


def _read(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as source:
        missing = [key for key in keys if key not in source.files]
        if missing:
            raise KeyError(f"{path} missing {missing}")
        return {key: np.asarray(source[key]) for key in keys}


def align(source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(source_ids)}
    missing = [str(value) for value in target_ids if str(value) not in lookup]
    if missing:
        raise KeyError(f"feature cache misses {len(missing)} samples")
    return np.asarray(values)[[lookup[str(value)] for value in target_ids]]


def build_cohort(
    stage: str,
    config: dict[str, Any],
    depth_path: Path,
    split_path: Path,
    e0_base_path: Path,
) -> tuple[Cohort, np.ndarray, np.ndarray]:
    depth = _read(
        depth_path,
        ("sample_ids", "labels", "users", "features", "action_logits"),
    )
    depth_ids = depth["sample_ids"].astype(str)
    depth_users = depth["users"].astype(str)
    depth_labels = depth["labels"].astype(np.int64)
    e0_base = _read(e0_base_path, ("sample_ids", "labels", "sequence_predictions"))
    if not np.array_equal(
        align(e0_base["sample_ids"].astype(str), e0_base["labels"], depth_ids),
        depth_labels,
    ):
        raise RuntimeError("Depth cache labels disagree with the independent E0 source")
    families = feature_families(depth["features"], depth["action_logits"])
    split_keys = (
        "H1_selection_sample_ids",
        "H1_selection_labels",
        "H1_selection_users",
        "H1_selection_safe_prediction",
    )
    if stage == "h2_confirmation":
        split_keys += (
            "H2_confirmation_sample_ids",
            "H2_confirmation_labels",
            "H2_confirmation_users",
            "H2_confirmation_safe_prediction",
        )
    split = _read(split_path, split_keys)
    h1_ids = split["H1_selection_sample_ids"].astype(str)
    h1_anchor = split["H1_selection_safe_prediction"].astype(np.int64)
    e0_users = set(map(str, config["cohorts"]["source_only_users"]))
    e0_ids = depth_ids[np.isin(depth_users, sorted(e0_users))]
    e0_anchor = align(
        e0_base["sample_ids"].astype(str), e0_base["sequence_predictions"], e0_ids
    ).astype(np.int64)
    universe_ids = np.concatenate((h1_ids, e0_ids))
    universe_anchor = np.concatenate((h1_anchor, e0_anchor))
    if len(set(universe_ids.tolist())) != len(universe_ids):
        raise RuntimeError("H1 and E0 overlap")
    if stage == "h1":
        eval_ids = h1_ids
        eval_labels = split["H1_selection_labels"].astype(np.int64)
        eval_users = split["H1_selection_users"].astype(str)
    else:
        eval_ids = split["H2_confirmation_sample_ids"].astype(str)
        eval_labels = split["H2_confirmation_labels"].astype(np.int64)
        eval_users = split["H2_confirmation_users"].astype(str)
        h2_anchor = split["H2_confirmation_safe_prediction"].astype(np.int64)
        universe_ids = np.concatenate((universe_ids, eval_ids))
        universe_anchor = np.concatenate((universe_anchor, h2_anchor))
    labels = align(depth_ids, depth_labels, universe_ids).astype(np.int64)
    users = align(depth_ids, depth_users, universe_ids).astype(str)
    expected_eval = np.arange(len(h1_ids)) if stage == "h1" else np.arange(
        len(h1_ids) + len(e0_ids), len(universe_ids)
    )
    if not np.array_equal(labels[expected_eval], eval_labels):
        raise RuntimeError("split labels disagree with Depth cache")
    cohort = Cohort(
        sample_ids=universe_ids,
        labels=labels,
        users=users,
        anchor_prediction=universe_anchor,
        features={name: align(depth_ids, values, universe_ids) for name, values in families.items()},
    )
    return cohort, expected_eval, eval_ids


def evaluate_recipe(
    cohort: Cohort,
    eval_indices: np.ndarray,
    recipe: dict[str, Any],
    recipe_name: str,
    stage: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    direct = np.zeros((len(eval_indices), NUM_CLASSES), dtype=np.float64)
    zero = np.zeros_like(direct)
    shuffled = np.zeros_like(direct)
    temperatures: dict[str, float] = {}
    evaluation_users = cohort.users[eval_indices]
    folds = sorted(set(evaluation_users.tolist())) if stage == "h1" else ["H2_all"]
    for fold_number, held_user in enumerate(folds):
        if stage == "h1":
            local_eval = np.flatnonzero(evaluation_users == held_user)
            outer_eval = eval_indices[local_eval]
            train = np.flatnonzero(cohort.users != held_user)
        else:
            local_eval = np.arange(len(eval_indices))
            outer_eval = eval_indices
            train = np.flatnonzero(~np.isin(np.arange(len(cohort.labels)), eval_indices))
        if set(cohort.users[train]) & set(cohort.users[outer_eval]):
            raise RuntimeError("outer train/evaluation users overlap")
        logits, _, temperature, model = calibrated_outer_prediction(
            cohort.features[recipe_name], cohort.labels, cohort.users,
            train, outer_eval, recipe,
        )
        direct[local_eval] = logits
        training_mean = cohort.features[recipe_name][train].mean(axis=0, keepdims=True)
        zero[local_eval] = decision_scores(
            model, np.repeat(training_mean, len(outer_eval), axis=0)
        ) / temperature
        rng = np.random.default_rng(seed + fold_number * 1009)
        permutation = np.arange(len(outer_eval))
        for user in sorted(set(cohort.users[outer_eval].tolist())):
            selected = np.flatnonzero(cohort.users[outer_eval] == user)
            permutation[selected] = selected[rng.permutation(len(selected))]
        shuffled[local_eval] = decision_scores(
            model, cohort.features[recipe_name][outer_eval][permutation]
        ) / temperature
        temperatures[str(held_user)] = temperature
    labels = cohort.labels[eval_indices]
    anchor = cohort.anchor_prediction[eval_indices]
    direct_prediction = direct.argmax(axis=1)
    result = {
        "recipe": recipe,
        "feature_dim": int(cohort.features[recipe_name].shape[1]),
        "metrics": metrics(direct, labels),
        "zero_metrics": metrics(zero, labels),
        "shuffle_metrics": metrics(shuffled, labels),
        "vs_anchor": change_audit(labels, anchor, direct_prediction),
        "temperature_by_outer_fold": temperatures,
        "per_user": {},
    }
    for user in sorted(set(evaluation_users.tolist())):
        selected = evaluation_users == user
        result["per_user"][user] = {
            "rows": int(selected.sum()),
            "correct": int(np.sum(direct_prediction[selected] == labels[selected])),
            "anchor_correct": int(np.sum(anchor[selected] == labels[selected])),
            "delta": int(
                np.sum(direct_prediction[selected] == labels[selected])
                - np.sum(anchor[selected] == labels[selected])
            ),
        }
    return result, {
        "direct_logits": direct.astype(np.float32),
        "direct_probability": softmax(direct).astype(np.float32),
        "zero_logits": zero.astype(np.float32),
        "shuffle_logits": shuffled.astype(np.float32),
    }


def selection_key(name: str, result: dict[str, Any]) -> tuple[Any, ...]:
    value = result["metrics"]
    return (
        int(value["correct"]),
        float(value["balanced_accuracy"]),
        float(value["top5"]),
        -float(value["log_loss"]),
        tuple(-ord(char) for char in name),
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    config_sha256 = canonical_hash(config)
    if args.stage == "h2_confirmation":
        if args.h1_summary is None:
            raise ValueError("H2 requires --h1-summary")
        frozen = json.loads(args.h1_summary.resolve().read_text(encoding="utf-8"))
        if frozen.get("stage") != "P99_D0_H1" or frozen.get("config_sha256") != config_sha256:
            raise ValueError("H1 summary stage/config differs")
        recipe_names = [str(frozen["selected_recipe"])]
    else:
        if args.h1_summary is not None:
            raise ValueError("--h1-summary is only valid for H2")
        recipe_names = list(config["recipes"])
    cohort, eval_indices, eval_ids = build_cohort(
        args.stage, config, args.depth_features, args.split_source, args.e0_base
    )
    results: dict[str, Any] = {}
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for recipe_name in recipe_names:
        result, arrays = evaluate_recipe(
            cohort, eval_indices, config["recipes"][recipe_name], recipe_name,
            args.stage, int(config["seed"]),
        )
        results[recipe_name] = result
        artifacts[recipe_name] = arrays
        print(
            f"recipe={recipe_name} correct={result['metrics']['correct']}/{result['metrics']['total']} "
            f"top5={result['metrics']['top5']:.6f} zero={result['zero_metrics']['correct']} "
            f"shuffle={result['shuffle_metrics']['correct']}",
            flush=True,
        )
    selected = recipe_names[0] if args.stage == "h2_confirmation" else max(
        recipe_names, key=lambda name: selection_key(name, results[name])
    )
    selected_arrays = artifacts[selected]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / ("h1_predictions.npz" if args.stage == "h1" else "h2_predictions.npz"),
        sample_ids=eval_ids,
        labels=cohort.labels[eval_indices],
        users=cohort.users[eval_indices],
        anchor_prediction=cohort.anchor_prediction[eval_indices],
        selected_recipe=np.asarray(selected),
        **selected_arrays,
    )
    summary = {
        "stage": "P99_D0_H1" if args.stage == "h1" else "P99_D0_H2_confirmation",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": config_sha256,
        "config_path": str(args.config.resolve()),
        "depth_feature_path": str(args.depth_features.resolve()),
        "protocol": (
            "H1 leave-one-exploration-user-out with E0 source-only users"
            if args.stage == "h1"
            else "frozen H1 recipe trained on H1+E0 and evaluated once on H2"
        ),
        "evaluated_rows": int(len(eval_indices)),
        "training_users_by_policy": sorted(
            set(config["cohorts"]["exploration_users"])
            | set(config["cohorts"]["source_only_users"])
        ),
        "selected_recipe": selected,
        "anchor": {
            "correct": int(
                np.sum(cohort.anchor_prediction[eval_indices] == cohort.labels[eval_indices])
            ),
            "total": int(len(eval_indices)),
        },
        "candidates": results,
        "leakage_audit": {
            "backbone_cache_label_free": True,
            "outer_user_disjoint": True,
            "scaler_fit_inside_outer_train": True,
            "temperature_fit_on_inner_user_oof": True,
            "h3_code_path_present": False,
        },
        "next_decision": (
            "Run the fixed-budget Student transfer controls for information-bearing D0 targets, then decide whether global Depth is useful or D1 hand-workspace geometry is required."
            if args.stage == "h1"
            else "Jointly audit frozen Teacher and Student confirmation; H3 remains unavailable."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
