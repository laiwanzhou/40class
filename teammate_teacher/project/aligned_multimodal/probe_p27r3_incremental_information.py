from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from p27r2_event_data import load_event_cache


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "p27_r3_incremental_probe.json"
DEFAULT_CACHE = (
    PROJECT_DIR / "runs" / "p27_r2_event_audit" / "event_cache_v2.npz"
)
DEFAULT_CORE_DIR = PROJECT_DIR / "runs" / "p27_r2_fold0"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_r3_incremental_probe"

SMALL_IDS = np.asarray(
    [1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39]
)
HARD_IDS = np.asarray(
    [7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 35, 37, 38, 39]
)
FOCUS_IDS = np.asarray([19, 24, 25, 26, 37, 9, 10, 21, 22])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P27-R3 outer-train-only incremental information probe"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--core-dir", type=Path, default=DEFAULT_CORE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_triplet(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def metric_bundle(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    subsets = {
        "overall": np.ones(len(labels), dtype=bool),
        "small": np.isin(labels, SMALL_IDS),
        "hard": np.isin(labels, HARD_IDS),
        "focus": np.isin(labels, FOCUS_IDS),
    }
    output: dict[str, Any] = {}
    for name, mask in subsets.items():
        output[name] = {
            "samples": int(mask.sum()),
            **metric_triplet(labels[mask], predictions[mask]),
        }
    return output


def dense_probabilities(model: ExtraTreesClassifier, features: np.ndarray) -> np.ndarray:
    source = model.predict_proba(features)
    probabilities = np.full((len(features), 40), 1e-7, dtype=np.float64)
    probabilities[:, model.classes_.astype(np.int64)] = source
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def new_classifier(config: dict[str, Any], seed: int) -> ExtraTreesClassifier:
    spec = config["classifier"]
    return ExtraTreesClassifier(
        n_estimators=int(spec["n_estimators"]),
        max_depth=int(spec["max_depth"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        max_features=str(spec["max_features"]),
        class_weight=str(spec["class_weight"]),
        n_jobs=int(spec["n_jobs"]),
        random_state=int(seed),
    )


def explicit_event_sequence(cache, indices: np.ndarray) -> np.ndarray:
    skeleton = cache.skeleton_tokens[indices, :, 27:43]
    imu = cache.imu_tokens[indices, :, :25]
    visual = cache.visual_tokens[indices, :, 96:106]
    return np.concatenate([skeleton, imu, visual], axis=2).astype(
        np.float32, copy=False
    )


def raw_token_sequence(cache, indices: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            cache.skeleton_tokens[indices],
            cache.imu_tokens[indices],
            cache.visual_tokens[indices],
        ],
        axis=2,
    ).astype(np.float32, copy=False)


def scalar_features(cache, indices: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            cache.event_targets[indices],
            cache.event_quality[indices],
            cache.modality_mask[indices],
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def flatten_sequence(sequence: np.ndarray, presence: np.ndarray) -> np.ndarray:
    return np.concatenate([sequence.reshape(len(sequence), -1), presence], axis=1)


def transformed_sequence(
    sequence: np.ndarray,
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "normal":
        return sequence
    if mode == "zero":
        return np.zeros_like(sequence)
    if mode == "cross_sample_shuffle":
        if len(sequence) <= 1:
            return sequence.copy()
        order = rng.permutation(len(sequence))
        if np.any(order == np.arange(len(sequence))):
            order = np.roll(order, 1)
        return sequence[order]
    if mode == "time_reverse":
        return sequence[:, ::-1].copy()
    if mode == "within_sample_time_permutation":
        output = np.empty_like(sequence)
        for index in range(len(sequence)):
            output[index] = sequence[index, rng.permutation(sequence.shape[1])]
        return output
    raise ValueError(mode)


def locate(cache, sample_ids: np.ndarray) -> np.ndarray:
    lookup = {
        sample_id: index
        for index, sample_id in enumerate(cache.sample_ids.astype(str).tolist())
    }
    missing = [sample_id for sample_id in sample_ids.astype(str) if sample_id not in lookup]
    if missing:
        raise KeyError(f"Core rows missing from event cache: {missing[:3]}")
    return np.asarray([lookup[value] for value in sample_ids.astype(str)], dtype=np.int64)


def load_fold_core(
    cache,
    core_dir: Path,
    fold: int,
) -> dict[str, np.ndarray]:
    path = core_dir / f"development_core_loso_fold_{fold}.npz"
    with np.load(path, allow_pickle=False) as source:
        train_ids = source["train_sample_ids"].astype(str)
        held_ids = source["held_sample_ids"].astype(str)
        result = {
            "train_ids": train_ids,
            "held_ids": held_ids,
            "train_indices": locate(cache, train_ids),
            "held_indices": locate(cache, held_ids),
            "train_logits": source["train_logits"].astype(np.float32),
            "held_logits": source["held_logits"].astype(np.float32),
        }
    return result


def feature_layers(
    cache,
    indices: np.ndarray,
    base_logits: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    presence = cache.modality_mask[indices].astype(np.float32, copy=False)
    explicit = explicit_event_sequence(cache, indices)
    raw = raw_token_sequence(cache, indices)
    scalar = scalar_features(cache, indices)
    layers = {
        "base": base_logits,
        "event_scalar": scalar,
        "event_sequence": flatten_sequence(explicit, presence),
        "raw_token": flatten_sequence(raw, presence),
        "base_plus_event_scalar": np.concatenate([base_logits, scalar], axis=1),
        "base_plus_event_sequence": np.concatenate(
            [base_logits, flatten_sequence(explicit, presence)], axis=1
        ),
        "base_plus_raw_token": np.concatenate(
            [base_logits, flatten_sequence(raw, presence)], axis=1
        ),
    }
    sequences = {
        "event_sequence": explicit,
        "raw_token": raw,
        "base_plus_event_sequence": explicit,
        "base_plus_raw_token": raw,
    }
    return layers, sequences


def sequence_features_for_mode(
    name: str,
    base_logits: np.ndarray,
    sequence: np.ndarray,
    presence: np.ndarray,
    mode: str,
    seed: int,
) -> np.ndarray:
    transformed = transformed_sequence(sequence, mode, seed)
    flattened = flatten_sequence(transformed, presence)
    if name.startswith("base_plus_"):
        return np.concatenate([base_logits, flattened], axis=1)
    return flattened


def pairwise_probe(
    config: dict[str, Any],
    fold: int,
    labels_train: np.ndarray,
    labels_held: np.ndarray,
    train_layers: dict[str, np.ndarray],
    held_layers: dict[str, np.ndarray],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = (
        "base",
        "event_scalar",
        "event_sequence",
        "raw_token",
        "base_plus_event_sequence",
        "base_plus_raw_token",
    )
    for left, right in config["pairwise_confusions"]:
        train_mask = np.isin(labels_train, [left, right])
        held_mask = np.isin(labels_held, [left, right])
        if train_mask.sum() < 12 or held_mask.sum() < 4:
            continue
        for method_index, name in enumerate(names):
            if name == "base":
                columns = np.asarray([left, right], dtype=np.int64)
                restricted = held_layers[name][held_mask][:, columns]
                predictions = columns[restricted.argmax(axis=1)]
            else:
                model = new_classifier(config, seed + 1000 * method_index + left * 41 + right)
                model.fit(train_layers[name][train_mask], labels_train[train_mask])
                predictions = model.predict(held_layers[name][held_mask])
            metrics = metric_triplet(labels_held[held_mask], predictions)
            rows.append(
                {
                    "inner_fold": fold,
                    "left_class": left,
                    "right_class": right,
                    "method": name,
                    "train_samples": int(train_mask.sum()),
                    "held_samples": int(held_mask.sum()),
                    **metrics,
                }
            )
    return rows


def per_group_rows(
    records: list[dict[str, Any]],
    cache,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    methods = sorted({record["method"] for record in records})
    for method in methods:
        selected = [record for record in records if record["method"] == method]
        labels = np.concatenate([record["labels"] for record in selected])
        predictions = np.concatenate([record["predictions"] for record in selected])
        indices = np.concatenate([record["indices"] for record in selected])
        subjects = cache.subjects[indices].astype(str)
        for subject in sorted(np.unique(subjects).tolist()):
            mask = subjects == subject
            subject_rows.append(
                {
                    "method": method,
                    "subject": subject,
                    "samples": int(mask.sum()),
                    **metric_triplet(labels[mask], predictions[mask]),
                }
            )
        for class_id in range(40):
            mask = labels == class_id
            if not mask.any():
                continue
            class_rows.append(
                {
                    "method": method,
                    "class_id": class_id,
                    "samples": int(mask.sum()),
                    "recall": float(np.mean(predictions[mask] == class_id)),
                }
            )
    return subject_rows, class_rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    cache = load_event_cache(args.cache.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    outer_train = cache.outer_folds != int(config["outer_fold"])
    if int(outer_train.sum()) != 1947:
        raise RuntimeError(f"Unexpected outer-train size: {int(outer_train.sum())}")

    fold_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    fold_summaries: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(3):
        core = load_fold_core(cache, args.core_dir.resolve(), fold)
        train_indices = core["train_indices"]
        held_indices = core["held_indices"]
        if not np.all(outer_train[train_indices]) or not np.all(outer_train[held_indices]):
            raise RuntimeError("R3 probe attempted to include an outer-held sample")
        labels_train = cache.labels[train_indices]
        labels_held = cache.labels[held_indices]
        train_layers, train_sequences = feature_layers(
            cache, train_indices, core["train_logits"]
        )
        held_layers, held_sequences = feature_layers(
            cache, held_indices, core["held_logits"]
        )
        fold_models: dict[str, ExtraTreesClassifier] = {}
        fold_predictions: dict[str, np.ndarray] = {
            "base_argmax": core["held_logits"].argmax(axis=1)
        }
        base_probabilities = np.exp(
            core["held_logits"] - core["held_logits"].max(axis=1, keepdims=True)
        )
        base_probabilities /= base_probabilities.sum(axis=1, keepdims=True)
        fold_rows.append(
            {
                "inner_fold": fold,
                "method": "base_argmax",
                "feature_dim": 40,
                "log_loss": float(log_loss(labels_held, base_probabilities, labels=list(range(40)))),
                **{
                    f"{subset}_{metric}": value
                    for subset, subset_values in metric_bundle(
                        labels_held, fold_predictions["base_argmax"]
                    ).items()
                    for metric, value in subset_values.items()
                },
            }
        )
        for method_index, (name, train_features) in enumerate(train_layers.items()):
            model = new_classifier(config, seed + fold * 100 + method_index)
            model.fit(train_features, labels_train)
            probabilities = dense_probabilities(model, held_layers[name])
            predictions = probabilities.argmax(axis=1)
            fold_models[name] = model
            fold_predictions[name] = predictions
            metrics = metric_bundle(labels_held, predictions)
            fold_rows.append(
                {
                    "inner_fold": fold,
                    "method": name,
                    "feature_dim": int(train_features.shape[1]),
                    "log_loss": float(
                        log_loss(labels_held, probabilities, labels=list(range(40)))
                    ),
                    **{
                        f"{subset}_{metric}": value
                        for subset, subset_values in metrics.items()
                        for metric, value in subset_values.items()
                    },
                }
            )
            prediction_records.append(
                {
                    "method": name,
                    "labels": labels_held.copy(),
                    "predictions": predictions.copy(),
                    "indices": held_indices.copy(),
                }
            )
        prediction_records.append(
            {
                "method": "base_argmax",
                "labels": labels_held.copy(),
                "predictions": fold_predictions["base_argmax"].copy(),
                "indices": held_indices.copy(),
            }
        )

        presence = cache.modality_mask[held_indices].astype(np.float32, copy=False)
        for sequence_index, (name, sequence) in enumerate(held_sequences.items()):
            reference = fold_predictions[name]
            reference_metrics = metric_bundle(labels_held, reference)
            for mode_index, mode in enumerate(config["ablations"]):
                features = sequence_features_for_mode(
                    name,
                    core["held_logits"],
                    sequence,
                    presence,
                    mode,
                    seed + fold * 10000 + sequence_index * 100 + mode_index,
                )
                predictions = fold_models[name].predict(features)
                metrics = metric_bundle(labels_held, predictions)
                ablation_rows.append(
                    {
                        "inner_fold": fold,
                        "method": name,
                        "ablation": mode,
                        **{
                            f"{subset}_{metric}": value
                            for subset, subset_values in metrics.items()
                            for metric, value in subset_values.items()
                        },
                        "overall_delta_pp": 100.0
                        * (
                            metrics["overall"]["accuracy"]
                            - reference_metrics["overall"]["accuracy"]
                        ),
                        "hard_delta_pp": 100.0
                        * (
                            metrics["hard"]["accuracy"]
                            - reference_metrics["hard"]["accuracy"]
                        ),
                        "focus_delta_pp": 100.0
                        * (
                            metrics["focus"]["accuracy"]
                            - reference_metrics["focus"]["accuracy"]
                        ),
                    }
                )
        pair_rows.extend(
            pairwise_probe(
                config,
                fold,
                labels_train,
                labels_held,
                train_layers,
                held_layers,
                seed + fold * 10000,
            )
        )
        fold_summaries[str(fold)] = {
            "train_subjects": sorted(np.unique(cache.subjects[train_indices]).tolist()),
            "held_subjects": sorted(np.unique(cache.subjects[held_indices]).tolist()),
            "train_samples": int(len(train_indices)),
            "held_samples": int(len(held_indices)),
            "outer_held_rows_touched_for_prediction": 0,
        }

    subject_rows, class_rows = per_group_rows(prediction_records, cache)
    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "ablations.csv", ablation_rows)
    write_csv(output / "pairwise_confusions.csv", pair_rows)
    write_csv(output / "per_subject.csv", subject_rows)
    write_csv(output / "per_class.csv", class_rows)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        by_method[str(row["method"])].append(row)
    means = {}
    for method, rows in by_method.items():
        numeric = {}
        for key in rows[0]:
            if key in {"inner_fold", "method", "feature_dim"}:
                continue
            numeric[key] = float(np.mean([float(row[key]) for row in rows]))
        means[method] = {
            "folds": len(rows),
            "feature_dim": int(rows[0]["feature_dim"]),
            **numeric,
        }
    by_ablation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ablation_rows:
        by_ablation[(str(row["method"]), str(row["ablation"]))].append(row)
    ablation_means = {
        f"{method}/{ablation}": {
            "overall_delta_pp": float(
                np.mean([float(row["overall_delta_pp"]) for row in rows])
            ),
            "hard_delta_pp": float(
                np.mean([float(row["hard_delta_pp"]) for row in rows])
            ),
            "focus_delta_pp": float(
                np.mean([float(row["focus_delta_pp"]) for row in rows])
            ),
        }
        for (method, ablation), rows in by_ablation.items()
    }
    summary = {
        "protocol": config["protocol"],
        "status": "complete",
        "outer_fold": int(config["outer_fold"]),
        "outer_train_samples": int(outer_train.sum()),
        "outer_held_samples_excluded": int((~outer_train).sum()),
        "outer_held_predictions_generated": False,
        "feature_definition": config["feature_layers"],
        "folds": fold_summaries,
        "mean_metrics": means,
        "mean_ablation_deltas": ablation_means,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
