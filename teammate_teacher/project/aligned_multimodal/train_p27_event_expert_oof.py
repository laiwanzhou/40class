from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from p27r2_event_data import load_event_cache
from probe_p27r3_incremental_information import (
    explicit_event_sequence,
    flatten_sequence,
    metric_bundle,
    transformed_sequence,
    write_csv,
)
from sklearn.ensemble import ExtraTreesClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = (
    PROJECT_DIR / "runs" / "p27_r2_event_audit" / "event_cache_v2.npz"
)
DEFAULT_MANIFEST_DIR = PROJECT_DIR / "data" / "p27_strong_inner"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_strong_inner" / "event_expert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train explicit temporal-event expert on P27 inner folds"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=27041)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dense_probabilities(
    model: ExtraTreesClassifier, features: np.ndarray
) -> np.ndarray:
    source = model.predict_proba(features)
    probabilities = np.full((len(features), 40), 1e-7, dtype=np.float64)
    probabilities[:, model.classes_.astype(np.int64)] = source
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = load_event_cache(args.cache.resolve())
    cache_position = {
        str(sample_id): index
        for index, sample_id in enumerate(cache.sample_ids.astype(str))
    }
    rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    fold_info: dict[str, Any] = {}
    for fold in range(3):
        manifest_path = args.manifest_dir.resolve() / f"fold_{fold}.csv"
        manifest = read_csv(manifest_path)
        train_rows = [row for row in manifest if row["split"] == "train"]
        held_rows = [row for row in manifest if row["split"] == "val"]
        missing = [
            row["sample_id"]
            for row in manifest
            if row["sample_id"] not in cache_position
        ]
        if missing:
            raise KeyError(f"fold {fold}: {len(missing)} rows missing event cache")
        train_indices = np.asarray(
            [cache_position[row["sample_id"]] for row in train_rows],
            dtype=np.int64,
        )
        held_indices = np.asarray(
            [cache_position[row["sample_id"]] for row in held_rows],
            dtype=np.int64,
        )
        if np.any(cache.outer_folds[train_indices] == 0) or np.any(
            cache.outer_folds[held_indices] == 0
        ):
            raise RuntimeError("Event expert attempted to touch fold-0 outer-held")
        train_sequence = explicit_event_sequence(cache, train_indices)
        held_sequence = explicit_event_sequence(cache, held_indices)
        train_features = flatten_sequence(
            train_sequence,
            cache.modality_mask[train_indices].astype(np.float32),
        )
        held_presence = cache.modality_mask[held_indices].astype(np.float32)
        held_features = flatten_sequence(held_sequence, held_presence)
        train_labels = cache.labels[train_indices].astype(np.int64)
        held_labels = cache.labels[held_indices].astype(np.int64)
        model = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=16,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(args.seed) + fold,
        )
        started = time.perf_counter()
        model.fit(train_features, train_labels)
        fit_seconds = time.perf_counter() - started
        probabilities = dense_probabilities(model, held_features)
        predictions = probabilities.argmax(axis=1)
        rows.append(
            {
                "inner_fold": fold,
                "method": "explicit_event_sequence",
                "feature_dim": int(train_features.shape[1]),
                **flatten_metrics(metric_bundle(held_labels, predictions)),
            }
        )
        for mode_index, mode in enumerate(
            ("zero", "cross_sample_shuffle", "time_reverse", "within_sample_time_permutation")
        ):
            transformed = transformed_sequence(
                held_sequence,
                mode,
                int(args.seed) + fold * 100 + mode_index,
            )
            transformed_features = flatten_sequence(transformed, held_presence)
            transformed_predictions = model.predict(transformed_features)
            metrics = metric_bundle(held_labels, transformed_predictions)
            ablation_rows.append(
                {
                    "inner_fold": fold,
                    "ablation": mode,
                    **flatten_metrics(metrics),
                    "overall_delta_pp": 100.0
                    * (
                        metrics["overall"]["accuracy"]
                        - np.mean(predictions == held_labels)
                    ),
                    "hard_delta_pp": 100.0
                    * (
                        metrics["hard"]["accuracy"]
                        - metric_bundle(held_labels, predictions)["hard"]["accuracy"]
                    ),
                }
            )
        model_path = output / f"fold_{fold}_event_expert.joblib"
        joblib.dump(model, model_path, compress=3)
        np.savez_compressed(
            output / f"fold_{fold}_logits.npz",
            protocol=np.asarray("p27-strong-explicit-event-expert-inner-v1"),
            sample_ids=np.asarray([row["sample_id"] for row in held_rows]),
            labels=held_labels,
            subjects=np.asarray([row["user_id"] for row in held_rows]),
            logits=np.log(np.clip(probabilities, 1e-12, 1.0)).astype(np.float32),
            outer_held_predictions_generated=np.asarray(False),
        )
        fold_info[str(fold)] = {
            "train_samples": int(len(train_rows)),
            "held_samples": int(len(held_rows)),
            "train_subjects": sorted({row["user_id"] for row in train_rows}),
            "held_subjects": sorted({row["user_id"] for row in held_rows}),
            "fit_seconds": float(fit_seconds),
            "model_path": str(model_path.resolve()),
            "model_bytes": int(model_path.stat().st_size),
            "outer_held_predictions_generated": False,
        }
        print(
            f"fold={fold} event={np.mean(predictions == held_labels):.4f}",
            flush=True,
        )
    write_csv(output / "fold_metrics.csv", rows)
    write_csv(output / "ablations.csv", ablation_rows)
    summary = {
        "protocol": "p27-strong-explicit-event-expert-inner-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "feature_definition": "32 x (16 Skeleton relation/delta + 25 IMU invariants + 10 Depth/IR motion/context), plus four modality masks",
        "folds": fold_info,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
