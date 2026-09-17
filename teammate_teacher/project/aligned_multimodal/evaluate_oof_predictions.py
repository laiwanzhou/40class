from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 subject-disjoint OOF 预测并做 subject-cluster bootstrap")
    parser.add_argument(
        "--method",
        action="append",
        nargs=4,
        metavar=("NAME", "FOLD0", "FOLD1", "FOLD2"),
        required=True,
        help="可重复指定：方法名及三折预测 CSV",
    )
    parser.add_argument("--primary", required=True, help="需要与其他方法比较的主方法名")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        }


def read_fold(path: Path, fold: int) -> list[dict[str, object]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "label", "prediction"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"预测格式错误：{path}")
    result = []
    for row in rows:
        parts = row["sample_id"].split("__")
        if len(parts) < 4 or not parts[2].startswith("user"):
            raise ValueError(f"无法从 sample_id 解析 subject：{row['sample_id']}")
        result.append(
            {
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
                "user_id": parts[2],
                "fold": fold,
            }
        )
    if len({row["sample_id"] for row in result}) != len(result):
        raise ValueError(f"fold 内存在重复 sample_id：{path}")
    return result


def cluster_bootstrap_accuracy_delta(
    labels: np.ndarray,
    primary: np.ndarray,
    baseline: np.ndarray,
    users: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    unique_users = np.asarray(sorted(set(users.tolist()), key=lambda value: int(value[4:])))
    user_indices = {user: np.flatnonzero(users == user) for user in unique_users}
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        indices = np.concatenate([user_indices[user] for user in sampled])
        deltas[repeat] = (
            accuracy_score(labels[indices], primary[indices])
            - accuracy_score(labels[indices], baseline[indices])
        )
    return {
        "delta_pp": float(100 * (accuracy_score(labels, primary) - accuracy_score(labels, baseline))),
        "subject_cluster_bootstrap_95_ci_pp": [
            float(100 * np.percentile(deltas, 2.5)),
            float(100 * np.percentile(deltas, 97.5)),
        ],
        "probability_delta_positive": float(np.mean(deltas > 0)),
        "clusters": int(len(unique_users)),
        "repeats": repeats,
    }


def main() -> None:
    args = parse_args()
    methods: dict[str, list[dict[str, object]]] = {}
    sources: dict[str, list[str]] = {}
    for name, *paths in args.method:
        if name in methods:
            raise ValueError(f"方法名重复：{name}")
        folds = [read_fold(Path(path), fold) for fold, path in enumerate(paths)]
        combined = [row for fold_rows in folds for row in fold_rows]
        if len({row["sample_id"] for row in combined}) != len(combined):
            raise ValueError(f"三个 fold 间存在重复样本：{name}")
        methods[name] = combined
        sources[name] = [str(Path(path).resolve()) for path in paths]
    if args.primary not in methods:
        raise ValueError(f"主方法不存在：{args.primary}")

    reference_ids = [row["sample_id"] for row in methods[args.primary]]
    reference_set = set(reference_ids)
    for name, rows in methods.items():
        if {row["sample_id"] for row in rows} != reference_set:
            raise ValueError(f"方法 sample_id 集合不一致：{name}")
    ordered: dict[str, dict[str, dict[str, object]]] = {}
    for name, rows in methods.items():
        ordered[name] = {str(row["sample_id"]): row for row in rows}

    labels = np.asarray([ordered[args.primary][key]["label"] for key in reference_ids], dtype=np.int64)
    users = np.asarray([ordered[args.primary][key]["user_id"] for key in reference_ids])
    folds = np.asarray([ordered[args.primary][key]["fold"] for key in reference_ids], dtype=np.int64)
    result: dict[str, object] = {
        "primary": args.primary,
        "samples": len(reference_ids),
        "subjects": sorted(set(users.tolist()), key=lambda value: int(value[4:])),
        "methods": {},
        "comparisons": {},
        "sources": sources,
    }
    predictions_by_method: dict[str, np.ndarray] = {}
    for name in methods:
        predictions = np.asarray([ordered[name][key]["prediction"] for key in reference_ids], dtype=np.int64)
        predictions_by_method[name] = predictions
        per_fold = {
            str(fold): metrics(labels[folds == fold], predictions[folds == fold]) for fold in range(3)
        }
        per_subject = {
            user: metrics(labels[users == user], predictions[users == user]) for user in result["subjects"]
        }
        result["methods"][name] = {
            "pooled_oof": metrics(labels, predictions),
            "mean_fold_accuracy": float(np.mean([value["accuracy"] for value in per_fold.values()])),
            "subject_macro_accuracy": float(np.mean([value["accuracy"] for value in per_subject.values()])),
            "per_fold": per_fold,
            "per_subject": per_subject,
        }

    primary_predictions = predictions_by_method[args.primary]
    for index, (name, predictions) in enumerate(predictions_by_method.items()):
        if name == args.primary:
            continue
        comparison = cluster_bootstrap_accuracy_delta(
            labels,
            primary_predictions,
            predictions,
            users,
            args.bootstrap_repeats,
            args.seed + index,
        )
        comparison["folds_with_positive_accuracy_delta"] = sum(
            accuracy_score(labels[folds == fold], primary_predictions[folds == fold])
            > accuracy_score(labels[folds == fold], predictions[folds == fold])
            for fold in range(3)
        )
        result["comparisons"][f"{args.primary}_vs_{name}"] = comparison

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
