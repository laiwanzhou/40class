from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "p0_six_modality_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一分析 S+D 的三折 OOF 与无标签 Test 预测敏感性")
    parser.add_argument("--oof-root", type=Path, default=PROJECT_DIR / "runs" / "p5_oof_fusion")
    parser.add_argument("--test-logits", type=Path, default=DEFAULT_OUTPUT_DIR / "test_logits.npz")
    parser.add_argument("--training-manifest", type=Path, default=PROJECT_DIR / "data" / "manifest.csv")
    parser.add_argument("--test-union-manifest", type=Path, default=PROJECT_DIR / "data" / "six_modality_audit" / "test_union_manifest.csv")
    parser.add_argument("--class-mapping", type=Path, default=REPO_DIR / "class_mapping.csv")
    parser.add_argument("--taxonomy", type=Path, default=PROJECT_DIR / "data" / "six_modality_audit" / "small_action_taxonomy_v1.csv")
    parser.add_argument("--depth-weight", type=float, default=0.4)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def distribution(predictions: np.ndarray) -> np.ndarray:
    counts = np.bincount(predictions, minlength=40).astype(np.float64)
    return counts / counts.sum()


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    middle = 0.5 * (a + b)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / np.clip(right[mask], 1e-12, None))))

    return 0.5 * kl(a, middle) + 0.5 * kl(b, middle)


def load_oof(root: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "labels": [],
        "skeleton_logits": [],
        "depth_logits": [],
        "folds": [],
    }
    for fold in range(3):
        data = np.load(root / f"fold_{fold}" / "logits.npz")
        count = len(data["labels"])
        arrays["sample_ids"].append(data["sample_ids"].astype(str))
        arrays["labels"].append(data["labels"].astype(np.int64))
        arrays["skeleton_logits"].append(data["skeleton_logits"].astype(np.float64))
        arrays["depth_logits"].append(data["depth_logits"].astype(np.float64))
        arrays["folds"].append(np.full(count, fold, dtype=np.int64))
    return {key: np.concatenate(values) for key, values in arrays.items()}


def grouped_depth_donors(
    rng: np.random.Generator,
    users: np.ndarray,
    labels: np.ndarray,
    same_label: bool,
) -> np.ndarray:
    all_indices = np.arange(len(labels))
    donors = np.empty(len(labels), dtype=np.int64)
    for index in all_indices:
        if same_label:
            candidates = all_indices[(labels == labels[index]) & (all_indices != index)]
        else:
            candidates = all_indices[(users == users[index]) & (labels != labels[index])]
        if len(candidates) == 0:
            candidates = all_indices[all_indices != index]
        donors[index] = int(rng.choice(candidates))
    return donors


def repeated_oof_shuffle(
    skeleton_logits: np.ndarray,
    depth_logits: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    depth_weight: float,
    repeats: int,
    seed: int,
    same_label: bool,
) -> dict[str, Any]:
    baseline_predictions = ((1.0 - depth_weight) * skeleton_logits + depth_weight * depth_logits).argmax(1)
    metric_rows = []
    change_rates = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        donors = grouped_depth_donors(rng, users, labels, same_label=same_label)
        logits = (1.0 - depth_weight) * skeleton_logits + depth_weight * depth_logits[donors]
        predictions = logits.argmax(1)
        metric_rows.append(list(metrics(labels, predictions).values()))
        change_rates.append(float(np.mean(predictions != baseline_predictions)))
    values = np.asarray(metric_rows)
    names = ["accuracy", "balanced_accuracy", "macro_f1"]
    return {
        "donor_rule": "same class control" if same_label else "same subject, different class",
        "mean": {name: float(values[:, index].mean()) for index, name in enumerate(names)},
        "std": {name: float(values[:, index].std()) for index, name in enumerate(names)},
        "prediction_change_rate_mean": float(np.mean(change_rates)),
        "prediction_change_rate_std": float(np.std(change_rates)),
    }


def test_shuffle_sensitivity(
    skeleton_logits: np.ndarray,
    depth_logits: np.ndarray,
    depth_weight: float,
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    baseline_logits = (1.0 - depth_weight) * skeleton_logits + depth_weight * depth_logits
    baseline_probabilities = softmax(baseline_logits)
    baseline_predictions = baseline_probabilities.argmax(1)
    baseline_confidence = baseline_probabilities.max(1)
    baseline_entropy = entropy(baseline_probabilities)
    per_sample_changes = np.zeros(len(baseline_predictions), dtype=np.float64)
    change_rates = []
    confidence_deltas = []
    entropy_deltas = []
    class_change_counts = np.zeros(40, dtype=np.float64)
    class_totals = np.zeros(40, dtype=np.float64)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        donors = rng.permutation(len(depth_logits))
        shuffled_logits = (1.0 - depth_weight) * skeleton_logits + depth_weight * depth_logits[donors]
        probabilities = softmax(shuffled_logits)
        predictions = probabilities.argmax(1)
        changed = predictions != baseline_predictions
        per_sample_changes += changed
        change_rates.append(float(np.mean(changed)))
        confidence_deltas.append(float(np.mean(probabilities.max(1) - baseline_confidence)))
        entropy_deltas.append(float(np.mean(entropy(probabilities) - baseline_entropy)))
        for class_id in range(40):
            mask = baseline_predictions == class_id
            class_change_counts[class_id] += float(changed[mask].sum())
            class_totals[class_id] += float(mask.sum())
    class_sensitivity = {
        str(class_id): (float(class_change_counts[class_id] / class_totals[class_id]) if class_totals[class_id] else None)
        for class_id in range(40)
    }
    return (
        {
            "interpretation": "无 Test 标签；以下只表示预测敏感性，不表示准确率升降。",
            "repeats": repeats,
            "random_depth_shuffle": {
                "prediction_change_rate_mean": float(np.mean(change_rates)),
                "prediction_change_rate_std": float(np.std(change_rates)),
                "mean_confidence_delta": float(np.mean(confidence_deltas)),
                "mean_entropy_delta": float(np.mean(entropy_deltas)),
                "change_rate_by_baseline_predicted_class": class_sensitivity,
            },
        },
        per_sample_changes / repeats,
    )


def subset_metrics(labels: np.ndarray, predictions: np.ndarray, class_ids: set[int]) -> dict[str, Any]:
    mask = np.isin(labels, sorted(class_ids))
    subset_labels = labels[mask]
    subset_predictions = predictions[mask]
    per_class_recalls = [
        float(np.mean(subset_predictions[subset_labels == class_id] == class_id))
        for class_id in sorted(class_ids)
        if np.any(subset_labels == class_id)
    ]
    return {
        "samples": int(mask.sum()),
        "classes": len(class_ids),
        "accuracy": float(accuracy_score(subset_labels, subset_predictions)),
        "balanced_accuracy": float(np.mean(per_class_recalls)),
        "macro_f1_over_fixed_classes": float(
            f1_score(
                subset_labels,
                subset_predictions,
                labels=sorted(class_ids),
                average="macro",
                zero_division=0,
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.depth_weight <= 1:
        raise ValueError("--depth-weight must be in [0,1]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    oof = load_oof(args.oof_root.resolve())
    test_data = np.load(args.test_logits.resolve())
    test = {
        "sample_ids": test_data["sample_ids"].astype(str),
        "skeleton_logits": test_data["skeleton_logits"].astype(np.float64),
        "depth_logits": test_data["depth_logits"].astype(np.float64),
    }
    mapping_rows = read_csv(args.class_mapping.resolve())
    class_names = {int(row["action_id"]): row["action_name"].split("_", 1)[-1] for row in mapping_rows}
    manifest_rows = read_csv(args.training_manifest.resolve())
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    missing_ids = [sample_id for sample_id in oof["sample_ids"] if sample_id not in manifest_by_id]
    if missing_ids:
        raise KeyError(f"OOF sample_id not in manifest: {missing_ids[:5]}")
    users = np.asarray([manifest_by_id[sample_id]["user_id"] for sample_id in oof["sample_ids"]])
    manifest_labels = np.asarray([int(manifest_by_id[sample_id]["class_id"]) for sample_id in oof["sample_ids"]])
    if not np.array_equal(manifest_labels, oof["labels"]):
        raise ValueError("OOF labels disagree with training manifest")

    taxonomy_rows = read_csv(args.taxonomy.resolve())
    fixed_small_ids = {int(row["class_id"]) for row in taxonomy_rows if int(row["include_fixed_small_action"]) == 1}
    taxonomy_version = sorted(set(row["taxonomy_version"] for row in taxonomy_rows))

    weight = float(args.depth_weight)
    oof_skeleton_predictions = oof["skeleton_logits"].argmax(1)
    oof_depth_predictions = oof["depth_logits"].argmax(1)
    oof_fused_logits = (1.0 - weight) * oof["skeleton_logits"] + weight * oof["depth_logits"]
    oof_fused_predictions = oof_fused_logits.argmax(1)
    oof_methods = {
        "skeleton": oof_skeleton_predictions,
        "depth": oof_depth_predictions,
        "fusion_w040": oof_fused_predictions,
    }

    test_skeleton_predictions = test["skeleton_logits"].argmax(1)
    test_depth_predictions = test["depth_logits"].argmax(1)
    test_fused_logits = (1.0 - weight) * test["skeleton_logits"] + weight * test["depth_logits"]
    test_fused_probabilities = softmax(test_fused_logits)
    test_fused_predictions = test_fused_probabilities.argmax(1)
    test_sensitivity, test_shuffle_change_probability = test_shuffle_sensitivity(
        test["skeleton_logits"], test["depth_logits"], weight, int(args.repeats), int(args.seed)
    )

    oof_true_distribution = distribution(oof["labels"])
    oof_pred_distribution = distribution(oof_fused_predictions)
    test_pred_distribution = distribution(test_fused_predictions)

    per_class_rows = []
    matrix = confusion_matrix(oof["labels"], oof_fused_predictions, labels=np.arange(40))
    for class_id in range(40):
        row: dict[str, Any] = {
            "class_id": class_id,
            "action_name": class_names[class_id],
            "fixed_small_action": int(class_id in fixed_small_ids),
            "oof_support": int(np.sum(oof["labels"] == class_id)),
            "test_fused_prediction_count": int(np.sum(test_fused_predictions == class_id)),
            "test_skeleton_prediction_count": int(np.sum(test_skeleton_predictions == class_id)),
            "test_depth_prediction_count": int(np.sum(test_depth_predictions == class_id)),
        }
        for method, predictions in oof_methods.items():
            label_mask = oof["labels"] == class_id
            predicted_mask = predictions == class_id
            true_positive = int(np.sum(label_mask & predicted_mask))
            row[f"{method}_recall"] = float(true_positive / label_mask.sum()) if label_mask.sum() else None
            row[f"{method}_precision"] = float(true_positive / predicted_mask.sum()) if predicted_mask.sum() else None
            row[f"{method}_prediction_count"] = int(predicted_mask.sum())
        row["fusion_minus_skeleton_recall_pp"] = 100.0 * (
            float(row["fusion_w040_recall"]) - float(row["skeleton_recall"])
        )
        confusion_targets = [target for target in np.argsort(matrix[class_id])[::-1] if target != class_id and matrix[class_id, target] > 0]
        row["top_confusions"] = "|".join(
            f"{target}:{class_names[target]}:{int(matrix[class_id, target])}" for target in confusion_targets[:3]
        )
        per_class_rows.append(row)

    with (output_dir / "sd_oof_per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    test_union_rows = read_csv(args.test_union_manifest.resolve())
    test_union_by_id = {row["sample_id"]: row for row in test_union_rows}
    test_rows = []
    test_entropy = entropy(test_fused_probabilities)
    test_confidence = test_fused_probabilities.max(1)
    for index, sample_id in enumerate(test["sample_ids"]):
        if sample_id not in test_union_by_id:
            raise KeyError(f"Test sample missing from union manifest: {sample_id}")
        availability = test_union_by_id[sample_id]
        test_rows.append(
            {
                "sample_id": sample_id,
                "fusion_prediction": int(test_fused_predictions[index]),
                "skeleton_prediction": int(test_skeleton_predictions[index]),
                "depth_prediction": int(test_depth_predictions[index]),
                "drop_depth_changes_prediction": int(test_fused_predictions[index] != test_skeleton_predictions[index]),
                "drop_skeleton_changes_prediction": int(test_fused_predictions[index] != test_depth_predictions[index]),
                "random_depth_shuffle_change_probability": float(test_shuffle_change_probability[index]),
                "fusion_confidence": float(test_confidence[index]),
                "fusion_entropy": float(test_entropy[index]),
                "present_pattern": availability["present_pattern"],
                "usable_pattern": availability["usable_pattern"],
                "skeleton_file_count": availability["skeleton_file_count"],
                "radar_usable": availability["radar_usable"],
                "thermal_usable": availability["thermal_usable"],
            }
        )
    with (output_dir / "sd_test_prediction_sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)

    oof_shuffle_wrong = repeated_oof_shuffle(
        oof["skeleton_logits"], oof["depth_logits"], oof["labels"], users, weight,
        int(args.repeats), int(args.seed), same_label=False,
    )
    oof_shuffle_control = repeated_oof_shuffle(
        oof["skeleton_logits"], oof["depth_logits"], oof["labels"], users, weight,
        int(args.repeats), int(args.seed) + 10000, same_label=True,
    )
    test_sensitivity["drop_depth"] = {
        "prediction_change_rate": float(np.mean(test_fused_predictions != test_skeleton_predictions)),
        "mean_confidence_delta": float(softmax(test["skeleton_logits"]).max(1).mean() - test_confidence.mean()),
        "mean_entropy_delta": float(entropy(softmax(test["skeleton_logits"])).mean() - test_entropy.mean()),
    }
    test_sensitivity["drop_skeleton"] = {
        "prediction_change_rate": float(np.mean(test_fused_predictions != test_depth_predictions)),
        "mean_confidence_delta": float(softmax(test["depth_logits"]).max(1).mean() - test_confidence.mean()),
        "mean_entropy_delta": float(entropy(softmax(test["depth_logits"])).mean() - test_entropy.mean()),
    }

    summary = {
        "protocol": {
            "depth_weight": weight,
            "oof": "pooled three-fold subject-disjoint logits; all comparisons use identical samples",
            "test": "unlabelled sensitivity only; no Test accuracy is estimated",
            "shuffle_repeats": int(args.repeats),
        },
        "oof": {
            "samples": len(oof["labels"]),
            "subjects": len(set(users.tolist())),
            "metrics": {method: metrics(oof["labels"], predictions) for method, predictions in oof_methods.items()},
            "drop_depth": {
                **metrics(oof["labels"], oof_skeleton_predictions),
                "accuracy_delta_vs_fusion_pp": 100.0 * (
                    accuracy_score(oof["labels"], oof_skeleton_predictions)
                    - accuracy_score(oof["labels"], oof_fused_predictions)
                ),
            },
            "drop_skeleton": {
                **metrics(oof["labels"], oof_depth_predictions),
                "accuracy_delta_vs_fusion_pp": 100.0 * (
                    accuracy_score(oof["labels"], oof_depth_predictions)
                    - accuracy_score(oof["labels"], oof_fused_predictions)
                ),
            },
            "shuffle_depth_same_subject_different_class": oof_shuffle_wrong,
            "shuffle_depth_same_class_control": oof_shuffle_control,
            "fixed_small_action": {
                "taxonomy_version": taxonomy_version,
                "class_ids": sorted(fixed_small_ids),
                "metrics": {
                    method: subset_metrics(oof["labels"], predictions, fixed_small_ids)
                    for method, predictions in oof_methods.items()
                },
            },
        },
        "test": {
            "samples": len(test_fused_predictions),
            "fused_predicted_classes": len(set(test_fused_predictions.tolist())),
            "fused_missing_predicted_classes": [class_id for class_id in range(40) if class_id not in set(test_fused_predictions.tolist())],
            "prediction_counts": {
                "skeleton": {str(i): int(np.sum(test_skeleton_predictions == i)) for i in range(40)},
                "depth": {str(i): int(np.sum(test_depth_predictions == i)) for i in range(40)},
                "fusion_w040": {str(i): int(np.sum(test_fused_predictions == i)) for i in range(40)},
            },
            "distribution_divergence_bits": {
                "test_prediction_vs_oof_prediction_js": js_divergence(test_pred_distribution, oof_pred_distribution),
                "test_prediction_vs_oof_true_label_js": js_divergence(test_pred_distribution, oof_true_distribution),
                "oof_prediction_vs_oof_true_label_js": js_divergence(oof_pred_distribution, oof_true_distribution),
            },
            "sensitivity": test_sensitivity,
        },
        "outputs": {
            "per_class": str((output_dir / "sd_oof_per_class.csv").resolve()),
            "test_sensitivity": str((output_dir / "sd_test_prediction_sensitivity.csv").resolve()),
        },
    }
    (output_dir / "sd_oof_test_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
