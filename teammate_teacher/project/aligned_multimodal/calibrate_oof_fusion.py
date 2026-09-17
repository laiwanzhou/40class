from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用另外两个 subject folds 的 OOF logits 拟合温度，再应用到留出 fold"
    )
    parser.add_argument("--fold", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight", type=float, action="append", default=[])
    return parser.parse_args()


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = logits.astype(np.float64) / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1))
    return float(np.mean(logsumexp - scaled[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    # Scalar grid search is deterministic and sufficient for this one-parameter fit.
    coarse = np.exp(np.linspace(np.log(0.15), np.log(8.0), 240))
    losses = np.asarray([negative_log_likelihood(logits, labels, value) for value in coarse])
    best = float(coarse[int(losses.argmin())])
    low, high = max(0.05, best / 1.15), min(20.0, best * 1.15)
    fine = np.linspace(low, high, 240)
    fine_losses = np.asarray([negative_log_likelihood(logits, labels, value) for value in fine])
    temperature = float(fine[int(fine_losses.argmin())])
    return (
        temperature,
        negative_log_likelihood(logits, labels, 1.0),
        negative_log_likelihood(logits, labels, temperature),
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def write_predictions(
    path: Path, sample_ids: np.ndarray, labels: np.ndarray, predictions: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prediction"])
        writer.writerows(zip(sample_ids.tolist(), labels.tolist(), predictions.tolist()))


def main() -> None:
    args = parse_args()
    if len(args.fold) != 3:
        raise ValueError("必须恰好提供三个 --fold logits.npz")
    weights = args.weight or [0.5, 0.4]
    if any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("--weight 必须位于 [0, 1]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    all_sample_ids: list[str] = []
    for path in args.fold:
        with np.load(path.resolve()) as data:
            fold = {name: data[name].copy() for name in data.files}
        expected = {"sample_ids", "labels", "skeleton_logits", "depth_logits"}
        if set(fold) != expected:
            raise ValueError(f"{path} 字段不匹配：{sorted(fold)}")
        if len(set(fold["sample_ids"].tolist())) != len(fold["sample_ids"]):
            raise ValueError(f"{path} 内存在重复 sample_id")
        folds.append(fold)
        all_sample_ids.extend(fold["sample_ids"].tolist())
    if len(set(all_sample_ids)) != len(all_sample_ids):
        raise ValueError("三个 folds 之间存在重复 sample_id")

    summary: dict[str, object] = {
        "protocol": (
            "For each held-out subject fold, fit one scalar temperature per expert on the "
            "other two folds' OOF logits, then apply it to the held-out fold."
        ),
        "weights": weights,
        "folds": {},
        "sources": [str(path.resolve()) for path in args.fold],
    }
    pooled: dict[float, dict[str, list[np.ndarray]]] = {
        weight: {"labels": [], "predictions": []} for weight in weights
    }

    for held_out, fold in enumerate(folds):
        calibration_folds = [item for index, item in enumerate(folds) if index != held_out]
        calibration_labels = np.concatenate([item["labels"] for item in calibration_folds])
        calibration_skeleton = np.concatenate(
            [item["skeleton_logits"] for item in calibration_folds]
        )
        calibration_depth = np.concatenate([item["depth_logits"] for item in calibration_folds])
        skeleton_temperature, skeleton_nll_before, skeleton_nll_after = fit_temperature(
            calibration_skeleton, calibration_labels
        )
        depth_temperature, depth_nll_before, depth_nll_after = fit_temperature(
            calibration_depth, calibration_labels
        )
        skeleton_logits = fold["skeleton_logits"] / skeleton_temperature
        depth_logits = fold["depth_logits"] / depth_temperature
        fold_summary = {
            "calibration_samples": int(len(calibration_labels)),
            "held_out_samples": int(len(fold["labels"])),
            "skeleton_temperature": skeleton_temperature,
            "depth_temperature": depth_temperature,
            "calibration_nll": {
                "skeleton_before": skeleton_nll_before,
                "skeleton_after": skeleton_nll_after,
                "depth_before": depth_nll_before,
                "depth_after": depth_nll_after,
            },
            "results": {},
        }
        fold_dir = output_dir / f"fold_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for weight in weights:
            predictions = (
                (1.0 - weight) * skeleton_logits + weight * depth_logits
            ).argmax(axis=1)
            name = f"calibrated_w{round(weight * 100):03d}.csv"
            path = fold_dir / name
            write_predictions(path, fold["sample_ids"], fold["labels"], predictions)
            fold_summary["results"][f"{weight:.6g}"] = {
                "path": str(path),
                **metrics(fold["labels"], predictions),
            }
            pooled[weight]["labels"].append(fold["labels"])
            pooled[weight]["predictions"].append(predictions)
        summary["folds"][str(held_out)] = fold_summary

    summary["pooled_oof"] = {}
    for weight in weights:
        labels = np.concatenate(pooled[weight]["labels"])
        predictions = np.concatenate(pooled[weight]["predictions"])
        summary["pooled_oof"][f"{weight:.6g}"] = metrics(labels, predictions)

    output = output_dir / "calibration_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
