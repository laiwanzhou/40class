from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from probe_p27r3_incremental_information import metric_bundle, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "runs" / "p27_strong_inner" / "skeleton_imu"
DEFAULT_OUTPUT = DEFAULT_INPUT / "calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit Skeleton/RF-IMU temperature and fusion calibration"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-logit-key", default="skeleton_logits")
    return parser.parse_args()


def log_softmax(source: np.ndarray) -> np.ndarray:
    shifted = source - source.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def fuse(
    skeleton_logits: np.ndarray,
    imu_logits: np.ndarray,
    device_counts: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    skeleton_temperature, imu_temperature, base_weight = parameters
    weights = (
        base_weight * np.clip(device_counts.astype(np.float64) / 5.0, 0.0, 1.0)
    )[:, None]
    return (
        (1.0 - weights) * skeleton_logits / skeleton_temperature
        + weights * imu_logits / imu_temperature
    )


def fit_parameters(
    parts: list[dict[str, np.ndarray]], base_logit_key: str
) -> dict[str, Any]:
    skeleton_logits = np.concatenate([part[base_logit_key] for part in parts])
    imu_logits = np.concatenate([part["imu_logits"] for part in parts])
    device_counts = np.concatenate([part["imu_device_counts"] for part in parts])
    labels = np.concatenate([part["labels"] for part in parts]).astype(np.int64)

    def objective(parameters: np.ndarray) -> float:
        logits = fuse(
            skeleton_logits, imu_logits, device_counts, parameters
        )
        return float(-log_softmax(logits)[np.arange(len(labels)), labels].mean())

    initial = np.asarray([0.92, 0.63, 0.40], dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=((0.25, 3.0), (0.25, 3.0), (0.0, 0.8)),
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Fusion calibration failed: {result.message}")
    return {
        "skeleton_temperature": float(result.x[0]),
        "imu_temperature": float(result.x[1]),
        "base_weight": float(result.x[2]),
        "nll": float(result.fun),
        "iterations": int(result.nit),
        "samples": int(len(labels)),
    }


def flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def load_fold(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    if bool(archive["outer_held_predictions_generated"]):
        raise RuntimeError(f"Outer-held predictions are forbidden here: {path}")
    return {key: archive[key] for key in archive.files}


def main() -> None:
    args = parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = [load_fold(source / f"fold_{fold}_logits.npz") for fold in range(3)]

    fixed_parameters = np.asarray([0.9167410586298098, 0.6312607866282122, 0.4])
    if any(args.base_logit_key not in part for part in folds):
        raise KeyError(f"Missing base logit key: {args.base_logit_key}")
    full_fit = fit_parameters(folds, args.base_logit_key)
    cross_fit_parameters: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    calibrated_logits: list[np.ndarray] = []
    for target in range(3):
        fit = fit_parameters(
            [folds[index] for index in range(3) if index != target],
            args.base_logit_key,
        )
        cross_fit_parameters[str(target)] = fit
        parameters = np.asarray(
            [
                fit["skeleton_temperature"],
                fit["imu_temperature"],
                fit["base_weight"],
            ]
        )
        part = folds[target]
        cross_fitted = fuse(
            part[args.base_logit_key],
            part["imu_logits"],
            part["imu_device_counts"],
            parameters,
        )
        calibrated_logits.append(cross_fitted.astype(np.float32))
        methods = {
            "base": part[args.base_logit_key],
            "base_imu_fixed": fuse(
                part[args.base_logit_key],
                part["imu_logits"],
                part["imu_device_counts"],
                fixed_parameters,
            ),
            "base_imu_cross_fitted_nll": cross_fitted,
        }
        for method, logits in methods.items():
            rows.append(
                {
                    "inner_fold": target,
                    "method": method,
                    **flatten_metrics(
                        metric_bundle(part["labels"], logits.argmax(axis=1))
                    ),
                }
            )

    write_csv(output / "fold_metrics.csv", rows)
    aggregate_rows: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        aggregate_rows.append(
            {
                "method": method,
                **{
                    key: float(np.mean([float(row[key]) for row in selected]))
                    for key in selected[0]
                    if key not in {"inner_fold", "method"}
                },
            }
        )
    write_csv(output / "mean_metrics.csv", aggregate_rows)
    np.savez_compressed(
        output / "cross_fitted_logits.npz",
        protocol=np.asarray("p27-strong-skeleton-rfimu-cross-fitted-nll-v1"),
        sample_ids=np.concatenate([part["sample_ids"] for part in folds]),
        labels=np.concatenate([part["labels"] for part in folds]),
        subjects=np.concatenate([part["subjects"] for part in folds]),
        inner_folds=np.concatenate(
            [
                np.full(len(part["labels"]), fold, dtype=np.int64)
                for fold, part in enumerate(folds)
            ]
        ),
        logits=np.concatenate(calibrated_logits),
        outer_held_predictions_generated=np.asarray(False),
    )
    summary = {
        "protocol": "p27-strong-skeleton-rfimu-cross-fitted-nll-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "selection_target": "multiclass NLL only; no direct optimization of Overall/Small/Hard",
        "base_logit_key": args.base_logit_key,
        "fixed_parameters": {
            "skeleton_temperature": float(fixed_parameters[0]),
            "imu_temperature": float(fixed_parameters[1]),
            "base_weight": float(fixed_parameters[2]),
        },
        "cross_fit_parameters": cross_fit_parameters,
        "full_outer_train_oof_parameters_for_future_freeze": full_fit,
        "mean_metrics": {
            str(row["method"]): {
                key: value for key, value in row.items() if key != "method"
            }
            for row in aggregate_rows
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
