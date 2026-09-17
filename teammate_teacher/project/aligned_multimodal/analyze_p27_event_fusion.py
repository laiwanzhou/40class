from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from probe_p27r3_incremental_information import metric_bundle, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = PROJECT_DIR / "runs" / "p27_strong_inner" / "skeleton_imu"
DEFAULT_EVENT = PROJECT_DIR / "runs" / "p27_strong_inner" / "event_expert"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "runs" / "p27_strong_inner" / "skeleton_imu_event_fusion"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit strong-base plus explicit-event expert fusion"
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-logit-key", default="fused_logits")
    parser.add_argument("--event-dir", type=Path, default=DEFAULT_EVENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def log_softmax(source: np.ndarray) -> np.ndarray:
    shifted = source - source.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def align(
    base: np.lib.npyio.NpzFile,
    event: np.lib.npyio.NpzFile,
    base_logit_key: str,
) -> dict[str, np.ndarray]:
    if bool(base["outer_held_predictions_generated"]) or bool(
        event["outer_held_predictions_generated"]
    ):
        raise RuntimeError("Outer-held predictions are forbidden")
    position = {
        str(sample_id): index
        for index, sample_id in enumerate(event["sample_ids"].astype(str))
    }
    indices = np.asarray(
        [position[str(sample_id)] for sample_id in base["sample_ids"].astype(str)]
    )
    if not np.array_equal(base["labels"], event["labels"][indices]):
        raise RuntimeError("Base/event label mismatch")
    return {
        "sample_ids": base["sample_ids"],
        "labels": base["labels"].astype(np.int64),
        "subjects": base["subjects"],
        "base_logits": base[base_logit_key].astype(np.float64),
        "event_logits": event["logits"][indices].astype(np.float64),
    }


def fuse(part: dict[str, np.ndarray], parameters: np.ndarray) -> np.ndarray:
    base_temperature, event_temperature, event_weight = parameters
    return (
        (1.0 - event_weight) * part["base_logits"] / base_temperature
        + event_weight * part["event_logits"] / event_temperature
    )


def fit(parts: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    source = {
        key: np.concatenate([part[key] for part in parts])
        for key in ("labels", "base_logits", "event_logits")
    }

    def objective(parameters: np.ndarray) -> float:
        logits = fuse(source, parameters)
        return float(
            -log_softmax(logits)[
                np.arange(len(source["labels"])), source["labels"]
            ].mean()
        )

    result = minimize(
        objective,
        np.asarray([1.0, 1.0, 0.15], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((0.25, 3.0), (0.25, 3.0), (0.0, 0.5)),
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(str(result.message))
    return {
        "base_temperature": float(result.x[0]),
        "event_temperature": float(result.x[1]),
        "event_weight": float(result.x[2]),
        "nll": float(result.fun),
        "iterations": int(result.nit),
        "samples": int(len(source["labels"])),
    }


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
    parts = []
    for fold in range(3):
        base = np.load(
            args.base_dir.resolve() / f"fold_{fold}_logits.npz",
            allow_pickle=False,
        )
        event = np.load(
            args.event_dir.resolve() / f"fold_{fold}_logits.npz",
            allow_pickle=False,
        )
        parts.append(align(base, event, str(args.base_logit_key)))
    rows: list[dict[str, Any]] = []
    fused_logits: list[np.ndarray] = []
    cross_fit: dict[str, Any] = {}
    for target in range(3):
        parameters = fit(
            [parts[index] for index in range(3) if index != target]
        )
        cross_fit[str(target)] = parameters
        values = np.asarray(
            [
                parameters["base_temperature"],
                parameters["event_temperature"],
                parameters["event_weight"],
            ]
        )
        part = parts[target]
        fused = fuse(part, values)
        fused_logits.append(fused.astype(np.float32))
        methods = {
            "strong_base": part["base_logits"],
            "event_expert": part["event_logits"],
            "strong_base_event_cross_fitted_nll": fused,
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
    mean_metrics: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        mean_metrics[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in selected[0]
            if key not in {"inner_fold", "method"}
        }
    full_fit = fit(parts)
    np.savez_compressed(
        output / "cross_fitted_logits.npz",
        protocol=np.asarray("p27-strong-base-event-cross-fitted-nll-v1"),
        sample_ids=np.concatenate([part["sample_ids"] for part in parts]),
        labels=np.concatenate([part["labels"] for part in parts]),
        subjects=np.concatenate([part["subjects"] for part in parts]),
        inner_folds=np.concatenate(
            [
                np.full(len(part["labels"]), fold, dtype=np.int64)
                for fold, part in enumerate(parts)
            ]
        ),
        logits=np.concatenate(fused_logits),
        outer_held_predictions_generated=np.asarray(False),
    )
    summary = {
        "protocol": "p27-strong-base-event-cross-fitted-nll-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "base_logit_key": str(args.base_logit_key),
        "selection_target": "multiclass NLL only",
        "cross_fit_parameters": cross_fit,
        "full_outer_train_oof_parameters_for_future_freeze": full_fit,
        "mean_metrics": mean_metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
