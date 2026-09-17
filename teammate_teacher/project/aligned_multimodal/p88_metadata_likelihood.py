from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DEFAULT_TEACHER,
    DEFAULT_TRAIN_METADATA,
    classification_metrics,
    decode_sessions,
)
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_oof_candidate_ensemble import DEFAULT_H1_REPEAT, load_protocol
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
FEATURE_SETS = {
    "duration": (0,),
    "duration_rate": (0, 2),
    "duration_rows_rate": (0, 1, 2),
    "sensor": (0, 1, 2, 3, 4),
    "sensor_time": (0, 1, 2, 3, 4, 5, 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend frozen P87 with class-conditional recording metadata likelihood."
    )
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--repeat-config-summary", type=Path, default=DEFAULT_H1_REPEAT)
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument("--likelihood-temperatures", type=float, nargs="+", default=(1.0, 2.0, 4.0, 8.0))
    parser.add_argument("--weights", type=float, nargs="+", default=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30))
    parser.add_argument("--variance-prior", type=float, default=12.0)
    parser.add_argument("--aligned-shortlist", type=int, default=30)
    return parser.parse_args()


def read_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = np.asarray([row["sample_id"] for row in rows])
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in rows])
    feature_rows = []
    for row in rows:
        duration = max(float(row.get("duration_seconds") or 0.0), 0.0)
        imu_rows = max(float(row.get("imu_parsed_rows") or 0.0), 0.0)
        start = float(row.get("start_seconds") or 0.0)
        angle = 2.0 * np.pi * start / 86400.0
        feature_rows.append(
            [
                np.log1p(duration),
                np.log1p(imu_rows),
                np.log1p(imu_rows / max(duration, 0.1)),
                float(row.get("device_count") or 0.0),
                np.log1p(float(row.get("imu_csv_files") or 0.0)),
                np.sin(angle),
                np.cos(angle),
            ]
        )
    return ids, labels, users, np.asarray(feature_rows, dtype=np.float64)


def diagonal_gaussian_log_likelihood(
    train_feature: np.ndarray,
    train_label: np.ndarray,
    target_feature: np.ndarray,
    variance_prior: float,
) -> np.ndarray:
    global_variance = np.var(train_feature, axis=0) + 1e-4
    score = np.empty((len(target_feature), 40), dtype=np.float64)
    for class_id in range(40):
        values = train_feature[train_label == class_id]
        mean = values.mean(axis=0)
        class_variance = np.var(values, axis=0) + 1e-4
        variance = (
            len(values) * class_variance + variance_prior * global_variance
        ) / (len(values) + variance_prior)
        difference = target_feature - mean
        score[:, class_id] = -0.5 * np.sum(
            difference * difference / variance + np.log(variance), axis=1
        )
    return score


def softmax(score: np.ndarray, temperature: float) -> np.ndarray:
    scaled = score / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probability = np.exp(scaled)
    return probability / probability.sum(axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    (
        sample_ids,
        labels,
        base_probability,
        base_decoded,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        repeat,
    ) = load_protocol(args)
    all_ids, all_labels, all_users, all_feature = read_metadata(args.train_metadata.resolve())
    lookup = {sample_id: index for index, sample_id in enumerate(all_ids)}
    target_rows = np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
    fit_mask = ~np.isin(all_users, args.holdout_users)
    selected_here = args.fixed_summary is None
    if args.fixed_summary:
        source = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8"))
        configurations = [source["best"]["configuration"]]
    else:
        configurations = [
            {"feature_set": feature_set, "temperature": float(temperature), "weight": float(weight)}
            for feature_set in FEATURE_SETS
            for temperature in args.likelihood_temperatures
            for weight in args.weights
        ]
    likelihood_cache: dict[tuple[str, float], np.ndarray] = {}
    probability_cache: dict[tuple[str, float, float], np.ndarray] = {}
    conventional = []
    for configuration in configurations:
        feature_set = str(configuration["feature_set"])
        temperature = float(configuration["temperature"])
        weight = float(configuration["weight"])
        cache_key = (feature_set, temperature)
        columns = FEATURE_SETS[feature_set]
        if cache_key not in likelihood_cache:
            train_feature = all_feature[fit_mask][:, columns]
            target_feature = all_feature[target_rows][:, columns]
            score = diagonal_gaussian_log_likelihood(
                train_feature,
                all_labels[fit_mask],
                target_feature,
                float(args.variance_prior),
            )
            likelihood_cache[cache_key] = softmax(score, temperature)
        blended = (
            (1.0 - weight) * base_probability
            + weight * likelihood_cache[cache_key]
        )
        blended /= blended.sum(axis=1, keepdims=True)
        probability_cache[(feature_set, temperature, weight)] = blended
        raw = blended.argmax(axis=1)
        decoded = decode_sessions(
            np.log(np.maximum(blended, 1e-12)), sessions, transition, decoder
        )
        conventional.append(
            {
                "configuration": configuration,
                "metadata_only": classification_metrics(
                    labels, likelihood_cache[cache_key].argmax(axis=1)
                ),
                "raw": classification_metrics(labels, raw),
                "decoded": classification_metrics(labels, decoded),
                "decoded_rescue_harm_vs_base": rescue_harm(labels, base_decoded, decoded),
            }
        )
    conventional.sort(
        key=lambda result: (
            result["decoded"]["correct"],
            result["raw"]["correct"],
            -result["configuration"]["weight"],
        ),
        reverse=True,
    )
    aligned_results = []
    for result in conventional[: max(1, int(args.aligned_shortlist))]:
        configuration = result["configuration"]
        key = (
            str(configuration["feature_set"]),
            float(configuration["temperature"]),
            float(configuration["weight"]),
        )
        blended = probability_cache[key]
        prediction, grouping = decode_aligned_repeat(
            np.log(np.maximum(blended, 1e-12)),
            indices,
            metadata,
            transition,
            decoder,
            repeat,
        )
        aligned_results.append(
            {
                **result,
                "aligned": classification_metrics(labels, prediction),
                "aligned_rescue_harm_vs_base": rescue_harm(labels, base_decoded, prediction),
                "grouping": grouping,
            }
        )
    aligned_results.sort(
        key=lambda result: (
            result["aligned"]["correct"],
            result["aligned"]["balanced_accuracy"],
            result["decoded"]["correct"],
            -result["configuration"]["weight"],
        ),
        reverse=True,
    )
    summary = {
        "stage": "P88_recording_metadata_likelihood",
        "status": "complete",
        "selected_on_current_holdout": selected_here,
        "holdout_users": sorted(args.holdout_users),
        "base_decoded": classification_metrics(labels, base_decoded),
        "best": aligned_results[0],
        "repeat_configuration": asdict(repeat),
        "decoder_configuration": asdict(decoder),
        "grid_size": len(configurations),
        "all_aligned_candidates": aligned_results,
        "top_conventional_candidates": conventional[:50],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"all_aligned_candidates", "top_conventional_candidates"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
