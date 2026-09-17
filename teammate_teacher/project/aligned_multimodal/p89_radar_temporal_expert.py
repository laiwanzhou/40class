from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_MANIFEST = PROJECT_DIR / "data/six_modality_audit/train_union_manifest.csv"
TEST_MANIFEST = PROJECT_DIR / "data/p46_test_union_manifest.csv"
FOLDS = PROJECT_DIR / "data/subject_folds/folds_summary.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p89_radar_temporal_expert_v1"
SEED = 20260816
POINT_COLUMNS = ("x", "y", "z", "v", "snr", "noise")
FRAME_SIGNAL_NAMES = (
    "log_count",
    *(f"{name}_{stat}" for name in (*POINT_COLUMNS, "range", "abs_v") for stat in ("mean", "std", "q10", "q50", "q90")),
    "dynamic_005",
    "dynamic_010",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal point-cloud Radar expert with subject-disjoint OOF."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    values = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    frame = int(row["frame"])
                    point = [float(row[name]) for name in POINT_COLUMNS]
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(point).all():
                    frames.append(frame)
                    values.append(point)
    except (OSError, UnicodeError):
        pass
    if not values:
        return np.zeros(0, dtype=np.int64), np.zeros((0, len(POINT_COLUMNS)), dtype=np.float64)
    return np.asarray(frames, dtype=np.int64), np.asarray(values, dtype=np.float64)


def interpolate(sequence: np.ndarray, length: int = 32) -> np.ndarray:
    if len(sequence) == 1:
        return np.repeat(sequence, length, axis=0)
    source = np.linspace(0.0, 1.0, len(sequence))
    target = np.linspace(0.0, 1.0, length)
    return np.stack(
        [np.interp(target, source, sequence[:, index]) for index in range(sequence.shape[1])],
        axis=1,
    )


def distribution(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return np.zeros(12, dtype=np.float64)
    quantiles = np.quantile(values, (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99))
    return np.asarray(
        [values.mean(), values.std(), *quantiles.tolist()], dtype=np.float64
    )


def normalized_histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    histogram = np.histogram(np.clip(values, bins[0], bins[-1]), bins=bins)[0].astype(np.float64)
    return histogram / max(histogram.sum(), 1.0)


def extract_trial(path: Path) -> np.ndarray:
    files = sorted(path.glob("*.csv")) if path.is_dir() else []
    frame_parts = []
    value_parts = []
    for file_path in files:
        frames, values = numeric_file(file_path)
        if len(values):
            frame_parts.append(frames)
            value_parts.append(values)
    if not value_parts:
        raise RuntimeError(f"Radar trial has no points: {path}")
    frames = np.concatenate(frame_parts)
    values = np.concatenate(value_parts)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    values = values[order]
    unique_frames = np.unique(frames)
    frame_signals = []
    for frame in unique_frames:
        point = values[frames == frame]
        ranges = np.linalg.norm(point[:, :3], axis=1)
        abs_v = np.abs(point[:, 3])
        channels = [*point.T, ranges, abs_v]
        signal = [np.log1p(len(point))]
        for channel in channels:
            signal.extend(
                (
                    float(channel.mean()),
                    float(channel.std()),
                    *np.quantile(channel, (0.10, 0.50, 0.90)).tolist(),
                )
            )
        signal.extend((float(np.mean(abs_v > 0.05)), float(np.mean(abs_v > 0.10))))
        frame_signals.append(signal)
    sequence = np.asarray(frame_signals, dtype=np.float64)
    if sequence.shape[1] != len(FRAME_SIGNAL_NAMES):
        raise RuntimeError(f"Radar frame feature contract changed: {sequence.shape}")
    resampled = interpolate(sequence, 32)

    temporal_blocks = [
        sequence.mean(axis=0),
        sequence.std(axis=0),
        np.quantile(sequence, 0.10, axis=0),
        np.quantile(sequence, 0.50, axis=0),
        np.quantile(sequence, 0.90, axis=0),
        sequence.max(axis=0) - sequence.min(axis=0),
        np.mean(np.abs(np.diff(sequence, axis=0)), axis=0)
        if len(sequence) > 1
        else np.zeros(sequence.shape[1]),
        resampled.reshape(-1),
    ]
    frequency_columns = (0, 1, 6, 11, 16, 21, 36, 41, 42)
    frequency = []
    for column in frequency_columns:
        signal = resampled[:, column] - resampled[:, column].mean()
        frequency.extend(np.abs(np.fft.rfft(signal))[1:9].tolist())
    temporal_blocks.append(np.asarray(frequency, dtype=np.float64))

    ranges = np.linalg.norm(values[:, :3], axis=1)
    global_blocks = [distribution(values[:, index]) for index in range(values.shape[1])]
    global_blocks.extend((distribution(ranges), distribution(np.abs(values[:, 3]))))
    histogram_blocks = [
        normalized_histogram(values[:, 0], np.linspace(-5.0, 5.0, 25)),
        normalized_histogram(values[:, 1], np.linspace(-1.0, 10.0, 25)),
        normalized_histogram(values[:, 2], np.linspace(-3.0, 3.0, 25)),
        normalized_histogram(values[:, 3], np.linspace(-3.0, 3.0, 33)),
        normalized_histogram(ranges, np.linspace(0.0, 10.0, 33)),
    ]
    range_velocity = np.histogram2d(
        np.clip(ranges, 0.0, 10.0),
        np.clip(values[:, 3], -3.0, 3.0),
        bins=(np.linspace(0.0, 10.0, 13), np.linspace(-3.0, 3.0, 13)),
    )[0].astype(np.float64)
    range_velocity /= max(range_velocity.sum(), 1.0)
    metadata = np.asarray(
        [
            np.log1p(len(values)),
            np.log1p(len(unique_frames)),
            np.log1p(len(values) / len(unique_frames)),
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(
        np.concatenate(
            [metadata, *global_blocks, *histogram_blocks, range_velocity.reshape(-1), *temporal_blocks]
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def build_features(
    manifest_path: Path, output: Path, split: str, rebuild: bool
) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = [row for row in read_csv(manifest_path) if row["radar_usable"] == "1"]
    rows.sort(key=lambda row: row.get("official_sample_id", row["sample_id"]) or row["sample_id"])
    cache = output / f"{split}_features.npz"
    if cache.is_file() and not rebuild:
        with np.load(cache) as data:
            cached_ids = data["sample_ids"].astype(str)
            expected_ids = np.asarray([row.get("official_sample_id", row["sample_id"]) or row["sample_id"] for row in rows])
            if np.array_equal(cached_ids, expected_ids):
                return rows, np.asarray(data["features"], dtype=np.float32)
    features = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        features.append(extract_trial(Path(row["radar_path"])))
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "split": split,
                        "processed": index + 1,
                        "total": len(rows),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    matrix = np.stack(features)
    np.savez_compressed(
        cache,
        sample_ids=np.asarray([row.get("official_sample_id", row["sample_id"]) or row["sample_id"] for row in rows]),
        features=matrix,
    )
    return rows, matrix


def make_model(name: str):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=900,
            max_depth=24,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=40,
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=7,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.35,
            reg_alpha=1.0,
            reg_lambda=8.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(name)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_rows, train_x = build_features(
        TRAIN_MANIFEST, output, "train", args.rebuild_features
    )
    test_rows, test_x = build_features(
        TEST_MANIFEST, output, "test", args.rebuild_features
    )
    labels = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in train_rows]).astype(str)
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    results = {}
    oof_probabilities = {}
    test_probabilities = {}
    for name in ("extra_trees", "lightgbm"):
        oof = np.zeros((len(labels), 40), dtype=np.float64)
        per_fold = []
        for fold in folds:
            fit = np.isin(users, fold["train_users"])
            held = np.isin(users, fold["val_users"])
            estimator = make_model(name)
            estimator.fit(train_x[fit], labels[fit])
            partial = estimator.predict_proba(train_x[held])
            target = np.flatnonzero(held)
            oof[np.ix_(target, estimator.classes_.astype(np.int64))] = partial
            per_fold.append(
                {"fold": int(fold["fold"]), **metrics(labels[held], oof[target].argmax(1))}
            )
        final = make_model(name)
        final.fit(train_x, labels)
        partial = final.predict_proba(test_x)
        test_probability = np.zeros((len(test_rows), 40), dtype=np.float64)
        test_probability[:, final.classes_.astype(np.int64)] = partial
        oof_probabilities[name] = oof
        test_probabilities[name] = test_probability
        results[name] = {"overall": metrics(labels, oof.argmax(1)), "folds": per_fold}
        print(name, results[name], flush=True)
    selected = max(
        results,
        key=lambda name: (
            min(item["correct"] / item["total"] for item in results[name]["folds"]),
            results[name]["overall"]["accuracy"],
        ),
    )
    np.savez_compressed(
        output / "oof_logits.npz",
        sample_ids=np.asarray([row["sample_id"] for row in train_rows]),
        labels=labels,
        radar_present=np.ones(len(train_rows), dtype=np.uint8),
        radar_logits=np.log(np.maximum(oof_probabilities[selected], 1e-12)),
    )
    np.savez_compressed(
        output / "test_logits.npz",
        sample_ids=np.asarray([row["official_sample_id"] for row in test_rows]),
        radar_present=np.ones(len(test_rows), dtype=np.uint8),
        radar_logits=np.log(np.maximum(test_probabilities[selected], 1e-12)),
    )
    report = {
        "stage": "P89_temporal_point_cloud_Radar_expert_v1",
        "protocol": (
            "Only non-empty Radar files; frame-resampled point-cloud statistics and "
            "range-Doppler occupancy; fixed subject-disjoint folds; no timestamps, "
            "user IDs, test labels, or availability pattern are model features."
        ),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "feature_dimensions": int(train_x.shape[1]),
        "results": results,
        "selected": selected,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
