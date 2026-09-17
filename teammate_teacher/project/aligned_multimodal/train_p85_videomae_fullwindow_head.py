from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from train_p46_videomae_head import fit_temperature, l2_normalize, make_model, row_standardize
from train_p85_videomae_full40_head import aligned_scores_40, metrics, sample_weights


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FULL = PROJECT_DIR / "runs/p85_videomae_large_fullwindow_full40_v1/complete_features.npz"
DEFAULT_WINDOWS = PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
DEFAULT_TEST_FULL = PROJECT_DIR / "runs/p85_videomae_large_fullwindow_test_v1/complete_features.npz"
DEFAULT_TEST_WINDOWS = PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p85_videomae_large_fullwindow_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full+early+late full-40 VideoMAE heads.")
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--test-full", type=Path, default=DEFAULT_TEST_FULL)
    parser.add_argument("--test-windows", type=Path, default=DEFAULT_TEST_WINDOWS)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def align(reference: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = np.asarray(cache["sample_ids"]).astype(str)
    lookup = {value: index for index, value in enumerate(ids)}
    if set(reference) != set(lookup):
        raise RuntimeError("Full and early/late caches contain different samples")
    order = np.asarray([lookup[value] for value in reference], dtype=np.int64)
    return {
        key: values[order] if values.ndim > 0 and values.shape[0] == len(ids) else values
        for key, values in cache.items()
    }


def matrices(
    full_features: np.ndarray,
    full_logits: np.ndarray,
    window_features: np.ndarray,
    window_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    full = l2_normalize(np.asarray(full_features, dtype=np.float32))
    windows = l2_normalize(np.asarray(window_features, dtype=np.float32))
    if full.shape[1:] != (3, 1024) or windows.shape[1:] != (2, 3, 1024):
        raise RuntimeError(f"Unexpected full/window shapes: {full.shape}, {windows.shape}")
    early, late = windows[:, 0], windows[:, 1]
    window_mean = l2_normalize(windows.mean(axis=1))
    three_mean = l2_normalize(np.concatenate((full[:, None], windows), axis=1).mean(axis=1))
    difference = late - early
    kinetics = row_standardize(
        np.concatenate(
            (
                np.asarray(full_logits, dtype=np.float32)[:, None],
                np.asarray(window_logits, dtype=np.float32),
            ),
            axis=1,
        ).reshape(len(full), -1)
    )
    return {
        "full": full.reshape(len(full), -1),
        "three_mean": three_mean.reshape(len(full), -1),
        "full_window_mean": np.concatenate((full, window_mean), axis=1).reshape(len(full), -1),
        "full_temporal_delta": np.concatenate((full, difference), axis=1).reshape(len(full), -1),
        "full_early_late": np.concatenate((full[:, None], windows), axis=1).reshape(len(full), -1),
        "three_clip_kinetics": kinetics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    full = load(args.full)
    sample_ids = np.asarray(full["sample_ids"]).astype(str)
    windows = align(sample_ids, load(args.windows))
    labels = np.asarray(full["labels"], dtype=np.int64)
    users = np.asarray(full["users"]).astype(str)
    if len(sample_ids) != 2914 or set(labels.tolist()) != set(range(40)):
        raise RuntimeError("Frozen full-40 cache changed")
    for key in ("labels", "users"):
        if not np.array_equal(np.asarray(full[key]).astype(str), np.asarray(windows[key]).astype(str)):
            raise RuntimeError(f"Full/windows metadata mismatch: {key}")
    values = matrices(
        full["features"], full["kinetics_logits"], windows["features"], windows["kinetics_logits"]
    )

    test_full = load(args.test_full)
    test_ids = np.asarray(test_full["sample_ids"]).astype(str)
    test_windows = align(test_ids, load(args.test_windows))
    test_values = matrices(
        test_full["features"],
        test_full["kinetics_logits"],
        test_windows["features"],
        test_windows["kinetics_logits"],
    )
    if len(test_ids) != 401:
        raise RuntimeError("Expected 401 readable Test rows")

    p12 = load(args.p12_oof)
    p12_lookup = {str(value): index for index, value in enumerate(p12["sample_ids"].astype(str))}
    order = np.asarray([p12_lookup[value] for value in sample_ids], dtype=np.int64)
    folds = np.asarray(p12["folds"], dtype=np.int64)[order]
    if not np.array_equal(labels, np.asarray(p12["labels"], dtype=np.int64)[order]):
        raise RuntimeError("P12 labels disagree with full-window cache")

    rows: list[dict[str, Any]] = []
    oof_by_config: dict[tuple[str, float, float], np.ndarray] = {}
    for name, matrix in values.items():
        for power in (0.5, 0.75):
            for alpha in (1000.0, 3000.0, 10000.0):
                oof = np.full((len(labels), 40), np.nan, dtype=np.float64)
                for held_fold in (0, 1, 2):
                    fit_indices = np.flatnonzero(folds != held_fold)
                    held_indices = np.flatnonzero(folds == held_fold)
                    model = make_model(alpha)
                    model.fit(
                        matrix[fit_indices],
                        labels[fit_indices],
                        ridge__sample_weight=sample_weights(labels[fit_indices], power),
                    )
                    oof[held_indices] = aligned_scores_40(model, matrix[held_indices])
                result = metrics(labels, oof.argmax(axis=1))
                row = {
                    "feature_set": name,
                    "class_weight_power": power,
                    "alpha": alpha,
                    "dimensions": int(matrix.shape[1]),
                    **result,
                }
                rows.append(row)
                oof_by_config[(name, power, alpha)] = oof
                print(
                    f"{name:21s} power={power:.2f} alpha={alpha:7g} "
                    f"acc={100*float(result['accuracy']):.2f}%",
                    flush=True,
                )
    write_csv(output / "group_oof_results.csv", rows)

    oof_payload: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "labels": labels,
        "users": users,
        "folds": folds,
    }
    test_payload: dict[str, np.ndarray] = {"sample_ids": test_ids}
    selected: dict[str, Any] = {}
    for name, matrix in values.items():
        row = max(
            (item for item in rows if item["feature_set"] == name),
            key=lambda item: (
                float(item["accuracy"]),
                float(item["balanced_accuracy"]),
                float(item["macro_f1"]),
            ),
        )
        power = float(row["class_weight_power"])
        alpha = float(row["alpha"])
        raw_oof = oof_by_config[(name, power, alpha)]
        temperature = fit_temperature(raw_oof, labels)
        model = make_model(alpha)
        model.fit(
            matrix,
            labels,
            ridge__sample_weight=sample_weights(labels, power),
        )
        oof_payload[f"{name}_logits"] = (raw_oof / temperature).astype(np.float32)
        test_payload[f"{name}_logits"] = (
            aligned_scores_40(model, test_values[name]) / temperature
        ).astype(np.float32)
        joblib.dump(model, output / f"final_{name}_head.joblib", compress=3)
        selected[name] = {"selected_oof": row, "temperature": temperature}
    for filename, payload in (
        ("candidate_oof_logits.npz", oof_payload),
        ("candidate_test_logits.npz", test_payload),
    ):
        path = output / filename
        temporary = path.with_suffix(path.suffix + ".building")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
    best = max(selected, key=lambda name: float(selected[name]["selected_oof"]["accuracy"]))
    summary = {
        "protocol": "full+early+late Large VideoMAE; frozen three subject folds",
        "deployment_rule_note": (
            "This head is a distillation-teacher diagnostic because final inference "
            "would otherwise require Large VideoMAE. A compliant student must remove "
            "that dependency and keep all deployed weights below 100 MB."
        ),
        "selected_by_feature": selected,
        "best_feature": best,
        "best_oof": selected[best]["selected_oof"],
        "oof_logits": str(output / "candidate_oof_logits.npz"),
        "test_logits": str(output / "candidate_test_logits.npz"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
