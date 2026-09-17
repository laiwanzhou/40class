from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_p46_70_ceiling import (
    DEFAULT_MANIFEST,
    DEFAULT_P12,
    DEFAULT_P46,
    DEFAULT_RUNS,
    DEFAULT_V3,
    greedy_oracle,
    load_manifest,
    load_required_predictions,
    metric_bundle,
    risk_reason,
    scan_saved_experts,
    write_csv,
    write_json,
)
from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DEFAULT_RUNS / "p46_70_ceiling_audit_v1" / "saved_oof"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory every saved OOF/logit NPZ aligned to frozen P46 validation."
    )
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12", type=Path, default=DEFAULT_P12)
    parser.add_argument("--p46", type=Path, default=DEFAULT_P46)
    parser.add_argument("--v3", type=Path, default=DEFAULT_V3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def source_ids(values: np.ndarray, sample_to_source: dict[str, str]) -> list[str | None]:
    return [sample_to_source.get(str(value), str(value)) for value in values]


def mapped_labels(values: np.ndarray, canonical_values: set[int]) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64)
    if set(np.unique(labels)).issubset(canonical_values):
        return labels
    if labels.size and labels.min() >= 0 and labels.max() < len(HARD_CLASS_IDS):
        return np.asarray(HARD_CLASS_IDS, dtype=np.int64)[labels]
    return labels


def predictions_from_array(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] in {len(HARD_CLASS_IDS), 40}:
        prediction = values.argmax(axis=1).astype(np.int64)
        if values.shape[1] == len(HARD_CLASS_IDS):
            prediction = np.asarray(HARD_CLASS_IDS, dtype=np.int64)[prediction]
        return prediction
    if values.ndim == 1 and np.issubdtype(values.dtype, np.number):
        return values.astype(np.int64)
    return None


def fold_purity(
    identifiers: list[str | None],
    folds: np.ndarray | None,
    manifest: dict[str, dict[str, str]],
    target_set: set[str],
) -> tuple[bool, str]:
    if folds is None or len(folds) != len(identifiers):
        return False, "missing_fold_vector"
    by_user: dict[str, set[int]] = {}
    for source_id, fold in zip(identifiers, folds):
        if source_id not in target_set:
            continue
        by_user.setdefault(manifest[source_id]["user_id"], set()).add(int(fold))
    if not by_user or any(len(values) != 1 for values in by_user.values()):
        return False, "target_user_split_across_folds"
    if len({next(iter(values)) for values in by_user.values()}) < 2:
        return False, "target_users_do_not_span_multiple_held_folds"
    return True, "subject_pure_fold_vector"


def common_candidates(
    path: Path,
    data: Any,
    sample_to_source: dict[str, str],
    manifest: dict[str, dict[str, str]],
    target_ids: list[str],
    canonical_labels: dict[str, int],
) -> list[tuple[dict[str, Any], np.ndarray]]:
    files = set(data.files)
    id_key = next((key for key in ("sample_ids", "source_ids") if key in files), None)
    label_key = next((key for key in ("labels", "label", "targets") if key in files), None)
    if id_key is None or label_key is None:
        return []
    identifiers = source_ids(data[id_key], sample_to_source)
    index = {value: row for row, value in enumerate(identifiers) if value is not None}
    if not set(target_ids).issubset(index):
        return []
    labels_all = mapped_labels(data[label_key], set(canonical_labels.values()))
    selected = np.asarray([index[value] for value in target_ids], dtype=np.int64)
    if any(int(labels_all[row]) != canonical_labels[source_id] for row, source_id in zip(selected, target_ids)):
        return []
    fold_key = next((key for key in ("folds", "held_fold", "held_folds") if key in files), None)
    fold_values = np.asarray(data[fold_key]) if fold_key else None
    verified, verification = fold_purity(
        identifiers, fold_values, manifest, set(target_ids)
    )
    output: list[tuple[dict[str, Any], np.ndarray]] = []
    for key in data.files:
        lowered = key.lower()
        if not ("logit" in lowered or "prediction" in lowered):
            continue
        if lowered in {"route_probability", "route_to_thermal"}:
            continue
        array = np.asarray(data[key])
        if len(array.shape) == 0 or array.shape[0] != len(identifiers):
            continue
        predictions = predictions_from_array(array)
        if predictions is None:
            continue
        selected_predictions = predictions[selected]
        output.append(
            (
                {
                    "path": str(path.resolve()),
                    "field": key,
                    "id_key": id_key,
                    "label_key": label_key,
                    "fold_key": fold_key or "",
                    "fold_pure_verified": verified,
                    "verification": verification,
                },
                selected_predictions,
            )
        )
    return output


def prefixed_candidates(
    path: Path,
    data: Any,
    sample_to_source: dict[str, str],
    manifest: dict[str, dict[str, str]],
    target_ids: list[str],
    canonical_labels: dict[str, int],
) -> list[tuple[dict[str, Any], np.ndarray]]:
    files = set(data.files)
    output: list[tuple[dict[str, Any], np.ndarray]] = []
    for prediction_key in data.files:
        if not prediction_key.endswith("__predictions"):
            continue
        prefix = prediction_key[: -len("__predictions")]
        id_key = prefix + "__sample_ids"
        label_key = prefix + "__labels"
        if id_key not in files or label_key not in files:
            continue
        identifiers = source_ids(data[id_key], sample_to_source)
        index = {value: row for row, value in enumerate(identifiers) if value is not None}
        if not set(target_ids).issubset(index):
            continue
        labels_all = mapped_labels(data[label_key], set(canonical_labels.values()))
        selected = np.asarray([index[value] for value in target_ids], dtype=np.int64)
        if any(
            int(labels_all[row]) != canonical_labels[source_id]
            for row, source_id in zip(selected, target_ids)
        ):
            continue
        predictions = predictions_from_array(data[prediction_key])
        if predictions is None:
            continue
        output.append(
            (
                {
                    "path": str(path.resolve()),
                    "field": prediction_key,
                    "id_key": id_key,
                    "label_key": label_key,
                    "fold_key": "",
                    "fold_pure_verified": False,
                    "verification": "prefixed_cross_subject_diagnostic_without_fold_vector",
                },
                predictions[selected],
            )
        )
    return output


def main() -> None:
    args = parse_args()
    runs = args.runs.resolve()
    output = args.output_dir.resolve()
    manifest, sample_to_source = load_manifest(args.manifest.resolve())
    required = {
        "p12": load_required_predictions(args.p12.resolve()),
        "p46": load_required_predictions(args.p46.resolve()),
        "v3": load_required_predictions(args.v3.resolve()),
    }
    target_ids = sorted(required["v3"])
    canonical_labels = {source_id: required["v3"][source_id][0] for source_id in target_ids}
    labels = np.asarray([canonical_labels[value] for value in target_ids])
    required_correct = [
        np.asarray(
            [values[source_id][1] == canonical_labels[source_id] for source_id in target_ids]
        )
        for values in required.values()
    ]
    initial_solved = np.logical_or.reduce(required_correct)
    triple_wrong = {
        source_id for source_id, solved in zip(target_ids, initial_solved) if not solved
    }
    _, csv_unique, csv_vectors = scan_saved_experts(
        runs,
        target_ids,
        canonical_labels,
        sample_to_source,
        triple_wrong,
        {manifest[source_id]["user_id"] for source_id in target_ids},
    )
    unique: dict[str, dict[str, Any]] = {
        row["prediction_hash"]: dict(row) for row in csv_unique
    }
    vectors = dict(csv_vectors)
    npz_rows: list[dict[str, Any]] = []
    paths = [
        path
        for path in sorted(runs.rglob("*.npz"))
        if any(token in path.name.lower() for token in ("oof", "logit", "prediction"))
    ]
    for number, path in enumerate(paths, start=1):
        try:
            with np.load(path, allow_pickle=False) as data:
                candidates = common_candidates(
                    path,
                    data,
                    sample_to_source,
                    manifest,
                    target_ids,
                    canonical_labels,
                )
                candidates.extend(
                    prefixed_candidates(
                        path,
                        data,
                        sample_to_source,
                        manifest,
                        target_ids,
                        canonical_labels,
                    )
                )
        except (OSError, ValueError, KeyError, EOFError):
            continue
        for metadata, predictions in candidates:
            metrics = metric_bundle(labels, predictions)
            digest = hashlib.sha256(predictions.astype(np.int16).tobytes()).hexdigest()
            corrections = int(
                sum(
                    predictions[index] == labels[index]
                    for index, source_id in enumerate(target_ids)
                    if source_id in triple_wrong
                )
            )
            risk = risk_reason(path)
            if not metadata["fold_pure_verified"]:
                risk = ";".join(
                    value for value in (risk, "oof_subject_purity_unverified") if value
                )
            row = {
                **metadata,
                **metrics,
                "correct": int((predictions == labels).sum()),
                "triple_wrong_corrections": corrections,
                "prediction_hash": digest,
                "risk_reason": risk,
                "eligible_for_ceiling": not bool(risk),
                "strong_for_ceiling": not bool(risk)
                and metrics["accuracy"] >= 0.30
                and metrics["macro_f1"] >= 0.25,
            }
            npz_rows.append(row)
            if digest not in unique:
                unique[digest] = dict(row)
                unique[digest]["aliases"] = f"{metadata['path']}::{metadata['field']}"
                unique[digest]["alias_count"] = 1
                vectors[digest] = predictions
            elif row["strong_for_ceiling"]:
                unique[digest]["strong_for_ceiling"] = True
                unique[digest]["eligible_for_ceiling"] = True
        if number % 50 == 0:
            print(json.dumps({"stage": "npz_scan", "completed": number, "total": len(paths)}), flush=True)
    unique_rows = list(unique.values())
    unique_rows.sort(
        key=lambda row: (
            -int(row.get("strong_for_ceiling", False)),
            -float(row.get("accuracy", 0.0)),
            -int(row.get("triple_wrong_corrections", 0)),
            str(row.get("path", "")),
        )
    )
    strong_oracle = greedy_oracle(
        target_ids,
        labels,
        initial_solved,
        unique_rows,
        vectors,
        eligible_only=True,
        strong_only=True,
    )
    best_single = max(npz_rows, key=lambda row: row["accuracy"], default=None)
    best_verified = max(
        (row for row in npz_rows if row["strong_for_ceiling"]),
        key=lambda row: row["accuracy"],
        default=None,
    )
    summary = {
        "protocol": "saved OOF/logit inventory aligned to frozen P46 validation",
        "npz_files_scanned": len(paths),
        "aligned_npz_fields": len(npz_rows),
        "unique_vectors_including_csv": len(unique_rows),
        "strong_verified_vectors_including_csv": sum(
            bool(row.get("strong_for_ceiling", False)) for row in unique_rows
        ),
        "best_aligned_npz_field": best_single,
        "best_strong_verified_npz_field": best_verified,
        "strong_verified_oracle": strong_oracle,
        "warning": (
            "All oracle figures are label-aware diagnostics. Only NPZ fields with a fold vector "
            "that keeps every target user inside exactly one held fold are marked verified."
        ),
    }
    npz_rows.sort(
        key=lambda row: (
            -int(row["strong_for_ceiling"]),
            -float(row["accuracy"]),
            -int(row["triple_wrong_corrections"]),
            row["path"],
            row["field"],
        )
    )
    write_csv(output / "aligned_npz_experts.csv", npz_rows)
    write_csv(output / "combined_unique_experts.csv", unique_rows)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
