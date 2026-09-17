from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from p46_event_data import safe_path
from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = PROJECT_DIR / "runs"
DEFAULT_OUTPUT = DEFAULT_RUNS / "p46_70_ceiling_audit_v1"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "p46_single_split.csv"
DEFAULT_EVENT_RUN = DEFAULT_RUNS / "p46_event_inputs_full"
DEFAULT_P12 = (
    DEFAULT_RUNS
    / "p46_step10_detail21_fullcoverage_v2"
    / "p12_restricted_predictions.csv"
)
DEFAULT_P46 = (
    DEFAULT_RUNS
    / "p46_step10_detail21_fullcoverage_v2"
    / "best_macro_f1_predictions.csv"
)
DEFAULT_V3 = DEFAULT_RUNS / "p46_unified_repair_v3_clean" / "best_predictions.csv"

PREDICTION_COLUMNS = (
    "predicted_class_id",
    "prediction",
    "predicted_label",
    "pred_class_id",
    "y_pred",
)
LABEL_COLUMNS = (
    "true_class_id",
    "label",
    "true_label",
    "target",
    "class_id",
    "y_true",
)
RISK_TOKENS = (
    "smoke",
    "oracle",
    "manual",
    "reviewed",
    "annotation",
    "assisted",
    "test_prediction",
    "failure",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the empirical 70% ceiling on the frozen P46 validation trials, "
            "including every aligned saved expert and multimodal cache quality."
        )
    )
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--p12", type=Path, default=DEFAULT_P12)
    parser.add_argument("--p46", type=Path, default=DEFAULT_P46)
    parser.add_argument("--v3", type=Path, default=DEFAULT_V3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def first_present(fieldnames: Iterable[str], choices: Iterable[str]) -> str | None:
    available = set(fieldnames)
    return next((value for value in choices if value in available), None)


def load_manifest(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    rows = read_csv(path)
    by_source = {row["source_id"]: row for row in rows}
    sample_to_source = {row["sample_id"]: row["source_id"] for row in rows}
    if len(by_source) != len(rows) or len(sample_to_source) != len(rows):
        raise RuntimeError("manifest IDs are not unique")
    return by_source, sample_to_source


def prediction_rows(
    path: Path,
    sample_to_source: dict[str, str],
) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            id_column = first_present(fields, ("source_id", "sample_id"))
            label_column = first_present(fields, LABEL_COLUMNS)
            prediction_column = first_present(fields, PREDICTION_COLUMNS)
            detail_prediction = None
            detail_label = None
            if prediction_column is None and "predicted_detail_index" in fields:
                prediction_column = "predicted_detail_index"
                detail_prediction = True
            if label_column is None and "true_detail_index" in fields:
                label_column = "true_detail_index"
                detail_label = True
            if id_column is None or label_column is None or prediction_column is None:
                return {}, {"status": "unsupported_schema", "fields": fields}
            result: dict[str, tuple[int, int]] = {}
            duplicates = 0
            invalid = 0
            for row in reader:
                try:
                    identifier = row[id_column]
                    source_id = (
                        identifier
                        if id_column == "source_id"
                        else sample_to_source.get(identifier)
                    )
                    if source_id is None:
                        continue
                    label = int(float(row[label_column]))
                    prediction = int(float(row[prediction_column]))
                    if detail_label:
                        label = int(HARD_CLASS_IDS[label])
                    if detail_prediction:
                        prediction = int(HARD_CLASS_IDS[prediction])
                except (KeyError, TypeError, ValueError, IndexError):
                    invalid += 1
                    continue
                if source_id in result:
                    duplicates += 1
                    continue
                result[source_id] = (label, prediction)
            return result, {
                "status": "parsed",
                "id_column": id_column,
                "label_column": label_column,
                "prediction_column": prediction_column,
                "duplicates": duplicates,
                "invalid_rows": invalid,
            }
    except (OSError, UnicodeError, csv.Error) as error:
        return {}, {"status": "read_error", "error": str(error)}


def load_required_predictions(path: Path) -> dict[str, tuple[int, int]]:
    rows = read_csv(path)
    output: dict[str, tuple[int, int]] = {}
    for row in rows:
        label_key = "true_class_id"
        prediction_key = "predicted_class_id"
        output[row["source_id"]] = (int(row[label_key]), int(row[prediction_key]))
    if len(output) != len(rows):
        raise RuntimeError(f"duplicate required prediction IDs: {path}")
    return output


def metric_bundle(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.asarray(HARD_CLASS_IDS),
                average="macro",
                zero_division=0,
            )
        ),
    }


def risk_reason(path: Path) -> str:
    lowered = str(path).lower().replace("\\", "/")
    matches = [token for token in RISK_TOKENS if token in lowered]
    return ";".join(matches)


def nearest_protocol_metadata(path: Path, runs: Path) -> dict[str, Any]:
    current = path.parent
    while current == runs or runs in current.parents:
        for name in ("frozen_config.json", "summary.json", "resume_config.json"):
            candidate = current / name
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if "train_subjects" in payload or "val_subjects" in payload:
                return {"path": str(candidate.resolve()), "payload": payload}
        if current == runs:
            break
        current = current.parent
    return {}


def scan_saved_experts(
    runs: Path,
    target_ids: list[str],
    canonical_labels: dict[str, int],
    sample_to_source: dict[str, str],
    triple_wrong: set[str],
    target_users: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    target_set = set(target_ids)
    inventory: list[dict[str, Any]] = []
    unique: dict[str, dict[str, Any]] = {}
    prediction_vectors: dict[str, np.ndarray] = {}
    for path in sorted(runs.rglob("*.csv")):
        parsed, schema = prediction_rows(path, sample_to_source)
        overlap = target_set.intersection(parsed)
        if not overlap:
            continue
        label_mismatches = sum(
            parsed[source_id][0] != canonical_labels[source_id] for source_id in overlap
        )
        coverage = len(overlap)
        row: dict[str, Any] = {
            "path": str(path.resolve()),
            "status": schema["status"],
            "coverage": coverage,
            "coverage_rate": coverage / len(target_ids),
            "label_mismatches": label_mismatches,
            "risk_reason": risk_reason(path),
            "eligible_for_ceiling": False,
            **{f"schema_{key}": value for key, value in schema.items() if key != "status"},
        }
        if coverage == len(target_ids) and label_mismatches == 0:
            metadata = nearest_protocol_metadata(path, runs)
            payload = metadata.get("payload", {})
            train_subjects = {str(value) for value in payload.get("train_subjects", [])}
            val_subjects = {str(value) for value in payload.get("val_subjects", [])}
            subject_disjoint = bool(train_subjects and val_subjects) and (
                not target_users.intersection(train_subjects)
                and target_users.issubset(val_subjects)
            )
            if not subject_disjoint:
                row["risk_reason"] = ";".join(
                    value
                    for value in (row["risk_reason"], "subject_protocol_unknown_or_overlap")
                    if value
                )
            labels = np.asarray([canonical_labels[value] for value in target_ids])
            predictions = np.asarray([parsed[value][1] for value in target_ids])
            metrics = metric_bundle(labels, predictions)
            digest = hashlib.sha256(predictions.astype(np.int16).tobytes()).hexdigest()
            corrections = sum(
                parsed[source_id][1] == canonical_labels[source_id]
                for source_id in triple_wrong
            )
            row.update(
                {
                    **metrics,
                    "correct": int((labels == predictions).sum()),
                    "prediction_hash": digest,
                    "triple_wrong_corrections": corrections,
                    "eligible_for_ceiling": not bool(row["risk_reason"]),
                    "subject_disjoint_verified": subject_disjoint,
                    "protocol_metadata": metadata.get("path", ""),
                    "train_subjects": ";".join(sorted(train_subjects)),
                    "val_subjects": ";".join(sorted(val_subjects)),
                }
            )
            row["strong_for_ceiling"] = bool(row["eligible_for_ceiling"]) and (
                metrics["accuracy"] >= 0.30 and metrics["macro_f1"] >= 0.25
            )
            if digest not in unique:
                unique[digest] = dict(row)
                unique[digest]["aliases"] = [str(path.resolve())]
                prediction_vectors[digest] = predictions
            else:
                unique[digest]["aliases"].append(str(path.resolve()))
                if not row["risk_reason"]:
                    unique[digest]["eligible_for_ceiling"] = True
                if row.get("strong_for_ceiling", False):
                    unique[digest]["strong_for_ceiling"] = True
                    unique[digest]["subject_disjoint_verified"] = True
                    unique[digest]["protocol_metadata"] = row["protocol_metadata"]
                    unique[digest]["train_subjects"] = row["train_subjects"]
                    unique[digest]["val_subjects"] = row["val_subjects"]
        inventory.append(row)
    unique_rows = []
    for digest, row in unique.items():
        aliases = list(row.pop("aliases"))
        row["alias_count"] = len(aliases)
        row["aliases"] = " | ".join(aliases)
        unique_rows.append(row)
    inventory.sort(
        key=lambda row: (
            -int(row.get("coverage", 0)),
            -int(row.get("triple_wrong_corrections", -1)),
            -float(row.get("accuracy", -1.0)),
            str(row["path"]),
        )
    )
    unique_rows.sort(
        key=lambda row: (
            -int(row.get("triple_wrong_corrections", 0)),
            -float(row.get("accuracy", 0.0)),
            str(row["path"]),
        )
    )
    return inventory, unique_rows, prediction_vectors


def greedy_oracle(
    target_ids: list[str],
    labels: np.ndarray,
    initial_solved: np.ndarray,
    unique_rows: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    *,
    eligible_only: bool,
    strong_only: bool = False,
) -> dict[str, Any]:
    solved = initial_solved.copy()
    steps: list[dict[str, Any]] = []
    available = {
        row["prediction_hash"]: row
        for row in unique_rows
        if not eligible_only or row["eligible_for_ceiling"]
        if not strong_only or row.get("strong_for_ceiling", False)
    }
    while available:
        best_digest = ""
        best_gain = 0
        for digest in available:
            correct = vectors[digest] == labels
            gain = int((correct & ~solved).sum())
            if gain > best_gain:
                best_digest, best_gain = digest, gain
        if not best_digest:
            break
        row = available.pop(best_digest)
        solved |= vectors[best_digest] == labels
        steps.append(
            {
                "step": len(steps) + 1,
                "path": row["path"],
                "risk_reason": row["risk_reason"],
                "new_correct": best_gain,
                "oracle_correct": int(solved.sum()),
                "oracle_accuracy": float(solved.mean()),
            }
        )
    return {
        "correct": int(solved.sum()),
        "accuracy": float(solved.mean()),
        "steps": steps,
    }


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    return float(values.std() / max(abs(values.mean()), 1e-6))


def event_quality(event_path: Path) -> dict[str, float]:
    with np.load(event_path, allow_pickle=False) as cache:
        frame_time = cache["frame_time_seconds"].astype(np.float64)
        roi_valid = cache["roi_valid"].astype(bool)
        roi_quality = cache["roi_quality"].astype(np.float64)
        geometry = cache["oriented_roi_geometry"].astype(np.float64)
        region_names = [str(value) for value in cache["local_region_names"]]
        result: dict[str, float] = {
            "frames": float(len(frame_time)),
            "duration_seconds": (
                float(frame_time[-1] - frame_time[0]) if len(frame_time) > 1 else 0.0
            ),
            "frame_delta_cv": (
                coefficient_of_variation(np.diff(frame_time))
                if len(frame_time) > 2
                else float("nan")
            ),
            "roi_valid_rate": float(roi_valid.mean()),
            "roi_quality_mean": finite_mean(roi_quality[roi_valid]),
            "roi_clipped_ratio_mean": finite_mean(cache["roi_clipped_ratio"]),
            "oriented_angle_valid_rate": float(
                cache["oriented_angle_valid"].astype(bool).mean()
            ),
            "pose_quality_mean": finite_mean(cache["pose_quality_factor"]),
            "skeleton_joint_valid_rate": float(
                cache["skeleton_joint_mask"].astype(bool).mean()
            ),
            "skeleton_frame_quality_mean": finite_mean(
                cache["skeleton_frame_quality"]
            ),
            "body_axes_raw_valid_rate": float(
                cache["body_axes_raw_valid"].astype(bool).mean()
            ),
            "imu_points": float(len(cache["imu_values"])),
            "imu_points_per_frame": float(
                len(cache["imu_values"]) / max(len(frame_time), 1)
            ),
            "imu_device_valid_rate": float(cache["imu_device_mask"].astype(bool).mean()),
        }
        center_steps: list[float] = []
        size_cvs: list[float] = []
        for index, name in enumerate(region_names):
            safe_name = name.lower().replace(" ", "_")
            valid = roi_valid[:, index]
            result[f"roi_{safe_name}_valid_rate"] = float(valid.mean())
            result[f"roi_{safe_name}_quality_mean"] = finite_mean(
                roi_quality[valid, index]
            )
            pair = valid[1:] & valid[:-1]
            if pair.any():
                step = np.linalg.vector_norm(
                    geometry[1:, index, :2] - geometry[:-1, index, :2], axis=1
                )[pair]
                center_steps.extend(step.tolist())
            if valid.any():
                sizes = np.sqrt(
                    np.clip(geometry[valid, index, 2] * geometry[valid, index, 3], 0.0, None)
                )
                size_cvs.append(coefficient_of_variation(sizes))
        result["roi_center_step_mean"] = finite_mean(np.asarray(center_steps))
        result["roi_size_cv_mean"] = finite_mean(np.asarray(size_cvs))
        return result


def raw_file_count(path_text: str) -> int:
    path = Path(path_text)
    if not path.is_dir():
        return 0
    try:
        return sum(value.is_file() for value in path.iterdir())
    except OSError:
        return 0


def quality_rows(
    target_ids: list[str],
    manifest: dict[str, dict[str, str]],
    event_run: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    cache_root = event_run / "trial_event_cache"
    for index, source_id in enumerate(target_ids, start=1):
        row = manifest[source_id]
        event_path = cache_root / safe_path(source_id).with_suffix(".npz")
        quality: dict[str, Any] = {
            "source_id": source_id,
            "sample_id": row["sample_id"],
            "user_id": row["user_id"],
            "class_id": int(row["class_id"]),
            "trial_id": row["trial_id"],
            "event_cache_exists": event_path.is_file(),
            "raw_depth_files": raw_file_count(row["depth_dir"]),
            "raw_ir_files": raw_file_count(row["ir_dir"]),
            "raw_skeleton_files": raw_file_count(row["skeleton_dir"]),
        }
        if event_path.is_file():
            quality.update(event_quality(event_path))
        result[source_id] = quality
        if index % 50 == 0:
            print(json.dumps({"stage": "quality", "completed": index}), flush=True)
    return result


def quantile(values: list[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if len(array) else float("nan")


def failure_flags(row: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    checks = (
        ("low_roi_valid", row.get("roi_valid_rate"), "lt", thresholds["roi_valid_q25"]),
        ("low_roi_quality", row.get("roi_quality_mean"), "lt", thresholds["roi_quality_q25"]),
        ("unstable_roi_center", row.get("roi_center_step_mean"), "gt", thresholds["roi_step_q75"]),
        ("unstable_roi_size", row.get("roi_size_cv_mean"), "gt", thresholds["roi_size_q75"]),
        ("high_roi_clipping", row.get("roi_clipped_ratio_mean"), "gt", thresholds["roi_clip_q75"]),
        ("low_skeleton_quality", row.get("skeleton_frame_quality_mean"), "lt", thresholds["skeleton_q25"]),
        ("low_skeleton_coverage", row.get("skeleton_joint_valid_rate"), "lt", thresholds["skeleton_coverage_q25"]),
        ("sparse_imu", row.get("imu_points_per_frame"), "lt", thresholds["imu_q25"]),
        ("short_clip", row.get("frames"), "lt", thresholds["frames_q10"]),
    )
    output: list[str] = []
    for name, raw_value, direction, threshold in checks:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or not math.isfinite(threshold):
            continue
        if (direction == "lt" and value < threshold) or (
            direction == "gt" and value > threshold
        ):
            output.append(name)
    return output


def aggregate_rows(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        output.append(
            {
                **dict(zip(group_fields, key)),
                "samples": len(values),
                "existing_expert_rescues": sum(
                    int(row["historical_expert_rescue_count"] > 0) for row in values
                ),
                "pipeline_quality_failures": sum(
                    row["audit_category"] == "pipeline_quality_failure" for row in values
                ),
                "representation_gaps": sum(
                    row["audit_category"] == "new_representation_needed" for row in values
                ),
                "roi_valid_rate_mean": finite_mean(
                    np.asarray([row.get("roi_valid_rate", np.nan) for row in values])
                ),
                "skeleton_quality_mean": finite_mean(
                    np.asarray(
                        [row.get("skeleton_frame_quality_mean", np.nan) for row in values]
                    )
                ),
                "imu_points_per_frame_mean": finite_mean(
                    np.asarray(
                        [row.get("imu_points_per_frame", np.nan) for row in values]
                    )
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    manifest, sample_to_source = load_manifest(args.manifest.resolve())
    required = {
        "p12": load_required_predictions(args.p12.resolve()),
        "p46": load_required_predictions(args.p46.resolve()),
        "v3": load_required_predictions(args.v3.resolve()),
    }
    target_ids = sorted(required["v3"])
    if any(set(values) != set(target_ids) for values in required.values()):
        raise RuntimeError("required expert prediction IDs do not align")
    canonical_labels = {source_id: required["v3"][source_id][0] for source_id in target_ids}
    for name, values in required.items():
        mismatches = [
            source_id
            for source_id in target_ids
            if values[source_id][0] != canonical_labels[source_id]
        ]
        if mismatches:
            raise RuntimeError(f"{name} label alignment failed: {mismatches[:3]}")
    labels = np.asarray([canonical_labels[source_id] for source_id in target_ids])
    required_correct = {
        name: np.asarray(
            [values[source_id][1] == canonical_labels[source_id] for source_id in target_ids]
        )
        for name, values in required.items()
    }
    initial_solved = np.logical_or.reduce(list(required_correct.values()))
    triple_wrong = {
        source_id for source_id, solved in zip(target_ids, initial_solved) if not solved
    }
    inventory, unique_rows, vectors = scan_saved_experts(
        args.runs.resolve(),
        target_ids,
        canonical_labels,
        sample_to_source,
        triple_wrong,
        {manifest[source_id]["user_id"] for source_id in target_ids},
    )
    all_oracle = greedy_oracle(
        target_ids, labels, initial_solved, unique_rows, vectors, eligible_only=False
    )
    eligible_oracle = greedy_oracle(
        target_ids, labels, initial_solved, unique_rows, vectors, eligible_only=True
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
    quality = quality_rows(target_ids, manifest, args.event_run.resolve())
    numeric = [
        row for row in quality.values() if row.get("event_cache_exists")
    ]
    thresholds = {
        "roi_valid_q25": quantile([row["roi_valid_rate"] for row in numeric], 0.25),
        "roi_quality_q25": quantile([row["roi_quality_mean"] for row in numeric], 0.25),
        "roi_step_q75": quantile([row["roi_center_step_mean"] for row in numeric], 0.75),
        "roi_size_q75": quantile([row["roi_size_cv_mean"] for row in numeric], 0.75),
        "roi_clip_q75": quantile([row["roi_clipped_ratio_mean"] for row in numeric], 0.75),
        "skeleton_q25": quantile(
            [row["skeleton_frame_quality_mean"] for row in numeric], 0.25
        ),
        "skeleton_coverage_q25": quantile(
            [row["skeleton_joint_valid_rate"] for row in numeric], 0.25
        ),
        "imu_q25": quantile([row["imu_points_per_frame"] for row in numeric], 0.25),
        "frames_q10": quantile([row["frames"] for row in numeric], 0.10),
    }
    eligible_candidates = [
        row for row in unique_rows if bool(row.get("strong_for_ceiling", False))
    ]
    triple_rows: list[dict[str, Any]] = []
    for source_id in sorted(triple_wrong):
        index = target_ids.index(source_id)
        rescue_paths = [
            row["path"]
            for row in eligible_candidates
            if vectors[row["prediction_hash"]][index] == labels[index]
        ]
        row = {
            **quality[source_id],
            "p12_prediction": required["p12"][source_id][1],
            "p46_prediction": required["p46"][source_id][1],
            "v3_prediction": required["v3"][source_id][1],
            "historical_expert_rescue_count": len(rescue_paths),
            "historical_expert_rescue_paths": " | ".join(rescue_paths[:20]),
        }
        flags = failure_flags(row, thresholds)
        row["quality_flags"] = ";".join(flags)
        if rescue_paths:
            row["audit_category"] = "existing_expert_rescue"
        elif flags:
            row["audit_category"] = "pipeline_quality_failure"
        else:
            row["audit_category"] = "new_representation_needed"
        triple_rows.append(row)
    triple_rows.sort(
        key=lambda row: (
            {"existing_expert_rescue": 0, "pipeline_quality_failure": 1}.get(
                row["audit_category"], 2
            ),
            -int(row["historical_expert_rescue_count"]),
            int(row["class_id"]),
            str(row["source_id"]),
        )
    )
    category_counts = Counter(row["audit_category"] for row in triple_rows)
    required_metrics = {
        name: {
            **metric_bundle(
                labels,
                np.asarray([values[source_id][1] for source_id in target_ids]),
            ),
            "correct": int(correct.sum()),
        }
        for (name, values), correct in zip(required.items(), required_correct.values())
    }
    summary = {
        "protocol": "P46 frozen validation 70% ceiling audit v1",
        "target": {
            "samples": len(target_ids),
            "accuracy": 0.70,
            "correct_required": math.ceil(0.70 * len(target_ids)),
        },
        "required_experts": required_metrics,
        "required_expert_oracle": {
            "correct": int(initial_solved.sum()),
            "accuracy": float(initial_solved.mean()),
            "triple_wrong": len(triple_wrong),
            "shortfall_to_70": max(math.ceil(0.70 * len(target_ids)) - int(initial_solved.sum()), 0),
        },
        "saved_prediction_inventory": {
            "aligned_files": sum(row["coverage"] == len(target_ids) for row in inventory),
            "unique_prediction_vectors": len(unique_rows),
            "eligible_unique_vectors": sum(
                bool(row["eligible_for_ceiling"]) for row in unique_rows
            ),
            "strong_subject_disjoint_vectors": sum(
                bool(row.get("strong_for_ceiling", False)) for row in unique_rows
            ),
            "all_saved_oracle": all_oracle,
            "eligible_saved_oracle": eligible_oracle,
            "strong_subject_disjoint_oracle": strong_oracle,
            "provenance_warning": (
                "Oracle numbers use validation labels and are diagnostic only. Paths without an "
                "obvious risk token are excluded unless their saved protocol proves that all "
                "target users were held out. Weak experts below 30% accuracy or 25% macro-F1 "
                "are excluded from the strong oracle because chance corrections are not routable."
            ),
        },
        "triple_error_audit": {
            "samples": len(triple_rows),
            "category_counts": dict(category_counts),
            "quality_thresholds": thresholds,
        },
        "sources": {
            "manifest": str(args.manifest.resolve()),
            "event_run": str(args.event_run.resolve()),
            "p12": str(args.p12.resolve()),
            "p46": str(args.p46.resolve()),
            "v3": str(args.v3.resolve()),
        },
    }
    write_csv(output / "saved_prediction_inventory.csv", inventory)
    write_csv(output / "unique_expert_predictions.csv", unique_rows)
    write_csv(output / "triple_error_samples.csv", triple_rows)
    write_csv(output / "triple_error_by_class.csv", aggregate_rows(triple_rows, ("class_id",)))
    write_csv(output / "triple_error_by_user.csv", aggregate_rows(triple_rows, ("user_id",)))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
