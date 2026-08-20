from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.data.thermal_v2_inventory import (
    image_files,
    load_canonical_thermal_records,
    resolve_thermal_trial_path,
)
from src.data.thermal_v2_sampling import normalized_probe_indices
from src.roi.thermal_trial_context import build_trial_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_T0_REPORT = PROJECT_ROOT / "reports" / "thermal_stage0_data_alignment_audit.json"
DEFAULT_WEIGHTS = Path(r"D:\work\2026.7.14_kaggle\40class\yolo11n-pose.pt")
DEFAULT_JSONL = PROJECT_ROOT / "metadata" / "thermal" / "thermal_v2_trial_context.jsonl"
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "metadata" / "thermal" / "thermal_v2_trial_context_summary.json"
)

Detection = tuple[float, float, float, float, float]
Predictor = Callable[[list[Path]], list[list[Detection]]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_record(record: dict[str, Any], *, data_root: Path) -> dict[str, Any]:
    trial_path = resolve_thermal_trial_path(data_root, record)
    return {
        "sample_id": str(record["sample_id"]),
        "class_id": int(record["class_id"]),
        "action_name": str(record["action_name"]),
        "user_id": str(record["user_id"]),
        "trial_id": str(record["trial_id"]),
        "development_split": str(record["development_split"]),
        "duration_bucket": str(record.get("duration_bucket", "unknown")),
        "directory_present": bool(record.get("directory_present", False)),
        "usable": bool(record.get("usable", False)),
        "file_count": int(record.get("file_count", 0)),
        "decodable_frame_count": int(record.get("decodable_frame_count", 0)),
        "duplicate_frame_count": int(record.get("duplicate_frame_count", 0)),
        "thermal_relative_path": str(trial_path.relative_to(data_root)).replace("\\", "/"),
    }


def build_context_records(
    records: Sequence[dict[str, Any]],
    *,
    data_root: Path,
    predictor: Predictor,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    output = [_base_record(record, data_root=data_root) for record in records]
    jobs: list[tuple[int, int, Path]] = []
    files_by_row: dict[int, tuple[Path, ...]] = {}
    frame_size_by_row: dict[int, tuple[int, int]] = {}
    for row_index, (source, target) in enumerate(zip(records, output, strict=True)):
        if not source.get("usable", False):
            target.update(
                {
                    "context_available": False,
                    "probe_indices": [],
                    "accepted_indices": [],
                    "accepted_detections": [],
                    "bbox_xyxy": None,
                    "detection_hit_ratio": 0.0,
                    "median_confidence": 0.0,
                    "bbox_area_ratio": 0.0,
                    "fallback_reason": "thermal_unavailable",
                    "frame_size": None,
                    "probe_detections": [],
                }
            )
            continue
        files = image_files(resolve_thermal_trial_path(data_root, source))
        if not files:
            target.update(
                {
                    "context_available": False,
                    "probe_indices": [],
                    "accepted_indices": [],
                    "accepted_detections": [],
                    "bbox_xyxy": None,
                    "detection_hit_ratio": 0.0,
                    "median_confidence": 0.0,
                    "bbox_area_ratio": 0.0,
                    "fallback_reason": "thermal_files_missing",
                    "frame_size": None,
                    "probe_detections": [],
                }
            )
            continue
        with Image.open(files[0]) as image:
            frame_size_by_row[row_index] = image.size
        files_by_row[row_index] = files
        for frame_index in normalized_probe_indices(len(files)):
            jobs.append((row_index, frame_index, files[frame_index]))

    detections_by_row: dict[int, dict[int, list[Detection]]] = defaultdict(dict)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        predictions = predictor([path for _, _, path in batch])
        if len(predictions) != len(batch):
            raise ValueError("predictor output length does not match probe batch")
        for (row_index, frame_index, _), detections in zip(
            batch, predictions, strict=True
        ):
            detections_by_row[row_index][frame_index] = list(detections)

    for row_index, files in files_by_row.items():
        target = output[row_index]
        detections = detections_by_row[row_index]
        context = build_trial_context(
            sample_id=target["sample_id"],
            frame_size=frame_size_by_row[row_index],
            frame_count=len(files),
            detections_by_index=detections,
        )
        probe_detections: list[dict[str, Any]] = []
        for frame_index in context.probe_indices:
            candidates = detections.get(frame_index, [])
            valid = [
                row
                for row in candidates
                if len(row) == 5
                and row[-1] >= 0.25
                and row[2] > row[0]
                and row[3] > row[1]
            ]
            best = max(valid, key=lambda row: row[-1]) if valid else None
            probe_detections.append(
                {
                    "frame_index": frame_index,
                    "normalized_time": (
                        0.0 if len(files) == 1 else frame_index / (len(files) - 1)
                    ),
                    "candidate_count_at_0_25": len(valid),
                    "best_bbox_xyxy_confidence": list(best) if best else None,
                }
        )
        target.update(context.to_dict())
        target["context_available"] = target.pop("available")
        target["probe_indices"] = list(target["probe_indices"])
        target["accepted_indices"] = list(target["accepted_indices"])
        target["accepted_detections"] = [
            list(row) for row in target["accepted_detections"]
        ]
        if target["bbox_xyxy"] is not None:
            target["bbox_xyxy"] = list(target["bbox_xyxy"])
        target["frame_size"] = list(frame_size_by_row[row_index])
        target["probe_detections"] = probe_detections

    return output


class YoloBoxPredictor:
    def __init__(self, weights: Path, *, device: str, imgsz: int = 640) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz

    def __call__(self, paths: list[Path]) -> list[list[Detection]]:
        images = [
            cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            for path in paths
        ]
        if any(image is None for image in images):
            raise ValueError("probe batch contains an undecodable Thermal image")
        results = self.model.predict(
            images, device=self.device, imgsz=self.imgsz, conf=0.01, verbose=False
        )
        output: list[list[Detection]] = []
        for result in results:
            if result.boxes is None or not len(result.boxes):
                output.append([])
                continue
            boxes = result.boxes.xyxy.detach().float().cpu().numpy()
            confidences = result.boxes.conf.detach().float().cpu().numpy()
            output.append(
                [
                    (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                        float(confidence),
                    )
                    for box, confidence in zip(boxes, confidences, strict=True)
                ]
            )
        return output


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "p25", "median", "p75", "p95", "max", "mean")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _group_summary(records: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return [
        {
            key: value,
            "canonical_trials": len(rows),
            "usable_trials": sum(bool(row["usable"]) for row in rows),
            "context_available_trials": sum(bool(row["context_available"]) for row in rows),
            "context_availability_rate_over_usable": (
                sum(bool(row["context_available"]) for row in rows)
                / max(1, sum(bool(row["usable"]) for row in rows))
            ),
        }
        for value, rows in sorted(groups.items())
    ]


def summarize_context_records(
    records: Sequence[dict[str, Any]], *, weights: Path, t0_report: Path
) -> dict[str, Any]:
    usable = [record for record in records if record["usable"]]
    available = [record for record in usable if record["context_available"]]
    return {
        "schema_version": 1,
        "artifact": "thermal_v2_fixed_trial_context",
        "canonical_trials": len(records),
        "usable_trials": len(usable),
        "context_available_trials": len(available),
        "context_availability_rate_over_usable": len(available) / max(1, len(usable)),
        "fallback_reason_counts": dict(
            sorted(Counter(str(row["fallback_reason"]) for row in records if row["fallback_reason"]).items())
        ),
        "detection_hit_ratio": _quantiles(
            [float(row["detection_hit_ratio"]) for row in usable]
        ),
        "median_confidence": _quantiles(
            [float(row["median_confidence"]) for row in available]
        ),
        "bbox_area_ratio": _quantiles(
            [float(row["bbox_area_ratio"]) for row in available]
        ),
        "by_development_split": _group_summary(records, "development_split"),
        "by_user": _group_summary(records, "user_id"),
        "by_class": _group_summary(records, "class_id"),
        "by_duration_bucket": _group_summary(records, "duration_bucket"),
        "policy": {
            "probe_count": 8,
            "confidence_threshold": 0.25,
            "expansion": 1.4,
            "minimum_side_ratio": 0.35,
            "minimum_hits_singleton": 1,
            "minimum_hits_other": 2,
            "fixed_box_for_complete_trial": True,
            "imports_ir_bbox": False,
            "uses_motion_peak_sampling": False,
        },
        "yolo": {
            "model": "YOLO11n-pose",
            "weights_bytes": weights.stat().st_size,
            "weights_sha256": sha256_file(weights),
            "imgsz": 640,
        },
        "source_t0_report_sha256": sha256_file(t0_report),
        "forbidden_access": {
            "heldout4_labels": False,
            "competition_test": False,
            "quarantined_evidence": False,
            "ir_or_depth_inputs_indices_or_boxes": False,
        },
    }


def write_artifacts(
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    *,
    jsonl_path: Path,
    summary_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True) for record in records]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = dict(summary)
    summary["records_jsonl_sha256"] = sha256_file(jsonl_path)
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    summary["artifact_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed trial-level Thermal YOLO context boxes."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--t0-report", type=Path, default=DEFAULT_T0_REPORT)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    t0_report = args.t0_report.resolve()
    records = load_canonical_thermal_records(t0_report)
    probe_jobs = sum(
        len(normalized_probe_indices(len(image_files(resolve_thermal_trial_path(data_root, record)))))
        for record in records
        if record.get("usable", False)
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "canonical_trials": len(records),
                    "probe_jobs": probe_jobs,
                    "model_loaded": False,
                },
                sort_keys=True,
            )
        )
        return

    weights = args.yolo_weights.resolve()
    predictor = YoloBoxPredictor(weights, device=args.device)
    contexts = build_context_records(
        records,
        data_root=data_root,
        predictor=predictor,
        batch_size=args.batch_size,
    )
    summary = summarize_context_records(contexts, weights=weights, t0_report=t0_report)
    write_artifacts(
        contexts,
        summary,
        jsonl_path=args.jsonl_output.resolve(),
        summary_path=args.summary_output.resolve(),
    )
    print(
        json.dumps(
            {
                "canonical_trials": len(contexts),
                "context_available_trials": summary["context_available_trials"],
                "probe_jobs": probe_jobs,
                "jsonl_output": str(args.jsonl_output.resolve()),
                "summary_output": str(args.summary_output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
