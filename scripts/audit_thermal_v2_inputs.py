from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_CONTEXT = PROJECT_ROOT / "metadata/thermal/thermal_v2_trial_context.jsonl"
DEFAULT_CONTEXT_SUMMARY = PROJECT_ROOT / "metadata/thermal/thermal_v2_trial_context_summary.json"
DEFAULT_NORMALIZATION = PROJECT_ROOT / "metadata/thermal/thermal_v2_train12_normalization.json"
DEFAULT_WEIGHTS = Path(r"D:\work\2026.7.14_kaggle\40class\yolo11n-pose.pt")
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports/thermal_v2_input_audit.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports/thermal_v2_input_audit.md"
DEFAULT_MONTAGE_DIR = PROJECT_ROOT / "reports/thermal_v2_input_montages"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox_iou(
    left: Sequence[float] | None, right: Sequence[float] | None
) -> float | None:
    if left is None or right is None:
        return None
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def build_quality_vector(
    record: dict[str, Any],
    sampled_indices: Sequence[int],
    pose_mask: Sequence[bool],
) -> list[float]:
    decodable = int(record.get("decodable_frame_count", 0))
    file_count = int(record.get("file_count", 0))
    return [
        decodable / max(1, file_count),
        len(set(sampled_indices)) / max(1, len(sampled_indices)),
        float(record.get("detection_hit_ratio", 0.0)),
        float(record.get("median_confidence", 0.0)),
        float(record.get("bbox_area_ratio", 0.0)),
        sum(bool(value) for value in pose_mask) / max(1, len(pose_mask)),
        int(record.get("duplicate_frame_count", 0)) / max(1, decodable),
        min(16 / max(1, decodable), 1.0),
    ]


def _stable_key(record: dict[str, Any]) -> str:
    return hashlib.sha256(str(record["sample_id"]).encode("utf-8")).hexdigest()


def select_representative_records(
    records: Sequence[dict[str, Any]], count: int = 56
) -> list[dict[str, Any]]:
    eligible = [record for record in records if record.get("usable", False)]
    if count < 1 or count > len(eligible):
        raise ValueError("representative count must fit the usable population")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(record: dict[str, Any]) -> None:
        sample_id = str(record["sample_id"])
        if sample_id not in selected_ids and len(selected) < count:
            selected.append(record)
            selected_ids.add(sample_id)

    for class_id in sorted({int(record["class_id"]) for record in eligible}):
        candidates = [record for record in eligible if int(record["class_id"]) == class_id]
        add(
            min(
                candidates,
                key=lambda record: (
                    not bool(record["context_available"]),
                    str(record["user_id"]),
                    _stable_key(record),
                ),
            )
        )
    for record in sorted(
        (record for record in eligible if not record["context_available"]),
        key=_stable_key,
    ):
        add(record)
    for field in ("user_id", "duration_bucket", "development_split"):
        for value in sorted({str(record.get(field, "unknown")) for record in eligible}):
            if not any(str(record.get(field, "unknown")) == value for record in selected):
                candidates = [
                    record
                    for record in eligible
                    if str(record.get(field, "unknown")) == value
                    and record["sample_id"] not in selected_ids
                ]
                if candidates:
                    add(min(candidates, key=_stable_key))
    for record in sorted(eligible, key=_stable_key):
        add(record)
    return selected


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {name: None for name in ("min", "p25", "median", "p75", "p95", "max", "mean")}
    return {
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("context JSONL contains duplicate sample IDs")
    return records


def trial_path(data_root: Path, record: dict[str, Any]) -> Path:
    resolved = (data_root / str(record["thermal_relative_path"])).resolve()
    if data_root.resolve() not in resolved.parents:
        raise ValueError("Thermal trial path escapes the data root")
    return resolved


def decode_rgb(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"undecodable representative frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_view(
    image_rgb: np.ndarray,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    bbox_xyxy: Sequence[int] | None = None,
) -> torch.Tensor:
    import torch

    view = image_rgb
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
        view = view[y1:y2, x1:x2]
    height, width = view.shape[:2]
    scale = 176 / min(height, width)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(view, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    top, left = (resized_height - 160) // 2, (resized_width - 160) // 2
    crop = resized[top : top + 160, left : left + 160]
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - mean[:, None, None]) / std[:, None, None]


class YoloPosePredictor:
    def __init__(self, weights: Path, *, device: str) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device

    def __call__(
        self, paths: list[Path]
    ) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
        import torch

        images = [cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) for path in paths]
        if any(image is None for image in images):
            raise ValueError("pose batch contains an undecodable Thermal image")
        results = self.model.predict(images, device=self.device, imgsz=640, conf=0.01, verbose=False)
        output: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        for result in results:
            if result.boxes is None or not len(result.boxes) or result.keypoints is None:
                output.append((None, None))
                continue
            confidences = result.boxes.conf.detach().float().cpu()
            best = int(torch.argmax(confidences))
            if float(confidences[best]) < 0.25:
                output.append((None, None))
                continue
            xy = result.keypoints.xy[best].detach().float().cpu()
            keypoint_confidence = (
                torch.ones(17, 1)
                if result.keypoints.conf is None
                else result.keypoints.conf[best].detach().float().cpu().reshape(17, 1)
            )
            keypoints = torch.cat((xy, keypoint_confidence), dim=1)
            bbox = torch.cat(
                (
                    result.boxes.xyxy[best].detach().float().cpu(),
                    confidences[best].reshape(1),
                )
            )
            output.append((keypoints, bbox))
        return output


def build_pose_lookup(
    selected: Sequence[dict[str, Any]],
    *,
    data_root: Path,
    predictor: YoloPosePredictor,
    batch_size: int,
) -> tuple[dict[tuple[str, int], tuple[torch.Tensor | None, torch.Tensor | None]], int]:
    from src.data.thermal_v2_inventory import image_files
    from src.data.thermal_v2_sampling import normalized_window_indices

    jobs: list[tuple[str, int, Path]] = []
    for record in selected:
        files = image_files(trial_path(data_root, record))
        indices = {index for window in normalized_window_indices(len(files)) for index in window}
        jobs.extend((str(record["sample_id"]), index, files[index]) for index in sorted(indices))
    lookup: dict[tuple[str, int], tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        predictions = predictor([path for _, _, path in batch])
        for (sample_id, index, _), prediction in zip(batch, predictions, strict=True):
            lookup[(sample_id, index)] = prediction
    return lookup, len(jobs)


def audit_trial_tensors(
    record: dict[str, Any],
    *,
    data_root: Path,
    pose_lookup: dict[tuple[str, int], tuple[torch.Tensor | None, torch.Tensor | None]],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, Any]:
    import torch

    from src.data.thermal_v2_features import (
        encode_pose_step,
        signed_grayscale_differences,
    )
    from src.data.thermal_v2_inventory import image_files
    from src.data.thermal_v2_sampling import normalized_window_indices

    files = image_files(trial_path(data_root, record))
    windows = normalized_window_indices(len(files))
    full_windows, crop_windows, motion_windows = [], [], []
    pose_windows, pose_mask_windows = [], []
    flat_indices: list[int] = []
    flat_pose_mask: list[bool] = []
    for window in windows:
        full_frames, crop_frames, pose_steps, pose_valid = [], [], [], []
        for index in window:
            image = decode_rgb(files[index])
            full_frames.append(preprocess_view(image, mean=mean, std=std))
            crop_frames.append(
                preprocess_view(image, mean=mean, std=std, bbox_xyxy=record["bbox_xyxy"])
                if record["context_available"]
                else torch.zeros(3, 160, 160)
            )
            keypoints, bbox = pose_lookup[(str(record["sample_id"]), index)]
            pose_steps.append(
                encode_pose_step(keypoints, bbox, frame_size=(image.shape[1], image.shape[0]))
            )
            pose_valid.append(keypoints is not None and bbox is not None)
            flat_indices.append(index)
            flat_pose_mask.append(pose_valid[-1])
        full = torch.stack(full_frames)
        full_windows.append(full)
        crop_windows.append(torch.stack(crop_frames))
        motion_windows.append(signed_grayscale_differences(full))
        pose_windows.append(torch.stack(pose_steps))
        pose_mask_windows.append(torch.tensor(pose_valid, dtype=torch.bool))
    tensors = {
        "full_rgb": torch.stack(full_windows),
        "crop_rgb": torch.stack(crop_windows),
        "motion": torch.stack(motion_windows),
        "pose": torch.stack(pose_windows),
        "pose_mask": torch.stack(pose_mask_windows),
        "availability": torch.tensor(
            [True, bool(record["context_available"]), True, any(flat_pose_mask)],
            dtype=torch.bool,
        ),
        "quality": torch.tensor(
            build_quality_vector(record, flat_indices, flat_pose_mask), dtype=torch.float32
        ),
    }
    return {
        "sample_id": record["sample_id"],
        "class_id": int(record["class_id"]),
        "shapes": {name: list(value.shape) for name, value in tensors.items()},
        "all_tensors_finite": all(
            bool(torch.isfinite(value).all())
            for value in tensors.values()
            if value.dtype != torch.bool
        ),
        "availability": tensors["availability"].tolist(),
        "quality": tensors["quality"].tolist(),
        "pose_valid_steps": sum(flat_pose_mask),
        "sampled_steps": len(flat_pose_mask),
    }


def context_continuity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    for record in records:
        boxes = [probe["best_bbox_xyxy_confidence"] for probe in record["probe_detections"]]
        for left, right in zip(boxes, boxes[1:]):
            value = bbox_iou(left, right)
            if value is not None:
                values.append(value)
    return {
        "adjacent_detected_pair_count": len(values),
        "iou": quantiles(values),
        "rate_iou_ge_0_5": sum(value >= 0.5 for value in values) / len(values),
    }


def save_montages(
    records: Sequence[dict[str, Any]], *, data_root: Path, montage_dir: Path
) -> list[str]:
    from src.data.thermal_v2_inventory import image_files
    from src.data.thermal_v2_sampling import normalized_window_indices

    montage_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    for record in records:
        files = image_files(trial_path(data_root, record))
        index = normalized_window_indices(
            len(files), windows=((0.5, 0.5),), frames_per_window=1
        )[0][0]
        image = Image.fromarray(decode_rgb(files[index]))
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        if record["context_available"]:
            draw.rectangle(tuple(record["bbox_xyxy"]), outline=(0, 255, 0), width=3)
            crop = image.crop(tuple(record["bbox_xyxy"])).resize((320, 240))
            route = "fixed_context"
        else:
            draw.rectangle((1, 1, 318, 238), outline=(255, 0, 0), width=3)
            crop, route = image.copy(), "full_fallback"
        panel = Image.new("RGB", (640, 278), "white")
        panel.paste(overlay, (0, 38))
        panel.paste(crop, (320, 38))
        label = (
            f"{record['sample_id']} | {route} | hits={record['detection_hit_ratio']:.2f} "
            f"conf={record['median_confidence']:.2f}"
        )
        ImageDraw.Draw(panel).text((6, 7), label, fill="black", font=font)
        panels.append(panel)
    paths: list[str] = []
    for page_number, start in enumerate(range(0, len(panels), 10), start=1):
        page = Image.new("RGB", (1280, 1390), (224, 224, 224))
        for panel_index, panel in enumerate(panels[start : start + 10]):
            page.paste(panel, ((panel_index % 2) * 640, (panel_index // 2) * 278))
        path = montage_dir / f"thermal_v2_context_{page_number:02d}.jpg"
        page.save(path, quality=90)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return paths


def build_report(
    *,
    records: Sequence[dict[str, Any]],
    context_summary: dict[str, Any],
    normalization: dict[str, Any],
    tensor_rows: Sequence[dict[str, Any]],
    pose_job_count: int,
    montage_paths: Sequence[str],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    from src.data.thermal_v2_features import QUALITY_NAMES

    usable = [record for record in records if record["usable"]]
    probe_rows = [probe for record in usable for probe in record["probe_detections"]]
    detected = [probe for probe in probe_rows if probe["best_bbox_xyxy_confidence"]]
    shapes = tensor_rows[0]["shapes"]
    quality_columns = list(zip(*(row["quality"] for row in tensor_rows), strict=True))
    pose_valid = sum(row["pose_valid_steps"] for row in tensor_rows)
    pose_steps = sum(row["sampled_steps"] for row in tensor_rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "thermal_route_a_a0",
        "status": "pending_human_montage_approval",
        "zero_training": True,
        "canonical_population": {
            "canonical_trials": len(records),
            "directory_present_trials": sum(bool(row["directory_present"]) for row in records),
            "usable_trials": len(usable),
            "retained_unavailable_trials": len(records) - len(usable),
            "by_development_split": context_summary["by_development_split"],
        },
        "normalization": normalization,
        "fixed_context": {
            key: context_summary[key]
            for key in (
                "context_available_trials",
                "context_availability_rate_over_usable",
                "fallback_reason_counts",
                "detection_hit_ratio",
                "median_confidence",
                "bbox_area_ratio",
                "policy",
                "by_user",
                "by_class",
                "by_duration_bucket",
            )
        },
        "manual_montage_review": {
            "required": True,
            "approved": False,
            "reviewer": None,
            "reviewed_at": None,
        },
        "artifacts": artifacts,
        "sampling_contract": {
            "timeline": "thermal_native_normalized_time",
            "windows": [[0.0, 0.5], [0.25, 0.75], [0.5, 1.0]],
            "frames_per_window": 16,
            "motion_peak_sampling": False,
            "imports_ir_indices": False,
            "singleton_and_short_trials": "nearest_index_repeats_with_explicit_uniqueness",
        },
        "evidence_boundary": {
            "heldout4_labels_read": False,
            "competition_test_read": False,
            "quarantined_evidence_read": False,
            "ir_or_depth_inputs_read": False,
            "frozen_ir_x3d_modified": False,
        },
        "decision": "a0_implementation_complete_pending_human_montage_approval",
    }
    report["fixed_context"].update(
        {
            "probe_frame_count": len(probe_rows),
            "probe_frame_detection_rate": len(detected) / len(probe_rows),
            "bbox_continuity": context_continuity(usable),
        }
    )
    report["representative_tensor_audit"] = {
        "trial_count": len(tensor_rows),
        "class_count": len({row["class_id"] for row in tensor_rows}),
        "pose_unique_frame_jobs": pose_job_count,
        "pose_valid_step_rate": pose_valid / pose_steps,
        "all_tensors_finite": all(row["all_tensors_finite"] for row in tensor_rows),
        "shapes": shapes,
        "shape_consistent": all(row["shapes"] == shapes for row in tensor_rows),
        "availability_counts": {
            name: sum(row["availability"][index] for row in tensor_rows)
            for index, name in enumerate(("full", "crop", "motion", "pose"))
        },
        "quality": {
            name: quantiles([float(value) for value in column])
            for name, column in zip(QUALITY_NAMES, quality_columns, strict=True)
        },
        "sample_ids": [row["sample_id"] for row in tensor_rows],
        "montage_paths": list(montage_paths),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["artifact_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    population, context = report["canonical_population"], report["fixed_context"]
    tensors, norm = report["representative_tensor_audit"], report["normalization"]
    lines = [
        "# Thermal Route A A0 Input Audit",
        "",
        "**Status:** `pending_human_montage_approval`",
        "",
        "No model was trained. Heldout labels, competition test, quarantined evidence, IR/Depth inputs, and frozen IR/X3D evidence were not read or modified.",
        "",
        "## Canonical population",
        "",
        f"- Canonical: **{population['canonical_trials']}**; usable: **{population['usable_trials']}**; unavailable retained: **{population['retained_unavailable_trials']}**.",
        f"- Train12 normalization: **{norm['sampled_usable_trials']} trials / {norm['decoded_sample_frames']} frames**.",
        f"- RGB mean `{norm['rgb_mean']}`; std `{norm['rgb_std']}`.",
        "",
        "## Fixed Thermal context",
        "",
        f"- Available: **{context['context_available_trials']}/{population['usable_trials']} ({context['context_availability_rate_over_usable']:.1%})**.",
        f"- Fallbacks: `{json.dumps(context['fallback_reason_counts'], sort_keys=True)}`.",
        f"- Probe detection: **{context['probe_frame_detection_rate']:.1%}** over {context['probe_frame_count']} frames.",
        f"- Median confidence `{context['median_confidence']['median']:.3f}`; crop area `{context['bbox_area_ratio']['median']:.3f}`; adjacent IoU `{context['bbox_continuity']['iou']['median']:.3f}`.",
        "",
        "One square union box is fixed for each Thermal trial. No IR bbox or motion-peak frame selection is used; every failure remains a full-frame fallback.",
        "",
        "## Representative tensors",
        "",
        f"- **{tensors['trial_count']} trials / {tensors['class_count']} classes / {tensors['pose_unique_frame_jobs']} unique pose frames**.",
        f"- Pose valid-step rate: **{tensors['pose_valid_step_rate']:.1%}**.",
        f"- Shapes: `{json.dumps(tensors['shapes'], sort_keys=True)}`.",
        f"- Finite and shape-consistent: `{tensors['all_tensors_finite'] and tensors['shape_consistent']}`.",
        "",
        "Montage pages:",
        "",
        *[f"- `{value}`" for value in tensors["montage_paths"]],
        "",
        "## Stop",
        "",
        "Machine audit is complete, but A0 is not approved. A human must inspect every montage page before the workflow may advance to A1.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Thermal v2 inputs without training.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--context-summary", type=Path, default=DEFAULT_CONTEXT_SUMMARY)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--montage-dir", type=Path, default=DEFAULT_MONTAGE_DIR)
    parser.add_argument("--representative-trials", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    data_root, weights = args.data_root.resolve(), args.yolo_weights.resolve()
    context_path, summary_path = args.context.resolve(), args.context_summary.resolve()
    normalization_path = args.normalization.resolve()
    records = load_jsonl(context_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    selected = select_representative_records(records, args.representative_trials)
    pose_lookup, pose_jobs = build_pose_lookup(
        selected,
        data_root=data_root,
        predictor=YoloPosePredictor(weights, device=args.device),
        batch_size=args.batch_size,
    )
    mean = torch.tensor(normalization["rgb_mean"], dtype=torch.float32)
    std = torch.tensor(normalization["rgb_std"], dtype=torch.float32)
    tensor_rows = [
        audit_trial_tensors(
            record,
            data_root=data_root,
            pose_lookup=pose_lookup,
            mean=mean,
            std=std,
        )
        for record in selected
    ]
    montage_paths = save_montages(
        selected, data_root=data_root, montage_dir=args.montage_dir.resolve()
    )
    artifacts = {
        "context_jsonl": str(context_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "context_jsonl_sha256": sha256_file(context_path),
        "context_summary_sha256": sha256_file(summary_path),
        "normalization_sha256": sha256_file(normalization_path),
        "yolo_weights_sha256": sha256_file(weights),
        "yolo_weights_bytes": weights.stat().st_size,
        "new_learned_weights_bytes": 0,
    }
    report = build_report(
        records=records,
        context_summary=summary,
        normalization=normalization,
        tensor_rows=tensor_rows,
        pose_job_count=pose_jobs,
        montage_paths=montage_paths,
        artifacts=artifacts,
    )
    json_output, markdown_output = args.json_output.resolve(), args.markdown_output.resolve()
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, markdown_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "representative_trials": len(selected),
                "pose_unique_frame_jobs": pose_jobs,
                "montage_pages": len(montage_paths),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
