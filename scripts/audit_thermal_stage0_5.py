from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "c42bb43091c79903e5fde5655c2846c87305895a"
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_T0_REPORT = PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "reports/thermal_stage0_5_localization_route_audit.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "reports/thermal_stage0_5_localization_route_audit.md"
DEFAULT_MONTAGE_DIR = PROJECT_ROOT / "reports/thermal_stage0_5_montages"


MANUAL_REVIEW_OVERRIDES: dict[str, dict[str, Any]] = {
    "train__c00__user6__1-1-1": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["near_field_edge_clipping", "low_confidence_gap"],
    },
    "train__c02__user5__7-1-2": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["far_field_small_subject", "tight_crop_context_loss"],
    },
    "train__c04__user8__1-1-2": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["near_field_edge_clipping", "tight_crop_context_loss"],
    },
    "train__c05__user9__1-1-3": {
        "verdict": "mixed_with_background_false_candidate",
        "failure_types": ["subject_exits_frame", "background_hot_object_candidate"],
    },
    "train__c09__user1__2-1-3": {
        "verdict": "mixed_with_localization_collapse",
        "failure_types": ["hand_or_object_only_candidate", "low_confidence_gap"],
    },
    "train__c09__user20__6-3-2": {
        "verdict": "mixed_with_background_false_candidate",
        "failure_types": ["sofa_or_table_candidate", "low_confidence_gap"],
    },
    "train__c15__user16__6-2-1": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["recurrent_low_confidence", "partial_body_crop"],
    },
    "train__c16__user18__2-1-1": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["far_field_small_subject", "recurrent_low_confidence"],
    },
    "train__c18__user18__4-1-2": {
        "verdict": "mixed_with_localization_collapse",
        "failure_types": ["head_only_candidate", "extremely_short_trial"],
    },
    "train__c21__user6__4-1-3": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["oversized_box_includes_furniture", "tight_crop_instability"],
    },
    "train__c22__user18__4-1-3": {
        "verdict": "mixed_with_background_false_candidate",
        "failure_types": ["wrong_small_background_candidate", "low_confidence_gap"],
    },
    "train__c26__user16__5-3-1": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["far_field_small_subject", "tight_crop_context_loss"],
    },
    "train__c29__user1__3-2-1": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["large_pose_change", "edge_clipping"],
    },
    "train__c33__user16__6-1-1": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["standing_to_horizontal_pose_change", "low_confidence_gap"],
    },
    "train__c36__user7__6-1-1": {
        "verdict": "correct_subject_with_context_risk",
        "failure_types": ["near_field_edge_clipping"],
    },
}

GROSS_MOTION_CLASS_IDS = set(range(28, 37))
SEATED_OR_LYING_CLASS_IDS = {
    2,
    6,
    7,
    8,
    9,
    10,
    11,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    33,
    34,
}


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
        if part
    )


def _image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )


def find_duplicate_frame_groups(trial_path: Path) -> list[list[str]]:
    """Return exact decoded-image duplicate groups without altering source data."""
    by_hash: dict[str, list[str]] = {}
    for path in _image_files(trial_path):
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if len(encoded) else None
        if image is None:
            continue
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        by_hash.setdefault(digest, []).append(path.name)
    return [names for names in by_hash.values() if len(names) > 1]


def classify_count_relation(
    thermal_count: int, ir_count: int, depth_count: int
) -> str:
    """Classify frame-count asymmetry without asserting an unavailable root cause."""
    counts = (int(thermal_count), int(ir_count), int(depth_count))
    if max(counts) <= 8 and min(counts) > 0:
        return "short_capture_across_modalities"
    reference_max = max(ir_count, depth_count)
    if 0 < thermal_count <= 4 and reference_max >= 20:
        return "thermal_capture_truncated_or_export_incomplete"
    if thermal_count >= 20 and 0 < reference_max <= 4:
        return "ir_depth_capture_truncated_or_export_incomplete"
    available_reference = [value for value in (ir_count, depth_count) if value > 0]
    if thermal_count > 0 and available_reference:
        reference = float(np.median(available_reference))
        ratio = thermal_count / reference
        if 1.5 <= ratio <= 4.0:
            return "plausible_asynchronous_rate_difference"
        return "asynchronous_or_partial_capture_needs_metadata"
    return "missing_modality_prevents_count_relation_assessment"


def expand_context_bbox(
    bbox_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    expansion_fraction: float = 0.25,
) -> list[float]:
    """Expand a Thermal-native box on each side and clip it to Thermal bounds."""
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not (x2 > x1 and y2 > y1):
        raise ValueError("bbox_xyxy must have positive area")
    width = x2 - x1
    height = y2 - y1
    return [
        max(0.0, x1 - expansion_fraction * width),
        max(0.0, y1 - expansion_fraction * height),
        min(float(image_width), x2 + expansion_fraction * width),
        min(float(image_height), y2 + expansion_fraction * height),
    ]


def _bbox_area_ratio(
    bbox_xyxy: Sequence[float], image_width: int, image_height: int
) -> float:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_width / image_height)


def build_heat_motion_context(frames_bgr: Sequence[np.ndarray]) -> dict[str, Any]:
    """Build one trial-level context proposal from uniform Thermal samples."""
    if not frames_bgr:
        return {
            "valid": False,
            "invalid_reason": "no_decodable_frames",
            "bbox_xyxy": None,
            "component_area_ratio": None,
            "bbox_area_ratio": None,
        }
    height, width = frames_bgr[0].shape[:2]
    if any(frame.ndim != 3 or frame.shape[:2] != (height, width) for frame in frames_bgr):
        raise ValueError("heat/motion frames must share HxWxC dimensions")
    luminance = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_bgr]
    )
    heat_union = np.zeros((height, width), dtype=np.uint8)
    for frame in luminance:
        threshold = float(np.quantile(frame, 0.75))
        comparison = frame >= threshold if float(np.max(frame)) == threshold else frame > threshold
        heat_union |= comparison.astype(np.uint8)
    motion_union = np.zeros((height, width), dtype=np.uint8)
    for left, right in zip(luminance, luminance[1:]):
        difference = cv2.absdiff(left, right)
        threshold = float(np.quantile(difference, 0.75))
        if threshold > 0:
            motion_union |= (difference >= threshold).astype(np.uint8)
    combined = ((heat_union | motion_union) * 255).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )
    if component_count <= 1:
        return {
            "valid": False,
            "invalid_reason": "no_connected_component",
            "bbox_xyxy": None,
            "component_area_ratio": 0.0,
            "bbox_area_ratio": None,
        }
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    component_area_ratio = float(area / width / height)
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    box_width = int(stats[component, cv2.CC_STAT_WIDTH])
    box_height = int(stats[component, cv2.CC_STAT_HEIGHT])
    raw_bbox = [float(x), float(y), float(x + box_width), float(y + box_height)]
    bbox = expand_context_bbox(raw_bbox, width, height, expansion_fraction=0.25)
    bbox_area_ratio = _bbox_area_ratio(bbox, width, height)
    invalid_reason = None
    if component_area_ratio < 0.05:
        invalid_reason = "component_area_below_0_05"
    elif component_area_ratio > 0.95:
        invalid_reason = "component_area_above_0_95"
    return {
        "valid": invalid_reason is None,
        "invalid_reason": invalid_reason,
        "bbox_xyxy": bbox,
        "raw_bbox_xyxy": raw_bbox,
        "component_area_ratio": component_area_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "uniform_frame_count": len(frames_bgr),
        "uses_motion_peak_sampling": False,
    }


def choose_localization_route(
    yolo: dict[str, Any], heat_motion: dict[str, Any]
) -> dict[str, Any]:
    """Choose the frozen Thermal-native route and preserve fallback causes."""
    confidence = yolo.get("confidence")
    bbox = yolo.get("bbox_xyxy")
    area_ratio = yolo.get("bbox_area_ratio")
    yolo_reason = None
    if confidence is None:
        yolo_reason = "yolo_no_candidate"
    elif float(confidence) < 0.25:
        yolo_reason = "yolo_confidence_below_0_25"
    elif bbox is None:
        yolo_reason = "yolo_missing_bbox"
    elif area_ratio is None or float(area_ratio) < 0.05:
        yolo_reason = "yolo_bbox_area_below_0_05"
    if yolo_reason is None:
        return {
            "route": "thermal_yolo_context",
            "bbox_xyxy": bbox,
            "fallback_reason": None,
        }
    if bool(heat_motion.get("valid")):
        return {
            "route": "thermal_heat_motion_context",
            "bbox_xyxy": heat_motion.get("bbox_xyxy"),
            "fallback_reason": yolo_reason,
        }
    heat_reason = str(heat_motion.get("invalid_reason") or "invalid")
    return {
        "route": "full_frame",
        "bbox_xyxy": None,
        "fallback_reason": f"{yolo_reason};heat_motion_{heat_reason}",
    }


def assess_rendered_auto_scale_signals(
    frames_bgr: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Measure rendered RGB drift without claiming radiometric calibration."""
    if not frames_bgr:
        return {
            "evidence_strength": "unavailable",
            "absolute_temperature_scale_proven": False,
            "reason": "no_decodable_frames",
        }
    if len(frames_bgr) < 2:
        return {
            "evidence_strength": "unavailable",
            "absolute_temperature_scale_proven": False,
            "reason": "fewer_than_two_uniform_time_frames",
        }
    height, width = frames_bgr[0].shape[:2]
    if any(frame.ndim != 3 or frame.shape[:2] != (height, width) for frame in frames_bgr):
        raise ValueError("auto-scale frames must share HxWxC dimensions")
    luminance = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) for frame in frames_bgr]
    )
    temporal_std = np.std(luminance, axis=0)
    static_cutoff = float(np.quantile(temporal_std, 0.25))
    static_mask = temporal_std <= static_cutoff
    if int(static_mask.sum()) < 32:
        static_mask = np.ones((height, width), dtype=bool)
    static_medians = np.asarray(
        [float(np.median(frame[static_mask])) for frame in luminance], dtype=np.float64
    )
    hues = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float32) for frame in frames_bgr]
    )
    saturations = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1] for frame in frames_bgr]
    )
    static_hue_medians = []
    for hue, saturation in zip(hues, saturations, strict=True):
        valid_hue = static_mask & (saturation >= 20)
        static_hue_medians.append(
            float(np.median(hue[valid_hue])) if int(valid_hue.sum()) >= 32 else np.nan
        )
    percentile_rows = np.asarray(
        [np.quantile(frame, [0.01, 0.05, 0.50, 0.95, 0.99]) for frame in luminance],
        dtype=np.float64,
    )
    percentile_span = np.ptp(percentile_rows, axis=0)
    endpoint_low_fraction = np.mean(luminance <= 5.0, axis=(1, 2))
    endpoint_high_fraction = np.mean(luminance >= 250.0, axis=(1, 2))
    static_median_span = float(np.ptp(static_medians))
    finite_hues = np.asarray(
        [value for value in static_hue_medians if np.isfinite(value)], dtype=np.float64
    )
    if len(finite_hues):
        ordered_hues = np.sort(finite_hues % 180.0)
        circular_gaps = np.diff(
            np.concatenate([ordered_hues, ordered_hues[:1] + 180.0])
        )
        static_hue_span = float(180.0 - np.max(circular_gaps))
    else:
        static_hue_span = None
    median_quantile_span = float(percentile_span[2])
    endpoint_fraction_span = float(
        max(np.ptp(endpoint_low_fraction), np.ptp(endpoint_high_fraction))
    )
    score = 0
    if static_median_span >= 20.0:
        score += 1
    if median_quantile_span >= 25.0:
        score += 1
    if endpoint_fraction_span >= 0.10:
        score += 1
    if static_hue_span is not None and static_hue_span >= 10.0:
        score += 1
    evidence_strength = "weak" if score == 0 else "moderate" if score == 1 else "strong"
    return {
        "evidence_strength": evidence_strength,
        "absolute_temperature_scale_proven": False,
        "static_background_pixel_fraction": float(np.mean(static_mask)),
        "static_background_luminance_median_span": static_median_span,
        "static_background_hue_median_span": static_hue_span,
        "global_luminance_percentile_span": {
            name: float(value)
            for name, value in zip(("p01", "p05", "p50", "p95", "p99"), percentile_span)
        },
        "endpoint_fraction_span": endpoint_fraction_span,
        "interpretation": "rendered_auto_scale_indication_only_without_raw_temperature_or_scale_metadata",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Thermal Stage T0.5 localization-route audit"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--t0-report", type=Path, default=DEFAULT_T0_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--montage-dir", type=Path, default=DEFAULT_MONTAGE_DIR)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: Sequence[float | int | None]) -> dict[str, float | None]:
    finite = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if not len(finite):
        return {
            key: None
            for key in ("min", "p25", "median", "p75", "p95", "max", "mean")
        }
    return {
        "min": float(np.min(finite)),
        "p25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _decode(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if len(encoded) else None
    if image is None:
        raise RuntimeError(f"Previously audited frame became undecodable: {path}")
    return image


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    x1, y1 = max(lx1, rx1), max(ly1, ry1)
    x2, y2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        - intersection
    )
    return float(intersection / union) if union else 0.0


def _reference_box_coverage(
    reference: Sequence[float], candidate: Sequence[float]
) -> float:
    rx1, ry1, rx2, ry2 = (float(value) for value in reference)
    cx1, cy1, cx2, cy2 = (float(value) for value in candidate)
    intersection = max(0.0, min(rx2, cx2) - max(rx1, cx1)) * max(
        0.0, min(ry2, cy2) - max(ry1, cy1)
    )
    reference_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return float(intersection / reference_area) if reference_area else 0.0


def _ordered_groups(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["sample_id"], []).append(dict(record))
    return [
        sorted(group, key=lambda row: row["frame_order"]) for group in groups.values()
    ]


def _manual_reviews(
    frame_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    for index, group in enumerate(_ordered_groups(frame_records)):
        first = group[0]
        detected_areas = [
            row["bbox_area_ratio"]
            for row in group
            if row.get("detected") and row.get("bbox_area_ratio") is not None
        ]
        factors: list[str] = []
        if any("edge_clipped_bbox" in row.get("failure_types", []) for row in group):
            factors.append("near_field_or_edge_contact")
        if detected_areas and float(np.median(detected_areas)) < 0.12:
            factors.append("far_field_small_subject")
        if int(first["class_id"]) in SEATED_OR_LYING_CLASS_IDS:
            factors.append("seated_or_lying_action_context")
        if int(first["class_id"]) in GROSS_MOTION_CLASS_IDS:
            factors.append("gross_motion_action_context")
        if first["duration_bucket"] in {"single_frame", "2_to_8"}:
            factors.append("single_or_extremely_short")
        override = MANUAL_REVIEW_OVERRIDES.get(first["sample_id"], {})
        reviews.append(
            {
                "sample_id": first["sample_id"],
                "class_id": first["class_id"],
                "action_name": first["action_name"],
                "user_id": first["user_id"],
                "duration_bucket": first["duration_bucket"],
                "montage_page": index // 10 + 1,
                "reviewed_frames": len(group),
                "verdict": override.get(
                    "verdict", "correct_subject_localization_on_reviewed_frames"
                ),
                "failure_types": override.get("failure_types", []),
                "factors": factors,
                "accepted_detection_rate": float(
                    np.mean([bool(row.get("detected")) for row in group])
                ),
            }
        )
    factor_summary: list[dict[str, Any]] = []
    for factor in (
        "near_field_or_edge_contact",
        "far_field_small_subject",
        "seated_or_lying_action_context",
        "gross_motion_action_context",
        "single_or_extremely_short",
    ):
        selected_ids = {
            row["sample_id"] for row in reviews if factor in row["factors"]
        }
        selected_frames = [
            row for row in frame_records if row["sample_id"] in selected_ids
        ]
        factor_summary.append(
            {
                "factor": factor,
                "trials": len(selected_ids),
                "frames": len(selected_frames),
                "detection_rate": (
                    float(np.mean([bool(row.get("detected")) for row in selected_frames]))
                    if selected_frames
                    else None
                ),
            }
        )
    failure_counts = Counter(
        failure for review in reviews for failure in review["failure_types"]
    )
    verdict_counts = Counter(review["verdict"] for review in reviews)
    return {
        "reviewer_type": "codex_structured_visual_inspection_not_independent_human_signoff",
        "factor_definitions": {
            "near_field_or_edge_contact": "at least one T0 accepted box touched a Thermal image edge",
            "far_field_small_subject": "median accepted raw YOLO box area below 12 percent",
            "seated_or_lying_action_context": "action-context reporting stratum only; never a route input",
            "gross_motion_action_context": "gross-motion action reporting stratum only; never a sampling or route input",
            "single_or_extremely_short": "T0 duration bucket single_frame or 2_to_8",
        },
        "pages_reviewed": [
            f"reports/thermal_stage0_montages/thermal_yolo_pose_montage_{index:02d}.jpg"
            for index in range(1, 7)
        ],
        "reviewed_trials": len(reviews),
        "reviewed_frames": len(frame_records),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "factor_summary": factor_summary,
        "trials": reviews,
        "decision": "yolo_is_quality_bearing_conditional_locator_not_a_mandatory_crop",
    }


def _crop_or_placeholder(
    frame_bgr: np.ndarray,
    bbox: Sequence[float] | None,
    label: str,
    cell_size: tuple[int, int] = (320, 245),
) -> Image.Image:
    cell_width, cell_height = cell_size
    canvas = Image.new("RGB", cell_size, "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, cell_width, 25), fill="black")
    draw.text((5, 6), label, fill="white")
    if bbox is None:
        draw.text((95, 115), "unavailable", fill="gray")
        return canvas
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        draw.text((95, 115), "invalid crop", fill="red")
        return canvas
    crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop)
    image.thumbnail((cell_width, cell_height - 25))
    offset = ((cell_width - image.width) // 2, 25 + (cell_height - 25 - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def _save_route_montages(
    trial_payloads: Sequence[dict[str, Any]], montage_dir: Path
) -> list[str]:
    montage_dir.mkdir(parents=True, exist_ok=True)
    output: list[str] = []
    cell_width, cell_height = 320, 245
    row_height = 285
    for page_start in range(0, len(trial_payloads), 10):
        page = trial_payloads[page_start : page_start + 10]
        canvas = Image.new("RGB", (4 * cell_width, len(page) * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, payload in enumerate(page):
            middle = payload["frames"][len(payload["frames"]) // 2]
            frame = middle["image_bgr"]
            height, width = frame.shape[:2]
            full_bbox = [0.0, 0.0, float(width), float(height)]
            views = [
                (full_bbox, "full_frame"),
                (middle["yolo_context_bbox"], "thermal_yolo_context"),
                (payload["heat_motion"]["bbox_xyxy"], "thermal_heat_motion_context"),
                (middle["selected_bbox"], f"selected: {middle['route']}")
            ]
            draw.text(
                (5, row_index * row_height + 3),
                f"{payload['sample_id']} | t={middle['normalized_time']:.2f}",
                fill="black",
            )
            for column, (bbox, label) in enumerate(views):
                panel = _crop_or_placeholder(frame, bbox, label)
                canvas.paste(panel, (column * cell_width, row_index * row_height + 35))
        page_number = page_start // 10 + 1
        path = montage_dir / f"thermal_route_comparison_montage_{page_number:02d}.jpg"
        canvas.save(path, quality=90)
        output.append(str(path.relative_to(PROJECT_ROOT).as_posix()))
    return output


def _route_audit(
    registry_by_id: dict[str, dict[str, Any]],
    frame_records: Sequence[dict[str, Any]],
    montage_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trial_payloads: list[dict[str, Any]] = []
    scale_records: list[dict[str, Any]] = []
    serializable_frames: list[dict[str, Any]] = []
    yolo_continuity: list[float] = []
    final_continuity: list[float] = []
    heat_coverages: list[float] = []
    route_counts: Counter[str] = Counter()
    for group in _ordered_groups(frame_records):
        sample_id = group[0]["sample_id"]
        registry = registry_by_id[sample_id]
        files = _image_files(Path(registry["paths"]["Thermal"]))
        frames: list[np.ndarray] = []
        for row in group:
            index = int(row["frame_index"])
            if index >= len(files):
                raise ValueError(f"T0 frame index no longer exists for {sample_id}")
            frames.append(_decode(files[index]))
        heat_motion = build_heat_motion_context(frames)
        scale_record = assess_rendered_auto_scale_signals(frames)
        scale_records.append({"sample_id": sample_id, **scale_record})
        payload_frames: list[dict[str, Any]] = []
        previous_yolo: Sequence[float] | None = None
        previous_final: Sequence[float] | None = None
        for row, image in zip(group, frames, strict=True):
            height, width = image.shape[:2]
            yolo_context_bbox = None
            raw_bbox = row.get("bbox_xyxy")
            if raw_bbox is not None:
                yolo_context_bbox = expand_context_bbox(raw_bbox, width, height)
            yolo_candidate = {
                "confidence": row.get("confidence"),
                "bbox_area_ratio": row.get("bbox_area_ratio"),
                "bbox_xyxy": yolo_context_bbox,
            }
            route = choose_localization_route(yolo_candidate, heat_motion)
            selected_bbox = route["bbox_xyxy"]
            if route["route"] == "full_frame":
                selected_bbox = [0.0, 0.0, float(width), float(height)]
            route_counts[route["route"]] += 1
            if route["route"] == "thermal_yolo_context":
                if previous_yolo is not None:
                    yolo_continuity.append(_bbox_iou(previous_yolo, selected_bbox))
                previous_yolo = selected_bbox
            else:
                previous_yolo = None
            if previous_final is not None:
                final_continuity.append(_bbox_iou(previous_final, selected_bbox))
            previous_final = selected_bbox
            heat_coverage = None
            if row.get("detected") and raw_bbox is not None and heat_motion.get("valid"):
                heat_coverage = _reference_box_coverage(
                    raw_bbox, heat_motion["bbox_xyxy"]
                )
                heat_coverages.append(heat_coverage)
            selected_area_ratio = _bbox_area_ratio(selected_bbox, width, height)
            serializable_frames.append(
                {
                    "sample_id": sample_id,
                    "class_id": row["class_id"],
                    "user_id": row["user_id"],
                    "duration_bucket": row["duration_bucket"],
                    "frame_index": row["frame_index"],
                    "normalized_time": row["normalized_time"],
                    "yolo_confidence": row.get("confidence"),
                    "yolo_raw_bbox_area_ratio": row.get("bbox_area_ratio"),
                    "yolo_context_bbox_xyxy": yolo_context_bbox,
                    "heat_motion_context_bbox_xyxy": heat_motion.get("bbox_xyxy"),
                    "heat_motion_yolo_box_coverage_proxy": heat_coverage,
                    "route": route["route"],
                    "selected_bbox_area_ratio": selected_area_ratio,
                    "fallback_reason": route["fallback_reason"],
                }
            )
            payload_frames.append(
                {
                    **row,
                    "image_bgr": image,
                    "yolo_context_bbox": yolo_context_bbox
                    if route["route"] == "thermal_yolo_context"
                    else None,
                    "route": route["route"],
                    "selected_bbox": selected_bbox,
                }
            )
        trial_payloads.append(
            {
                "sample_id": sample_id,
                "heat_motion": heat_motion,
                "frames": payload_frames,
            }
        )
    montage_paths = _save_route_montages(trial_payloads, montage_dir)
    heat_trials = [payload["heat_motion"] for payload in trial_payloads]
    heat_selective = [
        bool(row["valid"]) and float(row.get("bbox_area_ratio") or 1.0) < 0.95
        for row in heat_trials
    ]
    heat_full_frame_equivalent = [
        row.get("bbox_area_ratio") is not None
        and float(row["bbox_area_ratio"]) >= 0.95
        for row in heat_trials
    ]
    auto_scale_counts = Counter(row["evidence_strength"] for row in scale_records)
    total_frames = len(serializable_frames)
    return (
        {
            "contract": {
                "timeline": "Thermal-native natural order and normalized time only",
                "uniform_timepoints_reused_from_t0": True,
                "motion_peak_sampling": False,
                "yolo_acceptance": "confidence>=0.25 and raw bbox area ratio>=0.05",
                "yolo_context_expansion": "25 percent on each side clipped to Thermal bounds",
                "heat_motion": "per-frame luminance top quartile plus adjacent uniform-frame difference p75; temporal union; 5x5 close; largest component; 25 percent expansion",
                "full_frame_always_valid": True,
                "ir_indices_or_bboxes_used": False,
            },
            "representative_trials": len(trial_payloads),
            "representative_frames": total_frames,
            "route_counts": dict(sorted(route_counts.items())),
            "route_rates": {
                key: value / total_frames for key, value in sorted(route_counts.items())
            },
            "fallback_rate_from_yolo": 1.0
            - route_counts.get("thermal_yolo_context", 0) / total_frames,
            "heat_motion_valid_trial_rate": float(
                np.mean([bool(row["valid"]) for row in heat_trials])
            ),
            "heat_motion_selective_trial_rate": float(np.mean(heat_selective)),
            "heat_motion_full_frame_equivalent_trial_rate": float(
                np.mean(heat_full_frame_equivalent)
            ),
            "heat_motion_component_area_ratio": _quantiles(
                [row.get("component_area_ratio") for row in heat_trials]
            ),
            "heat_motion_bbox_area_ratio": _quantiles(
                [row.get("bbox_area_ratio") for row in heat_trials]
            ),
            "heat_motion_yolo_box_coverage_proxy": _quantiles(heat_coverages),
            "yolo_context_continuity_iou": _quantiles(yolo_continuity),
            "final_route_continuity_iou": _quantiles(final_continuity),
            "selected_bbox_area_ratio": _quantiles(
                [row["selected_bbox_area_ratio"] for row in serializable_frames]
            ),
            "montage_paths": montage_paths,
            "frame_records": serializable_frames,
            "human_ground_truth_claimed": False,
            "coverage_proxy_note": "Heat/motion coverage is measured against accepted YOLO boxes for diagnosis only; YOLO is not ground truth.",
            "auto_scale_signal_summary": {
                "trial_counts": dict(sorted(auto_scale_counts.items())),
                "static_background_luminance_median_span": _quantiles(
                    [row.get("static_background_luminance_median_span") for row in scale_records]
                ),
                "static_background_hue_median_span": _quantiles(
                    [row.get("static_background_hue_median_span") for row in scale_records]
                ),
                "endpoint_fraction_span": _quantiles(
                    [row.get("endpoint_fraction_span") for row in scale_records]
                ),
                "absolute_temperature_scale_proven": False,
                "trials": scale_records,
            },
        },
        trial_payloads,
    )


def _save_short_trial_montages(
    records: Sequence[dict[str, Any]],
    registry_by_id: dict[str, dict[str, Any]],
    montage_dir: Path,
) -> list[str]:
    output: list[str] = []
    cell_width, cell_height = 320, 220
    per_page = 20
    for page_start in range(0, len(records), per_page):
        page = records[page_start : page_start + per_page]
        canvas = Image.new("RGB", (4 * cell_width, 5 * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, record in enumerate(page):
            row, column = divmod(offset, 4)
            files = _image_files(
                Path(registry_by_id[record["sample_id"]]["paths"]["Thermal"])
            )
            selected = sorted(set(np.rint(np.linspace(0, len(files) - 1, min(3, len(files)))).astype(int)))
            panel = Image.new("RGB", (cell_width, cell_height), "white")
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.rectangle((0, 0, cell_width, 38), fill="black")
            panel_draw.text((4, 4), record["sample_id"], fill="white")
            panel_draw.text(
                (4, 20),
                f"T/IR/D={record['thermal_frame_count']}/{record['ir_frame_count']}/{record['depth_frame_count']}",
                fill="white",
            )
            tile_width = cell_width // max(1, len(selected))
            for tile_index, frame_index in enumerate(selected):
                image = Image.fromarray(
                    cv2.cvtColor(_decode(files[int(frame_index)]), cv2.COLOR_BGR2RGB)
                )
                image.thumbnail((tile_width, cell_height - 38))
                x = tile_index * tile_width + (tile_width - image.width) // 2
                y = 38 + (cell_height - 38 - image.height) // 2
                panel.paste(image, (x, y))
            canvas.paste(panel, (column * cell_width, row * cell_height))
        page_number = page_start // per_page + 1
        path = montage_dir / f"thermal_short_trial_montage_{page_number:02d}.jpg"
        canvas.save(path, quality=88)
        output.append(str(path.relative_to(PROJECT_ROOT).as_posix()))
    return output


def _anomaly_audit(
    t0: dict[str, Any],
    registry_by_id: dict[str, dict[str, Any]],
    montage_dir: Path,
) -> dict[str, Any]:
    canonical = t0["thermal_data_audit"]["canonical_trial_records"]
    canonical_by_id = {row["sample_id"]: row for row in canonical}
    short_records: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    for row in canonical:
        thermal_count = int(row["decodable_frame_count"])
        if not 0 < thermal_count < 13:
            continue
        registry = registry_by_id[row["sample_id"]]
        ir_files = _image_files(Path(registry["paths"]["IR"])) if "IR" in registry["paths"] else []
        depth_files = (
            _image_files(Path(registry["paths"]["Depth_Color"]))
            if "Depth_Color" in registry["paths"]
            else []
        )
        relation = classify_count_relation(
            thermal_count, len(ir_files), len(depth_files)
        )
        relation_counts[relation] += 1
        short_records.append(
            {
                "sample_id": row["sample_id"],
                "class_id": row["class_id"],
                "user_id": row["user_id"],
                "development_split": row["development_split"],
                "oof_fold": row["oof_fold"],
                "category": (
                    "single_frame"
                    if thermal_count == 1
                    else "extremely_short_le4"
                    if thermal_count <= 4
                    else "short_lt13"
                ),
                "thermal_frame_count": thermal_count,
                "thermal_first_frame_number": row["first_frame_number"],
                "thermal_last_frame_number": row["last_frame_number"],
                "ir_frame_count": len(ir_files),
                "ir_first_name": ir_files[0].name if ir_files else None,
                "ir_last_name": ir_files[-1].name if ir_files else None,
                "depth_frame_count": len(depth_files),
                "relation_assessment": relation,
                "canonical_row_retained": True,
                "availability": True,
            }
        )
    short_records.sort(key=lambda row: row["sample_id"])
    for index, row in enumerate(short_records):
        page_number = index // 20 + 1
        row["visual_montage_path"] = (
            f"reports/thermal_stage0_5_montages/thermal_short_trial_montage_{page_number:02d}.jpg"
        )
        row["visual_review_status"] = (
            "included_in_codex_structured_visual_review;independent_human_signoff_pending"
        )
    short_montages = _save_short_trial_montages(
        short_records, registry_by_id, montage_dir
    )
    duplicate_records: list[dict[str, Any]] = []
    for row in t0["thermal_data_audit"]["anomalies"]["duplicate_frame_trials"]:
        registry = registry_by_id[row["sample_id"]]
        groups = find_duplicate_frame_groups(Path(registry["paths"]["Thermal"]))
        duplicate_records.append(
            {
                **row,
                "duplicate_file_groups": groups,
                "canonical_row_retained": True,
                "quality_only": True,
            }
        )
    lower = float(t0["temporal_alignment_audit"]["ir_thermal"]["tukey_lower"])
    upper = float(t0["temporal_alignment_audit"]["ir_thermal"]["tukey_upper"])
    outlier_records: list[dict[str, Any]] = []
    for sample_id, registry in registry_by_id.items():
        if "Thermal" not in registry["paths"] or "IR" not in registry["paths"]:
            continue
        thermal_count = int(canonical_by_id[sample_id]["decodable_frame_count"])
        ir_files = _image_files(Path(registry["paths"]["IR"]))
        if not thermal_count or not ir_files:
            continue
        ratio = thermal_count / len(ir_files)
        if lower <= ratio <= upper:
            continue
        depth_count = (
            len(_image_files(Path(registry["paths"]["Depth_Color"])))
            if "Depth_Color" in registry["paths"]
            else 0
        )
        outlier_records.append(
            {
                "sample_id": sample_id,
                "class_id": registry["class_id"],
                "user_id": registry["user_id"],
                "thermal_frame_count": thermal_count,
                "ir_frame_count": len(ir_files),
                "depth_frame_count": depth_count,
                "thermal_to_ir_ratio": ratio,
                "relation_assessment": classify_count_relation(
                    thermal_count, len(ir_files), depth_count
                ),
                "directory_error_supported": False,
                "frame_level_registration_supported": False,
            }
        )
    outlier_records.sort(key=lambda row: row["thermal_to_ir_ratio"])
    outlier_relation_counts = Counter(
        row["relation_assessment"] for row in outlier_records
    )
    extremes = outlier_records[:5] + outlier_records[-5:]
    return {
        "short_trial_summary": {
            "single_frame": sum(row["category"] == "single_frame" for row in short_records),
            "extremely_short_le4_including_single": sum(
                row["thermal_frame_count"] <= 4 for row in short_records
            ),
            "short_lt13": len(short_records),
            "relation_assessment_counts": dict(sorted(relation_counts.items())),
            "canonical_rows_deleted": 0,
        },
        "short_trial_records": short_records,
        "short_trial_montage_paths": short_montages,
        "duplicate_frame_summary": {
            "trials": len(duplicate_records),
            "duplicate_frames": sum(
                row["duplicate_frame_count"] for row in duplicate_records
            ),
            "canonical_rows_deleted": 0,
        },
        "duplicate_frame_records": duplicate_records,
        "ir_thermal_outlier_summary": {
            "tukey_lower": lower,
            "tukey_upper": upper,
            "trials": len(outlier_records),
            "relation_assessment_counts": dict(sorted(outlier_relation_counts.items())),
            "conclusion": "Counts and boundaries identify partial-capture candidates but cannot distinguish sensor stop from export truncation without acquisition metadata.",
        },
        "ir_thermal_extreme_records": extremes,
        "ir_thermal_outlier_records": outlier_records,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    manual = report["manual_yolo_montage_review"]
    route = report["localization_route_audit"]
    anomaly = report["anomaly_audit"]
    scale = route["auto_scale_signal_summary"]
    decision = report["frozen_preprocessing_decision"]
    lines = [
        "# Thermal Stage T0.5 Localization Route Audit",
        "",
        "## Scope and boundaries",
        "",
        f"- Baseline commit: `{report['provenance']['baseline_commit_sha']}`; branch: `{report['provenance']['branch']}`.",
        "- Population is the official train-14 users only. Sealed heldout users, heldout labels, competition test, and quarantined evidence were not enumerated or opened.",
        "- IR/X3D source, weights, configs, and ExpertEvidence remained read-only. No model, detector, or classifier was trained or tuned.",
        "- T0.5 reused the T0 YOLO outputs at the same Thermal-native uniform normalized-time points. It did not import IR indices or IR bboxes and did not use motion peaks for frame selection.",
        "",
        "## Executive findings",
        "",
        f"- Codex structured visual review covered **{manual['reviewed_trials']} trials / {manual['reviewed_frames']} frames** across all six T0 YOLO montage pages. Verdict: `{manual['decision']}`. Independent human sign-off remains pending.",
        f"- Candidate-chain route coverage on the same frames: `{json.dumps(route['route_counts'])}`; fallback from YOLO was **{route['fallback_rate_from_yolo']:.1%}**.",
        f"- Heat/motion proposal valid-trial rate was **{route['heat_motion_valid_trial_rate']:.1%}**. Against accepted YOLO boxes, its diagnostic coverage proxy was `{json.dumps(route['heat_motion_yolo_box_coverage_proxy'])}`; YOLO is not treated as ground truth.",
        f"- Heat/motion was selective (valid expanded bbox below 95% of frame) on only **{route['heat_motion_selective_trial_rate']:.1%}** of trials; **{route['heat_motion_full_frame_equivalent_trial_rate']:.1%}** expanded to at least 95% of the frame. It is not promoted into the frozen online route.",
        f"- T0 anomalies were retained: {anomaly['short_trial_summary']['single_frame']} singleton, {anomaly['short_trial_summary']['extremely_short_le4_including_single']} at most four frames, {anomaly['short_trial_summary']['short_lt13']} below 13 frames, and {anomaly['duplicate_frame_summary']['duplicate_frames']} exact duplicate frames in {anomaly['duplicate_frame_summary']['trials']} trials.",
        f"- Rendered auto-scale indication counts were `{json.dumps(scale['trial_counts'])}`. These are RGB drift indicators only; absolute temperature-scale stability remains unproven.",
        f"- Frozen first-experiment input: **{decision['first_iformer_t_tsm_view']}**. Conditional localization is retained as a later matched ablation and as label-free quality, not silently mixed into the first baseline.",
        "",
        "## 1. Structured YOLO montage review",
        "",
        f"Verdict counts: `{json.dumps(manual['verdict_counts'])}`.",
        "",
        f"Observed failure types: `{json.dumps(manual['failure_type_counts'])}`.",
        "",
        "| Factor | Trials | Frames | Detection rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in manual["factor_summary"]:
        rate = "n/a" if row["detection_rate"] is None else f"{row['detection_rate']:.1%}"
        lines.append(
            f"| {row['factor']} | {row['trials']} | {row['frames']} | {rate} |"
        )
    lines.extend(
        [
            "",
            "Structured visual review confirms that many high-confidence boxes identify the person, but near-field clipping removes hands/objects, far seated subjects create confidence gaps, and a few low-confidence candidates attach to furniture or hot objects. A numeric person detection is therefore not accepted as a complete action crop. This Codex review does not replace the still-pending independent human authorization gate.",
            "Factor strata are descriptive audit summaries only. Class/action context never enters localization routing or Thermal frame selection.",
            "",
            "## 2. Full-frame, YOLO-context, and heat/motion-context audit",
            "",
            f"- Route rates: `{json.dumps(route['route_rates'])}`.",
            f"- Selected bbox area ratio: `{json.dumps(route['selected_bbox_area_ratio'])}`.",
            f"- YOLO-context continuity: `{json.dumps(route['yolo_context_continuity_iou'])}`.",
            f"- Final conditional-route continuity: `{json.dumps(route['final_route_continuity_iou'])}`.",
            f"- Heat/motion component area: `{json.dumps(route['heat_motion_component_area_ratio'])}`; expanded area: `{json.dumps(route['heat_motion_bbox_area_ratio'])}`.",
            "- Heat/motion uses only the uniformly sampled Thermal frames. Motion differences contribute to the union mask, but peaks never choose training frames.",
            f"- Heat/motion selectivity rate: **{route['heat_motion_selective_trial_rate']:.1%}**; full-frame-equivalent expanded-box rate: **{route['heat_motion_full_frame_equivalent_trial_rate']:.1%}**. Its high box-coverage proxy mostly comes from retaining almost the whole frame, not from reliable person localization.",
            "- Full-frame retains all action context and is valid for every decodable trial. YOLO context is accepted only at confidence >=0.25 and raw bbox area >=5%; heat/motion rejects components below 5% or above 95%; every failure falls back explicitly.",
            "",
            "Route montages:",
            "",
            *[f"- `{value}`" for value in route["montage_paths"]],
            "",
            "## 3. Anomaly review",
            "",
            f"- Short-trial relation assessments: `{json.dumps(anomaly['short_trial_summary']['relation_assessment_counts'])}`.",
            f"- IR/Thermal Tukey outliers: **{anomaly['ir_thermal_outlier_summary']['trials']}**; assessments: `{json.dumps(anomaly['ir_thermal_outlier_summary']['relation_assessment_counts'])}`.",
            "- Present and decodable single/short directories are not directory errors. Where Thermal has 1-4 frames and IR/Depth have tens or hundreds, the evidence supports a partial Thermal capture/export candidate, not a proven sensor fault. The reverse asymmetry similarly implicates IR/Depth capture/export. Acquisition metadata is required to distinguish sensor stop from export truncation.",
            "- All canonical rows remain. Availability stays true for decodable short trials; frame scarcity, source uniqueness, duplicate ratio, and fallback route remain quality fields.",
            "- Visual review of all seven short-trial montage pages found genuine short action snippets, but also some room-only frames, edge-only bodies, and partial action fragments. These are visibility/quality defects, not grounds for deleting canonical samples.",
            "",
            "Short-trial montages:",
            "",
            *[f"- `{value}`" for value in anomaly["short_trial_montage_paths"]],
            "",
            "## 4. Rendered pseudocolor and auto-scale indications",
            "",
            f"- Static-background luminance median span: `{json.dumps(scale['static_background_luminance_median_span'])}`.",
            f"- Static-background hue median span: `{json.dumps(scale['static_background_hue_median_span'])}`.",
            f"- Rendered endpoint-fraction span: `{json.dumps(scale['endpoint_fraction_span'])}`.",
            "- Drift is assessed on low-temporal-variance background pixels and global rendered luminance percentiles. Motion and scene changes can still contaminate these metrics. Without raw temperatures, emissivity, camera range, or palette metadata, neither weak nor strong RGB drift proves an absolute temperature scale.",
            "",
            "## 5. Frozen preprocessing decision",
            "",
            f"- First iFormer-T+TSM development view: `{decision['first_iformer_t_tsm_view']}`.",
            f"- Conditional ablation priority: `{decision['conditional_ablation_priority']}`.",
            f"- Fallback order: `{decision['conditional_fallback_order']}`.",
            f"- Heat/motion status: `{decision['heat_motion_status']}`.",
            f"- Required quality fields: `{', '.join(decision['quality_fields'])}`.",
            "- The first development baseline uses the simple full-frame path for every decodable Thermal trial. This avoids route-dependent context loss and makes the matched MobileNetV3 control exact. A later YOLO-context/full-frame ablation may use the same split and sampler, but may not replace the baseline without a controlled result. The audited heat/motion proposal is too close to full-frame to justify online route complexity.",
            "- Singleton and short trials use the frozen 16-target Thermal sampler with explicit repeated-source mask. No canonical sample is deleted, and motion peaks, IR indices, and IR bboxes remain prohibited.",
            "",
            "**Stage T0.5 stop:** preprocessing is frozen for the first development experiment, but no training is authorized or performed in this stage.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from scripts.audit_thermal_stage0 import (
        _oof_train_users,
        inventory_train14_trials,
        validate_train14_users,
    )

    args = parse_args()
    t0 = _load_json(args.t0_report)
    if t0["provenance"]["baseline_commit_sha"] != BASELINE_COMMIT:
        raise ValueError("T0 baseline commit does not match the frozen Thermal baseline")
    if t0["provenance"]["population"] != "official_train14_only":
        raise ValueError("T0 report is not scoped to official train-14")
    oof_path = PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json"
    oof = _load_json(oof_path)
    train14_users = validate_train14_users(_oof_train_users(oof))
    registry = inventory_train14_trials(args.data_root, train14_users)
    registry_by_id = {row["sample_id"]: row for row in registry}
    frame_records = t0["localization_audit"]["frame_records"]
    missing = {row["sample_id"] for row in frame_records} - set(registry_by_id)
    if missing:
        raise ValueError(f"T0 representative trials missing from train-14 registry: {sorted(missing)}")
    manual = _manual_reviews(frame_records)
    route, _ = _route_audit(registry_by_id, frame_records, args.montage_dir)
    anomalies = _anomaly_audit(t0, registry_by_id, args.montage_dir)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
    ).strip()
    audit_code_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, audit_code_head],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError(
            f"Frozen Thermal baseline {BASELINE_COMMIT} is not an ancestor of {audit_code_head}"
        )
    report = {
        "schema_version": "thermal-stage0.5-audit-v1",
        "stage": "thermal_stage_t0_5",
        "status": "completed_no_training",
        "provenance": {
            "generated_local": datetime.now().astimezone().isoformat(),
            "baseline_commit_sha": BASELINE_COMMIT,
            "audit_code_base_commit_sha": audit_code_head,
            "audit_script_sha256": _sha256_file(Path(__file__)),
            "branch": branch,
            "worktree": str(PROJECT_ROOT),
            "data_root": str(args.data_root),
            "population": "official_train14_only",
            "train14_users": train14_users,
            "sealed_heldout_users_not_enumerated": [
                "user4",
                "user17",
                "user23",
                "user24",
            ],
            "competition_test_accessed": False,
            "quarantined_evidence_accessed": False,
            "ir_x3d_modified": False,
            "training_or_tuning_performed": False,
            "new_learned_weights_bytes": 0,
            "t0_report_path": str(args.t0_report.relative_to(PROJECT_ROOT).as_posix()),
            "t0_report_sha256": _sha256_file(args.t0_report),
            "t0_yolo_weights_sha256": t0["localization_audit"]["weights_sha256"],
        },
        "manual_yolo_montage_review": manual,
        "localization_route_audit": route,
        "anomaly_audit": anomalies,
        "frozen_preprocessing_decision": {
            "first_iformer_t_tsm_view": "full_frame",
            "reason": "universal decodable-trial coverage and action-context retention; avoids route-dependent crop failures in the first matched baseline",
            "conditional_ablation_priority": "thermal_yolo_context_then_full_frame",
            "conditional_fallback_order": "thermal_yolo_context -> full_frame",
            "conditional_routes_status": "quality_bearing_later_matched_ablation_not_first_baseline",
            "heat_motion_status": "offline_quality_diagnostic_not_default_route_due_low_selectivity",
            "sampling": "16 uniform normalized Thermal time targets; no motion-peak sampling; no IR indices",
            "short_trial_policy": "retain canonical sample; repeat nearest Thermal source only for fixed tensor; emit source-uniqueness mask",
            "quality_fields": [
                "directory_present",
                "decodable_frame_fraction",
                "distinct_frame_ratio",
                "unique_sampled_source_ratio",
                "duration_bucket",
                "localizer_confidence",
                "raw_bbox_area_ratio",
                "expanded_bbox_area_ratio",
                "bbox_continuity",
                "heat_motion_component_area_ratio",
                "heat_motion_valid",
                "fallback_route",
                "fallback_reason",
                "rendered_auto_scale_evidence_strength",
            ],
            "color_jitter_allowed": False,
            "canonical_rows_deleted": 0,
        },
        "deployment_and_evidence_boundary": {
            "known_frozen_ir_plus_yolo_bytes": 20644200,
            "new_t0_5_learned_bytes": 0,
            "complete_package_ceiling_exclusive": 95000000,
            "complete_package_pass_claimed": False,
            "expert_evidence_written": False,
            "stage_t1_training_authorized": False,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(report, args.markdown_output)
    print(
        json.dumps(
            {
                "json": str(args.json_output),
                "markdown": str(args.markdown_output),
                "route_counts": route["route_counts"],
                "short_trials": len(anomalies["short_trial_records"]),
            }
        )
    )


if __name__ == "__main__":
    main()
