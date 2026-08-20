from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.audit_thermal_stage0_5 import (
    assess_rendered_auto_scale_signals,
    build_heat_motion_context,
    classify_count_relation,
    choose_localization_route,
    expand_context_bbox,
    find_duplicate_frame_groups,
)


def test_yolo_context_expands_and_clips_thermal_native_bbox() -> None:
    result = expand_context_bbox(
        bbox_xyxy=[10.0, 20.0, 50.0, 60.0],
        image_width=80,
        image_height=70,
        expansion_fraction=0.25,
    )

    assert result == pytest.approx([0.0, 10.0, 60.0, 70.0])


def test_heat_motion_context_tracks_moving_hot_subject() -> None:
    frames: list[np.ndarray] = []
    for x in (12, 20, 28, 36):
        gray = np.full((64, 96), 20, dtype=np.uint8)
        cv2.rectangle(gray, (x, 18), (x + 18, 54), 230, thickness=-1)
        frames.append(cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA))

    result = build_heat_motion_context(frames)

    assert result["valid"] is True
    assert result["component_area_ratio"] >= 0.05
    assert result["bbox_area_ratio"] < 0.95
    x1, y1, x2, y2 = result["bbox_xyxy"]
    assert x1 <= 12 and x2 >= 54
    assert y1 <= 18 and y2 >= 54


def test_heat_motion_context_rejects_uniform_full_frame_component() -> None:
    frames = [np.full((48, 64, 3), 127, dtype=np.uint8) for _ in range(3)]

    result = build_heat_motion_context(frames)

    assert result["valid"] is False
    assert result["invalid_reason"] == "component_area_above_0_95"


@pytest.mark.parametrize(
    ("yolo", "heat_motion", "expected_route", "expected_reason"),
    [
        (
            {"confidence": 0.8, "bbox_area_ratio": 0.2, "bbox_xyxy": [1, 2, 20, 30]},
            {"valid": True, "bbox_xyxy": [2, 3, 30, 40]},
            "thermal_yolo_context",
            None,
        ),
        (
            {"confidence": 0.2, "bbox_area_ratio": 0.2, "bbox_xyxy": [1, 2, 20, 30]},
            {"valid": True, "bbox_xyxy": [2, 3, 30, 40]},
            "thermal_heat_motion_context",
            "yolo_confidence_below_0_25",
        ),
        (
            {"confidence": 0.2, "bbox_area_ratio": 0.2, "bbox_xyxy": [1, 2, 20, 30]},
            {"valid": False, "invalid_reason": "component_area_below_0_05"},
            "full_frame",
            "yolo_confidence_below_0_25;heat_motion_component_area_below_0_05",
        ),
    ],
)
def test_route_priority_and_fallback_reason_are_explicit(
    yolo: dict[str, object],
    heat_motion: dict[str, object],
    expected_route: str,
    expected_reason: str | None,
) -> None:
    result = choose_localization_route(yolo, heat_motion)

    assert result["route"] == expected_route
    assert result["fallback_reason"] == expected_reason


def test_auto_scale_audit_distinguishes_stable_rendering_from_global_drift() -> None:
    base = np.tile(np.arange(96, dtype=np.uint8), (64, 1))
    stable = [cv2.applyColorMap(base, cv2.COLORMAP_MAGMA) for _ in range(5)]
    drifting = [
        cv2.applyColorMap(np.clip(base.astype(np.int16) + shift, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        for shift in (0, 25, 50, 75, 100)
    ]

    stable_result = assess_rendered_auto_scale_signals(stable)
    drifting_result = assess_rendered_auto_scale_signals(drifting)

    assert stable_result["evidence_strength"] == "weak"
    assert drifting_result["evidence_strength"] in {"moderate", "strong"}
    assert stable_result["absolute_temperature_scale_proven"] is False
    assert drifting_result["absolute_temperature_scale_proven"] is False


def test_auto_scale_hue_drift_uses_circular_hue_distance() -> None:
    frames: list[np.ndarray] = []
    for hue in (178, 179, 0, 1, 2):
        hsv = np.full((32, 48, 3), (hue, 180, 160), dtype=np.uint8)
        frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    result = assess_rendered_auto_scale_signals(frames)

    assert result["static_background_hue_median_span"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("thermal", "ir", "depth", "expected"),
    [
        (1, 72, 72, "thermal_capture_truncated_or_export_incomplete"),
        (57, 1, 1, "ir_depth_capture_truncated_or_export_incomplete"),
        (2, 3, 3, "short_capture_across_modalities"),
        (48, 20, 20, "plausible_asynchronous_rate_difference"),
    ],
)
def test_frame_count_relation_keeps_sensor_asymmetry_explicit(
    thermal: int, ir: int, depth: int, expected: str
) -> None:
    assert classify_count_relation(thermal, ir, depth) == expected


def test_duplicate_scan_identifies_exact_file_names(tmp_path) -> None:
    image = np.full((12, 16, 3), (30, 100, 220), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "frame_1.jpg"), image)
    assert cv2.imwrite(str(tmp_path / "frame_2.jpg"), image)
    image[0, 0] = (0, 0, 0)
    assert cv2.imwrite(str(tmp_path / "frame_3.jpg"), image)

    groups = find_duplicate_frame_groups(tmp_path)

    assert groups == [["frame_1.jpg", "frame_2.jpg"]]
