from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.audit_thermal_stage0 import (
    _pearson_correlation,
    audit_trial_images,
    classify_thermal_rendering,
    estimate_motion_alignment,
    normalized_time_candidate_pairs,
    serializable_trial_record,
    sample_thermal_indices,
    summarize_localization_groups,
    validate_train14_users,
)


def test_train14_boundary_rejects_sealed_heldout_users() -> None:
    with pytest.raises(ValueError, match="sealed heldout"):
        validate_train14_users(["user1", "user4"])


@pytest.mark.parametrize(
    ("frame_count", "sample_count", "expected"),
    [
        (1, 5, [0, 0, 0, 0, 0]),
        (2, 5, [0, 0, 0, 1, 1]),
        (5, 5, [0, 1, 2, 3, 4]),
    ],
)
def test_thermal_sampler_uses_own_normalized_timeline(
    frame_count: int, sample_count: int, expected: list[int]
) -> None:
    assert sample_thermal_indices(frame_count, sample_count) == expected


def test_normalized_time_candidates_do_not_pair_by_raw_position() -> None:
    pairs = normalized_time_candidate_pairs(reference_count=3, candidate_count=5)

    assert pairs == [(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)]


def test_rendering_classifier_distinguishes_gray_copy_and_pseudocolor() -> None:
    gray = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
    gray_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    pseudo = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)[:, :, ::-1]

    gray_result = classify_thermal_rendering([gray_rgb, gray_rgb])
    pseudo_result = classify_thermal_rendering([pseudo, pseudo])

    assert gray_result["rendering"] == "grayscale_copy"
    assert pseudo_result["rendering"] == "stable_pseudocolor"
    assert pseudo_result["auto_scale_identifiable"] is False


def test_trial_audit_separates_present_decodable_usable_and_duplicates(
    tmp_path: Path,
) -> None:
    image = np.full((16, 20, 3), (20, 80, 180), dtype=np.uint8)
    first = tmp_path / "frame_000001.jpg"
    duplicate = tmp_path / "frame_000002.jpg"
    corrupt = tmp_path / "frame_000003.jpg"
    assert cv2.imwrite(str(first), image)
    shutil.copyfile(first, duplicate)
    corrupt.write_bytes(b"not-a-jpeg")

    result = audit_trial_images(tmp_path)

    assert result["directory_present"] is True
    assert result["file_count"] == 3
    assert result["decodable_frame_count"] == 2
    assert result["corrupt_jpeg_count"] == 1
    assert result["duplicate_frame_count"] == 1
    assert result["distinct_frame_ratio"] == pytest.approx(0.5)
    assert result["usable"] is True


def test_empty_trial_directory_is_present_but_not_decodable_or_usable(
    tmp_path: Path,
) -> None:
    result = audit_trial_images(tmp_path)

    assert result["directory_present"] is True
    assert result["decodable"] is False
    assert result["usable"] is False


def test_pearson_correlation_avoids_blas_and_handles_degenerate_input() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float64)

    assert _pearson_correlation(values, values) == pytest.approx(1.0)
    assert _pearson_correlation(values, -values) == pytest.approx(-1.0)
    assert _pearson_correlation(values, np.ones_like(values)) is None


def test_motion_alignment_recovers_synthetic_normalized_offset() -> None:
    times = np.linspace(0.0, 1.0, 80)
    reference = np.exp(-0.5 * ((times - 0.55) / 0.08) ** 2)
    candidate = np.exp(-0.5 * ((times - 0.45) / 0.08) ** 2)

    result = estimate_motion_alignment(reference, candidate)

    assert result["correlation"] > 0.98
    assert result["offset"] == pytest.approx(0.10, abs=0.02)
    assert result["scale"] == pytest.approx(1.0, abs=0.04)


def test_localization_groups_report_coverage_and_detection_rate() -> None:
    records = [
        {"user_id": "user1", "detected": True, "confidence": 0.8, "previous_bbox_iou": 0.7},
        {"user_id": "user1", "detected": False, "confidence": 0.1, "previous_bbox_iou": None},
        {"user_id": "user2", "detected": True, "confidence": 0.6, "previous_bbox_iou": None},
    ]

    groups = summarize_localization_groups(records, "user_id")

    assert groups[0]["user_id"] == "user1"
    assert groups[0]["frames"] == 2
    assert groups[0]["detection_rate"] == pytest.approx(0.5)
    assert groups[0]["detected_confidence"]["median"] == pytest.approx(0.8)


def test_serializable_trial_record_drops_runtime_absolute_paths() -> None:
    record = {
        "sample_id": "train__c00__user1__1-1-1",
        "paths": {"Thermal": r"D:\\private\\Thermal"},
        "thermal_path": r"D:\\private\\Thermal",
        "usable": True,
    }

    output = serializable_trial_record(record)

    assert output == {"sample_id": record["sample_id"], "usable": True}
