from __future__ import annotations

import math
import json
from pathlib import Path
import subprocess
import sys

from scripts.audit_thermal_v2_inputs import (
    bbox_iou,
    build_quality_vector,
    select_representative_records,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "thermal_v2_input_audit.json"
MARKDOWN_PATH = ROOT / "reports" / "thermal_v2_input_audit.md"
MONTAGE_DIR = ROOT / "reports" / "thermal_v2_input_montages"


def record(sample_id: str, class_id: int, user_id: str, *, available: bool) -> dict:
    return {
        "sample_id": sample_id,
        "class_id": class_id,
        "user_id": user_id,
        "duration_bucket": "9_to_32",
        "usable": True,
        "context_available": available,
        "decodable_frame_count": 10,
        "file_count": 10,
        "duplicate_frame_count": 1,
        "detection_hit_ratio": 0.5,
        "median_confidence": 0.7,
        "bbox_area_ratio": 0.4 if available else 0.0,
    }


def test_representative_selection_covers_classes_and_fallback() -> None:
    records = [
        record("a", 0, "user1", available=True),
        record("b", 0, "user6", available=False),
        record("c", 1, "user7", available=True),
    ]

    selected = select_representative_records(records, count=3)

    assert {row["class_id"] for row in selected} == {0, 1}
    assert any(not row["context_available"] for row in selected)
    assert len({row["sample_id"] for row in selected}) == 3


def test_quality_vector_has_finite_frozen_order() -> None:
    row = record("a", 0, "user1", available=True)
    sampled_indices = [0, 0, 1, 1, 2, 2, 3, 3]
    pose_mask = [True, False, True, False, True, False, True, False]

    quality = build_quality_vector(row, sampled_indices, pose_mask)

    assert len(quality) == 8
    assert all(math.isfinite(value) for value in quality)
    assert quality == [1.0, 0.5, 0.5, 0.7, 0.4, 0.5, 0.1, 1.0]


def test_bbox_iou_handles_overlap_and_missing() -> None:
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
    assert bbox_iou(None, (0, 0, 1, 1)) is None


def test_pure_audit_import_does_not_eagerly_load_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.audit_thermal_v2_inputs; "
            "print('torch' in sys.modules)",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_generated_a0_report_stops_for_human_montage_approval() -> None:
    assert REPORT_PATH.is_file()
    assert MARKDOWN_PATH.is_file()
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["status"] == "pending_human_montage_approval"
    assert report["canonical_population"]["canonical_trials"] == 2427
    assert report["canonical_population"]["usable_trials"] == 2299
    assert report["fixed_context"]["context_available_trials"] == 2207
    assert report["representative_tensor_audit"]["trial_count"] == 56
    assert report["representative_tensor_audit"]["all_tensors_finite"]
    assert report["representative_tensor_audit"]["shapes"] == {
        "full_rgb": [3, 16, 3, 160, 160],
        "crop_rgb": [3, 16, 3, 160, 160],
        "motion": [3, 16, 1, 160, 160],
        "pose": [3, 16, 56],
        "pose_mask": [3, 16],
        "availability": [4],
        "quality": [8],
    }
    assert report["manual_montage_review"] == {
        "required": True,
        "approved": False,
        "reviewer": None,
        "reviewed_at": None,
    }
    assert report["evidence_boundary"]["heldout4_labels_read"] is False
    assert report["evidence_boundary"]["competition_test_read"] is False
    assert report["evidence_boundary"]["ir_or_depth_inputs_read"] is False
    assert list(MONTAGE_DIR.glob("thermal_v2_context_*.jpg"))
