from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.export_depth_ordinal_assets import (
    content_metrics,
    deterministic_reliability,
    export_depth_ordinal_assets,
)
from src.data.ordinal_depth import JET_LUT_BGR, crop_and_letterbox_ordinal


def test_crop_letterbox_preserves_aspect_and_masks_padding() -> None:
    values = np.full((4, 8), 120, dtype=np.uint8)
    valid = np.ones_like(values, dtype=bool)
    output, output_valid = crop_and_letterbox_ordinal(values, valid, [0, 0, 8, 4], (8, 8))
    assert output_valid.sum() == 32
    assert np.all(output[output_valid] == 120)
    assert np.all(output[~output_valid] == 0)


def test_content_rule_and_reliability_are_deterministic() -> None:
    assert content_metrics(np.zeros((16, 16), dtype=np.uint8))["content_invalid_2of3"]
    assert deterministic_reliability(True, 1.0, 0.5) == 0.5
    assert deterministic_reliability(False, 1.0, 1.0) == 0.0
    assert deterministic_reliability(True, 0.25, 1.0) == 0.0


def test_explicit_sample_selection_rejects_unknown_ids(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    _write_csv(baseline / "all_frame_inputs.csv", [{"sample_id": "known", "source_frame_index": 0}])
    _write_csv(baseline / "roi_frame_audit.csv", [{
        "split": "train", "class_id": 0, "action_name": "Action", "sample_id": "known",
        "user_id": "u", "source_frame_index": 0, "view_name": "depth_context", "valid": 1,
        "source": "x", "confidence": 1.0, "x1": 0, "y1": 0, "x2": 1, "y2": 1,
    }])
    effective = tmp_path / "effective.csv"
    source = tmp_path / "source.csv"
    _write_csv(effective, [{
        "sample_id": "known", "source_frame_index": 0, "view_name": "depth_context",
        "valid_flag": 1, "content_invalid": False, "effective_valid": True,
    }])
    _write_csv(source, [{
        "sample_id": "known", "path": "missing.png", "readable": True, "unexpected_pixels": 0,
    }])
    import pytest

    with pytest.raises(ValueError, match="Requested sample IDs are absent"):
        export_depth_ordinal_assets(
            baseline_root=baseline,
            effective_audit_path=effective,
            source_depth_path=source,
            output_root=tmp_path / "out",
            sample_ids={"absent"},
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_exporter_writes_depth_only_assets_and_combined_manifest(tmp_path: Path) -> None:
    raw_root = tmp_path / "train"
    depth_trial = raw_root / "Depth_Color" / "0_Action" / "user1" / "1-1-1"
    ir_trial = raw_root / "IR" / "0_Action" / "user1" / "1-1-1"
    baseline = tmp_path / "baseline"
    output = tmp_path / "ordinal"
    sample_id = "train__c00__user1__1-1-1"
    timestamp = "2025-01-01_00-00-00.000"
    depth_path = depth_trial / f"Depth_{timestamp}_00000001_Color.png"
    ir_path = ir_trial / f"IR_{timestamp}_00000001.png"
    depth_trial.mkdir(parents=True)
    ir_trial.mkdir(parents=True)
    ordinal = np.arange(48, dtype=np.uint8).reshape(6, 8)
    depth_image = JET_LUT_BGR[ordinal]
    depth_image[0, 0] = 0
    assert cv2.imwrite(str(depth_path), depth_image)
    assert cv2.imwrite(str(ir_path), np.full((6, 8), 80, dtype=np.uint8))

    frame: dict[str, object] = {
        "split": "train", "class_id": 0, "action_name": "Action", "sample_id": sample_id,
        "user_id": "user1", "source_frame_index": 0,
    }
    effective_rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []
    for view in ("ir_context", "ir_left", "ir_right", "ir_relation"):
        relative = Path("train") / "c00" / sample_id / view / "f0000.png"
        target = baseline / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(target), np.full((4, 4), 80, dtype=np.uint8))
        frame[f"{view}_path"] = relative.as_posix()
        frame[f"{view}_valid"] = 1
        frame[f"{view}_source"] = "fixture"
        frame[f"{view}_confidence"] = 1.0
        effective_rows.append({
            "sample_id": sample_id, "source_frame_index": 0, "view_name": view,
            "valid_flag": 1, "content_invalid": False, "effective_valid": True,
        })
    for view in ("depth_context", "depth_relation"):
        frame[f"{view}_path"] = "unused_old_depth.png"
        frame[f"{view}_valid"] = 1
        frame[f"{view}_source"] = "fixture"
        roi_rows.append({
            "split": "train", "class_id": 0, "action_name": "Action", "sample_id": sample_id,
            "user_id": "user1", "source_frame_index": 0, "view_name": view, "valid": 1,
            "source": "fixture", "confidence": 1.0, "x1": 0, "y1": 0, "x2": 8, "y2": 6,
        })
        effective_rows.append({
            "sample_id": sample_id, "source_frame_index": 0, "view_name": view,
            "valid_flag": 1, "content_invalid": False, "effective_valid": True,
        })
    _write_csv(baseline / "all_frame_inputs.csv", [frame])
    _write_csv(baseline / "roi_frame_audit.csv", roi_rows)
    effective_path = tmp_path / "effective.csv"
    source_path = tmp_path / "source.csv"
    _write_csv(effective_path, effective_rows)
    _write_csv(source_path, [{
        "sample_id": sample_id, "path": str(depth_path), "readable": True, "unexpected_pixels": 0,
    }])

    result = export_depth_ordinal_assets(
        baseline_root=baseline,
        effective_audit_path=effective_path,
        source_depth_path=source_path,
        output_root=output,
        image_size=4,
    )
    assert result["samples"] == 1
    assert result["frames"] == 1
    assert result["ir_images_duplicated"] is False
    assert (output / "_SUCCESS").is_file()
    manifest = pd.read_csv(output / "combined_frame_manifest.csv")
    assert len(manifest) == 1
    assert Path(manifest.ir_context_path.iloc[0]).resolve() == (baseline / frame["ir_context_path"]).resolve()
    for view in ("depth_context", "depth_relation"):
        value = cv2.imread(manifest[f"{view}_ordinal_path"].iloc[0], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(manifest[f"{view}_pixel_valid_path"].iloc[0], cv2.IMREAD_UNCHANGED)
        assert value.shape == (4, 4)
        assert set(np.unique(mask)) <= {0, 255}
        assert not np.any(value[mask == 0])
    assert result["competition_test_read"] is False
    assert result["training_run"] is False
