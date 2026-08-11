from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.inference.x3d_s_ir_context_pipeline import X3DIRContextPipeline
from src.roi.ir_primary_input_builder import IRPrimaryInputROIBuilder
from src.roi.pose_locator import PoseDetection
from scripts.audit_x3d_s_run import crop_parity_metrics, run_online_parity


class FakePoseLocator:
    def __init__(self, detection: PoseDetection | None) -> None:
        self.detection = detection
        self.confidences: list[float | None] = []

    def predict(
        self,
        images: list[np.ndarray],
        *,
        detection_confidence: float | None = None,
    ) -> list[PoseDetection | None]:
        self.confidences.append(detection_confidence)
        return [self.detection for _ in images]


def _raw_frames(tmp_path: Path, count: int = 3) -> tuple[Path, ...]:
    paths = []
    for index in range(count):
        image = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) + index
        path = tmp_path / f"frame-{index:02d}.png"
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return tuple(paths)


def _detection() -> PoseDetection:
    return PoseDetection(
        bbox_xyxy=np.asarray([12.0, 8.0, 52.0, 60.0], dtype=np.float32),
        bbox_confidence=0.9,
        keypoints_xy=np.zeros((17, 2), dtype=np.float32),
        keypoints_confidence=np.zeros(17, dtype=np.float32),
    )


def test_online_preprocess_reuses_roi_builder_and_x3d_sampling(tmp_path: Path) -> None:
    paths = _raw_frames(tmp_path)
    detection = _detection()
    locator = FakePoseLocator(detection)
    builder = IRPrimaryInputROIBuilder()
    pipeline = X3DIRContextPipeline(pose_locator=locator, roi_builder=builder)

    result = pipeline.preprocess_trial(paths)

    expected = builder.build(
        np.stack([detection.bbox_xyxy] * len(paths)),
        np.stack([detection.keypoints_xy] * len(paths)),
        np.stack([detection.keypoints_confidence] * len(paths)),
        width=64,
        height=64,
    )
    np.testing.assert_allclose(result.boxes, expected.boxes[:, 0], atol=0.0, rtol=0.0)
    assert result.crops.shape == (3, 256, 256)
    assert tuple(result.clips.shape) == (1, 1, 3, 13, 182, 182)
    assert result.window_bounds.tolist() == [[0, 3]]
    assert result.source_indices.shape == (1, 1, 13)
    assert result.recovery_used is False
    assert locator.confidences == [0.25]
    assert torch.isfinite(result.clips).all()


def test_importing_pose_locator_does_not_patch_global_opencv_functions() -> None:
    code = """
import cv2
before = {name: getattr(cv2, name) for name in ('imread', 'imwrite', 'imshow')}
import src.roi.pose_locator
assert all(getattr(cv2, name) is function for name, function in before.items())
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_online_preprocess_forbids_silent_full_frame_fallback(tmp_path: Path) -> None:
    locator = FakePoseLocator(None)
    pipeline = X3DIRContextPipeline(pose_locator=locator)

    with pytest.raises(ValueError, match="full-frame fallback is forbidden"):
        pipeline.preprocess_trial(_raw_frames(tmp_path))

    assert locator.confidences == [0.25, 0.01]


def test_crop_parity_uses_distribution_gates_not_single_pixel_maximum() -> None:
    reference = np.zeros((1, 256, 256), dtype=np.uint8)
    candidate = reference.copy()
    candidate[0, 128, 128] = 168

    metrics = crop_parity_metrics(candidate, reference)

    assert metrics["max_absolute_error"] == 168
    assert metrics["mae"] <= 1.0
    assert metrics["p99_absolute_error"] <= 8.0
    assert metrics["psnr_db"] >= 40.0
    assert metrics["worst_frame_mae"] <= 2.0
    assert metrics["gate_passed"] is True


def test_real_online_pipeline_matches_exported_training_rois() -> None:
    config_path = Path("configs/experiments/x3d_s_ir_context_fold0.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    report = run_online_parity(
        config,
        Path("outputs/x3d_s_ir_context_fold0/x3d_s_ir_context_adaptive_smoke"),
    )

    assert report["status"] == "passed"
    assert report["sample_count"] == 6
    assert report["clip_counts"] == [1, 1, 2, 4, 8, 8]
    assert report["recovered_sample_count"] >= 1
    assert report["max_box_absolute_error"] <= 1.0
    assert report["max_crop_mae"] <= 1.0
    assert report["max_crop_p99_absolute_error"] <= 8.0
    assert report["min_crop_psnr_db"] >= 40.0
    assert report["max_worst_frame_crop_mae"] <= 2.0
    assert report["frame_order_gate_passed"] is True
    assert report["person_selection_recovery_gate_passed"] is True
    assert report["normalization_gate_passed"] is True
    assert report["normalization_contract"] == "shared_x3d_clip_transform"
    assert report["input_parity_gate_passed"] is True
    assert len(report["x3d_checkpoint_sha256"]) == 64
    assert report["model_sensitivity_thresholds_applied"] is False
    assert all(
        {
            "embedding_cosine_similarity",
            "probability_l1_distance",
            "jensen_shannon_divergence",
            "max_class_probability_delta",
            "top1_agreement",
        }
        <= set(row["model_sensitivity"])
        for row in report["sample_rows"]
    )
    assert set(report["latency_seconds_by_length_bucket"]) == {
        "<=13",
        "14-32",
        "33-64",
        ">64",
    }
    assert report["latency_seconds_by_length_bucket"][">64"]["sample_count"] == 3
