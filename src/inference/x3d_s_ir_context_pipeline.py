from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from src.data.x3d_clip_dataset import (
    LOCAL_FRAMES,
    MAX_CLIPS,
    TARGET_WINDOW_FRAMES,
    _transform_clip_frames,
    partition_trial_windows,
    stratified_temporal_indices,
)
from src.models.expert_contract import ExpertOutput
from src.roi.ir_primary_input_builder import IRPrimaryInputROIBuilder
from src.roi.pose_locator import PoseDetection


FROZEN_IR_CONTEXT_INTERACTION_CONFIG = {
    "keypoint_threshold": 0.25,
    "projection_alpha": 0.55,
    "long_axis_scale": 0.65,
    "short_axis_scale": 0.45,
    "wrist_fallback_person_width": 0.35,
    "minimum_pixel_side": 48.0,
    "merge_iou_threshold": 0.30,
    "merge_projected_distance": 0.60,
    "merge_wrist_distance": 0.70,
    "merge_elbow_distance": 0.80,
    "merge_wide_wrist_distance": 1.20,
    "two_hand_horizontal_padding": 0.30,
    "two_hand_top_padding": 0.10,
    "two_hand_bottom_padding": 0.45,
    "hand_head_distance": 0.90,
}


def build_frozen_ir_context_roi_builder() -> IRPrimaryInputROIBuilder:
    return IRPrimaryInputROIBuilder(
        keypoint_threshold=0.25,
        context_padding=0.22,
        context_quantile=0.95,
        minimum_context_side_ratio=0.45,
        local_padding=0.12,
        duplicate_iou_threshold=0.70,
        interaction_config=FROZEN_IR_CONTEXT_INTERACTION_CONFIG,
    )


class PoseLocator(Protocol):
    def predict(
        self,
        images: Sequence[np.ndarray],
        *,
        detection_confidence: float | None = None,
    ) -> list[PoseDetection | None]: ...


@dataclass(frozen=True)
class OnlineIRContextPreprocessResult:
    boxes: np.ndarray
    crops: np.ndarray
    clips: torch.Tensor
    source_indices: torch.Tensor
    window_bounds: torch.Tensor
    bbox_confidences: np.ndarray
    recovery_used: bool
    pose_roi_seconds: float


@dataclass(frozen=True)
class OnlineIRContextPrediction:
    expert: ExpertOutput
    preprocess: OnlineIRContextPreprocessResult
    x3d_seconds: float
    complete_trial_seconds: float


def crop_ir_context(path: Path, box: np.ndarray, *, output_size: int = 256) -> np.ndarray:
    with Image.open(path) as opened:
        crop = opened.convert("L").crop(tuple(float(value) for value in box))
    padded = ImageOps.pad(
        crop,
        (output_size, output_size),
        method=Image.Resampling.LANCZOS,
        color=0,
    )
    return np.asarray(padded, dtype=np.uint8)


class X3DIRContextPipeline:
    def __init__(
        self,
        *,
        pose_locator: PoseLocator,
        roi_builder: IRPrimaryInputROIBuilder | None = None,
        model: torch.nn.Module | None = None,
        device: torch.device | str = "cpu",
        base_detection_confidence: float = 0.25,
        recovery_detection_confidence: float = 0.01,
    ) -> None:
        self.pose_locator = pose_locator
        self.roi_builder = roi_builder or build_frozen_ir_context_roi_builder()
        self.model = model
        self.device = torch.device(device)
        self.base_detection_confidence = float(base_detection_confidence)
        self.recovery_detection_confidence = float(recovery_detection_confidence)

    def preprocess_trial(
        self,
        raw_ir_paths: Sequence[Path | str],
    ) -> OnlineIRContextPreprocessResult:
        paths = tuple(Path(path) for path in raw_ir_paths)
        if not paths:
            raise ValueError("Raw IR trial must contain at least one frame")
        started = time.perf_counter()
        gray_frames = []
        rgb_frames = []
        shape: tuple[int, int] | None = None
        for path in paths:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Could not decode raw IR frame: {path}")
            if gray.ndim == 3 and gray.shape[2] == 1:
                gray = gray[:, :, 0]
            if gray.ndim != 2:
                raise ValueError(f"Raw IR frame is not grayscale: {path}")
            if shape is None:
                shape = gray.shape
            elif gray.shape != shape:
                raise ValueError("Raw IR frame dimensions changed within trial")
            gray_frames.append(gray)
            rgb_frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        assert shape is not None
        detections = self.pose_locator.predict(
            rgb_frames,
            detection_confidence=self.base_detection_confidence,
        )
        try:
            boxes, bbox_confidences = self._context_boxes(detections, shape)
            recovery_used = False
        except ValueError:
            detections = self.pose_locator.predict(
                rgb_frames,
                detection_confidence=self.recovery_detection_confidence,
            )
            try:
                boxes, bbox_confidences = self._context_boxes(detections, shape)
            except ValueError as error:
                raise ValueError(
                    "Trial has no usable pose evidence; full-frame fallback is forbidden"
                ) from error
            recovery_used = True

        crops = np.stack(
            [crop_ir_context(path, box) for path, box in zip(paths, boxes, strict=True)]
        )
        windows = partition_trial_windows(len(paths))
        clips = []
        source_indices = []
        tensors = [
            torch.from_numpy(crop.copy()).unsqueeze(0).to(torch.float32).div_(255.0)
            for crop in crops
        ]
        for window_index, (start, end) in enumerate(windows):
            indices = stratified_temporal_indices(start, end, training=False)
            source_indices.append(indices)
            clip = _transform_clip_frames(
                [tensors[index] for index in indices.tolist()],
                training=False,
                generator=torch.Generator().manual_seed(window_index),
            )
            clips.append(clip.unsqueeze(0))
        return OnlineIRContextPreprocessResult(
            boxes=boxes,
            crops=crops,
            clips=torch.stack(clips),
            source_indices=torch.stack(source_indices).unsqueeze(1),
            window_bounds=torch.tensor(windows, dtype=torch.long),
            bbox_confidences=bbox_confidences,
            recovery_used=recovery_used,
            pose_roi_seconds=time.perf_counter() - started,
        )

    def predict_trial(
        self,
        raw_ir_paths: Sequence[Path | str],
    ) -> OnlineIRContextPrediction:
        if self.model is None:
            raise ValueError("X3D model is required for trial prediction")
        trial_started = time.perf_counter()
        preprocess = self.preprocess_trial(raw_ir_paths)
        return self.predict_preprocessed(
            preprocess,
            num_frames=len(raw_ir_paths),
            trial_started=trial_started,
        )

    def predict_preprocessed(
        self,
        preprocess: OnlineIRContextPreprocessResult,
        *,
        num_frames: int,
        trial_started: float | None = None,
    ) -> OnlineIRContextPrediction:
        if self.model is None:
            raise ValueError("X3D model is required for trial prediction")
        started = time.perf_counter() if trial_started is None else trial_started
        clips = preprocess.clips.squeeze(1).unsqueeze(0).to(self.device)
        clip_mask = torch.ones((1, clips.shape[1]), dtype=torch.bool, device=self.device)
        unique_fraction = preprocess.source_indices.unique().numel() / preprocess.source_indices.numel()
        quality = torch.tensor(
            [[
                1.0,
                1.0,
                float(np.mean(preprocess.bbox_confidences)),
                unique_fraction,
                1.0,
                min(num_frames / (TARGET_WINDOW_FRAMES * MAX_CLIPS), 1.0),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        x3d_started = time.perf_counter()
        self.model.to(self.device).eval()
        with torch.inference_mode():
            output = self.model(
                clips,
                clip_mask=clip_mask,
                quality=quality,
                quality_mask=torch.ones_like(quality, dtype=torch.bool),
                availability=torch.ones((1, 1), dtype=torch.bool, device=self.device),
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return OnlineIRContextPrediction(
            expert=output,
            preprocess=preprocess,
            x3d_seconds=time.perf_counter() - x3d_started,
            complete_trial_seconds=time.perf_counter() - started,
        )

    def _context_boxes(
        self,
        detections: Sequence[PoseDetection | None],
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = len(detections)
        person_boxes = np.full((frames, 4), np.nan, dtype=np.float32)
        keypoints = np.full((frames, 17, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((frames, 17), dtype=np.float32)
        bbox_confidences = np.zeros(frames, dtype=np.float32)
        for index, detection in enumerate(detections):
            if detection is None:
                continue
            person_boxes[index] = detection.bbox_xyxy
            keypoints[index] = detection.keypoints_xy
            confidence[index] = detection.keypoints_confidence
            bbox_confidences[index] = detection.bbox_confidence
        height, width = shape
        result = self.roi_builder.build(person_boxes, keypoints, confidence, width, height)
        return result.boxes[:, 0].copy(), bbox_confidences
