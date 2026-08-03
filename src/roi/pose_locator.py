from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ultralytics import YOLO


COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class PoseDetection:
    bbox_xyxy: np.ndarray
    bbox_confidence: float
    keypoints_xy: np.ndarray
    keypoints_confidence: np.ndarray


class UltralyticsPoseLocator:
    def __init__(
        self,
        weights: Path | str,
        device: int | str = 0,
        image_size: int = 640,
        detection_confidence: float = 0.25,
    ) -> None:
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"Pose weights not found: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.device = device
        self.image_size = image_size
        self.detection_confidence = detection_confidence

    def predict(self, images: Sequence[np.ndarray]) -> list[PoseDetection | None]:
        results = self.model.predict(
            list(images),
            device=self.device,
            imgsz=self.image_size,
            conf=self.detection_confidence,
            verbose=False,
        )
        detections: list[PoseDetection | None] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
                detections.append(None)
                continue
            confidences = result.boxes.conf.detach().float().cpu().numpy()
            best = int(confidences.argmax())
            keypoints = result.keypoints.data[best].detach().float().cpu().numpy()
            if keypoints.shape != (17, 3):
                raise ValueError(f"Expected 17x3 COCO keypoints, got {keypoints.shape}")
            detections.append(
                PoseDetection(
                    bbox_xyxy=result.boxes.xyxy[best].detach().float().cpu().numpy(),
                    bbox_confidence=float(confidences[best]),
                    keypoints_xy=keypoints[:, :2],
                    keypoints_confidence=keypoints[:, 2],
                )
            )
        return detections
