from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThermalRouteDecision:
    route: str
    bbox_xyxy: tuple[int, int, int, int] | None
    confidence: float
    area_ratio: float
    fallback_reason: str | None


class ThermalContextLocator:
    """Label-free Thermal YOLO geometry gate from the frozen T0.5 audit."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.25,
        area_ratio_threshold: float = 0.05,
        expansion_per_side: float = 0.25,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.area_ratio_threshold = area_ratio_threshold
        self.expansion_per_side = expansion_per_side

    def choose(
        self,
        detection_xyxy_confidence: tuple[float, float, float, float, float] | None,
        *,
        width: int,
        height: int,
    ) -> ThermalRouteDecision:
        if width <= 0 or height <= 0:
            raise ValueError("Thermal dimensions must be positive")
        if detection_xyxy_confidence is None:
            return ThermalRouteDecision("full_frame", None, 0.0, 0.0, "no_detection")
        x1, y1, x2, y2, confidence = map(float, detection_xyxy_confidence)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
            return ThermalRouteDecision("full_frame", None, confidence, 0.0, "invalid_bbox")
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return ThermalRouteDecision("full_frame", None, confidence, 0.0, "invalid_bbox")
        area_ratio = box_width * box_height / (width * height)
        if confidence < self.confidence_threshold:
            return ThermalRouteDecision("full_frame", None, confidence, area_ratio, "low_confidence")
        if area_ratio < self.area_ratio_threshold:
            return ThermalRouteDecision("full_frame", None, confidence, area_ratio, "small_bbox")
        expanded = (
            max(0, math.floor(x1 - box_width * self.expansion_per_side)),
            max(0, math.floor(y1 - box_height * self.expansion_per_side)),
            min(width, math.ceil(x2 + box_width * self.expansion_per_side)),
            min(height, math.ceil(y2 + box_height * self.expansion_per_side)),
        )
        if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
            return ThermalRouteDecision("full_frame", None, confidence, area_ratio, "invalid_expanded_bbox")
        return ThermalRouteDecision(
            "thermal_yolo_context", expanded, confidence, area_ratio, None
        )
