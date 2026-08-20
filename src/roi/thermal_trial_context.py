from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
import statistics

from src.data.thermal_v2_sampling import normalized_probe_indices


Detection = tuple[float, float, float, float, float]


@dataclass(frozen=True)
class ThermalTrialContext:
    sample_id: str
    available: bool
    probe_indices: tuple[int, ...]
    accepted_indices: tuple[int, ...]
    accepted_detections: tuple[Detection, ...]
    bbox_xyxy: tuple[int, int, int, int] | None
    detection_hit_ratio: float
    median_confidence: float
    bbox_area_ratio: float
    fallback_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _valid_detection(
    detection: Sequence[float],
    *,
    frame_size: tuple[int, int],
    confidence_threshold: float,
) -> Detection | None:
    if len(detection) != 5:
        return None
    x1, y1, x2, y2, confidence = (float(value) for value in detection)
    width, height = frame_size
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
        return None
    if confidence < confidence_threshold or x2 <= x1 or y2 <= y1:
        return None
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
        confidence,
    )


def _square_context_box(
    detections: Sequence[Detection],
    *,
    frame_size: tuple[int, int],
    expansion: float,
    minimum_side_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = frame_size
    union_x1 = min(row[0] for row in detections)
    union_y1 = min(row[1] for row in detections)
    union_x2 = max(row[2] for row in detections)
    union_y2 = max(row[3] for row in detections)
    center_x = (union_x1 + union_x2) / 2.0
    center_y = (union_y1 + union_y2) / 2.0
    side = max(
        (union_x2 - union_x1) * expansion,
        (union_y2 - union_y1) * expansion,
        minimum_side_ratio * max(width, height),
    )
    side_i = min(int(math.ceil(side)), min(width, height))
    left = min(max(int(math.floor(center_x - side_i / 2.0 + 0.5)), 0), width - side_i)
    top = min(max(int(math.floor(center_y - side_i / 2.0 + 0.5)), 0), height - side_i)
    return left, top, left + side_i, top + side_i


def build_trial_context(
    *,
    sample_id: str,
    frame_size: tuple[int, int],
    frame_count: int,
    detections_by_index: Mapping[int, Sequence[Sequence[float]]],
    confidence_threshold: float = 0.25,
    expansion: float = 1.4,
    minimum_side_ratio: float = 0.35,
) -> ThermalTrialContext:
    width, height = frame_size
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if width < 1 or height < 1:
        raise ValueError("frame dimensions must be positive")
    if expansion < 1.0:
        raise ValueError("expansion must be at least 1")
    if not (0.0 < minimum_side_ratio <= 1.0):
        raise ValueError("minimum_side_ratio must be in (0, 1]")

    probe_indices = normalized_probe_indices(frame_count)
    accepted_indices: list[int] = []
    accepted: list[Detection] = []
    for index in probe_indices:
        candidates = [
            valid
            for detection in detections_by_index.get(index, ())
            if (
                valid := _valid_detection(
                    detection,
                    frame_size=frame_size,
                    confidence_threshold=confidence_threshold,
                )
            )
            is not None
        ]
        if candidates:
            accepted_indices.append(index)
            accepted.append(max(candidates, key=lambda row: row[-1]))

    hit_ratio = len(accepted) / len(probe_indices)
    median_confidence = statistics.median(row[-1] for row in accepted) if accepted else 0.0
    required_hits = 1 if len(probe_indices) == 1 else 2
    if len(accepted) < required_hits:
        reason = "no_detection" if not accepted else "insufficient_detection_hits"
        return ThermalTrialContext(
            sample_id=sample_id,
            available=False,
            probe_indices=probe_indices,
            accepted_indices=tuple(accepted_indices),
            accepted_detections=tuple(accepted),
            bbox_xyxy=None,
            detection_hit_ratio=hit_ratio,
            median_confidence=median_confidence,
            bbox_area_ratio=0.0,
            fallback_reason=reason,
        )

    bbox = _square_context_box(
        accepted,
        frame_size=frame_size,
        expansion=expansion,
        minimum_side_ratio=minimum_side_ratio,
    )
    area_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / width / height
    return ThermalTrialContext(
        sample_id=sample_id,
        available=True,
        probe_indices=probe_indices,
        accepted_indices=tuple(accepted_indices),
        accepted_detections=tuple(accepted),
        bbox_xyxy=bbox,
        detection_hit_ratio=hit_ratio,
        median_confidence=median_confidence,
        bbox_area_ratio=area_ratio,
        fallback_reason=None,
    )
