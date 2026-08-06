from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.roi.object_interaction_builder import ObjectInteractionROIBuilder, ObjectInteractionROIResult


PERSON_VIEW_NAMES = (
    "person_context",
    "left_interaction",
    "right_interaction",
    "hand_head_union",
    "two_hand_table_context",
)


@dataclass(frozen=True)
class PersonInteractionCropResult:
    person_boxes: np.ndarray
    person_raw_boxes: np.ndarray
    person_sources: np.ndarray
    person_detected: np.ndarray
    person_interpolated: np.ndarray
    person_nearest_held: np.ndarray
    person_touches_boundary: np.ndarray
    detector_boxes: np.ndarray
    keypoints_xy: np.ndarray
    keypoint_valid: np.ndarray
    local_boxes_original: np.ndarray
    local_boxes_person: np.ndarray
    valid_mask: np.ndarray
    interaction: ObjectInteractionROIResult


def _box_valid(boxes: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(boxes).all(axis=-1)
        & (boxes[..., 2] > boxes[..., 0])
        & (boxes[..., 3] > boxes[..., 1])
    )


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = float(np.clip(x1, 0.0, width - 2.0))
    y1 = float(np.clip(y1, 0.0, height - 2.0))
    x2 = float(np.clip(x2, x1 + 2.0, width))
    y2 = float(np.clip(y2, y1 + 2.0, height))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _union_boxes(boxes: list[np.ndarray]) -> np.ndarray:
    values = np.stack(boxes)
    return np.asarray(
        [values[:, 0].min(), values[:, 1].min(), values[:, 2].max(), values[:, 3].max()],
        dtype=np.float32,
    )


def _interpolate_track(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not valid.any():
        raise ValueError("A trial has no valid person evidence; full-frame fallback is forbidden")
    indices = np.arange(len(values), dtype=np.float32)
    valid_indices = indices[valid]
    filled = values.copy()
    for column in range(values.shape[1]):
        filled[:, column] = np.interp(indices, valid_indices, values[valid, column])
    missing = ~valid
    interpolated = missing & (indices > valid_indices[0]) & (indices < valid_indices[-1])
    nearest = missing & ~interpolated
    return filled, interpolated, nearest


def _smooth_centers(centers: np.ndarray, radius: int, ema_alpha: float) -> np.ndarray:
    if radius < 0:
        raise ValueError("smoothing_radius must be non-negative")
    median = np.empty_like(centers)
    for index in range(len(centers)):
        start = max(0, index - radius)
        end = min(len(centers), index + radius + 1)
        median[index] = np.median(centers[start:end], axis=0)
    forward = median.copy()
    for index in range(1, len(forward)):
        forward[index] = ema_alpha * median[index] + (1.0 - ema_alpha) * forward[index - 1]
    backward = median.copy()
    for index in range(len(backward) - 2, -1, -1):
        backward[index] = ema_alpha * median[index] + (1.0 - ema_alpha) * backward[index + 1]
    return (forward + backward) / 2.0


def _fixed_box(center: np.ndarray, box_width: float, box_height: float, width: int, height: int) -> np.ndarray:
    box_width = min(max(box_width, 2.0), float(width))
    box_height = min(max(box_height, 2.0), float(height))
    x1 = float(np.clip(center[0] - box_width / 2.0, 0.0, width - box_width))
    y1 = float(np.clip(center[1] - box_height / 2.0, 0.0, height - box_height))
    return np.asarray([x1, y1, x1 + box_width, y1 + box_height], dtype=np.float32)


class PersonInteractionCropBuilder:
    def __init__(
        self,
        interaction_config: dict[str, float] | None = None,
        keypoint_threshold: float = 0.25,
        padding_ratio: float = 0.08,
        minimum_width_ratio: float = 0.30,
        minimum_height_ratio: float = 0.45,
        maximum_width_ratio: float = 0.96,
        maximum_height_ratio: float = 0.96,
        smoothing_radius: int = 2,
        smoothing_ema_alpha: float = 0.45,
    ) -> None:
        self.keypoint_threshold = float(keypoint_threshold)
        self.padding_ratio = float(padding_ratio)
        self.minimum_width_ratio = float(minimum_width_ratio)
        self.minimum_height_ratio = float(minimum_height_ratio)
        self.maximum_width_ratio = float(maximum_width_ratio)
        self.maximum_height_ratio = float(maximum_height_ratio)
        self.smoothing_radius = int(smoothing_radius)
        self.smoothing_ema_alpha = float(smoothing_ema_alpha)
        self.interaction_builder = ObjectInteractionROIBuilder(**(interaction_config or {}))

    def build(
        self,
        person_boxes: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_confidence: np.ndarray,
        width: int,
        height: int,
    ) -> PersonInteractionCropResult:
        interaction = self.interaction_builder.build(
            person_boxes, keypoints_xy, keypoints_confidence, width, height
        )
        detected = _box_valid(person_boxes)
        keypoint_valid = (
            (keypoints_confidence >= self.keypoint_threshold)
            & np.isfinite(keypoints_xy).all(axis=-1)
        )
        raw = np.full((len(person_boxes), 4), np.nan, dtype=np.float32)
        evidence_valid = np.zeros(len(person_boxes), dtype=bool)

        for frame in range(len(person_boxes)):
            evidence: list[np.ndarray] = []
            if detected[frame]:
                evidence.append(np.asarray(person_boxes[frame], dtype=np.float32))
            points = keypoints_xy[frame, keypoint_valid[frame]]
            if len(points):
                evidence.append(
                    np.asarray(
                        [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
                        dtype=np.float32,
                    )
                )
            for view in range(1, len(PERSON_VIEW_NAMES)):
                candidate = interaction.raw_boxes[frame, view]
                if interaction.valid_mask[frame, view] and _box_valid(candidate[None, :])[0]:
                    evidence.append(np.asarray(candidate, dtype=np.float32))
            if not evidence:
                continue
            union = _union_boxes(evidence)
            pad = self.padding_ratio * max(union[2] - union[0], union[3] - union[1])
            raw[frame] = np.asarray(
                [union[0] - pad, union[1] - pad, union[2] + pad, union[3] + pad], dtype=np.float32
            )
            evidence_valid[frame] = True

        filled, interpolated, nearest = _interpolate_track(raw, evidence_valid)
        centers = np.stack(((filled[:, 0] + filled[:, 2]) / 2.0, (filled[:, 1] + filled[:, 3]) / 2.0), axis=1)
        centers = _smooth_centers(centers, self.smoothing_radius, self.smoothing_ema_alpha)

        required_half_width = np.maximum(centers[:, 0] - filled[:, 0], filled[:, 2] - centers[:, 0])
        required_half_height = np.maximum(centers[:, 1] - filled[:, 1], filled[:, 3] - centers[:, 1])
        stable_width = float(np.clip(
            2.0 * required_half_width.max(), self.minimum_width_ratio * width, self.maximum_width_ratio * width
        ))
        stable_height = float(np.clip(
            2.0 * required_half_height.max(), self.minimum_height_ratio * height, self.maximum_height_ratio * height
        ))
        boxes = np.stack([_fixed_box(center, stable_width, stable_height, width, height) for center in centers])
        touches = (
            np.isclose(boxes[:, 0], 0.0)
            | np.isclose(boxes[:, 1], 0.0)
            | np.isclose(boxes[:, 2], float(width))
            | np.isclose(boxes[:, 3], float(height))
        )
        sources = np.full(len(boxes), "detected", dtype="U20")
        sources[interpolated] = "interpolated"
        sources[nearest] = "nearest_hold"

        local_original = np.zeros((len(boxes), len(PERSON_VIEW_NAMES), 4), dtype=np.float32)
        local_person = np.zeros_like(local_original)
        valid_mask = interaction.valid_mask.copy()
        valid_mask[:, 0] = True
        for frame, person in enumerate(boxes):
            local_original[frame, 0] = person
            local_person[frame, 0] = np.asarray([0.0, 0.0, person[2] - person[0], person[3] - person[1]])
            for view in range(1, len(PERSON_VIEW_NAMES)):
                if not valid_mask[frame, view]:
                    continue
                original = _clip_box(interaction.raw_boxes[frame, view], width, height)
                intersection = np.asarray(
                    [max(original[0], person[0]), max(original[1], person[1]),
                     min(original[2], person[2]), min(original[3], person[3])], dtype=np.float32
                )
                if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
                    valid_mask[frame, view] = False
                    continue
                local_original[frame, view] = intersection
                local_person[frame, view] = intersection - np.asarray(
                    [person[0], person[1], person[0], person[1]], dtype=np.float32
                )

        return PersonInteractionCropResult(
            person_boxes=boxes.astype(np.float32),
            person_raw_boxes=raw,
            person_sources=sources,
            person_detected=detected,
            person_interpolated=interpolated,
            person_nearest_held=nearest,
            person_touches_boundary=touches,
            detector_boxes=np.asarray(person_boxes, dtype=np.float32),
            keypoints_xy=np.asarray(keypoints_xy, dtype=np.float32),
            keypoint_valid=keypoint_valid,
            local_boxes_original=local_original,
            local_boxes_person=local_person,
            valid_mask=valid_mask,
            interaction=interaction,
        )
