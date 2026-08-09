from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.roi.object_interaction_builder import ObjectInteractionROIBuilder, box_iou


VIEW_NAMES = (
    "person_context",
    "left_directional_interaction",
    "right_directional_interaction",
    "adaptive_relation",
)


@dataclass(frozen=True)
class IRPrimaryInputROIResult:
    boxes: np.ndarray
    valid_mask: np.ndarray
    sources: np.ndarray
    context_raw_boxes: np.ndarray
    context_touches_boundary: np.ndarray
    left_right_iou: np.ndarray
    interaction: object


def _valid_box(box: np.ndarray) -> bool:
    return bool(
        np.isfinite(box).all()
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _union(boxes: list[np.ndarray]) -> np.ndarray:
    values = np.stack(boxes)
    return np.asarray(
        [values[:, 0].min(), values[:, 1].min(), values[:, 2].max(), values[:, 3].max()],
        dtype=np.float32,
    )


def _square_inside(box: np.ndarray, width: int, height: int, padding: float = 0.0) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width = max(x2 - x1, 2.0)
    box_height = max(y2 - y1, 2.0)
    side = max(box_width, box_height) * (1.0 + 2.0 * padding)
    side = min(max(side, 16.0), float(width), float(height))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = float(np.clip(center_x - side / 2.0, 0.0, width - side))
    top = float(np.clip(center_y - side / 2.0, 0.0, height - side))
    return np.asarray([left, top, left + side, top + side], dtype=np.float32)


def _box_inside(center: np.ndarray, box_width: float, box_height: float, width: int, height: int) -> np.ndarray:
    box_width = float(np.clip(box_width, 16.0, width))
    box_height = float(np.clip(box_height, 16.0, height))
    left = float(np.clip(center[0] - box_width / 2.0, 0.0, width - box_width))
    top = float(np.clip(center[1] - box_height / 2.0, 0.0, height - box_height))
    return np.asarray([left, top, left + box_width, top + box_height], dtype=np.float32)


def _smooth(values: np.ndarray, radius: int = 2, alpha: float = 0.45) -> np.ndarray:
    median = np.empty_like(values)
    for index in range(len(values)):
        median[index] = np.median(
            values[max(0, index - radius) : min(len(values), index + radius + 1)], axis=0,
        )
    forward = median.copy()
    backward = median.copy()
    for index in range(1, len(values)):
        forward[index] = alpha * median[index] + (1.0 - alpha) * forward[index - 1]
    for index in range(len(values) - 2, -1, -1):
        backward[index] = alpha * median[index] + (1.0 - alpha) * backward[index + 1]
    return (forward + backward) / 2.0


def _fill_boxes(boxes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        raise ValueError("Cannot stabilize a box track without valid evidence")
    centers = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0]
    sizes = np.c_[boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]
    values = np.c_[centers, sizes]
    indices = np.arange(len(values), dtype=np.float32)
    valid_indices = indices[valid]
    filled = values.copy()
    for column in range(values.shape[1]):
        filled[:, column] = np.interp(indices, valid_indices, values[valid, column])
    return _smooth(filled)


class IRPrimaryInputROIBuilder:
    """Build IR appearance and Depth geometry crops without a full-frame view."""

    def __init__(
        self,
        keypoint_threshold: float = 0.25,
        context_padding: float = 0.22,
        context_quantile: float = 0.95,
        minimum_context_side_ratio: float = 0.45,
        local_padding: float = 0.12,
        duplicate_iou_threshold: float = 0.70,
        interaction_config: dict[str, float] | None = None,
    ) -> None:
        self.keypoint_threshold = float(keypoint_threshold)
        self.context_padding = float(context_padding)
        self.context_quantile = float(context_quantile)
        self.minimum_context_side_ratio = float(minimum_context_side_ratio)
        self.local_padding = float(local_padding)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.interaction_builder = ObjectInteractionROIBuilder(**(interaction_config or {}))

    def build(
        self,
        person_boxes: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_confidence: np.ndarray,
        width: int,
        height: int,
    ) -> IRPrimaryInputROIResult:
        interaction = self.interaction_builder.build(
            person_boxes, keypoints_xy, keypoints_confidence, width, height,
        )
        frames = len(person_boxes)
        boxes = np.zeros((frames, len(VIEW_NAMES), 4), dtype=np.float32)
        valid = np.zeros((frames, len(VIEW_NAMES)), dtype=bool)
        sources = np.full((frames, len(VIEW_NAMES)), "invalid", dtype="U32")

        keypoint_valid = (
            (keypoints_confidence >= self.keypoint_threshold)
            & np.isfinite(keypoints_xy).all(axis=-1)
        )
        raw_context = np.full((frames, 4), np.nan, dtype=np.float32)
        context_valid = np.zeros(frames, dtype=bool)
        for frame in range(frames):
            evidence: list[np.ndarray] = []
            if _valid_box(person_boxes[frame]):
                evidence.append(np.asarray(person_boxes[frame], dtype=np.float32))
            points = keypoints_xy[frame, keypoint_valid[frame]]
            if len(points):
                evidence.append(np.r_[points.min(axis=0), points.max(axis=0)].astype(np.float32))
            for view in range(1, interaction.boxes.shape[1]):
                raw = interaction.raw_boxes[frame, view]
                if interaction.valid_mask[frame, view] and _valid_box(raw):
                    evidence.append(np.asarray(raw, dtype=np.float32))
            if evidence:
                raw_context[frame] = _union(evidence)
                context_valid[frame] = True
        context_track = _fill_boxes(raw_context, context_valid)
        centers = context_track[:, :2]
        required_half = np.maximum(
            centers - raw_context[:, :2], raw_context[:, 2:] - centers,
        )
        required_half[~context_valid] = np.nan
        robust_side = 2.0 * float(np.nanquantile(required_half, self.context_quantile))
        robust_side *= 1.0 + 2.0 * self.context_padding
        robust_side = max(robust_side, self.minimum_context_side_ratio * min(width, height))
        for frame in range(frames):
            raw = raw_context[frame] if context_valid[frame] else np.r_[centers[frame], centers[frame]]
            required = 2.0 * np.maximum(centers[frame] - raw[:2], raw[2:] - centers[frame])
            required *= 1.0 + 2.0 * self.context_padding
            context = _box_inside(
                centers[frame],
                max(robust_side, float(required[0])),
                max(robust_side, float(required[1])),
                width,
                height,
            )
            boxes[frame, 0] = context
            valid[frame, 0] = True
            sources[frame, 0] = "robust_person_interaction_context"

        for frame in range(frames):
            for output_view, interaction_view in ((1, 1), (2, 2)):
                if interaction.valid_mask[frame, interaction_view]:
                    boxes[frame, output_view] = _square_inside(
                        interaction.raw_boxes[frame, interaction_view], width, height, self.local_padding,
                    )
                    valid[frame, output_view] = True
                    sources[frame, output_view] = str(interaction.sources[frame, interaction_view])

            if interaction.valid_mask[frame, 3]:
                adaptive_raw = interaction.raw_boxes[frame, 3]
                adaptive_source = "hand_head_union"
            elif interaction.valid_mask[frame, 4]:
                adaptive_raw = interaction.raw_boxes[frame, 4]
                adaptive_source = "two_hand_table_context"
            elif valid[frame, 1] and valid[frame, 2]:
                adaptive_raw = _union([boxes[frame, 1], boxes[frame, 2]])
                adaptive_source = "two_hand_relation"
            elif valid[frame, 1] or valid[frame, 2]:
                selected = 1 if valid[frame, 1] else 2
                adaptive_raw = boxes[frame, selected]
                adaptive_source = "single_hand_context"
            else:
                adaptive_raw = None
                adaptive_source = "invalid"
            if adaptive_raw is not None:
                boxes[frame, 3] = _square_inside(adaptive_raw, width, height, self.local_padding)
                valid[frame, 3] = True
                sources[frame, 3] = adaptive_source

        left_right_iou = np.full(frames, np.nan, dtype=np.float32)
        for frame in range(frames):
            if valid[frame, 1] and valid[frame, 2]:
                left_right_iou[frame] = box_iou(boxes[frame, 1], boxes[frame, 2])
                if left_right_iou[frame] >= self.duplicate_iou_threshold:
                    valid[frame, 2] = False
                    sources[frame, 2] = "duplicate_suppressed"

        touches = (
            np.isclose(boxes[:, 0, 0], 0.0)
            | np.isclose(boxes[:, 0, 1], 0.0)
            | np.isclose(boxes[:, 0, 2], float(width))
            | np.isclose(boxes[:, 0, 3], float(height))
        )
        return IRPrimaryInputROIResult(
            boxes=boxes,
            valid_mask=valid,
            sources=sources,
            context_raw_boxes=raw_context,
            context_touches_boundary=touches,
            left_right_iou=left_right_iou,
            interaction=interaction,
        )
