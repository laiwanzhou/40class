from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VIEW_NAMES = (
    "global_context",
    "left_interaction",
    "right_interaction",
    "hand_head_union",
    "two_hand_table_context",
)


@dataclass(frozen=True)
class ObjectInteractionROIResult:
    boxes: np.ndarray
    raw_boxes: np.ndarray
    valid_mask: np.ndarray
    keypoint_valid: np.ndarray
    sources: np.ndarray
    projected_points: np.ndarray
    directional_iou: np.ndarray
    old_tight_iou: np.ndarray
    projected_out_of_bounds: np.ndarray
    wrist_edge_distance: np.ndarray
    two_hand_merge: np.ndarray
    hand_head_trigger: np.ndarray


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-6)


def _finite_valid(points: np.ndarray, confidence: np.ndarray, threshold: float) -> np.ndarray:
    return (confidence >= threshold) & np.isfinite(points).all(axis=-1)


def _clip_raw_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = float(np.clip(x1, 0, width - 2))
    y1 = float(np.clip(y1, 0, height - 2))
    x2 = float(np.clip(x2, x1 + 2, width))
    y2 = float(np.clip(y2, y1 + 2, height))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _square_inside(box: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = _clip_raw_box(box, width, height)
    center = np.asarray([(clipped[0] + clipped[2]) / 2, (clipped[1] + clipped[3]) / 2])
    side = min(max(float(clipped[2] - clipped[0]), float(clipped[3] - clipped[1]), 16.0), width, height)
    x1 = float(np.clip(center[0] - side / 2, 0, width - side))
    y1 = float(np.clip(center[1] - side / 2, 0, height - side))
    return np.asarray([x1, y1, x1 + side, y1 + side], dtype=np.float32)


def _points_box(points: np.ndarray, padding_x: float, padding_top: float, padding_bottom: float) -> np.ndarray:
    low = points.min(axis=0)
    high = points.max(axis=0)
    return np.asarray(
        [low[0] - padding_x, low[1] - padding_top, high[0] + padding_x, high[1] + padding_bottom],
        dtype=np.float32,
    )


def _directional_box(
    elbow: np.ndarray,
    wrist: np.ndarray,
    shoulder_width: float,
    alpha: float,
    long_axis_scale: float,
    short_axis_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    direction = wrist - elbow
    forearm = max(float(np.linalg.norm(direction)), 1.0)
    unit = direction / forearm
    perpendicular = np.asarray([-unit[1], unit[0]])
    projected = wrist + alpha * direction
    long_margin = max(long_axis_scale * forearm, 0.15 * shoulder_width)
    short_margin = max(short_axis_scale * forearm, 0.10 * shoulder_width)
    start = elbow - 0.20 * long_margin * unit
    end = projected + long_margin * unit
    corners = np.stack(
        (start + short_margin * perpendicular, start - short_margin * perpendicular,
         end + short_margin * perpendicular, end - short_margin * perpendicular)
    )
    return _points_box(corners, 0.0, 0.0, 0.0), projected.astype(np.float32)


class ObjectInteractionROIBuilder:
    def __init__(
        self,
        keypoint_threshold: float = 0.25,
        projection_alpha: float = 0.4,
        long_axis_scale: float = 0.7,
        short_axis_scale: float = 0.45,
        wrist_fallback_person_width: float = 0.35,
        minimum_pixel_side: float = 48.0,
        merge_iou_threshold: float = 0.30,
        merge_projected_distance: float = 0.60,
        merge_wrist_distance: float = 0.70,
        merge_elbow_distance: float = 0.80,
        merge_wide_wrist_distance: float = 1.20,
        two_hand_horizontal_padding: float = 0.25,
        two_hand_top_padding: float = 0.10,
        two_hand_bottom_padding: float = 0.35,
        hand_head_distance: float = 0.90,
    ) -> None:
        self.keypoint_threshold = keypoint_threshold
        self.projection_alpha = projection_alpha
        self.long_axis_scale = long_axis_scale
        self.short_axis_scale = short_axis_scale
        self.wrist_fallback_person_width = wrist_fallback_person_width
        self.minimum_pixel_side = minimum_pixel_side
        self.merge_iou_threshold = merge_iou_threshold
        self.merge_projected_distance = merge_projected_distance
        self.merge_wrist_distance = merge_wrist_distance
        self.merge_elbow_distance = merge_elbow_distance
        self.merge_wide_wrist_distance = merge_wide_wrist_distance
        self.two_hand_horizontal_padding = two_hand_horizontal_padding
        self.two_hand_top_padding = two_hand_top_padding
        self.two_hand_bottom_padding = two_hand_bottom_padding
        self.hand_head_distance = hand_head_distance

    def build(
        self,
        person_boxes: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_confidence: np.ndarray,
        width: int,
        height: int,
    ) -> ObjectInteractionROIResult:
        frames = len(person_boxes)
        kp_valid = _finite_valid(keypoints_xy, keypoints_confidence, self.keypoint_threshold)
        bbox_valid = np.isfinite(person_boxes).all(axis=1)
        boxes = np.zeros((frames, 5, 4), dtype=np.float32)
        raw_boxes = np.full((frames, 5, 4), np.nan, dtype=np.float32)
        valid = np.zeros((frames, 5), dtype=bool)
        sources = np.full((frames, 5), "invalid", dtype="U24")
        projected = np.full((frames, 2, 2), np.nan, dtype=np.float32)
        directional_iou = np.full(frames, np.nan, dtype=np.float32)
        old_tight_iou = np.full(frames, np.nan, dtype=np.float32)
        projected_oob = np.zeros((frames, 2), dtype=bool)
        wrist_edge_distance = np.full((frames, 2), np.nan, dtype=np.float32)
        two_hand_merge = np.zeros(frames, dtype=bool)
        hand_head_trigger = np.zeros(frames, dtype=bool)

        for frame in range(frames):
            full = np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float32)
            boxes[frame, 0] = full
            raw_boxes[frame, 0] = full
            valid[frame, 0] = True
            sources[frame, 0] = "global"
            person = person_boxes[frame] if bbox_valid[frame] else full
            person_width = max(float(person[2] - person[0]), width * 0.20)
            if kp_valid[frame, 5] and kp_valid[frame, 6]:
                shoulder_width = max(float(np.linalg.norm(keypoints_xy[frame, 5] - keypoints_xy[frame, 6])), 1.0)
            else:
                shoulder_width = max(0.45 * person_width, 1.0)

            arm_data: list[dict[str, object]] = []
            for side, (elbow_index, wrist_index, view_index) in enumerate(((7, 9, 1), (8, 10, 2))):
                elbow_ok = bool(kp_valid[frame, elbow_index])
                wrist_ok = bool(kp_valid[frame, wrist_index])
                entry: dict[str, object] = {"elbow_ok": elbow_ok, "wrist_ok": wrist_ok, "view": view_index}
                if elbow_ok and wrist_ok:
                    raw, point = _directional_box(
                        keypoints_xy[frame, elbow_index], keypoints_xy[frame, wrist_index], shoulder_width,
                        self.projection_alpha, self.long_axis_scale, self.short_axis_scale,
                    )
                    projected[frame, side] = point
                    projected_oob[frame, side] = not (0 <= point[0] <= width and 0 <= point[1] <= height)
                    sources[frame, view_index] = "directional"
                    entry.update({"raw": raw, "projected": point})
                elif wrist_ok:
                    wrist = keypoints_xy[frame, wrist_index]
                    side_length = max(self.wrist_fallback_person_width * person_width, self.minimum_pixel_side)
                    raw = np.asarray(
                        [wrist[0] - side_length / 2, wrist[1] - side_length / 2,
                         wrist[0] + side_length / 2, wrist[1] + side_length / 2], dtype=np.float32,
                    )
                    sources[frame, view_index] = "wrist_fallback"
                    entry.update({"raw": raw})
                else:
                    arm_data.append(entry)
                    continue
                raw_boxes[frame, view_index] = raw
                boxes[frame, view_index] = _square_inside(raw, width, height)
                valid[frame, view_index] = True
                wrist = keypoints_xy[frame, wrist_index]
                box = boxes[frame, view_index]
                distances = np.asarray([wrist[0] - box[0], box[2] - wrist[0], wrist[1] - box[1], box[3] - wrist[1]])
                wrist_edge_distance[frame, side] = float(distances.min() / max(box[2] - box[0], 1.0))
                arm_data.append(entry)

            left, right = arm_data
            if bool(left["wrist_ok"]) and bool(right["wrist_ok"]):
                tight_side = max(0.45 * person_width, self.minimum_pixel_side)
                tight_boxes = []
                for wrist_index in (9, 10):
                    wrist = keypoints_xy[frame, wrist_index]
                    tight_boxes.append(_square_inside(np.r_[wrist - tight_side / 2, wrist + tight_side / 2], width, height))
                old_tight_iou[frame] = box_iou(tight_boxes[0], tight_boxes[1])

            both_directional = all(bool(item["elbow_ok"]) and bool(item["wrist_ok"]) for item in arm_data)
            if both_directional:
                left_raw = np.asarray(left["raw"])
                right_raw = np.asarray(right["raw"])
                directional_iou[frame] = box_iou(left_raw, right_raw)
                left_projected = np.asarray(left["projected"])
                right_projected = np.asarray(right["projected"])
                wrist_distance = float(np.linalg.norm(keypoints_xy[frame, 9] - keypoints_xy[frame, 10]))
                elbow_distance = float(np.linalg.norm(keypoints_xy[frame, 7] - keypoints_xy[frame, 8]))
                projected_distance = float(np.linalg.norm(left_projected - right_projected))
                merge = (
                    directional_iou[frame] > self.merge_iou_threshold
                    or projected_distance < self.merge_projected_distance * shoulder_width
                    or wrist_distance < self.merge_wrist_distance * shoulder_width
                    or (
                        elbow_distance < self.merge_elbow_distance * shoulder_width
                        and wrist_distance < self.merge_wide_wrist_distance * shoulder_width
                    )
                )
                if merge:
                    points = np.stack(
                        (keypoints_xy[frame, 7], keypoints_xy[frame, 9], left_projected,
                         keypoints_xy[frame, 8], keypoints_xy[frame, 10], right_projected)
                    )
                    raw = _points_box(
                        points,
                        self.two_hand_horizontal_padding * shoulder_width,
                        self.two_hand_top_padding * shoulder_width,
                        self.two_hand_bottom_padding * shoulder_width,
                    )
                    raw_boxes[frame, 4] = raw
                    boxes[frame, 4] = _square_inside(raw, width, height)
                    valid[frame, 4] = True
                    sources[frame, 4] = "two_hand_merge"
                    two_hand_merge[frame] = True

            head_indices = [index for index in range(5) if kp_valid[frame, index]]
            nearby_sides: list[int] = []
            if head_indices:
                head_points = keypoints_xy[frame, head_indices]
                head_center = head_points.mean(axis=0)
                for side, wrist_index in enumerate((9, 10)):
                    if valid[frame, side + 1]:
                        interaction_center = np.asarray(
                            [(boxes[frame, side + 1, 0] + boxes[frame, side + 1, 2]) / 2,
                             (boxes[frame, side + 1, 1] + boxes[frame, side + 1, 3]) / 2]
                        )
                        if np.linalg.norm(interaction_center - head_center) < self.hand_head_distance * shoulder_width:
                            nearby_sides.append(side)
                if nearby_sides:
                    points = [*head_points]
                    for side in nearby_sides:
                        elbow_index, wrist_index = ((7, 9), (8, 10))[side]
                        if kp_valid[frame, elbow_index]:
                            points.append(keypoints_xy[frame, elbow_index])
                        points.append(keypoints_xy[frame, wrist_index])
                        if np.isfinite(projected[frame, side]).all():
                            points.append(projected[frame, side])
                    raw = _points_box(np.asarray(points), 0.15 * shoulder_width, 0.15 * shoulder_width, 0.20 * shoulder_width)
                    raw_boxes[frame, 3] = raw
                    boxes[frame, 3] = _square_inside(raw, width, height)
                    valid[frame, 3] = True
                    sources[frame, 3] = "hand_head_union"
                    hand_head_trigger[frame] = True

        return ObjectInteractionROIResult(
            boxes=boxes,
            raw_boxes=raw_boxes,
            valid_mask=valid,
            keypoint_valid=kp_valid,
            sources=sources,
            projected_points=projected,
            directional_iou=directional_iou,
            old_tight_iou=old_tight_iou,
            projected_out_of_bounds=projected_oob,
            wrist_edge_distance=wrist_edge_distance,
            two_hand_merge=two_hand_merge,
            hand_head_trigger=hand_head_trigger,
        )
