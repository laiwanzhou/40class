from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ROIBuildResult:
    boxes: np.ndarray
    sources: np.ndarray


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float64)
    x1 = float(np.clip(x1, 0, width - 2))
    y1 = float(np.clip(y1, 0, height - 2))
    x2 = float(np.clip(x2, x1 + 2, width))
    y2 = float(np.clip(y2, y1 + 2, height))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _box_from_points(points: np.ndarray, margin: float, width: int, height: int) -> np.ndarray:
    low = points.min(axis=0)
    high = points.max(axis=0)
    size = np.maximum(high - low, 16.0)
    return _clip_box(np.r_[low - size * margin, high + size * margin], width, height)


def _square(center: np.ndarray, side: float, width: int, height: int) -> np.ndarray:
    half = max(side, 16.0) / 2.0
    return _clip_box(np.asarray([center[0] - half, center[1] - half, center[0] + half, center[1] + half]), width, height)


def _interpolate_short_gaps(values: np.ndarray, valid: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    output = values.astype(np.float64, copy=True)
    interpolated = np.zeros(len(values), dtype=bool)
    index = 0
    while index < len(values):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(values) and not valid[index]:
            index += 1
        end = index
        gap = end - start
        left = start - 1
        right = end
        if gap <= max_gap and left >= 0 and right < len(values) and valid[left] and valid[right]:
            for position in range(start, end):
                fraction = (position - left) / (right - left)
                output[position] = output[left] * (1.0 - fraction) + output[right] * fraction
                interpolated[position] = True
        elif gap <= 2 and left >= 0 and valid[left]:
            output[start:end] = output[left]
            interpolated[start:end] = True
        elif gap <= 2 and right < len(values) and valid[right]:
            output[start:end] = output[right]
            interpolated[start:end] = True
    return output, interpolated


def _median_smooth_boxes(boxes: np.ndarray, width: int, height: int, window: int = 5) -> np.ndarray:
    centers = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    sizes = np.c_[boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]
    values = np.c_[centers, sizes]
    smoothed = np.empty_like(values)
    radius = window // 2
    for index in range(len(values)):
        smoothed[index] = np.median(values[max(0, index - radius) : min(len(values), index + radius + 1)], axis=0)
    output = np.c_[
        smoothed[:, 0] - smoothed[:, 2] / 2,
        smoothed[:, 1] - smoothed[:, 3] / 2,
        smoothed[:, 0] + smoothed[:, 2] / 2,
        smoothed[:, 1] + smoothed[:, 3] / 2,
    ]
    return np.stack([_clip_box(box, width, height) for box in output])


class PoseROIBuilder:
    def __init__(self, keypoint_threshold: float = 0.25) -> None:
        self.keypoint_threshold = keypoint_threshold

    def build(
        self,
        person_boxes: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_confidence: np.ndarray,
        width: int,
        height: int,
    ) -> ROIBuildResult:
        frames = len(person_boxes)
        bbox_valid = np.isfinite(person_boxes).all(axis=1)
        bbox_track, bbox_interpolated = _interpolate_short_gaps(person_boxes, bbox_valid)
        wrist_tracks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for keypoint_index in (9, 10):
            valid = (
                keypoints_confidence[:, keypoint_index] >= self.keypoint_threshold
            ) & np.isfinite(keypoints_xy[:, keypoint_index]).all(axis=1)
            track, interpolated = _interpolate_short_gaps(keypoints_xy[:, keypoint_index], valid)
            wrist_tracks[keypoint_index] = (track, valid, interpolated)

        boxes = np.empty((frames, 3, 4), dtype=np.float32)
        sources = np.empty((frames, 3), dtype="U32")
        central_upper = np.asarray([width * 0.15, height * 0.05, width * 0.85, height * 0.90])
        central_hands = (
            _square(np.asarray([width * 0.40, height * 0.50]), min(width, height) * 0.55, width, height),
            _square(np.asarray([width * 0.60, height * 0.50]), min(width, height) * 0.55, width, height),
        )
        for frame in range(frames):
            kp_ok = keypoints_confidence[frame] >= self.keypoint_threshold
            point_indices = [index for index in (0, 5, 6, 7, 8, 9, 10, 11, 12) if kp_ok[index] and np.isfinite(keypoints_xy[frame, index]).all()]
            if kp_ok[5] and kp_ok[6] and len(point_indices) >= 4:
                upper = _box_from_points(keypoints_xy[frame, point_indices], 0.25, width, height)
                upper_source = "keypoints"
            elif bbox_valid[frame] or bbox_interpolated[frame]:
                person = bbox_track[frame]
                upper = np.asarray([person[0], person[1], person[2], person[1] + 0.70 * (person[3] - person[1])])
                upper = _clip_box(upper, width, height)
                upper_source = "person_bbox" if bbox_valid[frame] else "interpolated_bbox"
            else:
                upper = _clip_box(central_upper, width, height)
                upper_source = "central"
            boxes[frame, 0] = upper
            sources[frame, 0] = upper_source

            person_available = bbox_valid[frame] or bbox_interpolated[frame]
            person = bbox_track[frame] if person_available else np.asarray([0, 0, width, height], dtype=np.float64)
            person_width = max(float(person[2] - person[0]), width * 0.2)
            head = keypoints_xy[frame, 0] if kp_ok[0] else np.asarray([(upper[0] + upper[2]) / 2, upper[1]])
            left_track, left_valid, left_interpolated = wrist_tracks[9]
            right_track, right_valid, right_interpolated = wrist_tracks[10]
            both_close = (
                (left_valid[frame] or left_interpolated[frame])
                and (right_valid[frame] or right_interpolated[frame])
                and np.linalg.norm(left_track[frame] - right_track[frame]) < person_width * 0.35
            )
            shared_center = (left_track[frame] + right_track[frame]) / 2 if both_close else None
            shared_side = (
                max(person_width * 0.45, np.linalg.norm(left_track[frame] - right_track[frame]) + person_width * 0.25)
                if both_close
                else 0.0
            )
            for local_index, keypoint_index in enumerate((9, 10), start=1):
                track, valid, interpolated = wrist_tracks[keypoint_index]
                if valid[frame] or interpolated[frame]:
                    center = shared_center if shared_center is not None else track[frame]
                    side = shared_side if shared_center is not None else person_width * 0.45
                    hand_box = _square(center, side, width, height)
                    if np.linalg.norm(track[frame] - head) < person_width * 0.35:
                        joint_center = (track[frame] + head) / 2
                        joint_side = max(side, float(np.abs(track[frame] - head).max()) * 1.5)
                        hand_box = _square(joint_center, joint_side, width, height)
                    source = "wrist" if valid[frame] else "interpolated_wrist"
                elif upper_source != "central":
                    fraction = 0.35 if local_index == 1 else 0.65
                    center = np.asarray([upper[0] + fraction * (upper[2] - upper[0]), upper[1] + 0.58 * (upper[3] - upper[1])])
                    hand_box = _square(center, person_width * 0.45, width, height)
                    source = "upper_fixed"
                elif person_available:
                    hand_box = _clip_box(person, width, height)
                    source = "person_bbox"
                else:
                    hand_box = central_hands[local_index - 1]
                    source = "central"
                boxes[frame, local_index] = hand_box
                sources[frame, local_index] = source

        for view in range(3):
            boxes[:, view] = _median_smooth_boxes(boxes[:, view], width, height)
        return ROIBuildResult(boxes=boxes, sources=sources)
