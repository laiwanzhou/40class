from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.data.pose_roi_dataset import PoseROIDataset, PoseTrackCache, depth_frame_key
from src.roi.person_interaction_crop_builder import PersonInteractionCropBuilder
from src.roi.roi_builder import PoseROIBuilder


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    box = np.asarray(
        [max(first[0], second[0]), max(first[1], second[1]),
         min(first[2], second[2]), min(first[3], second[3])], dtype=np.float32
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


class PersonCropPoseROIDataset(PoseROIDataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        hard_actions: list[str],
        num_frames: int,
        image_size: int,
        training: bool,
        pose_cache_path: Path,
        data_root: Path,
        interaction_config: dict[str, float],
        person_crop_config: dict[str, float | int],
    ) -> None:
        super().__init__(
            frame=frame,
            hard_actions=hard_actions,
            num_frames=num_frames,
            image_size=image_size,
            training=training,
            use_pose_roi=True,
            pose_cache_path=pose_cache_path,
            use_ir_input=True,
            data_root=data_root,
        )
        cache = PoseTrackCache(pose_cache_path)
        person_builder = PersonInteractionCropBuilder(
            interaction_config=interaction_config, **person_crop_config
        )
        pose_builder = PoseROIBuilder()
        for sample in self.samples:
            paths = sample["paths"]
            if not isinstance(paths, tuple):
                raise TypeError("Invalid Depth paths")
            with Image.open(paths[0]) as image:
                width, height = image.size
            keys = [depth_frame_key(path) for path in paths]
            person_boxes, keypoints, confidence = cache.trial_arrays(str(sample["sample_id"]), keys)
            try:
                person = person_builder.build(person_boxes, keypoints, confidence, width, height)
            except ValueError as error:
                raise ValueError(f"Person crop failed for {sample['sample_id']}: {error}") from error
            pose = pose_builder.build(person_boxes, keypoints, confidence, width, height)
            local_boxes = np.zeros_like(pose.boxes)
            local_valid = np.zeros((len(paths), 3), dtype=bool)
            for frame_index in range(len(paths)):
                for view_index in range(3):
                    overlap = _intersection(pose.boxes[frame_index, view_index], person.person_boxes[frame_index])
                    if overlap is not None and pose.sources[frame_index, view_index] != "central":
                        local_boxes[frame_index, view_index] = overlap
                        local_valid[frame_index, view_index] = True
            sample["person_crop"] = person
            sample["roi_boxes"] = local_boxes
            sample["roi_valid"] = local_valid
            sample["roi_sources"] = pose.sources
            sample["width"] = width
            sample["height"] = height

    def _tensor(self, path: Path, box: np.ndarray, mode: str) -> torch.Tensor:
        with Image.open(path) as opened:
            crop = opened.convert(mode).crop(tuple(float(value) for value in box))
            resized = TF.resize(crop, [self.image_size, self.image_size], antialias=True)
            return (TF.to_tensor(resized) - 0.5) / 0.5

    def _view_tensors(
        self, path: Path, boxes: list[np.ndarray], valid: list[bool], mode: str
    ) -> list[torch.Tensor]:
        channels = 3 if mode == "RGB" else 1
        tensors: list[torch.Tensor] = []
        with Image.open(path) as opened:
            image = opened.convert(mode)
            for box, is_valid in zip(boxes, valid, strict=True):
                if not is_valid:
                    tensors.append(torch.zeros((channels, self.image_size, self.image_size)))
                    continue
                crop = image.crop(tuple(float(value) for value in box))
                resized = TF.resize(crop, [self.image_size, self.image_size], antialias=True)
                tensors.append((TF.to_tensor(resized) - 0.5) / 0.5)
        return tensors

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        paths = sample["paths"]
        ir_paths = sample["ir_paths"]
        if not isinstance(paths, tuple) or not isinstance(ir_paths, tuple):
            raise TypeError("Invalid paired paths")
        indices, mask = self._window(len(paths))
        person = sample["person_crop"]
        local_boxes = sample["roi_boxes"]
        local_valid = sample["roi_valid"]
        depth_frames: list[torch.Tensor] = []
        ir_frames: list[torch.Tensor] = []
        valid = torch.zeros((self.num_frames, 4), dtype=torch.bool)
        valid[:, 0] = True
        for output_index, frame_index_value in enumerate(indices):
            frame_index = int(frame_index_value)
            boxes = [person.person_boxes[frame_index], *local_boxes[frame_index]]
            frame_valid = [True, *(bool(value) for value in local_valid[frame_index])]
            for view_index, is_valid in enumerate(frame_valid):
                valid[output_index, view_index] = is_valid
            depth_views = self._view_tensors(paths[frame_index], boxes, frame_valid, "RGB")
            ir_views = self._view_tensors(ir_paths[frame_index], boxes, frame_valid, "L")
            depth_frames.append(torch.stack(depth_views))
            ir_frames.append(torch.stack(ir_views))
        depth = torch.stack(depth_frames)
        ir = torch.stack(ir_frames)
        return {
            "input": {"depth_input": depth, "ir_input": ir},
            "depth_input": depth,
            "ir_input": ir,
            "temporal_mask": mask,
            "roi_valid_mask": valid,
            "label": int(sample["label"]),
            "sample_id": str(sample["sample_id"]),
            "user_id": str(sample["user_id"]),
            "user_index": int(sample["user_index"]),
            "length": int(sample["length"]),
        }

    def person_crop_audit_rows(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        action_by_id = dict(zip(self.original_class_ids, self.class_names, strict=True))
        for sample in self.samples:
            result = sample["person_crop"]
            area = (result.person_boxes[:, 2] - result.person_boxes[:, 0]) * (
                result.person_boxes[:, 3] - result.person_boxes[:, 1]
            )
            detected = result.person_detected
            original_area = float(sample["width"] * sample["height"])
            detector_area = np.zeros(len(detected), dtype=np.float32)
            detector_boxes = result.detector_boxes
            detector_area[detected] = (
                (detector_boxes[detected, 2] - detector_boxes[detected, 0])
                * (detector_boxes[detected, 3] - detector_boxes[detected, 1])
            )
            rows.append({
                "sample_id": sample["sample_id"],
                "class_id": int(sample["original_class_id"]),
                "action_name": action_by_id[int(sample["original_class_id"])],
                "frames": int(sample["length"]),
                "person_main_valid_rate": 1.0,
                "person_main_area_ratio_mean": float(area.mean() / original_area),
                "person_bbox_area_ratio_mean": float(detector_area[detected].mean() / original_area) if detected.any() else np.nan,
                "person_occupancy_before_mean": float(detector_area[detected].mean() / original_area) if detected.any() else np.nan,
                "person_occupancy_after_mean": float(np.mean(detector_area[detected] / area[detected])) if detected.any() else np.nan,
                "all_valid_keypoints_contained_rate": _keypoint_containment_rate(result, sample),
                "directional_roi_contained_rate": _roi_containment_rate(result, (1, 2)),
                "hand_head_roi_contained_rate": _roi_containment_rate(result, (3,)),
                "two_hand_roi_contained_rate": _roi_containment_rate(result, (4,)),
                "boundary_touch_rate": float(result.person_touches_boundary.mean()),
                "interpolated_rate": float(result.person_interpolated.mean()),
                "nearest_hold_rate": float(result.person_nearest_held.mean()),
                "full_frame_fallback_count": 0,
                "zero_person_context_count": 0,
                "artificial_keypoint_count": 0,
                "lie_down_false_two_hand_count": int((
                    result.interaction.two_hand_merge
                    & ~result.interaction.keypoint_valid[:, [7, 8, 9, 10]].all(axis=1)
                ).sum()) if action_by_id[int(sample["original_class_id"])] == "Lie_down" else 0,
            })
        return pd.DataFrame(rows)


def _contains(container: np.ndarray, item: np.ndarray) -> bool:
    return bool(
        item[0] >= container[0] - 1e-4 and item[1] >= container[1] - 1e-4
        and item[2] <= container[2] + 1e-4 and item[3] <= container[3] + 1e-4
    )


def _roi_containment_rate(result: object, views: tuple[int, ...]) -> float:
    total = contained = 0
    for frame in range(len(result.person_boxes)):
        for view in views:
            if result.interaction.valid_mask[frame, view]:
                total += 1
                contained += int(_contains(result.person_boxes[frame], result.interaction.raw_boxes[frame, view]))
    return float(contained / total) if total else np.nan


def _keypoint_containment_rate(result: object, sample: dict[str, object]) -> float:
    del sample
    total = contained = 0
    for frame, valid in enumerate(result.keypoint_valid):
        points = result.keypoints_xy[frame, valid]
        if not len(points):
            continue
        total += 1
        box = result.person_boxes[frame]
        contained += int(bool(
            (points[:, 0] >= box[0] - 1e-4).all() and (points[:, 0] <= box[2] + 1e-4).all()
            and (points[:, 1] >= box[1] - 1e-4).all() and (points[:, 1] <= box[3] + 1e-4).all()
        ))
    return float(contained / total) if total else np.nan
