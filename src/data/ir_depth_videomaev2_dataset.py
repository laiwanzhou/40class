from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.data.common import resolve_manifest_path
from src.data.pose_roi_dataset import (
    PoseTrackCache,
    depth_frame_key,
    paired_frame_paths,
)
from src.data.x3d_clip_dataset import fixed_trial_person_context_box
from src.roi.object_interaction_builder import ObjectInteractionROIBuilder


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]


def uniform_trial_indices(length: int, frames: int, *, jitter: float = 0.0) -> np.ndarray:
    if length < 1 or frames < 2 or not 0.0 <= jitter <= 0.25:
        raise ValueError("invalid fixed-frame sampling contract")
    positions = np.linspace(0.0, length - 1, frames, dtype=np.float64)
    if jitter and length > 1:
        spacing = (length - 1) / (frames - 1)
        noise = np.random.uniform(-jitter * spacing, jitter * spacing, frames)
        noise[[0, -1]] = 0.0
        positions += noise
    indices = np.rint(np.clip(positions, 0, length - 1)).astype(np.int64)
    indices[0], indices[-1] = 0, length - 1
    return np.maximum.accumulate(indices)


def _fill_boxes(boxes: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, bool]:
    if not valid.any():
        return np.zeros_like(boxes), False
    output = boxes.copy()
    positions = np.arange(len(boxes), dtype=np.float32)
    valid_positions = positions[valid]
    for column in range(4):
        output[:, column] = np.interp(positions, valid_positions, boxes[valid, column])
    return output, True


class IRDepthVideoMAEV2Dataset(Dataset[dict[str, object]]):
    """Fixed-16 IR/depth clips with global, fixed-person, and directional hands."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        split_path: Path,
        data_root: Path,
        pose_cache_path: Path,
        partition: str,
        training: bool,
        frames: int = 16,
        image_size: int = 224,
        temporal_jitter: float = 0.1,
        interaction_config: dict[str, float] | None = None,
    ) -> None:
        if partition not in {"train", "validation"}:
            raise ValueError("partition must be train or validation")
        if frames != 16 or image_size != 224:
            raise ValueError("P0/P1 freezes VideoMAE input at 16 frames and 224 pixels")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        key = "train_user_ids" if partition == "train" else "validation_user_ids"
        users = set(str(value) for value in split[key])
        manifest = pd.read_csv(
            manifest_path, encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str}
        )
        present = (
            manifest["depth_color_path"].fillna("").astype(str).str.strip().ne("")
            & manifest["ir_path"].fillna("").astype(str).str.strip().ne("")
        )
        selected = manifest.loc[present & manifest["user_id"].isin(users)].copy()
        selected = selected.sort_values(["class_id", "sample_id"]).reset_index(drop=True)
        if selected.empty or selected["class_id"].nunique() != 40:
            raise ValueError(f"{partition} must retain all 40 classes")
        self.samples = selected.to_dict(orient="records")
        self.data_root = data_root.resolve()
        self.pose_cache = PoseTrackCache(pose_cache_path)
        self.training = bool(training)
        self.frames = int(frames)
        self.image_size = int(image_size)
        self.temporal_jitter = float(temporal_jitter)
        self.interaction_builder = ObjectInteractionROIBuilder(**(interaction_config or {}))

    def __len__(self) -> int:
        return len(self.samples)

    def class_ids(self) -> list[int]:
        return [int(sample["class_id"]) for sample in self.samples]

    def _tensor(self, image: Image.Image, box: np.ndarray, *, grayscale: bool) -> torch.Tensor:
        converted = image.convert("L" if grayscale else "RGB")
        crop = converted.crop(tuple(float(value) for value in box))
        resized = TF.resize(crop, [self.image_size, self.image_size], antialias=True)
        tensor = TF.to_tensor(resized)
        if grayscale:
            tensor = tensor.repeat(3, 1, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.samples[index]
        sample_id = str(row["sample_id"])
        depth_dir = resolve_manifest_path(self.data_root, str(row["depth_color_path"]))
        ir_dir = resolve_manifest_path(self.data_root, str(row["ir_path"]))
        depth_paths, ir_paths = paired_frame_paths(depth_dir, ir_dir)
        with Image.open(depth_paths[0]) as first:
            width, height = first.size
        keys = [depth_frame_key(path) for path in depth_paths]
        for path in depth_paths:
            self.pose_cache.validate_frame(sample_id, path, width, height)
        person_boxes, keypoints, confidence = self.pose_cache.trial_arrays(sample_id, keys)
        interaction = self.interaction_builder.build(
            person_boxes, keypoints, confidence, width, height
        )
        person_box = fixed_trial_person_context_box(
            person_boxes,
            width=width,
            height=height,
            detection_frames=8,
            crop_margin=1.4,
            minimum_side_fraction=0.35,
        )
        left_boxes, left_available = _fill_boxes(
            interaction.boxes[:, 1], interaction.valid_mask[:, 1]
        )
        right_boxes, right_available = _fill_boxes(
            interaction.boxes[:, 2], interaction.valid_mask[:, 2]
        )
        if left_available:
            left_box = fixed_trial_person_context_box(
                left_boxes,
                width=width,
                height=height,
                detection_frames=8,
                crop_margin=1.2,
                minimum_side_fraction=0.12,
            )
            left_boxes[:] = left_box
        if right_available:
            right_box = fixed_trial_person_context_box(
                right_boxes,
                width=width,
                height=height,
                detection_frames=8,
                crop_margin=1.2,
                minimum_side_fraction=0.12,
            )
            right_boxes[:] = right_box
        indices = uniform_trial_indices(
            len(depth_paths),
            self.frames,
            jitter=self.temporal_jitter if self.training else 0.0,
        )
        full_box = np.asarray((0.0, 0.0, float(width), float(height)), dtype=np.float32)
        depth_views: list[list[torch.Tensor]] = [[] for _ in range(4)]
        ir_views: list[list[torch.Tensor]] = [[] for _ in range(4)]
        for frame_index in indices:
            boxes = (
                full_box,
                person_box,
                left_boxes[int(frame_index)],
                right_boxes[int(frame_index)],
            )
            with Image.open(depth_paths[int(frame_index)]) as depth_image, Image.open(
                ir_paths[int(frame_index)]
            ) as ir_image:
                for view_index, box in enumerate(boxes):
                    if (view_index == 2 and not left_available) or (
                        view_index == 3 and not right_available
                    ):
                        depth_tensor = torch.zeros(3, self.image_size, self.image_size)
                        ir_tensor = torch.zeros_like(depth_tensor)
                    else:
                        depth_tensor = self._tensor(depth_image, box, grayscale=False)
                        ir_tensor = self._tensor(ir_image, box, grayscale=True)
                    depth_views[view_index].append(depth_tensor)
                    ir_views[view_index].append(ir_tensor)
        depth = torch.stack([torch.stack(values, dim=1) for values in depth_views])
        ir = torch.stack([torch.stack(values, dim=1) for values in ir_views])
        clips = torch.stack((ir, depth), dim=0)
        availability = torch.tensor(
            [[True, True, left_available, right_available]] * 2, dtype=torch.bool
        )
        return {
            "clips": clips,
            "availability": availability,
            "label": int(row["class_id"]),
            "sample_id": sample_id,
            "user_id": str(row["user_id"]),
            "sampled_indices": torch.from_numpy(indices.copy()),
        }
