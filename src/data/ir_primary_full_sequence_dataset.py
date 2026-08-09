from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from src.data.pose_roi_dataset import PoseROIDataset, PoseTrackCache, depth_frame_key
from src.roi.ir_primary_input_builder import IRPrimaryInputROIBuilder


class IRPrimaryFullSequenceDataset(PoseROIDataset):
    """Full-clip IR appearance views and lightweight Depth geometry views."""

    def __init__(
        self,
        frame: pd.DataFrame,
        hard_actions: list[str],
        num_frames: int,
        image_size: int,
        training: bool,
        pose_cache_path: Path,
        data_root: Path,
        roi_config: dict[str, object],
    ) -> None:
        super().__init__(
            frame=frame,
            hard_actions=hard_actions,
            num_frames=num_frames,
            image_size=image_size,
            training=training,
            use_pose_roi=False,
            use_ir_input=True,
            data_root=data_root,
        )
        cache = PoseTrackCache(pose_cache_path)
        builder = IRPrimaryInputROIBuilder(
            keypoint_threshold=float(roi_config["keypoint_threshold"]),
            context_padding=float(roi_config["context_padding"]),
            context_quantile=float(roi_config["context_quantile"]),
            minimum_context_side_ratio=float(roi_config["minimum_context_side_ratio"]),
            local_padding=float(roi_config["local_padding"]),
            duplicate_iou_threshold=float(roi_config["duplicate_iou_threshold"]),
            interaction_config=dict(roi_config["interaction"]),
        )
        for sample in self.samples:
            paths = sample["paths"]
            if not isinstance(paths, tuple):
                raise TypeError("Invalid Depth paths")
            with Image.open(paths[0]) as image:
                width, height = image.size
            keys = [depth_frame_key(path) for path in paths]
            person, keypoints, confidence = cache.trial_arrays(str(sample["sample_id"]), keys)
            roi = builder.build(person, keypoints, confidence, width, height)
            sample["input_roi"] = roi
            sample["roi_confidence"] = self._view_confidence(roi, confidence)
            sample["width"] = width
            sample["height"] = height

    @staticmethod
    def _view_confidence(roi: object, confidence: np.ndarray) -> np.ndarray:
        output = np.zeros((len(confidence), 4), dtype=np.float32)
        output[:, 0] = 1.0
        for frame in range(len(confidence)):
            for view, elbow, wrist in ((1, 7, 9), (2, 8, 10)):
                source = str(roi.sources[frame, view])
                if not roi.valid_mask[frame, view]:
                    continue
                output[frame, view] = (
                    min(float(confidence[frame, elbow]), float(confidence[frame, wrist]))
                    if source == "directional"
                    else float(confidence[frame, wrist])
                )
            source = str(roi.sources[frame, 3])
            if not roi.valid_mask[frame, 3]:
                continue
            left = output[frame, 1]
            right = output[frame, 2]
            if source in {"two_hand_table_context", "two_hand_relation"}:
                output[frame, 3] = min(left, right) if left > 0 and right > 0 else max(left, right)
            else:
                output[frame, 3] = max(left, right)
        return np.clip(output, 0.0, 1.0)

    def _window(self, length: int) -> tuple[np.ndarray, torch.Tensor]:
        if length <= 0:
            raise ValueError("A visual sample must contain at least one frame")
        if length > self.num_frames:
            indices = np.rint(np.linspace(0, length - 1, self.num_frames)).astype(np.int64)
            if len(np.unique(indices)) != self.num_frames:
                raise RuntimeError("Full-sequence sampling produced duplicate indices")
            mask = torch.ones(self.num_frames, dtype=torch.bool)
        else:
            indices = np.r_[np.arange(length), np.full(self.num_frames - length, length - 1)]
            mask = torch.arange(self.num_frames) < length
        return indices.astype(np.int64), mask

    def _views(
        self,
        path: Path,
        boxes: np.ndarray,
        valid: np.ndarray,
        indices: tuple[int, ...],
        mode: str,
    ) -> torch.Tensor:
        channels = 1 if mode == "L" else 3
        fill: int | tuple[int, int, int] = 0 if mode == "L" else (0, 0, 0)
        tensors: list[torch.Tensor] = []
        with Image.open(path) as opened:
            image = opened.convert(mode)
            for view in indices:
                if not bool(valid[view]):
                    tensors.append(torch.zeros((channels, self.image_size, self.image_size)))
                    continue
                crop = image.crop(tuple(float(value) for value in boxes[view]))
                padded = ImageOps.pad(
                    crop,
                    (self.image_size, self.image_size),
                    method=Image.Resampling.LANCZOS,
                    color=fill,
                )
                tensors.append((TF.to_tensor(padded) - 0.5) / 0.5)
        return torch.stack(tensors)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        paths = sample["paths"]
        ir_paths = sample["ir_paths"]
        if not isinstance(paths, tuple) or not isinstance(ir_paths, tuple):
            raise TypeError("Invalid paired paths")
        indices, temporal_mask = self._window(len(paths))
        roi = sample["input_roi"]
        confidence = sample["roi_confidence"]
        ir_frames: list[torch.Tensor] = []
        depth_frames: list[torch.Tensor] = []
        ir_valid = []
        depth_valid = []
        view_confidence = []
        for frame_value in indices:
            frame = int(frame_value)
            ir_frames.append(self._views(ir_paths[frame], roi.boxes[frame], roi.valid_mask[frame], (0, 1, 2, 3), "L"))
            depth_frames.append(self._views(paths[frame], roi.boxes[frame], roi.valid_mask[frame], (0, 3), "RGB"))
            ir_valid.append(torch.from_numpy(roi.valid_mask[frame].copy()))
            depth_valid.append(torch.from_numpy(roi.valid_mask[frame, (0, 3)].copy()))
            view_confidence.append(torch.from_numpy(confidence[frame].copy()))
        return {
            "ir_input": torch.stack(ir_frames),
            "depth_input": torch.stack(depth_frames),
            "ir_valid_mask": torch.stack(ir_valid),
            "depth_valid_mask": torch.stack(depth_valid),
            "view_confidence": torch.stack(view_confidence),
            "temporal_mask": temporal_mask,
            "frame_indices": torch.from_numpy(indices.copy()),
            "label": int(sample["label"]),
            "sample_id": str(sample["sample_id"]),
            "user_id": str(sample["user_id"]),
            "length": int(sample["length"]),
        }
