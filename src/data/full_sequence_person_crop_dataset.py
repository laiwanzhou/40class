from __future__ import annotations

import numpy as np
import torch

from src.data.person_crop_pose_roi_dataset import PersonCropPoseROIDataset


class FullSequencePersonCropPoseROIDataset(PersonCropPoseROIDataset):
    """Person-crop dataset that deterministically covers the complete clip."""

    def _window(self, length: int) -> tuple[np.ndarray, torch.Tensor]:
        if length <= 0:
            raise ValueError("A visual sample must contain at least one frame.")
        if length > self.num_frames:
            indices = np.rint(np.linspace(0, length - 1, self.num_frames)).astype(np.int64)
            if len(np.unique(indices)) != self.num_frames:
                raise RuntimeError("Full-sequence sampling produced duplicate indices.")
            mask = torch.ones(self.num_frames, dtype=torch.bool)
        else:
            indices = np.r_[np.arange(length), np.full(self.num_frames - length, length - 1)]
            mask = torch.arange(self.num_frames) < length
        return indices.astype(int), mask

    def __getitem__(self, index: int) -> dict[str, object]:
        result = super().__getitem__(index)
        indices, _ = self._window(int(result["length"]))
        result["frame_indices"] = torch.from_numpy(indices.astype(np.int64))
        return result
