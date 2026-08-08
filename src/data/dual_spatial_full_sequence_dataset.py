from __future__ import annotations

import numpy as np
import torch

from src.data.person_crop_pose_roi_dataset import PersonCropPoseROIDataset


class DualSpatialFullSequenceDataset(PersonCropPoseROIDataset):
    """Aligned original-frame and person-interaction views over the complete clip."""

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
        return indices.astype(np.int64), mask

    def __getitem__(self, index: int) -> dict[str, object]:
        result = super().__getitem__(index)
        sample = self.samples[index]
        paths = sample["paths"]
        ir_paths = sample["ir_paths"]
        if not isinstance(paths, tuple) or not isinstance(ir_paths, tuple):
            raise TypeError("Invalid paired Depth/IR paths.")
        indices, _ = self._window(len(paths))
        full_box = np.asarray(
            [0.0, 0.0, float(sample["width"]), float(sample["height"])], dtype=np.float32,
        )
        result["global_depth_input"] = torch.stack(
            [self._tensor(paths[int(frame)], full_box, "RGB") for frame in indices]
        )
        result["global_ir_input"] = torch.stack(
            [self._tensor(ir_paths[int(frame)], full_box, "L") for frame in indices]
        )
        result["frame_indices"] = torch.from_numpy(indices.copy())
        return result
