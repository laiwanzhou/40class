from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .common import sorted_files


def six_patch_boxes(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return the fixed 3x2 overlapping layout as PIL crop boxes."""
    if width < 2 or height < 3:
        raise ValueError(f"Image is too small for six patches: {width}x{height}")
    x_edges = [round(width * fraction) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
    y_edges = [round(height * fraction) for fraction in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)]
    return tuple(
        (x_edges[column], y_edges[row], x_edges[column + 2], y_edges[row + 2])
        for row in range(2)
        for column in range(3)
    )


class VisualSixPatchDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        modality: str,
        num_frames: int = 12,
        image_size: int = 192,
    ) -> None:
        if modality != "Depth_Color":
            raise ValueError(f"Six-patch input only supports Depth_Color, got {modality!r}")
        self.modality = modality
        self.num_frames = num_frames
        self.image_size = image_size
        started = time.perf_counter()
        self.samples: list[dict[str, object]] = []
        for row in frame.reset_index(drop=True).to_dict(orient="records"):
            sample_id = str(row["sample_id"])
            trial_path = Path(row["trial_path"])
            files = sorted_files(trial_path, {".png", ".jpg", ".jpeg"})
            if not files:
                raise FileNotFoundError(
                    f"No images for sample_id={sample_id}: trial_path={trial_path}"
                )
            indices = np.linspace(0, len(files) - 1, self.num_frames).round().astype(int)
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "class_id": int(row["class_id"]),
                    "trial_path": trial_path,
                    "selected_paths": tuple(files[int(index)] for index in indices),
                    "original_length": len(files),
                }
            )
        print(
            f"VisualSixPatchDataset indexed {len(self.samples)} {self.modality} samples "
            f"in {time.perf_counter() - started:.2f} seconds"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(
        self,
        selected_paths: tuple[Path, ...],
        trial_path: Path,
        sample_id: str,
    ) -> torch.Tensor:
        frame_tensors: list[torch.Tensor] = []
        for image_path in selected_paths:
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    views = [image, *(image.crop(box) for box in six_patch_boxes(*image.size))]
                    tensors = []
                    for view in views:
                        resized = TF.resize(
                            view,
                            [self.image_size, self.image_size],
                            antialias=True,
                        )
                        tensors.append((TF.to_tensor(resized) - 0.5) / 0.5)
            except Exception as exc:
                raise RuntimeError(
                    f"Image read failed for sample_id={sample_id}, "
                    f"trial_path={trial_path}: {image_path}"
                ) from exc
            frame_tensors.append(torch.stack(tensors))
        return torch.stack(frame_tensors)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        sample_id = str(sample["sample_id"])
        trial_path = Path(sample["trial_path"])
        selected_paths = sample["selected_paths"]
        if not isinstance(selected_paths, tuple):
            raise TypeError(f"Invalid frame index for sample_id={sample_id}: {trial_path}")
        tensor = self._load_frames(selected_paths, trial_path, sample_id)
        return {
            "input": tensor,
            "temporal_mask": torch.ones(self.num_frames, dtype=torch.bool),
            "label": int(sample["class_id"]),
            "sample_id": sample_id,
            "length": int(sample["original_length"]),
        }
