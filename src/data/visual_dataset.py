from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .common import sorted_files


class VisualSequenceDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        modality: str,
        num_frames: int = 12,
        image_size: int = 192,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.modality = modality
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def _load_frames(self, trial_path: Path, sample_id: str) -> tuple[torch.Tensor, int]:
        files = sorted_files(trial_path, {".png", ".jpg", ".jpeg"})
        if not files:
            raise FileNotFoundError(f"No images for sample_id={sample_id}: {trial_path}")
        indices = np.linspace(0, len(files) - 1, self.num_frames).round().astype(int)
        tensors: list[torch.Tensor] = []
        for index in indices:
            image_path = files[int(index)]
            try:
                with Image.open(image_path) as image:
                    if self.modality == "IR":
                        image = image.convert("L").convert("RGB")
                    else:
                        image = image.convert("RGB")
                    image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
                    tensor = TF.to_tensor(image)
            except Exception as exc:
                raise RuntimeError(
                    f"Image read failed for sample_id={sample_id}: {image_path}"
                ) from exc
            tensors.append((tensor - 0.5) / 0.5)
        return torch.stack(tensors), len(files)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        sample_id = str(row["sample_id"])
        trial_path = Path(row["trial_path"])
        tensor, original_length = self._load_frames(trial_path, sample_id)
        return {
            "input": tensor,
            "temporal_mask": torch.ones(self.num_frames, dtype=torch.bool),
            "label": int(row["class_id"]),
            "sample_id": sample_id,
            "length": original_length,
        }
