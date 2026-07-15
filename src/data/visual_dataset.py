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


class VisualSequenceDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        modality: str,
        num_frames: int = 12,
        image_size: int = 192,
    ) -> None:
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
            f"VisualSequenceDataset indexed {len(self.samples)} {self.modality} samples "
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
        tensors: list[torch.Tensor] = []
        for image_path in selected_paths:
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
                    f"Image read failed for sample_id={sample_id}, "
                    f"trial_path={trial_path}: {image_path}"
                ) from exc
            tensors.append((tensor - 0.5) / 0.5)
        return torch.stack(tensors)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        sample_id = str(sample["sample_id"])
        trial_path = Path(sample["trial_path"])
        selected_paths = sample["selected_paths"]
        if not isinstance(selected_paths, tuple):
            raise TypeError(f"Invalid frame index for sample_id={sample_id}: trial_path={trial_path}")
        tensor = self._load_frames(selected_paths, trial_path, sample_id)
        return {
            "input": tensor,
            "temporal_mask": torch.ones(self.num_frames, dtype=torch.bool),
            "label": int(sample["class_id"]),
            "sample_id": sample_id,
            "length": int(sample["original_length"]),
        }
