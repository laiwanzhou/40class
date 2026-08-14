from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, TypedDict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as transform_functional
from torchvision.transforms.functional import InterpolationMode

from src.data.ir_primary_full_sequence_dataset import class_map_hash


LOCAL_FRAMES = 13
TARGET_WINDOW_FRAMES = 32
MAX_CLIPS = 8
OUTPUT_SIZE = 182
VALIDATION_RESIZE = 200
X3D_MEAN = (0.45, 0.45, 0.45)
X3D_STD = (0.225, 0.225, 0.225)
QUALITY_NAMES = (
    "temporal_valid_fraction",
    "context_effective_rate",
    "context_reliability_mean",
    "unique_frame_fraction",
    "temporal_coverage_fraction",
    "normalized_trial_length",
)
REQUIRED_COLUMNS = {
    "split",
    "class_id",
    "action_name",
    "sample_id",
    "user_id",
    "source_frame_index",
    "temporal_valid",
    "ir_context_path",
    "ir_context_effective_valid",
    "ir_context_reliability",
}


@dataclass(frozen=True)
class IRAugmentationConfig:
    brightness: tuple[float, float] = (1.0, 1.0)
    contrast: tuple[float, float] = (1.0, 1.0)
    gamma: tuple[float, float] = (1.0, 1.0)
    noise_std_max: float = 0.0
    blur_probability: float = 0.0
    blur_kernel_size: int = 3
    blur_sigma: tuple[float, float] = (0.1, 0.1)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "IRAugmentationConfig":
        if value is None:
            return cls()

        def pair(name: str, default: tuple[float, float]) -> tuple[float, float]:
            raw = value.get(name, default)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
                raise ValueError(f"augmentation.{name} must contain two values")
            result = (float(raw[0]), float(raw[1]))
            if result[0] <= 0.0 or result[0] > result[1]:
                raise ValueError(f"augmentation.{name} must be positive and ordered")
            return result

        config = cls(
            brightness=pair("brightness", (1.0, 1.0)),
            contrast=pair("contrast", (1.0, 1.0)),
            gamma=pair("gamma", (1.0, 1.0)),
            noise_std_max=float(value.get("noise_std_max", 0.0)),
            blur_probability=float(value.get("blur_probability", 0.0)),
            blur_kernel_size=int(value.get("blur_kernel_size", 3)),
            blur_sigma=pair("blur_sigma", (0.1, 0.1)),
        )
        if config.noise_std_max < 0.0:
            raise ValueError("augmentation.noise_std_max must be non-negative")
        if not 0.0 <= config.blur_probability <= 1.0:
            raise ValueError("augmentation.blur_probability must lie in [0, 1]")
        if config.blur_kernel_size <= 0 or config.blur_kernel_size % 2 == 0:
            raise ValueError("augmentation.blur_kernel_size must be a positive odd integer")
        return config


class X3DClipSample(TypedDict):
    clips: torch.Tensor
    clip_mask: torch.Tensor
    num_frames: int
    num_clips: int
    label: int
    sample_id: str
    user_id: str
    class_map_hash: str
    quality: torch.Tensor
    quality_mask: torch.Tensor
    availability: torch.Tensor
    source_indices: torch.Tensor
    window_bounds: torch.Tensor
    clip_unique_frame_fraction: torch.Tensor


def adaptive_clip_count(
    num_frames: int,
    *,
    target_window_frames: int = TARGET_WINDOW_FRAMES,
    max_clips: int = MAX_CLIPS,
) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if target_window_frames <= 0 or max_clips <= 0:
        raise ValueError("target_window_frames and max_clips must be positive")
    return min(max_clips, max(1, math.ceil(num_frames / target_window_frames)))


def partition_trial_windows(
    num_frames: int,
    *,
    target_window_frames: int = TARGET_WINDOW_FRAMES,
    max_clips: int = MAX_CLIPS,
) -> list[tuple[int, int]]:
    num_clips = adaptive_clip_count(
        num_frames,
        target_window_frames=target_window_frames,
        max_clips=max_clips,
    )
    base_length, remainder = divmod(num_frames, num_clips)
    windows: list[tuple[int, int]] = []
    start = 0
    for window_index in range(num_clips):
        length = base_length + int(window_index < remainder)
        end = start + length
        windows.append((start, end))
        start = end
    return windows


def stratified_temporal_indices(
    start: int,
    end: int,
    *,
    training: bool,
    generator: torch.Generator | None = None,
    local_frames: int = LOCAL_FRAMES,
) -> torch.Tensor:
    if start < 0 or end <= start:
        raise ValueError("Temporal window must be non-empty and non-negative")
    if local_frames <= 0:
        raise ValueError("local_frames must be positive")
    length = end - start
    edges = np.linspace(start, end, local_frames + 1, dtype=np.float64)
    selected: list[int] = []
    for bin_index in range(local_frames):
        low = max(start, int(math.floor(edges[bin_index])))
        high = min(end - 1, int(math.ceil(edges[bin_index + 1])) - 1)
        high = max(low, high)
        if training:
            if generator is None:
                raise ValueError("Training temporal sampling requires a generator")
            value = int(torch.randint(low, high + 1, (1,), generator=generator).item())
        else:
            midpoint = start + (bin_index + 0.5) * length / local_frames
            value = min(end - 1, max(start, int(math.floor(midpoint))))
        selected.append(value)
    return torch.tensor(selected, dtype=torch.long)


def _read_gray_tensor(path: str | Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image at {path}, got shape {image.shape}")
    return torch.from_numpy(image.copy()).unsqueeze(0).to(torch.float32).div_(255.0)


def _random_resized_crop_parameters(
    height: int,
    width: int,
    generator: torch.Generator,
) -> tuple[int, int, int, int]:
    area = height * width
    log_min_ratio = math.log(0.9)
    log_max_ratio = math.log(1.1)
    for _ in range(10):
        scale = 0.8 + 0.2 * float(torch.rand((), generator=generator).item())
        log_ratio = log_min_ratio + (log_max_ratio - log_min_ratio) * float(
            torch.rand((), generator=generator).item()
        )
        aspect_ratio = math.exp(log_ratio)
        crop_width = int(round(math.sqrt(area * scale * aspect_ratio)))
        crop_height = int(round(math.sqrt(area * scale / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(
                torch.randint(0, height - crop_height + 1, (1,), generator=generator).item()
            )
            left = int(
                torch.randint(0, width - crop_width + 1, (1,), generator=generator).item()
            )
            return top, left, crop_height, crop_width

    input_ratio = width / height
    if input_ratio < 0.9:
        crop_width = width
        crop_height = int(round(crop_width / 0.9))
    elif input_ratio > 1.1:
        crop_height = height
        crop_width = int(round(crop_height * 1.1))
    else:
        crop_height, crop_width = height, width
    return (
        (height - crop_height) // 2,
        (width - crop_width) // 2,
        crop_height,
        crop_width,
    )


def _transform_clip_frames(
    frames: list[torch.Tensor],
    *,
    training: bool,
    generator: torch.Generator,
    augmentation: IRAugmentationConfig | None = None,
) -> torch.Tensor:
    if not frames:
        raise ValueError("A clip must contain at least one frame")
    if training:
        _, height, width = frames[0].shape
        top, left, crop_height, crop_width = _random_resized_crop_parameters(
            height, width, generator
        )
        flip = bool(torch.rand((), generator=generator).item() < 0.5)
        transformed = [
            transform_functional.resized_crop(
                frame,
                top,
                left,
                crop_height,
                crop_width,
                [OUTPUT_SIZE, OUTPUT_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            for frame in frames
        ]
        if flip:
            transformed = [transform_functional.hflip(frame) for frame in transformed]
        if augmentation is not None:
            transformed = _apply_ir_photometric_augmentation(
                transformed,
                config=augmentation,
                generator=generator,
            )
    else:
        transformed = [
            transform_functional.center_crop(
                transform_functional.resize(
                    frame,
                    VALIDATION_RESIZE,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                [OUTPUT_SIZE, OUTPUT_SIZE],
            )
            for frame in frames
        ]
    normalized = [
        transform_functional.normalize(frame.repeat(3, 1, 1), X3D_MEAN, X3D_STD)
        for frame in transformed
    ]
    return torch.stack(normalized, dim=1)


def _sample_uniform(
    bounds: tuple[float, float], generator: torch.Generator
) -> float:
    low, high = bounds
    return low + (high - low) * float(torch.rand((), generator=generator).item())


def _apply_ir_photometric_augmentation(
    frames: list[torch.Tensor],
    *,
    config: IRAugmentationConfig,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    brightness = _sample_uniform(config.brightness, generator)
    contrast = _sample_uniform(config.contrast, generator)
    gamma = _sample_uniform(config.gamma, generator)
    apply_blur = bool(
        torch.rand((), generator=generator).item() < config.blur_probability
    )
    blur_sigma = _sample_uniform(config.blur_sigma, generator)
    result: list[torch.Tensor] = []
    for frame in frames:
        transformed = transform_functional.adjust_brightness(frame, brightness)
        transformed = transform_functional.adjust_contrast(transformed, contrast)
        transformed = transform_functional.adjust_gamma(transformed.clamp(0.0, 1.0), gamma)
        if apply_blur:
            transformed = transform_functional.gaussian_blur(
                transformed,
                [config.blur_kernel_size, config.blur_kernel_size],
                [blur_sigma, blur_sigma],
            )
        if config.noise_std_max > 0.0:
            noise_std = config.noise_std_max * float(
                torch.rand((), generator=generator).item()
            )
            noise = torch.randn(
                transformed.shape,
                generator=generator,
                dtype=transformed.dtype,
                device=transformed.device,
            )
            transformed = transformed + noise * noise_std
        result.append(transformed.clamp(0.0, 1.0))
    return result


class X3DClipDataset(Dataset[X3DClipSample]):
    """One complete trial represented by adaptive local X3D clips."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        split: str,
        training: bool,
        augmentation_enabled: bool = True,
        augmentation_config: Mapping[str, object] | IRAugmentationConfig | None = None,
        seed: int = 20260715,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("Only train and val splits are allowed")
        if training != (split == "train"):
            raise ValueError("training must be true only for the train split")
        frame = (
            manifest.copy()
            if isinstance(manifest, pd.DataFrame)
            else pd.read_csv(manifest, encoding="utf-8-sig")
        )
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
            raise ValueError("Manifest contains null required values")
        if set(frame["split"].astype(str)) - {"train", "val"}:
            raise ValueError("Manifest contains a non-train/val split")
        split_counts = frame.groupby("sample_id", sort=False)["split"].nunique()
        if (split_counts > 1).any():
            raise ValueError("A sample_id appears on both train and val split sides")
        if frame.duplicated(["sample_id", "source_frame_index"]).any():
            raise ValueError("Duplicate (sample_id, source_frame_index) rows")

        classes = frame[["class_id", "action_name"]].drop_duplicates()
        class_ids = sorted(classes["class_id"].astype(int).unique().tolist())
        if len(classes) != 40 or class_ids != list(range(40)):
            raise ValueError("Expected one class-map row for each class ID 0 through 39")
        self.class_rows = classes.sort_values("class_id").reset_index(drop=True)
        self.class_map_hash = class_map_hash(self.class_rows)
        self.class_names = self.class_rows["action_name"].astype(str).tolist()
        self.training = bool(training)
        self.augmentation_enabled = bool(augmentation_enabled) and self.training
        self.augmentation_config = (
            augmentation_config
            if isinstance(augmentation_config, IRAugmentationConfig)
            else IRAugmentationConfig.from_mapping(augmentation_config)
        )
        self.seed = int(seed)
        self.epoch = 0

        selected = frame[frame["split"].astype(str) == split].copy()
        if selected.empty:
            raise ValueError(f"Manifest has no samples for split {split}")
        missing_paths = [
            path for path in selected["ir_context_path"].astype(str).unique() if not Path(path).is_file()
        ]
        if missing_paths:
            raise ValueError(f"Selected IR context image does not exist: {missing_paths[0]}")

        self.samples: list[pd.DataFrame] = []
        self.sample_ids: list[str] = []
        self.lengths: list[int] = []
        self.num_clips: list[int] = []
        for sample_id, group in selected.groupby("sample_id", sort=False):
            ordered = group.sort_values("source_frame_index").reset_index(drop=True)
            indices = ordered["source_frame_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(indices, np.arange(len(ordered), dtype=np.int64)):
                raise ValueError(f"Non-contiguous frame order for {sample_id}")
            for field in ("class_id", "action_name", "user_id"):
                if ordered[field].nunique() != 1:
                    raise ValueError(f"Inconsistent {field} within sample {sample_id}")
            self.samples.append(ordered)
            self.sample_ids.append(str(sample_id))
            self.lengths.append(len(ordered))
            self.num_clips.append(adaptive_clip_count(len(ordered)))

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _generator(self, dataset_index: int, window_index: int) -> torch.Generator:
        derived_seed = (
            self.seed
            + self.epoch * 1_000_003
            + dataset_index * 10_007
            + window_index * 101
        )
        return torch.Generator().manual_seed(derived_seed)

    def __getitem__(self, index: int) -> X3DClipSample:
        frame = self.samples[index]
        num_frames = len(frame)
        windows = partition_trial_windows(num_frames)
        clips: list[torch.Tensor] = []
        selected_indices: list[torch.Tensor] = []
        clip_unique_fractions: list[float] = []
        image_cache: dict[int, torch.Tensor] = {}
        for window_index, (start, end) in enumerate(windows):
            generator = self._generator(index, window_index)
            indices = stratified_temporal_indices(
                start,
                end,
                training=self.augmentation_enabled,
                generator=generator,
            )
            selected_indices.append(indices)
            clip_unique_fractions.append(float(indices.unique().numel() / LOCAL_FRAMES))
            temporal_frames: list[torch.Tensor] = []
            for source_index in indices.tolist():
                if source_index not in image_cache:
                    image_cache[source_index] = _read_gray_tensor(
                        frame.iloc[source_index]["ir_context_path"]
                    )
                temporal_frames.append(image_cache[source_index])
            clip = _transform_clip_frames(
                temporal_frames,
                training=self.augmentation_enabled,
                generator=generator,
                augmentation=self.augmentation_config,
            )
            clips.append(clip.unsqueeze(0))

        source_indices = torch.stack(selected_indices).unsqueeze(1)
        all_unique = source_indices.unique().numel()
        temporal_valid_fraction = float(frame["temporal_valid"].astype(bool).mean())
        context_effective_rate = float(
            frame["ir_context_effective_valid"].astype(bool).mean()
        )
        context_reliability_mean = float(frame["ir_context_reliability"].astype(float).mean())
        quality = torch.tensor(
            [
                temporal_valid_fraction,
                context_effective_rate,
                context_reliability_mean,
                all_unique / source_indices.numel(),
                1.0,
                min(num_frames / (TARGET_WINDOW_FRAMES * MAX_CLIPS), 1.0),
            ],
            dtype=torch.float32,
        )
        row0 = frame.iloc[0]
        return {
            "clips": torch.stack(clips),
            "clip_mask": torch.ones(len(windows), dtype=torch.bool),
            "num_frames": num_frames,
            "num_clips": len(windows),
            "label": int(row0["class_id"]),
            "sample_id": str(row0["sample_id"]),
            "user_id": str(row0["user_id"]),
            "class_map_hash": self.class_map_hash,
            "quality": quality,
            "quality_mask": torch.ones(len(QUALITY_NAMES), dtype=torch.bool),
            "availability": torch.tensor(
                [bool(frame["ir_context_effective_valid"].astype(bool).any())],
                dtype=torch.bool,
            ),
            "source_indices": source_indices,
            "window_bounds": torch.tensor(windows, dtype=torch.long),
            "clip_unique_frame_fraction": torch.tensor(
                clip_unique_fractions, dtype=torch.float32
            ),
        }


def collate_x3d_clips(items: Sequence[X3DClipSample]) -> dict[str, object]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    hashes = {item["class_map_hash"] for item in items}
    if len(hashes) != 1:
        raise ValueError("Cannot collate samples with different class maps")
    batch_size = len(items)
    max_clips = max(item["num_clips"] for item in items)
    clip_shape = items[0]["clips"].shape[1:]
    source_shape = items[0]["source_indices"].shape[1:]
    clips = torch.zeros((batch_size, max_clips, *clip_shape), dtype=items[0]["clips"].dtype)
    clip_mask = torch.zeros((batch_size, max_clips), dtype=torch.bool)
    source_indices = torch.full(
        (batch_size, max_clips, *source_shape), -1, dtype=torch.long
    )
    window_bounds = torch.full((batch_size, max_clips, 2), -1, dtype=torch.long)
    clip_unique = torch.zeros((batch_size, max_clips), dtype=torch.float32)
    for batch_index, item in enumerate(items):
        count = item["num_clips"]
        clips[batch_index, :count] = item["clips"]
        clip_mask[batch_index, :count] = item["clip_mask"]
        source_indices[batch_index, :count] = item["source_indices"]
        window_bounds[batch_index, :count] = item["window_bounds"]
        clip_unique[batch_index, :count] = item["clip_unique_frame_fraction"]
    return {
        "clips": clips,
        "clip_mask": clip_mask,
        "num_frames": torch.tensor([item["num_frames"] for item in items], dtype=torch.long),
        "num_clips": torch.tensor([item["num_clips"] for item in items], dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "sample_ids": tuple(item["sample_id"] for item in items),
        "user_ids": tuple(item["user_id"] for item in items),
        "class_map_hash": next(iter(hashes)),
        "quality": torch.stack([item["quality"] for item in items]),
        "quality_mask": torch.stack([item["quality_mask"] for item in items]),
        "availability": torch.stack([item["availability"] for item in items]),
        "source_indices": source_indices,
        "window_bounds": window_bounds,
        "clip_unique_frame_fraction": clip_unique,
    }
