from __future__ import annotations

from collections.abc import Iterator, Sequence
import hashlib
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


IR_VIEWS = ("ir_context", "ir_left", "ir_right", "ir_relation")
DEPTH_VIEWS = ("depth_context", "depth_relation")
VIEW_NAMES = (*IR_VIEWS, *DEPTH_VIEWS)
DEPTH_REPRESENTATIONS = {"raw", "relative", "raw+relative"}
QUALITY_NAMES = (
    "temporal_valid_fraction",
    *(f"{view}_effective_rate" for view in VIEW_NAMES),
    *(f"{view}_reliability_mean" for view in VIEW_NAMES),
    "depth_context_pixel_coverage_mean",
    "depth_relation_pixel_coverage_mean",
    "relative_stats_valid",
)


def class_map_hash(class_rows: pd.DataFrame) -> str:
    canonical = [
        {"class_id": int(row.class_id), "action_name": str(row.action_name)}
        for row in class_rows.sort_values("class_id").itertuples(index=False)
    ]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


class IRPrimaryFullSequenceDataset(Dataset[dict[str, object]]):
    """One complete train/validation trial per item, with no offline or online sampling."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        split: str,
        depth_representation: str,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("Only train and val splits are allowed")
        if depth_representation not in DEPTH_REPRESENTATIONS:
            raise ValueError(f"Unknown Depth representation: {depth_representation}")
        frame = (
            manifest.copy()
            if isinstance(manifest, pd.DataFrame)
            else pd.read_csv(manifest, encoding="utf-8-sig")
        )
        if set(frame.split.astype(str)) - {"train", "val"}:
            raise ValueError("Manifest contains a non-train/val split")
        classes = frame[["class_id", "action_name"]].drop_duplicates()
        if classes.class_id.nunique() != 40 or len(classes) != 40:
            raise ValueError("Expected one class-map row for each of 40 classes")
        self.class_rows = classes.sort_values("class_id").reset_index(drop=True)
        self.class_map_hash = class_map_hash(self.class_rows)
        self.class_names = self.class_rows.action_name.astype(str).tolist()
        self.depth_representation = depth_representation
        selected = frame[frame.split == split].copy()
        self.user_ids = sorted(selected.user_id.astype(str).unique())
        self.samples: list[pd.DataFrame] = []
        self.lengths: list[int] = []
        self.sample_ids: list[str] = []
        for sample_id, group in selected.groupby("sample_id", sort=False):
            ordered = group.sort_values("source_frame_index").reset_index(drop=True)
            indices = ordered.source_frame_index.to_numpy(dtype=np.int64)
            if not np.array_equal(indices, np.arange(len(ordered))):
                raise ValueError(f"Non-contiguous frame order for {sample_id}")
            if not ordered.temporal_valid.astype(bool).all():
                raise ValueError(f"Offline temporal-invalid frame found for {sample_id}")
            self.samples.append(ordered)
            self.lengths.append(len(ordered))
            self.sample_ids.append(str(sample_id))
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Duplicate sample IDs")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _relative_values(context: np.ndarray, context_valid: np.ndarray) -> tuple[float, float, bool]:
        valid_values = context[context_valid]
        frames, height, width = context.shape
        required = max(1024, math.ceil(0.01 * frames * height * width))
        if len(valid_values) < required:
            return 0.0, 1.0, False
        median = float(np.median(valid_values))
        q25, q75 = np.percentile(valid_values, [25, 75])
        return median, max(float(q75 - q25), 8.0), True

    def __getitem__(self, index: int) -> dict[str, object]:
        frame = self.samples[index]
        length = len(frame)
        ir = np.zeros((length, 4, 1, 256, 256), dtype=np.float32)
        depth_values = np.zeros((length, 2, 256, 256), dtype=np.uint8)
        depth_pixel_valid = np.zeros((length, 2, 1, 256, 256), dtype=bool)
        view_valid = np.zeros((length, 6), dtype=bool)
        view_reliability = np.zeros((length, 6), dtype=np.float32)

        for time, row in enumerate(frame.itertuples(index=False)):
            for view_index, view in enumerate(IR_VIEWS):
                valid = bool(getattr(row, f"{view}_effective_valid"))
                image = _read_gray(getattr(row, f"{view}_path")).astype(np.float32) / 255.0
                ir[time, view_index, 0] = (2.0 * image - 1.0) * valid
                view_valid[time, view_index] = valid
                view_reliability[time, view_index] = float(getattr(row, f"{view}_reliability")) * valid
            for local_index, view in enumerate(DEPTH_VIEWS):
                valid = bool(getattr(row, f"{view}_effective_valid"))
                values = _read_gray(getattr(row, f"{view}_ordinal_path"))
                pixel_valid = _read_gray(getattr(row, f"{view}_pixel_valid_path")) > 0
                depth_values[time, local_index] = values
                depth_pixel_valid[time, local_index, 0] = pixel_valid
                output_index = 4 + local_index
                view_valid[time, output_index] = valid
                view_reliability[time, output_index] = float(getattr(row, f"{view}_reliability")) * valid

        median, scale, stats_valid = self._relative_values(
            depth_values[:, 0], depth_pixel_valid[:, 0, 0],
        )
        raw = depth_values.astype(np.float32) / 255.0
        if stats_valid:
            relative = np.clip((depth_values.astype(np.float32) - median) / scale, -4.0, 4.0)
        else:
            relative = 2.0 * raw - 1.0
        pixel_valid_float = depth_pixel_valid[:, :, 0].astype(np.float32)
        effective = view_valid[:, 4:].astype(np.float32)[..., None, None]
        depth = np.zeros((length, 2, 3, 256, 256), dtype=np.float32)
        if self.depth_representation in {"raw", "raw+relative"}:
            depth[:, :, 0] = raw * pixel_valid_float * effective
        if self.depth_representation in {"relative", "raw+relative"}:
            depth[:, :, 1] = relative * pixel_valid_float * effective
        depth[:, :, 2] = pixel_valid_float * effective

        effective_rates = view_valid.mean(axis=0, dtype=np.float32)
        reliability_means = view_reliability.mean(axis=0, dtype=np.float32)
        pixel_coverage = depth_pixel_valid[:, :, 0].mean(axis=(0, 2, 3), dtype=np.float32)
        quality = np.r_[
            np.float32(1.0), effective_rates, reliability_means, pixel_coverage,
            np.float32(stats_valid),
        ].astype(np.float32)
        if len(quality) != len(QUALITY_NAMES):
            raise RuntimeError("Quality schema mismatch")
        timestamps = np.cumsum(frame.inter_frame_delta_seconds.to_numpy(dtype=np.float32))
        row0 = frame.iloc[0]
        return {
            "ir": torch.from_numpy(ir),
            "depth": torch.from_numpy(depth),
            "depth_pixel_valid": torch.from_numpy(depth_pixel_valid),
            "view_valid": torch.from_numpy(view_valid),
            "view_reliability": torch.from_numpy(view_reliability),
            "quality": torch.from_numpy(quality),
            "quality_mask": torch.ones(len(QUALITY_NAMES), dtype=torch.bool),
            "availability": torch.ones(1, dtype=torch.bool),
            "timestamps": torch.from_numpy(timestamps),
            "frame_indices": torch.arange(length, dtype=torch.long),
            "label": int(row0.class_id),
            "sample_id": str(row0.sample_id),
            "user_id": str(row0.user_id),
            "length": length,
            "relative_median": median,
            "relative_scale": scale,
            "relative_stats_valid": stats_valid,
        }


def collate_full_sequences(items: Sequence[dict[str, object]]) -> dict[str, object]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch = len(items)
    maximum = max(int(item["length"]) for item in items)

    def padded(name: str, fill: float | bool = 0) -> torch.Tensor:
        first = items[0][name]
        if not isinstance(first, torch.Tensor):
            raise TypeError(f"{name} is not a tensor")
        output = torch.full((batch, maximum, *first.shape[1:]), fill, dtype=first.dtype)
        for row, item in enumerate(items):
            value = item[name]
            assert isinstance(value, torch.Tensor)
            output[row, : len(value)] = value
        return output

    lengths = torch.tensor([int(item["length"]) for item in items], dtype=torch.long)
    temporal_mask = torch.arange(maximum).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "ir": padded("ir"),
        "depth": padded("depth"),
        "depth_pixel_valid": padded("depth_pixel_valid", False),
        "view_valid": padded("view_valid", False),
        "view_reliability": padded("view_reliability"),
        "timestamps": padded("timestamps"),
        "frame_indices": padded("frame_indices", -1),
        "temporal_mask": temporal_mask,
        "quality": torch.stack([item["quality"] for item in items]),
        "quality_mask": torch.stack([item["quality_mask"] for item in items]),
        "availability": torch.stack([item["availability"] for item in items]),
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.long),
        "lengths": lengths,
        "sample_ids": tuple(str(item["sample_id"]) for item in items),
        "user_ids": tuple(str(item["user_id"]) for item in items),
        "relative_stats_valid": torch.tensor(
            [bool(item["relative_stats_valid"]) for item in items], dtype=torch.bool,
        ),
    }


class FrameBudgetBatchSampler(Sampler[list[int]]):
    """Length-bucketed batches bounded by padded frames (max_length * batch_size)."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        max_frames: int,
        max_samples: int,
        shuffle: bool,
        seed: int,
        bucket_size: int = 128,
    ) -> None:
        if not lengths or min(lengths) <= 0:
            raise ValueError("All sequence lengths must be positive")
        if max_frames < max(lengths):
            raise ValueError("max_frames is smaller than the longest sequence")
        if max_samples <= 0 or bucket_size <= 0:
            raise ValueError("max_samples and bucket_size must be positive")
        self.lengths = tuple(int(value) for value in lengths)
        self.max_frames = int(max_frames)
        self.max_samples = int(max_samples)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = int(bucket_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        ordered = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        buckets = [ordered[start : start + self.bucket_size] for start in range(0, len(ordered), self.bucket_size)]
        if self.shuffle:
            generator = random.Random(self.seed + self.epoch)
            for bucket in buckets:
                generator.shuffle(bucket)
            generator.shuffle(buckets)
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for index in (value for bucket in buckets for value in bucket):
            proposed_max = max(current_max, self.lengths[index])
            if current and (
                len(current) >= self.max_samples
                or proposed_max * (len(current) + 1) > self.max_frames
            ):
                batches.append(current)
                current = []
                current_max = 0
            current.append(index)
            current_max = max(current_max, self.lengths[index])
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())
