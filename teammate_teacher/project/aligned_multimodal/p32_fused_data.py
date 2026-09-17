from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Sampler

from p30_shared_dir_roi_data import P30SharedDIRFeatureDataset, collate_p30_trials
from p31_skeleton_imu_data import P31SkeletonIMUDataset, collate_p31_trials


class P32FusedTrialDataset(Dataset[dict[str, Any]]):
    """Join P30 visual features and P31 motion inputs by exact sample/frame id."""

    def __init__(
        self,
        visual_run: str | Path,
        motion_run: str | Path,
        sample_ids: set[str] | None = None,
    ) -> None:
        self.visual = P30SharedDIRFeatureDataset(visual_run, sample_ids=sample_ids)
        self.motion = P31SkeletonIMUDataset(motion_run, sample_ids=sample_ids)
        motion_lookup = {
            row["sample_id"]: index for index, row in enumerate(self.motion.rows)
        }
        self.pairs: list[tuple[int, int]] = []
        self.rows: list[dict[str, str]] = []
        for visual_index, row in enumerate(self.visual.rows):
            sample_id = row["sample_id"]
            if sample_id not in motion_lookup:
                raise RuntimeError(f"P31 motion trial missing for {sample_id}")
            self.pairs.append((visual_index, motion_lookup[sample_id]))
            self.rows.append(row)
        if len(self.pairs) != len(self.motion):
            visual_ids = {row["sample_id"] for row in self.visual.rows}
            extra = [row["sample_id"] for row in self.motion.rows if row["sample_id"] not in visual_ids]
            raise RuntimeError(f"P31 contains trials absent from P30: {extra[:3]}")

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def frame_lengths(self) -> list[int]:
        return [int(row["frames"]) for row in self.rows]

    def __getitem__(self, index: int) -> dict[str, Any]:
        visual_index, motion_index = self.pairs[index]
        visual = self.visual[visual_index]
        motion = self.motion[motion_index]
        if visual["sample_id"] != motion["sample_id"]:
            raise RuntimeError(f"P30/P31 sample mismatch at dataset index {index}")
        if visual["frame_ids"] != motion["frame_ids"]:
            raise RuntimeError(f"P30/P31 frame mismatch: {visual['sample_id']}")
        if int(visual["class_id"]) != int(motion["class_id"]):
            raise RuntimeError(f"P30/P31 label mismatch: {visual['sample_id']}")
        return {"visual": visual, "motion": motion}


def collate_p32_trials(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty P32 batch")
    visual = collate_p30_trials([item["visual"] for item in items])
    motion = collate_p31_trials([item["motion"] for item in items])
    if visual["sample_id"] != motion["sample_id"]:
        raise RuntimeError("P30/P31 sample order changed during collation")
    if visual["frame_ids"] != motion["frame_ids"]:
        raise RuntimeError("P30/P31 frame ids changed during collation")
    if not visual["label"].equal(motion["label"]):
        raise RuntimeError("P30/P31 labels differ during collation")
    output = dict(visual)
    for key, value in motion.items():
        if key not in {"sample_id", "user_id", "frame_ids", "time_position", "frame_mask", "label"}:
            output[key] = value
    return output


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle locally by length to reduce full-sequence padding without frame loss."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_multiplier: int = 20,
        seed: int = 20260804,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            generator.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches


class BalancedLengthBucketBatchSampler(Sampler[list[int]]):
    """Class-balanced sampling plus local length bucketing.

    Each epoch contains exactly ``len(labels)`` sampled trials. Classes are drawn
    uniformly and trials within a class are drawn uniformly with replacement.
    The sampled indices are then length-bucketed, so class balancing does not
    require giving up the padding reduction used by the full-sequence model.
    """

    def __init__(
        self,
        lengths: list[int],
        labels: list[int],
        batch_size: int,
        drop_last: bool = False,
        bucket_multiplier: int = 20,
        seed: int = 20260804,
        samples_per_epoch: int | None = None,
    ) -> None:
        if len(lengths) != len(labels):
            raise ValueError("lengths and labels must have the same size")
        if not lengths:
            raise ValueError("balanced sampler requires at least one sample")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = lengths
        self.labels = [int(value) for value in labels]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.bucket_size = max(self.batch_size, self.batch_size * bucket_multiplier)
        self.seed = int(seed)
        self.epoch = 0
        self.samples_per_epoch = int(samples_per_epoch or len(labels))
        self.by_class: dict[int, list[int]] = {}
        for index, label in enumerate(self.labels):
            self.by_class.setdefault(label, []).append(index)
        self.classes = sorted(self.by_class)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.samples_per_epoch // self.batch_size
        return (self.samples_per_epoch + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        sampled = [
            generator.choice(self.by_class[generator.choice(self.classes)])
            for _ in range(self.samples_per_epoch)
        ]
        generator.shuffle(sampled)
        batches: list[list[int]] = []
        for start in range(0, len(sampled), self.bucket_size):
            bucket = sampled[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        generator.shuffle(batches)
        yield from batches


class BalancedCostBucketBatchSampler(Sampler[list[int]]):
    """Class-balanced sampling with a stable padded frame/IMU-point budget.

    A fixed trial count is unsafe for this dataset: the collator pads both the
    camera sequence and each raw IMU stream to the longest item in the batch.
    This sampler keeps the same class-balanced epoch distribution while using
    smaller batches only for expensive long trials.
    """

    def __init__(
        self,
        frame_lengths: list[int],
        point_lengths: list[int],
        labels: list[int],
        maximum_batch_size: int = 32,
        frame_budget: int = 2240,
        point_budget: int = 11200,
        bucket_multiplier: int = 20,
        seed: int = 20260804,
        samples_per_epoch: int | None = None,
    ) -> None:
        if not (len(frame_lengths) == len(point_lengths) == len(labels)):
            raise ValueError("frame_lengths, point_lengths and labels must align")
        self.frame_lengths = [max(1, int(value)) for value in frame_lengths]
        self.point_lengths = [max(1, int(value)) for value in point_lengths]
        self.labels = [int(value) for value in labels]
        self.maximum_batch_size = int(maximum_batch_size)
        self.frame_budget = int(frame_budget)
        self.point_budget = int(point_budget)
        self.bucket_size = max(
            self.maximum_batch_size,
            self.maximum_batch_size * int(bucket_multiplier),
        )
        self.seed = int(seed)
        self.samples_per_epoch = int(samples_per_epoch or len(labels))
        self.epoch = 0
        self.by_class: dict[int, list[int]] = {}
        for index, label in enumerate(self.labels):
            self.by_class.setdefault(label, []).append(index)
        self.classes = sorted(self.by_class)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # A lower bound; DataLoader does not require an exact value for iteration.
        return (self.samples_per_epoch + self.maximum_batch_size - 1) // self.maximum_batch_size

    def _cost(self, index: int) -> float:
        return max(
            self.frame_lengths[index] / 70.0,
            self.point_lengths[index] / 350.0,
        )

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        sampled = [
            generator.choice(self.by_class[generator.choice(self.classes)])
            for _ in range(self.samples_per_epoch)
        ]
        generator.shuffle(sampled)
        batches: list[list[int]] = []
        for start in range(0, len(sampled), self.bucket_size):
            bucket = sampled[start : start + self.bucket_size]
            bucket.sort(key=self._cost)
            batch: list[int] = []
            maximum_frames = 1
            maximum_points = 1
            for index in bucket:
                proposed_size = len(batch) + 1
                proposed_frames = max(maximum_frames, self.frame_lengths[index])
                proposed_points = max(maximum_points, self.point_lengths[index])
                exceeds = (
                    proposed_size > self.maximum_batch_size
                    or proposed_size * proposed_frames > self.frame_budget
                    or proposed_size * proposed_points > self.point_budget
                )
                if batch and exceeds:
                    batches.append(batch)
                    batch = []
                    proposed_size = 1
                    proposed_frames = self.frame_lengths[index]
                    proposed_points = self.point_lengths[index]
                batch.append(index)
                maximum_frames = proposed_frames
                maximum_points = proposed_points
            if batch:
                batches.append(batch)
        generator.shuffle(batches)
        yield from batches
