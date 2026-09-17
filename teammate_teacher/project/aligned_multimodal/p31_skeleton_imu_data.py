from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from p31_skeleton_imu_preprocessing import (
    COMMON_PART_NAMES,
    H36M_JOINT_NAMES,
    IMU_CHANNEL_NAMES,
    IMU_DEVICE_NAMES,
    SKELETON_FEATURE_NAMES,
    SKELETON_RELATION_NAMES,
    safe_trial_path,
)


class P31SkeletonIMUDataset(Dataset[dict[str, Any]]):
    """Variable-length Step 8/9 trials with every valid raw IMU point."""

    def __init__(
        self,
        cache_run: str | Path,
        sample_ids: set[str] | None = None,
    ) -> None:
        self.cache_run = Path(cache_run).resolve()
        summary_path = self.cache_run / "trial_summary.csv"
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if sample_ids is not None:
            rows = [row for row in rows if row["sample_id"] in sample_ids]
        self.rows = sorted(
            rows,
            key=lambda row: (int(row["class_id"]), row["user_id"], row["trial_id"]),
        )
        if not self.rows:
            raise RuntimeError(f"no P31 trials selected from {summary_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def cache_path(self, sample_id: str) -> Path:
        return (
            self.cache_run
            / "trial_motion_cache"
            / safe_trial_path(sample_id).with_suffix(".npz")
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with np.load(self.cache_path(row["sample_id"]), allow_pickle=False) as cache:
            contracts = (
                ("joint_names", H36M_JOINT_NAMES),
                ("common_part_names", COMMON_PART_NAMES),
                ("skeleton_feature_names", SKELETON_FEATURE_NAMES),
                ("skeleton_relation_names", SKELETON_RELATION_NAMES),
                ("imu_device_names", IMU_DEVICE_NAMES),
                ("imu_channel_names", IMU_CHANNEL_NAMES),
            )
            for key, expected in contracts:
                actual = tuple(str(value) for value in cache[key])
                if actual != expected:
                    raise RuntimeError(
                        f"P31 cache contract {key} differs for {row['sample_id']}"
                    )

            frame_ids = [str(value) for value in cache["frame_ids"]]
            frame_times = torch.from_numpy(cache["frame_time_seconds"].astype(np.float32))
            skeleton_features = torch.from_numpy(
                cache["skeleton_features"].astype(np.float32)
            )
            skeleton_feature_mask = torch.from_numpy(
                cache["skeleton_feature_mask"].astype(bool)
            )
            skeleton_joint_mask = torch.from_numpy(cache["skeleton_joint_mask"].astype(bool))
            skeleton_relations = torch.from_numpy(
                cache["skeleton_relations"].astype(np.float32)
            )
            skeleton_relation_mask = torch.from_numpy(
                cache["skeleton_relation_mask"].astype(bool)
            )
            skeleton_frame_quality = torch.from_numpy(
                cache["skeleton_frame_quality"].astype(np.float32)
            )
            flat_values = cache["imu_values"].astype(np.float32)
            flat_times = cache["imu_time_seconds"].astype(np.float32)
            flat_frame_index = cache["imu_frame_index"].astype(np.int64)
            offsets = cache["imu_device_offsets"].astype(np.int64)
            imu_interval_counts = torch.from_numpy(
                cache["imu_interval_counts"].astype(np.int64)
            )
            imu_device_mask = torch.from_numpy(cache["imu_device_mask"].astype(bool))

        lengths = np.diff(offsets)
        maximum_points = int(lengths.max(initial=0))
        imu_values = torch.zeros(
            len(IMU_DEVICE_NAMES), maximum_points, len(IMU_CHANNEL_NAMES)
        )
        imu_time_seconds = torch.zeros(len(IMU_DEVICE_NAMES), maximum_points)
        imu_frame_index = torch.full(
            (len(IMU_DEVICE_NAMES), maximum_points), -1, dtype=torch.long
        )
        imu_point_mask = torch.zeros(
            len(IMU_DEVICE_NAMES), maximum_points, dtype=torch.bool
        )
        for device_index, length_value in enumerate(lengths):
            length = int(length_value)
            if length == 0:
                continue
            start, end = int(offsets[device_index]), int(offsets[device_index + 1])
            imu_values[device_index, :length] = torch.from_numpy(flat_values[start:end])
            imu_time_seconds[device_index, :length] = torch.from_numpy(flat_times[start:end])
            imu_frame_index[device_index, :length] = torch.from_numpy(
                flat_frame_index[start:end]
            )
            imu_point_mask[device_index, :length] = True

        if len(frame_times) > 1:
            duration = frame_times[-1].clamp_min(1e-6)
            time_position = frame_times / duration
        else:
            time_position = torch.zeros_like(frame_times)
        return {
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
            "user_id": row["user_id"],
            "frame_ids": frame_ids,
            "frame_time_seconds": frame_times,
            "time_position": time_position,
            "skeleton_features": skeleton_features,
            "skeleton_feature_mask": skeleton_feature_mask,
            "skeleton_joint_mask": skeleton_joint_mask,
            "skeleton_relations": skeleton_relations,
            "skeleton_relation_mask": skeleton_relation_mask,
            "skeleton_frame_quality": skeleton_frame_quality,
            "imu_values": imu_values,
            "imu_time_seconds": imu_time_seconds,
            "imu_frame_index": imu_frame_index,
            "imu_point_mask": imu_point_mask,
            "imu_interval_counts": imu_interval_counts,
            "imu_device_mask": imu_device_mask,
        }


def collate_p31_trials(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty P31 batch")
    batch_size = len(items)
    maximum_frames = max(len(item["frame_ids"]) for item in items)
    maximum_points = max(item["imu_values"].shape[1] for item in items)
    maximum_points = max(maximum_points, 1)

    skeleton_features = torch.zeros(
        batch_size,
        maximum_frames,
        len(H36M_JOINT_NAMES),
        len(SKELETON_FEATURE_NAMES),
    )
    skeleton_feature_mask = torch.zeros_like(skeleton_features, dtype=torch.bool)
    skeleton_joint_mask = torch.zeros(
        batch_size, maximum_frames, len(H36M_JOINT_NAMES), dtype=torch.bool
    )
    skeleton_relations = torch.zeros(
        batch_size, maximum_frames, len(SKELETON_RELATION_NAMES)
    )
    skeleton_relation_mask = torch.zeros_like(skeleton_relations, dtype=torch.bool)
    skeleton_frame_quality = torch.zeros(batch_size, maximum_frames)
    frame_times = torch.zeros(batch_size, maximum_frames)
    time_position = torch.zeros(batch_size, maximum_frames)
    frame_mask = torch.zeros(batch_size, maximum_frames, dtype=torch.bool)

    imu_values = torch.zeros(
        batch_size,
        len(IMU_DEVICE_NAMES),
        maximum_points,
        len(IMU_CHANNEL_NAMES),
    )
    imu_times = torch.zeros(batch_size, len(IMU_DEVICE_NAMES), maximum_points)
    imu_frame_index = torch.full(
        (batch_size, len(IMU_DEVICE_NAMES), maximum_points), -1, dtype=torch.long
    )
    imu_point_mask = torch.zeros(
        batch_size, len(IMU_DEVICE_NAMES), maximum_points, dtype=torch.bool
    )
    imu_interval_counts = torch.zeros(
        batch_size, maximum_frames, len(IMU_DEVICE_NAMES), dtype=torch.long
    )
    imu_device_mask = torch.zeros(batch_size, len(IMU_DEVICE_NAMES), dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)

    for batch_index, item in enumerate(items):
        frame_count = len(item["frame_ids"])
        point_count = item["imu_values"].shape[1]
        skeleton_features[batch_index, :frame_count] = item["skeleton_features"]
        skeleton_feature_mask[batch_index, :frame_count] = item[
            "skeleton_feature_mask"
        ]
        skeleton_joint_mask[batch_index, :frame_count] = item["skeleton_joint_mask"]
        skeleton_relations[batch_index, :frame_count] = item["skeleton_relations"]
        skeleton_relation_mask[batch_index, :frame_count] = item[
            "skeleton_relation_mask"
        ]
        skeleton_frame_quality[batch_index, :frame_count] = item[
            "skeleton_frame_quality"
        ]
        frame_times[batch_index, :frame_count] = item["frame_time_seconds"]
        time_position[batch_index, :frame_count] = item["time_position"]
        frame_mask[batch_index, :frame_count] = True
        if point_count:
            imu_values[batch_index, :, :point_count] = item["imu_values"]
            imu_times[batch_index, :, :point_count] = item["imu_time_seconds"]
            imu_frame_index[batch_index, :, :point_count] = item["imu_frame_index"]
            imu_point_mask[batch_index, :, :point_count] = item["imu_point_mask"]
        imu_interval_counts[batch_index, :frame_count] = item["imu_interval_counts"]
        imu_device_mask[batch_index] = item["imu_device_mask"]
        labels[batch_index] = int(item["class_id"])

    return {
        "sample_id": [item["sample_id"] for item in items],
        "user_id": [item["user_id"] for item in items],
        "frame_ids": [item["frame_ids"] for item in items],
        "frame_time_seconds": frame_times,
        "time_position": time_position,
        "frame_mask": frame_mask,
        "skeleton_features": skeleton_features,
        "skeleton_feature_mask": skeleton_feature_mask,
        "skeleton_joint_mask": skeleton_joint_mask,
        "skeleton_relations": skeleton_relations,
        "skeleton_relation_mask": skeleton_relation_mask,
        "skeleton_frame_quality": skeleton_frame_quality,
        "imu_values": imu_values,
        "imu_time_seconds": imu_times,
        "imu_frame_index": imu_frame_index,
        "imu_point_mask": imu_point_mask,
        "imu_interval_counts": imu_interval_counts,
        "imu_device_mask": imu_device_mask,
        "label": labels,
    }
