from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from p30_shared_dir_roi_model import MODALITY_NAMES, REGION_NAMES
from p31_skeleton_imu_preprocessing import (
    IMU_DEVICE_NAMES,
    SKELETON_FEATURE_NAMES,
    SKELETON_RELATION_NAMES,
)
from p46_event_preprocessing import P46_LOCAL_REGIONS
from p46_protocol import HARD_CLASS_TO_INDEX, p46_split_for_user


CONTEXT_REGIONS = ("full_body", "global_fallback")


def safe_path(sample_id: str) -> Path:
    parts = sample_id.split("/")
    if len(parts) != 3 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe source id: {sample_id}")
    return Path(*parts)


class P46EventDataset(Dataset[dict[str, Any]]):
    """Exact all-frame join of P46 local/event inputs and P30 low-res context."""

    def __init__(
        self,
        event_run: str | Path,
        context_run: str | Path,
        split: str | None = None,
        sample_ids: set[str] | None = None,
        load_context: bool = True,
    ) -> None:
        self.event_run = Path(event_run).resolve()
        self.context_run = Path(context_run).resolve()
        self.load_context = bool(load_context)
        with (self.event_run / "trial_summary.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        rows = [
            {**row, "p46_split": p46_split_for_user(row["user_id"])}
            for row in rows
        ]
        if split is not None:
            if split not in {"train", "val"}:
                raise ValueError("split must be train, val or None")
            rows = [row for row in rows if row["p46_split"] == split]
        if sample_ids is not None:
            rows = [
                row
                for row in rows
                if row["sample_id"] in sample_ids or row["source_id"] in sample_ids
            ]
        self.rows = sorted(
            rows,
            key=lambda row: (
                0 if row["p46_split"] == "train" else 1,
                int(row["class_id"]),
                row["user_id"],
                row["trial_id"],
            ),
        )
        if not self.rows:
            raise RuntimeError("no P46 event trials selected")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def frame_lengths(self) -> list[int]:
        return [int(row["frames"]) for row in self.rows]

    @property
    def imu_point_lengths(self) -> list[int]:
        return [int(row["imu_points"]) for row in self.rows]

    def _event_path(self, source_id: str) -> Path:
        return self.event_run / "trial_event_cache" / safe_path(source_id).with_suffix(".npz")

    def _context_path(self, source_id: str) -> Path:
        relative = safe_path(source_id).with_suffix(".npz")
        lean = self.context_run / "trial_context_cache" / relative
        return lean if lean.is_file() else self.context_run / "trial_feature_cache" / relative

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source_id = row["source_id"]
        with np.load(self._event_path(source_id), allow_pickle=False) as cache:
            local_names = tuple(str(value) for value in cache["local_region_names"])
            if local_names != P46_LOCAL_REGIONS:
                raise RuntimeError(f"P46 local region contract changed: {source_id}")
            frame_ids = [str(value) for value in cache["frame_ids"]]
            frame_time_seconds = torch.from_numpy(
                cache["frame_time_seconds"].astype(np.float32)
            )
            item: dict[str, Any] = {
                "arm_spatial_features": torch.from_numpy(
                    cache["arm_spatial_features"].astype(np.float32)
                ),
                "detail_spatial_features": torch.from_numpy(
                    cache["detail_spatial_features"].astype(np.float32)
                ),
                "local_geometry_features": torch.from_numpy(
                    cache["local_geometry_features"].astype(np.float32)
                ),
                "oriented_roi_geometry": torch.from_numpy(
                    cache["oriented_roi_geometry"].astype(np.float32)
                ),
                "oriented_angle_valid": torch.from_numpy(
                    cache["oriented_angle_valid"].astype(bool)
                ),
                "local_roi_valid": torch.from_numpy(cache["roi_valid"].astype(bool)),
                "local_roi_quality": torch.from_numpy(
                    cache["roi_quality"].astype(np.float32)
                ),
                "local_roi_source": torch.from_numpy(
                    cache["roi_source"].astype(np.int64)
                ),
                "local_roi_clipped_ratio": torch.from_numpy(
                    cache["roi_clipped_ratio"].astype(np.float32)
                ),
                "pose_quality_factor": torch.from_numpy(
                    cache["pose_quality_factor"].astype(np.float32)
                ),
                "skeleton_features": torch.from_numpy(
                    cache["skeleton_features"].astype(np.float32)
                ),
                "skeleton_feature_mask": torch.from_numpy(
                    cache["skeleton_feature_mask"].astype(bool)
                ),
                "skeleton_joint_mask": torch.from_numpy(
                    cache["skeleton_joint_mask"].astype(bool)
                ),
                "skeleton_relations": torch.from_numpy(
                    cache["skeleton_relations"].astype(np.float32)
                ),
                "skeleton_relation_mask": torch.from_numpy(
                    cache["skeleton_relation_mask"].astype(bool)
                ),
                "skeleton_frame_quality": torch.from_numpy(
                    cache["skeleton_frame_quality"].astype(np.float32)
                ),
                "body_axes_camera": torch.from_numpy(
                    cache["body_axes_camera"].astype(np.float32)
                ),
                "body_axes_raw_valid": torch.from_numpy(
                    cache["body_axes_raw_valid"].astype(bool)
                ),
                "imu_interval_counts": torch.from_numpy(
                    cache["imu_interval_counts"].astype(np.int64)
                ),
                "imu_device_mask": torch.from_numpy(
                    cache["imu_device_mask"].astype(bool)
                ),
            }
            flat_values = cache["imu_values"].astype(np.float32)
            flat_raw = cache["imu_raw_vectors"].astype(np.float32)
            flat_times = cache["imu_time_seconds"].astype(np.float32)
            flat_frame_index = cache["imu_frame_index"].astype(np.int64)
            offsets = cache["imu_device_offsets"].astype(np.int64)

        if self.load_context:
            with np.load(self._context_path(source_id), allow_pickle=False) as context:
                modalities = tuple(str(value) for value in context["modality_names"])
                context_frame_ids = [str(value) for value in context["frame_ids"]]
                if "context_features" in context.files:
                    regions = tuple(str(value) for value in context["context_region_names"])
                    if modalities != MODALITY_NAMES or regions != CONTEXT_REGIONS:
                        raise RuntimeError(f"P46 lean context contract changed: {source_id}")
                    item["context_features"] = torch.from_numpy(
                        context["context_features"].astype(np.float32)
                    )
                    item["context_valid"] = torch.from_numpy(
                        context["context_valid"].astype(bool)
                    )
                    item["context_quality"] = torch.from_numpy(
                        context["context_quality"].astype(np.float32)
                    )
                else:
                    regions = tuple(str(value) for value in context["region_names"])
                    if modalities != MODALITY_NAMES or regions != REGION_NAMES:
                        raise RuntimeError(f"P30 context contract changed: {source_id}")
                    indices = [regions.index(name) for name in CONTEXT_REGIONS]
                    item["context_features"] = torch.from_numpy(
                        context["features"][:, :, indices].astype(np.float32)
                    )
                    item["context_valid"] = torch.from_numpy(
                        context["roi_valid"][:, indices].astype(bool)
                    )
                    item["context_quality"] = torch.from_numpy(
                        context["roi_quality"][:, indices].astype(np.float32)
                    )
            if context_frame_ids != frame_ids:
                raise RuntimeError(f"P30/P46 frame mismatch: {source_id}")

        lengths = np.diff(offsets)
        maximum_points = int(lengths.max(initial=0))
        imu_values = torch.zeros(len(IMU_DEVICE_NAMES), maximum_points, 10)
        imu_raw_vectors = torch.zeros(len(IMU_DEVICE_NAMES), maximum_points, 6)
        imu_times = torch.zeros(len(IMU_DEVICE_NAMES), maximum_points)
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
            start, stop = int(offsets[device_index]), int(offsets[device_index + 1])
            imu_values[device_index, :length] = torch.from_numpy(flat_values[start:stop])
            imu_raw_vectors[device_index, :length] = torch.from_numpy(flat_raw[start:stop])
            imu_times[device_index, :length] = torch.from_numpy(flat_times[start:stop])
            imu_frame_index[device_index, :length] = torch.from_numpy(
                flat_frame_index[start:stop]
            )
            imu_point_mask[device_index, :length] = True
        item.update(
            {
                "sample_id": row["sample_id"],
                "source_id": source_id,
                "user_id": row["user_id"],
                "p46_split": row["p46_split"],
                "class_id": int(row["class_id"]),
                "detail_index": HARD_CLASS_TO_INDEX[int(row["class_id"])],
                "frame_ids": frame_ids,
                "frame_time_seconds": frame_time_seconds,
                "time_position": (
                    frame_time_seconds / frame_time_seconds[-1].clamp_min(1e-6)
                    if len(frame_time_seconds) > 1
                    else torch.zeros_like(frame_time_seconds)
                ),
                "imu_values": imu_values,
                "imu_raw_vectors": imu_raw_vectors,
                "imu_time_seconds": imu_times,
                "imu_frame_index": imu_frame_index,
                "imu_point_mask": imu_point_mask,
            }
        )
        return item


def collate_p46_events(
    items: list[dict[str, Any]], *, include_context: bool = True
) -> dict[str, Any]:
    if not items:
        raise ValueError("empty P46 batch")
    batch = len(items)
    frames = max(len(item["frame_ids"]) for item in items)
    points = max(max(item["imu_values"].shape[1] for item in items), 1)

    output: dict[str, Any] = {
        "sample_id": [item["sample_id"] for item in items],
        "source_id": [item["source_id"] for item in items],
        "user_id": [item["user_id"] for item in items],
        "p46_split": [item["p46_split"] for item in items],
        "frame_ids": [item["frame_ids"] for item in items],
        "label": torch.tensor([item["class_id"] for item in items], dtype=torch.long),
        "detail_index": torch.tensor(
            [item["detail_index"] for item in items], dtype=torch.long
        ),
        "frame_mask": torch.zeros(batch, frames, dtype=torch.bool),
        "frame_time_seconds": torch.zeros(batch, frames),
        "time_position": torch.zeros(batch, frames),
        "arm_spatial_features": torch.zeros(batch, frames, 2, 2, 3, 3, 128),
        "detail_spatial_features": torch.zeros(batch, frames, 2, 3, 5, 5, 128),
        "local_geometry_features": torch.zeros(batch, frames, 5, 5, 5, 6),
        "oriented_roi_geometry": torch.zeros(batch, frames, 5, 6),
        "oriented_angle_valid": torch.zeros(batch, frames, 5, dtype=torch.bool),
        "local_roi_valid": torch.zeros(batch, frames, 5, dtype=torch.bool),
        "local_roi_quality": torch.zeros(batch, frames, 5),
        "local_roi_source": torch.zeros(batch, frames, 5, dtype=torch.long),
        "local_roi_clipped_ratio": torch.zeros(batch, frames, 5),
        "pose_quality_factor": torch.zeros(batch, frames),
        "skeleton_features": torch.zeros(
            batch, frames, 17, len(SKELETON_FEATURE_NAMES)
        ),
        "skeleton_feature_mask": torch.zeros(
            batch, frames, 17, len(SKELETON_FEATURE_NAMES), dtype=torch.bool
        ),
        "skeleton_joint_mask": torch.zeros(batch, frames, 17, dtype=torch.bool),
        "skeleton_relations": torch.zeros(
            batch, frames, len(SKELETON_RELATION_NAMES)
        ),
        "skeleton_relation_mask": torch.zeros(
            batch, frames, len(SKELETON_RELATION_NAMES), dtype=torch.bool
        ),
        "skeleton_frame_quality": torch.zeros(batch, frames),
        "body_axes_camera": torch.zeros(batch, frames, 3, 3),
        "body_axes_raw_valid": torch.zeros(batch, frames, dtype=torch.bool),
        "imu_values": torch.zeros(batch, len(IMU_DEVICE_NAMES), points, 10),
        "imu_raw_vectors": torch.zeros(batch, len(IMU_DEVICE_NAMES), points, 6),
        "imu_time_seconds": torch.zeros(batch, len(IMU_DEVICE_NAMES), points),
        "imu_frame_index": torch.full(
            (batch, len(IMU_DEVICE_NAMES), points), -1, dtype=torch.long
        ),
        "imu_point_mask": torch.zeros(
            batch, len(IMU_DEVICE_NAMES), points, dtype=torch.bool
        ),
        "imu_interval_counts": torch.zeros(
            batch, frames, len(IMU_DEVICE_NAMES), dtype=torch.long
        ),
        "imu_device_mask": torch.zeros(batch, len(IMU_DEVICE_NAMES), dtype=torch.bool),
    }
    if include_context:
        output.update(
            {
                "context_features": torch.zeros(batch, frames, 2, 2, 896),
                "context_valid": torch.zeros(batch, frames, 2, dtype=torch.bool),
                "context_quality": torch.zeros(batch, frames, 2),
            }
        )
    frame_keys = [
        "frame_time_seconds",
        "time_position",
        "arm_spatial_features",
        "detail_spatial_features",
        "local_geometry_features",
        "oriented_roi_geometry",
        "oriented_angle_valid",
        "local_roi_valid",
        "local_roi_quality",
        "local_roi_source",
        "local_roi_clipped_ratio",
        "pose_quality_factor",
        "skeleton_features",
        "skeleton_feature_mask",
        "skeleton_joint_mask",
        "skeleton_relations",
        "skeleton_relation_mask",
        "skeleton_frame_quality",
        "body_axes_camera",
        "body_axes_raw_valid",
        "imu_interval_counts",
    ]
    if include_context:
        frame_keys.extend(("context_features", "context_valid", "context_quality"))
    for index, item in enumerate(items):
        frame_count = len(item["frame_ids"])
        point_count = item["imu_values"].shape[1]
        output["frame_mask"][index, :frame_count] = True
        for key in frame_keys:
            output[key][index, :frame_count] = item[key]
        if point_count:
            for key in (
                "imu_values",
                "imu_raw_vectors",
                "imu_time_seconds",
                "imu_frame_index",
                "imu_point_mask",
            ):
                output[key][index, :, :point_count] = item[key]
        output["imu_device_mask"][index] = item["imu_device_mask"]
    return output
