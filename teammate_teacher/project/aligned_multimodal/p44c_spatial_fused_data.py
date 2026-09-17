from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from audit_yolo11_pose_skeleton import safe_name
from p32_fused_data import P32FusedTrialDataset, collate_p32_trials


SPATIAL_MODALITIES = ("depth", "ir")
SPATIAL_REGIONS = ("left_hand", "right_hand", "hand_workspace")


class P44CSpatialFusedDataset(Dataset[dict[str, Any]]):
    """Exact sample/frame join of P30 GAP, P31 motion and P44-C spatial ROI."""

    def __init__(
        self,
        visual_run: str | Path,
        motion_run: str | Path,
        spatial_run: str | Path,
        sample_ids: set[str],
    ) -> None:
        self.base = P32FusedTrialDataset(visual_run, motion_run, sample_ids=sample_ids)
        self.spatial_run = Path(spatial_run).resolve()
        with (self.spatial_run / "trial_summary.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            spatial_rows = list(csv.DictReader(handle))
        self.spatial_by_id = {row["source_id"]: row for row in spatial_rows}
        missing = [row["sample_id"] for row in self.base.rows if row["sample_id"] not in self.spatial_by_id]
        if missing:
            raise RuntimeError(f"P44-C spatial trials missing: {missing[:3]}")

    def __len__(self) -> int:
        return len(self.base)

    @property
    def rows(self) -> list[dict[str, str]]:
        return self.base.rows

    @property
    def frame_lengths(self) -> list[int]:
        return self.base.frame_lengths

    @property
    def imu_point_lengths(self) -> list[int]:
        return [int(self.base.motion.rows[motion_index]["imu_accepted_points"]) for _, motion_index in self.base.pairs]

    def spatial_path(self, sample_id: str) -> Path:
        return (
            self.spatial_run
            / "trial_spatial_cache"
            / safe_name(sample_id).with_suffix(".npz")
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        fused = self.base[index]
        sample_id = fused["visual"]["sample_id"]
        with np.load(self.spatial_path(sample_id), allow_pickle=False) as data:
            if tuple(str(value) for value in data["modality_names"]) != SPATIAL_MODALITIES:
                raise RuntimeError(f"spatial modality mismatch: {sample_id}")
            if tuple(str(value) for value in data["region_names"]) != SPATIAL_REGIONS:
                raise RuntimeError(f"spatial region mismatch: {sample_id}")
            frame_ids = [str(value) for value in data["frame_ids"]]
            spatial = {
                "frame_ids": frame_ids,
                "features": torch.from_numpy(data["spatial_features"].astype(np.float32)),
                "valid": torch.from_numpy(data["roi_valid"].astype(bool)),
                "quality": torch.from_numpy(data["roi_quality"].astype(np.float32)),
                "clipped": torch.from_numpy(data["roi_clipped_ratio"].astype(np.float32)),
            }
        if frame_ids != fused["visual"]["frame_ids"]:
            raise RuntimeError(f"P30/P31/P44-C frame mismatch: {sample_id}")
        return {"fused": fused, "spatial": spatial}


def collate_p44c(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty P44-C batch")
    output = collate_p32_trials([item["fused"] for item in items])
    batch_size, maximum = output["frame_mask"].shape
    feature_dim = items[0]["spatial"]["features"].shape[-1]
    spatial = torch.zeros(batch_size, maximum, 2, 3, 3, 3, feature_dim)
    valid = torch.zeros(batch_size, maximum, 3, dtype=torch.bool)
    quality = torch.zeros(batch_size, maximum, 3)
    clipped = torch.zeros(batch_size, maximum, 3)
    for index, item in enumerate(items):
        length = len(item["spatial"]["frame_ids"])
        spatial[index, :length] = item["spatial"]["features"]
        valid[index, :length] = item["spatial"]["valid"]
        quality[index, :length] = item["spatial"]["quality"]
        clipped[index, :length] = item["spatial"]["clipped"]
    output.update(
        {
            "spatial_features": spatial,
            "spatial_valid": valid,
            "spatial_quality": quality,
            "spatial_clipped_ratio": clipped,
        }
    )
    return output
