from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.roi.roi_builder import PoseROIBuilder

from .common import sorted_files


def depth_frame_key(path: Path) -> str:
    if not path.stem.startswith("Depth_"):
        raise ValueError(f"Unexpected Depth filename: {path.name}")
    return path.stem[len("Depth_") :].removesuffix("_Color")


class PoseTrackCache:
    def __init__(self, path: Path) -> None:
        with np.load(path) as data:
            sample_ids = data["sample_ids"].astype(str)
            frame_keys = data["frame_keys"].astype(str)
            boxes = data["bbox_xyxy"].astype(np.float32)
            keypoints = data["keypoints_xy"].astype(np.float32)
            confidence = data["keypoints_confidence"].astype(np.float32)
        self.lookup = {
            (sample_id, frame_key): (boxes[index], keypoints[index], confidence[index])
            for index, (sample_id, frame_key) in enumerate(zip(sample_ids, frame_keys, strict=True))
        }

    def trial_arrays(self, sample_id: str, frame_keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = np.full((len(frame_keys), 4), np.nan, dtype=np.float32)
        keypoints = np.full((len(frame_keys), 17, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((len(frame_keys), 17), dtype=np.float32)
        for index, key in enumerate(frame_keys):
            value = self.lookup.get((sample_id, key))
            if value is not None:
                boxes[index], keypoints[index], confidence[index] = value
        return boxes, keypoints, confidence


class PoseROIDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        hard_actions: list[str],
        num_frames: int,
        image_size: int,
        training: bool,
        use_pose_roi: bool,
        pose_cache_path: Path | None = None,
    ) -> None:
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training
        self.use_pose_roi = use_pose_roi
        present = frame[frame["action_name"].isin(hard_actions)].copy()
        found = set(present["action_name"])
        missing = set(hard_actions) - found
        if missing:
            raise ValueError(f"Hard actions absent from split: {sorted(missing)}")
        class_rows = present[["class_id", "action_name"]].drop_duplicates().sort_values("class_id")
        self.class_names = class_rows["action_name"].tolist()
        self.original_class_ids = class_rows["class_id"].astype(int).tolist()
        class_map = {class_id: index for index, class_id in enumerate(self.original_class_ids)}
        cache = PoseTrackCache(pose_cache_path) if use_pose_roi and pose_cache_path is not None else None
        if use_pose_roi and cache is None:
            raise ValueError("Pose ROI dataset requires a pose cache.")
        started = time.perf_counter()
        self.samples: list[dict[str, object]] = []
        roi_builder = PoseROIBuilder()
        for row in present.reset_index(drop=True).to_dict(orient="records"):
            trial_path = Path(row["trial_path"])
            paths = sorted_files(trial_path, {".png", ".jpg", ".jpeg"})
            if not paths:
                raise FileNotFoundError(f"No Depth frames for {row['sample_id']}: {trial_path}")
            sample: dict[str, object] = {
                "sample_id": str(row["sample_id"]),
                "label": class_map[int(row["class_id"])],
                "original_class_id": int(row["class_id"]),
                "paths": tuple(paths),
                "length": len(paths),
            }
            if use_pose_roi:
                assert cache is not None
                with Image.open(paths[0]) as image:
                    width, height = image.size
                keys = [depth_frame_key(path) for path in paths]
                person_boxes, keypoints, confidence = cache.trial_arrays(str(row["sample_id"]), keys)
                roi = roi_builder.build(person_boxes, keypoints, confidence, width, height)
                sample["roi_boxes"] = roi.boxes
                sample["roi_sources"] = roi.sources
            self.samples.append(sample)
        print(
            f"PoseROIDataset indexed {len(self.samples)} samples (training={training}, "
            f"pose_roi={use_pose_roi}) in {time.perf_counter() - started:.2f}s"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _window(self, length: int) -> tuple[np.ndarray, torch.Tensor]:
        if length >= self.num_frames:
            max_start = length - self.num_frames
            start = int(torch.randint(max_start + 1, (1,)).item()) if self.training and max_start else max_start // 2
            indices = np.arange(start, start + self.num_frames)
            mask = torch.ones(self.num_frames, dtype=torch.bool)
        else:
            indices = np.r_[np.arange(length), np.full(self.num_frames - length, length - 1)]
            mask = torch.arange(self.num_frames) < length
        return indices.astype(int), mask

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        paths = sample["paths"]
        if not isinstance(paths, tuple):
            raise TypeError("Invalid path index")
        indices, mask = self._window(len(paths))
        frames: list[torch.Tensor] = []
        roi_boxes = sample.get("roi_boxes")
        for frame_index in indices:
            with Image.open(paths[int(frame_index)]) as opened:
                image = opened.convert("RGB")
                views = [image]
                if self.use_pose_roi:
                    assert isinstance(roi_boxes, np.ndarray)
                    views.extend(image.crop(tuple(float(value) for value in box)) for box in roi_boxes[int(frame_index)])
                tensors = []
                for view in views:
                    resized = TF.resize(view, [self.image_size, self.image_size], antialias=True)
                    tensors.append((TF.to_tensor(resized) - 0.5) / 0.5)
                frames.append(torch.stack(tensors) if self.use_pose_roi else tensors[0])
        return {
            "input": torch.stack(frames),
            "temporal_mask": mask,
            "label": int(sample["label"]),
            "sample_id": str(sample["sample_id"]),
            "length": int(sample["length"]),
        }

    def roi_source_counts(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for sample in self.samples:
            sources = sample.get("roi_sources")
            if not isinstance(sources, np.ndarray):
                continue
            for view_index, view_name in enumerate(("upper_body", "left_hand", "right_hand")):
                values, counts = np.unique(sources[:, view_index], return_counts=True)
                rows.extend(
                    {"sample_id": sample["sample_id"], "view": view_name, "source": value, "frames": int(count)}
                    for value, count in zip(values, counts, strict=True)
                )
        return pd.DataFrame(rows)
