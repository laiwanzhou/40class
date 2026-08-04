from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.roi.object_interaction_builder import ObjectInteractionROIBuilder, VIEW_NAMES

from .pose_roi_dataset import PoseTrackCache, depth_frame_key, paired_frame_paths


class ObjectInteractionROIDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        pose_cache_path: Path,
        num_frames: int,
        image_size: int,
        training: bool,
        roi_config: dict[str, float],
        base_logits: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training
        self.base_logits = base_logits or {}
        self.class_names = (
            frame[["class_id", "action_name"]].drop_duplicates().sort_values("class_id")["action_name"].tolist()
        )
        self.original_class_ids = list(range(len(self.class_names)))
        if len(self.class_names) != 40:
            raise ValueError(f"Expected all 40 classes, got {len(self.class_names)}")
        cache = PoseTrackCache(pose_cache_path)
        builder = ObjectInteractionROIBuilder(**roi_config)
        self.samples: list[dict[str, object]] = []
        started = time.perf_counter()
        for row in frame.sort_values(["class_id", "sample_id"]).to_dict(orient="records"):
            sample_id = str(row["sample_id"])
            depth_path = Path(row["trial_path"])
            ir_path = data_root.joinpath(*Path(str(row["ir_path"]).replace("\\", "/")).parts)
            depth_paths, ir_paths = paired_frame_paths(depth_path, ir_path)
            with Image.open(depth_paths[0]) as depth_image, Image.open(ir_paths[0]) as ir_image:
                width, height = depth_image.size
                if ir_image.size != (width, height):
                    raise ValueError(f"Depth/IR size mismatch for {sample_id}")
            keys = [depth_frame_key(path) for path in depth_paths]
            if any((sample_id, key) not in cache.lookup for key in keys):
                raise KeyError(f"Pose cache is incomplete for {sample_id}")
            person_boxes, keypoints, confidence = cache.trial_arrays(sample_id, keys)
            roi = builder.build(person_boxes, keypoints, confidence, width, height)
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "label": int(row["class_id"]),
                    "action_name": str(row["action_name"]),
                    "depth_paths": depth_paths,
                    "ir_paths": ir_paths,
                    "length": len(depth_paths),
                    "width": width,
                    "height": height,
                    "roi": roi,
                }
            )
        print(
            f"ObjectInteractionROIDataset indexed {len(self.samples)} samples "
            f"(training={training}) in {time.perf_counter() - started:.2f}s"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def sampled_indices(self, length: int) -> tuple[np.ndarray, torch.Tensor]:
        if length > self.num_frames:
            anchors = np.linspace(0, length - 1, self.num_frames)
            if self.training:
                spacing = (length - 1) / max(self.num_frames - 1, 1)
                maximum = max(int(np.floor(spacing * 0.25)), 0)
                if maximum:
                    jitter = torch.randint(-maximum, maximum + 1, (self.num_frames,)).numpy()
                    jitter[0] = 0
                    jitter[-1] = 0
                    anchors = anchors + jitter
            indices = np.sort(np.clip(np.rint(anchors), 0, length - 1).astype(np.int64))
            return indices, torch.ones(self.num_frames, dtype=torch.bool)
        indices = np.full(self.num_frames, -1, dtype=np.int64)
        indices[:length] = np.arange(length)
        return indices, torch.arange(self.num_frames) < length

    def _image_tensor(self, path: Path, box: np.ndarray, mode: str) -> torch.Tensor:
        with Image.open(path) as opened:
            image = opened.convert(mode)
            crop = image.crop(tuple(float(value) for value in box))
            resized = TF.resize(crop, [self.image_size, self.image_size], antialias=True)
            return (TF.to_tensor(resized) - 0.5) / 0.5

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        length = int(sample["length"])
        indices, temporal_mask = self.sampled_indices(length)
        roi = sample["roi"]
        depth_paths = sample["depth_paths"]
        ir_paths = sample["ir_paths"]
        if not isinstance(depth_paths, tuple) or not isinstance(ir_paths, tuple):
            raise TypeError("Invalid paired frame paths")
        depth_frames: list[torch.Tensor] = []
        ir_frames: list[torch.Tensor] = []
        view_valid = torch.zeros((self.num_frames, len(VIEW_NAMES)), dtype=torch.bool)
        two_hand = torch.zeros(self.num_frames, dtype=torch.bool)
        hand_head = torch.zeros(self.num_frames, dtype=torch.bool)
        patterns: list[str] = []
        for output_index, frame_index in enumerate(indices):
            if frame_index < 0:
                depth_frames.append(torch.zeros((len(VIEW_NAMES), 3, self.image_size, self.image_size)))
                ir_frames.append(torch.zeros((len(VIEW_NAMES), 1, self.image_size, self.image_size)))
                patterns.append("padding")
                continue
            frame = int(frame_index)
            depth_views: list[torch.Tensor] = []
            ir_views: list[torch.Tensor] = []
            for view_index in range(len(VIEW_NAMES)):
                is_valid = bool(roi.valid_mask[frame, view_index])
                view_valid[output_index, view_index] = is_valid
                if is_valid:
                    box = roi.boxes[frame, view_index]
                    depth_views.append(self._image_tensor(depth_paths[frame], box, "RGB"))
                    ir_views.append(self._image_tensor(ir_paths[frame], box, "L"))
                else:
                    depth_views.append(torch.zeros((3, self.image_size, self.image_size)))
                    ir_views.append(torch.zeros((1, self.image_size, self.image_size)))
            depth_frames.append(torch.stack(depth_views))
            ir_frames.append(torch.stack(ir_views))
            two_hand[output_index] = bool(roi.valid_mask[frame, 4])
            hand_head[output_index] = bool(roi.valid_mask[frame, 3])
            left = (bool(roi.keypoint_valid[frame, 7]), bool(roi.keypoint_valid[frame, 9]))
            right = (bool(roi.keypoint_valid[frame, 8]), bool(roi.keypoint_valid[frame, 10]))
            patterns.append(f"L{int(left[0])}{int(left[1])}_R{int(right[0])}{int(right[1])}")
        sample_id = str(sample["sample_id"])
        base = self.base_logits.get(sample_id)
        if base is None:
            base = np.zeros(40, dtype=np.float32)
        depth_input = torch.stack(depth_frames)
        ir_input = torch.stack(ir_frames)
        return {
            "depth_input": depth_input,
            "ir_input": ir_input,
            "view_valid_mask": view_valid,
            "temporal_mask": temporal_mask,
            "base_logits": torch.from_numpy(np.asarray(base, dtype=np.float32)),
            "label": int(sample["label"]),
            "sample_id": sample_id,
            "length": length,
            "sampled_indices": torch.from_numpy(indices),
            "two_hand_roi_valid": two_hand.any(),
            "hand_head_roi_valid": hand_head.any(),
            "valid_keypoint_pattern": "|".join(sorted(set(patterns))),
        }

    def roi_audit_rows(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for sample in self.samples:
            roi = sample["roi"]
            keypoint = roi.keypoint_valid
            left_elbow, right_elbow = keypoint[:, 7], keypoint[:, 8]
            left_wrist, right_wrist = keypoint[:, 9], keypoint[:, 10]
            directional = roi.sources[:, 1:3] == "directional"
            fallback = roi.sources[:, 1:3] == "wrist_fallback"
            valid_iou = np.isfinite(roi.directional_iou)
            area = (roi.boxes[..., 2] - roi.boxes[..., 0]) * (roi.boxes[..., 3] - roi.boxes[..., 1])
            image_area = float(sample["width"]) * float(sample["height"])
            rows.append(
                {
                    "sample_id": sample["sample_id"], "class_id": sample["label"], "action_name": sample["action_name"],
                    "frames": sample["length"], "left_elbow_valid_rate": left_elbow.mean(),
                    "right_elbow_valid_rate": right_elbow.mean(), "left_wrist_valid_rate": left_wrist.mean(),
                    "right_wrist_valid_rate": right_wrist.mean(), "both_elbow_valid_rate": (left_elbow & right_elbow).mean(),
                    "one_elbow_only_rate": np.logical_xor(left_elbow, right_elbow).mean(),
                    "no_elbow_rate": (~left_elbow & ~right_elbow).mean(),
                    "both_wrist_valid_rate": (left_wrist & right_wrist).mean(),
                    "one_wrist_only_rate": np.logical_xor(left_wrist, right_wrist).mean(),
                    "direction_roi_valid_rate": directional.mean(), "wrist_only_fallback_rate": fallback.mean(),
                    "hand_head_roi_valid_rate": roi.valid_mask[:, 3].mean(),
                    "two_hand_roi_valid_rate": roi.valid_mask[:, 4].mean(),
                    "two_hand_merge_trigger_rate": roi.two_hand_merge.mean(),
                    "left_right_directional_iou_mean": np.nanmean(roi.directional_iou) if valid_iou.any() else np.nan,
                    "old_tight_iou_mean": np.nanmean(roi.old_tight_iou) if np.isfinite(roi.old_tight_iou).any() else np.nan,
                    "old_tight_overlap_rate": np.nanmean(roi.old_tight_iou > 0.30),
                    "new_directional_overlap_rate": np.nanmean(roi.directional_iou > 0.30),
                    "projected_point_out_of_bounds_rate": roi.projected_out_of_bounds.mean(),
                    "interaction_roi_area_ratio_mean": area[:, 1:][roi.valid_mask[:, 1:]].mean() / image_area
                    if roi.valid_mask[:, 1:].any() else np.nan,
                    "wrist_edge_distance_mean": np.nanmean(roi.wrist_edge_distance)
                    if np.isfinite(roi.wrist_edge_distance).any() else np.nan,
                    "artificial_keypoint_count": 0,
                }
            )
        return pd.DataFrame(rows)

    def temporal_diagnostic_rows(self) -> pd.DataFrame:
        rows = []
        was_training = self.training
        self.training = False
        try:
            for sample in self.samples:
                indices, mask = self.sampled_indices(int(sample["length"]))
                length = int(sample["length"])
                rows.append(
                    {
                        "sample_id": sample["sample_id"], "class_id": sample["label"], "action_name": sample["action_name"],
                        "original_frame_count": length, "sampled_indices": " ".join(map(str, indices.tolist())),
                        "used_padding": bool((indices < 0).any()), "temporal_valid_count": int(mask.sum()),
                        "full_trial_start_covered": bool(indices[0] == 0),
                        "full_trial_end_covered": bool(indices[int(mask.sum()) - 1] == length - 1),
                        "center24_coverage_ratio": min(24, length) / length,
                        "full48_coverage_ratio": min(48, length) / length if length <= 48 else 1.0,
                    }
                )
        finally:
            self.training = was_training
        return pd.DataFrame(rows)
