from __future__ import annotations

from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.roi.roi_builder import PoseROIBuilder

from .common import sorted_files


FRAME_PATTERN = re.compile(
    r"^(?P<modality>Depth|IR)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})_"
    r"(?P<frame_id>\d+)(?:_Color)?$"
)


def depth_frame_key(path: Path) -> str:
    if not path.stem.startswith("Depth_"):
        raise ValueError(f"Unexpected Depth filename: {path.name}")
    return path.stem[len("Depth_") :].removesuffix("_Color")


def paired_frame_key(path: Path, modality: str) -> tuple[str, int]:
    match = FRAME_PATTERN.fullmatch(path.stem)
    if match is None or match.group("modality") != modality:
        raise ValueError(f"Unparseable {modality} frame name: {path.name}")
    return match.group("timestamp"), int(match.group("frame_id"))


def paired_frame_paths(depth_path: Path, ir_path: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    depth_files = sorted_files(depth_path, {".png", ".jpg", ".jpeg"})
    ir_files = sorted_files(ir_path, {".png", ".jpg", ".jpeg"})
    depth = {paired_frame_key(path, "Depth"): path for path in depth_files}
    ir = {paired_frame_key(path, "IR"): path for path in ir_files}
    if len(depth) != len(depth_files) or len(ir) != len(ir_files):
        raise ValueError(f"Duplicate parsed frame key in {depth_path} or {ir_path}")
    if not depth or depth.keys() != ir.keys():
        raise ValueError(
            f"Depth/IR pairing mismatch for {depth_path}: "
            f"depth_only={len(depth.keys() - ir.keys())}, ir_only={len(ir.keys() - depth.keys())}"
        )
    keys = sorted(depth)
    return tuple(depth[key] for key in keys), tuple(ir[key] for key in keys)


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
        use_ir_input: bool = False,
        data_root: Path | None = None,
    ) -> None:
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training
        self.use_pose_roi = use_pose_roi
        self.use_ir_input = use_ir_input
        self.ir_mode = "normal"
        self.ir_permutation = np.arange(len(frame), dtype=np.int64)
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
            if use_ir_input:
                if data_root is None or not str(row.get("ir_path", "")).strip():
                    raise ValueError("Dual-input dataset requires data_root and manifest ir_path.")
                ir_trial_path = data_root.joinpath(*Path(str(row["ir_path"]).replace("\\", "/")).parts)
                paths, ir_paths = paired_frame_paths(trial_path, ir_trial_path)
            else:
                paths = tuple(sorted_files(trial_path, {".png", ".jpg", ".jpeg"}))
                ir_paths = ()
            if not paths:
                raise FileNotFoundError(f"No Depth frames for {row['sample_id']}: {trial_path}")
            sample: dict[str, object] = {
                "sample_id": str(row["sample_id"]),
                "label": class_map[int(row["class_id"])],
                "original_class_id": int(row["class_id"]),
                "paths": tuple(paths),
                "ir_paths": tuple(ir_paths),
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
        self.ir_permutation = np.arange(len(self.samples), dtype=np.int64)
        print(
            f"PoseROIDataset indexed {len(self.samples)} samples (training={training}, "
            f"pose_roi={use_pose_roi}, ir={use_ir_input}) in {time.perf_counter() - started:.2f}s"
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
        depth_frames: list[torch.Tensor] = []
        ir_frames: list[torch.Tensor] = []
        roi_boxes = sample.get("roi_boxes")
        ir_sample = self.samples[int(self.ir_permutation[index])]
        ir_paths = ir_sample.get("ir_paths")
        if self.use_ir_input and not isinstance(ir_paths, tuple):
            raise TypeError("Invalid IR path index")
        if self.use_ir_input and self.ir_mode == "shuffled":
            ir_indices, _ = self._window(len(ir_paths))
        else:
            ir_indices = indices
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
                depth_frames.append(torch.stack(tensors) if self.use_pose_roi else tensors[0])
        if self.use_ir_input:
            assert isinstance(ir_paths, tuple)
            for output_index, frame_index in enumerate(ir_indices):
                with Image.open(ir_paths[int(frame_index)]) as opened:
                    image = opened.convert("L")
                    views = [image]
                    if self.use_pose_roi:
                        assert isinstance(roi_boxes, np.ndarray)
                        depth_index = int(indices[output_index])
                        views.extend(image.crop(tuple(float(value) for value in box)) for box in roi_boxes[depth_index])
                    tensors = []
                    for view in views:
                        resized = TF.resize(view, [self.image_size, self.image_size], antialias=True)
                        tensor = (TF.to_tensor(resized) - 0.5) / 0.5
                        tensors.append(torch.zeros_like(tensor) if self.ir_mode == "masked" else tensor)
                    ir_frames.append(torch.stack(tensors) if self.use_pose_roi else tensors[0])
        depth_input = torch.stack(depth_frames)
        input_value: torch.Tensor | dict[str, torch.Tensor]
        input_value = (
            {"depth_input": depth_input, "ir_input": torch.stack(ir_frames)}
            if self.use_ir_input
            else depth_input
        )
        roi_valid = torch.ones((self.num_frames, 4), dtype=torch.bool)
        if isinstance(sample.get("roi_sources"), np.ndarray):
            sources = sample["roi_sources"][indices]
            roi_valid[:, 1:] = torch.from_numpy(sources != "central")
        return {
            "input": input_value,
            "depth_input": depth_input,
            **({"ir_input": torch.stack(ir_frames)} if self.use_ir_input else {}),
            "temporal_mask": mask,
            "roi_valid_mask": roi_valid,
            "label": int(sample["label"]),
            "sample_id": str(sample["sample_id"]),
            "length": int(sample["length"]),
        }

    def set_ir_mode(self, mode: str, seed: int = 0) -> None:
        if not self.use_ir_input:
            raise ValueError("IR modes require a dual-input dataset.")
        if mode not in {"normal", "masked", "shuffled"}:
            raise ValueError(f"Unsupported IR mode: {mode}")
        self.ir_mode = mode
        self.ir_permutation = np.arange(len(self.samples), dtype=np.int64)
        if mode == "shuffled":
            rng = np.random.default_rng(seed)
            for _ in range(1000):
                candidate = rng.permutation(len(self.samples))
                if np.all(candidate != np.arange(len(self.samples))):
                    self.ir_permutation = candidate
                    break
            else:
                raise RuntimeError("Could not construct a fixed-point-free IR permutation.")

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
