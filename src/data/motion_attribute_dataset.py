from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.canonical_multimodal_index import CanonicalTrial, build_canonical_trials
from src.data.clean_skeleton_segments import frame_bone_scale, segment_local_velocity
from src.experiments.motion_attribute_config import motion_family_targets, project_path


@dataclass(frozen=True)
class MotionTrial:
    features: torch.Tensor
    mask: torch.Tensor
    segment_ids: torch.Tensor
    attributes: torch.Tensor


def fit_apply_attribute_normalization(
    raw: np.ndarray, *, train_mask: np.ndarray, available: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    train_mask = np.asarray(train_mask, dtype=bool)
    available = np.asarray(available, dtype=bool)
    if raw.ndim != 2 or raw.shape[1] != 16:
        raise ValueError("motion attributes must have shape [N,16]")
    if train_mask.shape != (len(raw),) or available.shape != (len(raw),):
        raise ValueError("motion attribute normalization masks changed")
    selected = train_mask & available
    if not selected.any():
        raise ValueError("no train-supported motion attributes")
    mean = raw[selected].mean(axis=0).astype(np.float32)
    std = np.maximum(raw[selected].std(axis=0), 1e-6).astype(np.float32)
    normalized = (raw - mean[None]) / std[None]
    normalized[~available] = 0.0
    if not np.isfinite(normalized).all():
        raise ValueError("non-finite normalized motion attributes")
    return normalized.astype(np.float32), mean, std


def _truthy(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.casefold().isin({"true", "1", "yes"})


def _read_pose(path: Path, candidate_index: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not 0 <= candidate_index < len(payload):
        raise ValueError(f"invalid Skeleton candidate {candidate_index}: {path}")
    pose = np.asarray(payload[candidate_index].get("keypoints"), dtype=np.float64)
    if pose.shape != (17, 3) or not np.isfinite(pose).all():
        raise ValueError(f"invalid H36M-17 pose: {path}")
    return pose


def _angle(first: torch.Tensor, center: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    left = first - center
    right = last - center
    cosine = (left * right).sum(-1) / (
        left.norm(dim=-1) * right.norm(dim=-1)
    ).clamp_min(1e-6)
    return torch.acos(cosine.clamp(-1.0, 1.0))


def compute_motion_attributes(
    features: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if features.shape != (96, 17, 6) or mask.shape != (96,):
        raise ValueError("motion attributes require [96,17,6] and [96]")
    valid = mask.bool()
    if not bool(valid.any()):
        return torch.zeros(16, dtype=features.dtype)
    position = features[valid, :, :3]
    velocity = features[valid, :, 3:]
    hips = (position[:, 1] + position[:, 4]) * 0.5
    root_delta = hips[-1] - hips[0]
    root_velocity = (velocity[:, 1] + velocity[:, 4]) * 0.5
    root_speed = root_velocity.norm(dim=-1)
    joint_speed = velocity.norm(dim=-1)
    upper_energy = joint_speed[:, 8:17].square().mean()
    lower_energy = joint_speed[:, 1:7].square().mean()
    pairs = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))
    asymmetry = torch.stack(
        [
            (joint_speed[:, left] - joint_speed[:, right]).abs().mean()
            for left, right in pairs
        ]
    ).mean()
    knee_left = torch.pi - _angle(position[:, 1], position[:, 2], position[:, 3])
    knee_right = torch.pi - _angle(position[:, 4], position[:, 5], position[:, 6])
    knee = torch.cat((knee_left, knee_right)) / torch.pi
    hand_head = torch.stack(
        (
            (position[:, 13] - position[:, 10]).norm(dim=-1),
            (position[:, 16] - position[:, 10]).norm(dim=-1),
        ),
        dim=1,
    ).mean()
    hand_torso = torch.stack(
        (
            (position[:, 13] - position[:, 8]).norm(dim=-1),
            (position[:, 16] - position[:, 8]).norm(dim=-1),
        ),
        dim=1,
    ).mean()
    frame_energy = joint_speed.mean(dim=1)
    static_fraction = (frame_energy < 0.01).to(features.dtype).mean()
    if len(frame_energy) >= 3 and float(frame_energy.abs().max()) > 1e-8:
        spectrum = torch.fft.rfft(frame_energy - frame_energy.mean()).abs()
        if len(spectrum) > 1:
            dominant = spectrum[1:].argmax().to(features.dtype) + 1
            dominant_frequency = dominant / max(len(spectrum) - 1, 1)
        else:
            dominant_frequency = features.new_tensor(0.0)
    else:
        dominant_frequency = features.new_tensor(0.0)
    attributes = torch.stack(
        (
            root_delta[[0, 2]].norm(),
            root_delta[1],
            root_speed.mean(),
            root_speed.max(),
            joint_speed.mean(),
            joint_speed.max(),
            upper_energy,
            lower_energy,
            torch.log((upper_energy + 1e-6) / (lower_energy + 1e-6)),
            asymmetry,
            knee.mean(),
            knee.max(),
            hand_head,
            hand_torso,
            static_fraction,
            dominant_frequency,
        )
    )
    if not bool(torch.isfinite(attributes).all()):
        raise ValueError("non-finite motion attributes")
    return attributes


def resample_motion_trial(
    clean_rows: pd.DataFrame, *, data_root: Path, frames: int = 96
) -> MotionTrial:
    required = {
        "sample_id",
        "frame_id",
        "retained_segment_index",
        "skeleton_json_path",
        "candidate_index",
        "use_for_frame_training",
    }
    if not required.issubset(clean_rows.columns) or frames != 96:
        raise ValueError("motion trial input contract changed")
    rows = clean_rows[_truthy(clean_rows["use_for_frame_training"])].copy()
    rows = rows.dropna(subset=["candidate_index", "retained_segment_index"])
    rows = rows.sort_values("frame_id")
    if rows.empty or rows["sample_id"].astype(str).nunique() != 1:
        raise ValueError("motion trial expects one retained sample")
    frame_ids = rows["frame_id"].to_numpy(dtype=np.float64)
    segment_ids = rows["retained_segment_index"].to_numpy(dtype=np.int64)
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("motion trial frame IDs are not unique")
    poses = np.stack(
        [
            _read_pose(
                data_root / str(row.skeleton_json_path), int(row.candidate_index)
            )
            for row in rows.itertuples()
        ]
    )
    scales = frame_bone_scale(poses)
    normalized = poses / scales[:, None, None]
    velocity = segment_local_velocity(normalized, segment_ids)
    source = np.concatenate((normalized, velocity), axis=2)
    targets = np.linspace(frame_ids[0], frame_ids[-1], frames)
    output = np.zeros((frames, 17, 6), dtype=np.float32)
    output_mask = np.zeros(frames, dtype=bool)
    output_segments = np.full(frames, -1, dtype=np.int64)
    for segment in pd.unique(segment_ids):
        selected = segment_ids == segment
        positions = frame_ids[selected]
        values = source[selected]
        target_indices = np.flatnonzero(
            (targets >= positions[0]) & (targets <= positions[-1])
        )
        if not len(target_indices):
            target_indices = np.asarray(
                [int(np.argmin(np.abs(targets - np.median(positions))))]
            )
        for joint in range(17):
            for channel in range(6):
                output[target_indices, joint, channel] = np.interp(
                    targets[target_indices], positions, values[:, joint, channel]
                )
        output[target_indices[0], :, 3:] = 0.0
        output_mask[target_indices] = True
        output_segments[target_indices] = int(segment)
    features = torch.from_numpy(output)
    mask = torch.from_numpy(output_mask)
    segments = torch.from_numpy(output_segments)
    attributes = compute_motion_attributes(features, mask)
    return MotionTrial(features, mask, segments, attributes)


class MotionAttributeDataset(Dataset[dict[str, object]]):
    def __init__(self, config: dict[str, Any], *, partition: str) -> None:
        if partition not in {"train", "validation"}:
            raise ValueError("partition must be train or validation")
        population, data = config["population"], config["data"]
        self.trials = build_canonical_trials(
            project_path(str(population["manifest"])),
            project_path(str(population["split"])),
            Path(str(data["root"])),
            partition=partition,
        )
        clean = pd.read_csv(
            project_path(str(data["clean_view"])),
            encoding="utf-8-sig",
            dtype={"sample_id": str, "user_id": str},
        )
        self.lookup = {
            str(sample_id): rows.reset_index(drop=True)
            for sample_id, rows in clean.groupby("sample_id", sort=False)
        }
        self.data_root = Path(str(data["root"]))
        self.labels = np.asarray([trial.class_id for trial in self.trials], dtype=np.int64)
        self.user_ids = np.asarray([trial.user_id for trial in self.trials])
        self.sample_ids = np.asarray([trial.sample_id for trial in self.trials])
        self.supported_count = sum(trial.sample_id in self.lookup for trial in self.trials)
        expected = 1956 if partition == "train" else 385
        if self.supported_count != expected:
            raise ValueError(f"motion attribute supported population changed: {self.supported_count}")

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index: int) -> dict[str, object]:
        trial: CanonicalTrial = self.trials[index]
        rows = self.lookup.get(trial.sample_id)
        available = rows is not None
        if available:
            motion = resample_motion_trial(rows, data_root=self.data_root)
        else:
            motion = MotionTrial(
                torch.zeros(96, 17, 6),
                torch.zeros(96, dtype=torch.bool),
                torch.full((96,), -1, dtype=torch.long),
                torch.zeros(16),
            )
        return {
            "features": motion.features,
            "mask": motion.mask,
            "segment_ids": motion.segment_ids,
            "attributes": motion.attributes,
            "families": motion_family_targets(torch.tensor(trial.class_id)),
            "available": torch.tensor(available),
            "label": trial.class_id,
            "sample_id": trial.sample_id,
            "user_id": trial.user_id,
        }
