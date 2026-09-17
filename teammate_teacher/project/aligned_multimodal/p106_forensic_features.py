"""Frozen fine-grained, label-free evidence blocks for P106 forensics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from p100a_global_teacher_data import H3_USERS, P100AData
from p103_local_feature_data import (
    VJEPA_VIEW_NAMES,
    VMAE_VIEW_NAMES,
    load_p103_local_data,
)
from p104_modality_data import DEFAULT_DEPTH, DEFAULT_THERMAL


FINE_BLOCKS = (
    "local_interaction_temporal",
    "local_hand_phase_delta",
    "global_visual_phase_delta",
    "skeleton_wrist_arm_temporal",
    "imu_arm_phase_direction",
    "depth_person_workspace_geometry",
    "thermal_person_workspace_interaction",
)


@dataclass(frozen=True)
class FineFeatureBlock:
    name: str
    aligned: np.ndarray
    available: np.ndarray
    audit: dict[str, Any]

    def validate(self, rows: int) -> None:
        if self.name not in FINE_BLOCKS:
            raise ValueError(f"unknown P106 forensic block: {self.name}")
        if self.aligned.ndim != 2 or self.aligned.shape[0] != rows:
            raise ValueError(f"invalid {self.name} shape: {self.aligned.shape}")
        if self.available.shape != (rows,):
            raise ValueError(f"invalid {self.name} availability")
        if not np.isfinite(self.aligned).all():
            raise ValueError(f"non-finite P106 block: {self.name}")


def _flatten(*values: np.ndarray) -> np.ndarray:
    rows = len(values[0])
    return np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(rows, -1) for value in values],
        axis=1,
    ).astype(np.float32, copy=False)


def _view(values: np.ndarray, names: tuple[str, ...], selected: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    return np.asarray(values[:, [lookup[name] for name in selected]], dtype=np.float32)


def _temporal_summary(
    signals: np.ndarray, mask: np.ndarray, *, spectral: bool = False
) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    count = np.maximum(valid.sum(axis=1), 1)
    safe = np.where(valid, values, 0.0)
    mean = safe.sum(axis=1) / count
    centered = np.where(valid, values - mean[:, None], 0.0)
    std = np.sqrt(np.sum(centered**2, axis=1) / count)
    rms = np.sqrt(np.sum(safe**2, axis=1) / count)
    minimum = np.min(np.where(valid, values, np.inf), axis=1)
    maximum = np.max(np.where(valid, values, -np.inf), axis=1)
    minimum[~np.isfinite(minimum)] = 0.0
    maximum[~np.isfinite(maximum)] = 0.0
    pair_valid = valid[:, 1:] & valid[:, :-1]
    difference = np.where(pair_valid, values[:, 1:] - values[:, :-1], 0.0)
    mean_abs_difference = np.sum(np.abs(difference), axis=1) / np.maximum(
        pair_valid.sum(axis=1), 1
    )
    midpoint = values.shape[1] // 2
    early_valid = valid[:, :midpoint]
    late_valid = valid[:, midpoint:]
    early = np.where(early_valid, values[:, :midpoint], 0.0).sum(axis=1) / np.maximum(
        early_valid.sum(axis=1), 1
    )
    late = np.where(late_valid, values[:, midpoint:], 0.0).sum(axis=1) / np.maximum(
        late_valid.sum(axis=1), 1
    )
    blocks = [mean, std, rms, maximum - minimum, mean_abs_difference, early, late, late - early]
    if spectral:
        power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
        non_dc = power[:, 1:]
        bins = non_dc.shape[1]
        total = np.maximum(non_dc.sum(axis=1), 1e-12)
        first = max(1, bins // 3)
        second = max(first + 1, 2 * bins // 3)
        blocks.extend(
            (
                non_dc[:, :first].sum(axis=1) / total,
                non_dc[:, first:second].sum(axis=1) / total,
                non_dc[:, second:].sum(axis=1) / total,
                (np.argmax(non_dc, axis=1) + 1) / max(values.shape[1], 1),
            )
        )
    return np.nan_to_num(
        np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def _local_blocks(data: P100AData) -> dict[str, FineFeatureBlock]:
    local = load_p103_local_data(data.sample_ids, data.users)
    vmae_interaction = ["hand_full_interaction", "hand_motion_peak_interaction"]
    vjepa_interaction = [
        "hand_full_interaction",
        "hand_early_interaction",
        "hand_late_interaction",
        "hand_motion_peak_interaction",
    ]
    interaction = _flatten(
        _view(local.vmae_features, VMAE_VIEW_NAMES, vmae_interaction),
        _view(local.vmae_actions, VMAE_VIEW_NAMES, vmae_interaction),
        _view(local.vjepa_features, VJEPA_VIEW_NAMES, vjepa_interaction),
        _view(local.vjepa_actions, VJEPA_VIEW_NAMES, vjepa_interaction),
    )

    def delta(
        values: np.ndarray, names: tuple[str, ...], later: list[str], earlier: list[str]
    ) -> np.ndarray:
        return _view(values, names, later) - _view(values, names, earlier)

    parts = ("left", "right", "interaction")
    vmae_peak = [f"hand_motion_peak_{part}" for part in parts]
    vmae_full = [f"hand_full_{part}" for part in parts]
    vjepa_late = [f"hand_late_{part}" for part in parts]
    vjepa_early = [f"hand_early_{part}" for part in parts]
    vjepa_peak = [f"hand_motion_peak_{part}" for part in parts]
    vjepa_full = [f"hand_full_{part}" for part in parts]
    phase = _flatten(
        delta(local.vmae_features, VMAE_VIEW_NAMES, vmae_peak, vmae_full),
        delta(local.vmae_actions, VMAE_VIEW_NAMES, vmae_peak, vmae_full),
        delta(local.vjepa_features, VJEPA_VIEW_NAMES, vjepa_late, vjepa_early),
        delta(local.vjepa_actions, VJEPA_VIEW_NAMES, vjepa_late, vjepa_early),
        delta(local.vjepa_features, VJEPA_VIEW_NAMES, vjepa_peak, vjepa_full),
        delta(local.vjepa_actions, VJEPA_VIEW_NAMES, vjepa_peak, vjepa_full),
    )
    available = np.ones(len(data.sample_ids), dtype=bool)
    return {
        "local_interaction_temporal": FineFeatureBlock(
            "local_interaction_temporal",
            interaction,
            available,
            {
                "evidence": "full/early/late/motion-peak hand interaction crops",
                "sources": ["P103 VideoMAEv2 local", "P96 V-JEPA2 local"],
            },
        ),
        "local_hand_phase_delta": FineFeatureBlock(
            "local_hand_phase_delta",
            phase,
            available,
            {
                "evidence": "late-minus-early and motion-peak-minus-full hand descriptors",
                "sources": ["P103 VideoMAEv2 local", "P96 V-JEPA2 local"],
            },
        ),
    }


def _global_phase(data: P100AData) -> FineFeatureBlock:
    # Frozen P90 caches are ordered early(scene/person/workspace), then late(...).
    aligned = _flatten(
        data.visual_vmae[:, 3:] - data.visual_vmae[:, :3],
        data.visual_vmae_action[:, 3:] - data.visual_vmae_action[:, :3],
        data.visual_iv2[:, 3:] - data.visual_iv2[:, :3],
        data.visual_iv2_action[:, 3:] - data.visual_iv2_action[:, :3],
    )
    return FineFeatureBlock(
        "global_visual_phase_delta",
        aligned,
        np.ones(len(data.sample_ids), dtype=bool),
        {
            "evidence": "late-minus-early scene/person/workspace visual feature and action delta",
            "sources": ["P90 VideoMAEv2", "P90 InternVideo2-L"],
        },
    )


def _skeleton_block(data: P100AData) -> FineFeatureBlock:
    selected = data.skeleton_sequence[:, :, 11:17]
    joint_mask = data.skeleton_mask[:, :, 11:17].astype(bool)
    position = selected[..., :3].reshape(len(selected), selected.shape[1], -1)
    position_mask = np.repeat(joint_mask[..., None], 3, axis=-1).reshape(position.shape)
    bone = np.linalg.norm(selected[..., 3:6], axis=-1)
    speed = np.linalg.norm(selected[..., 6:9], axis=-1)
    acceleration = np.linalg.norm(selected[..., 9:12], axis=-1)
    left_wrist = data.skeleton_sequence[:, :, 13, :3]
    right_wrist = data.skeleton_sequence[:, :, 16, :3]
    head = data.skeleton_sequence[:, :, 10, :3]
    relation = np.stack(
        (
            np.linalg.norm(left_wrist - right_wrist, axis=-1),
            np.linalg.norm(left_wrist - head, axis=-1),
            np.linalg.norm(right_wrist - head, axis=-1),
        ),
        axis=-1,
    )
    relation_mask = np.stack(
        (
            data.skeleton_mask[:, :, 13].astype(bool)
            & data.skeleton_mask[:, :, 16].astype(bool),
            data.skeleton_mask[:, :, 13].astype(bool)
            & data.skeleton_mask[:, :, 10].astype(bool),
            data.skeleton_mask[:, :, 16].astype(bool)
            & data.skeleton_mask[:, :, 10].astype(bool),
        ),
        axis=-1,
    ).astype(bool)
    signals = np.concatenate((position, bone, speed, acceleration, relation), axis=-1)
    mask = np.concatenate(
        (position_mask, joint_mask, joint_mask, joint_mask, relation_mask), axis=-1
    )
    return FineFeatureBlock(
        "skeleton_wrist_arm_temporal",
        _temporal_summary(signals, mask),
        data.skeleton_available.astype(bool),
        {
            "evidence": "arm/wrist position, velocity, acceleration, trajectory phase and hand/head relations",
            "joints": [11, 12, 13, 14, 15, 16],
        },
    )


def _imu_block(data: P100AData) -> FineFeatureBlock:
    values = data.imu_sequence[:, :, 1:3]
    point_mask = data.imu_mask[:, :, 1:3].astype(bool)
    count = np.maximum(point_mask.sum(axis=3), 1)
    mean = np.sum(values * point_mask[..., None], axis=3) / count[..., None]
    device_mask = point_mask.any(axis=3)
    vector = mean[..., :12].reshape(len(mean), mean.shape[1], -1)
    vector_mask = np.repeat(device_mask[..., None], 12, axis=-1).reshape(vector.shape)
    magnitude = np.stack(
        [
            np.linalg.norm(mean[..., start : start + 3], axis=-1)
            for start in (0, 3, 6, 9)
        ],
        axis=-1,
    ).reshape(len(mean), mean.shape[1], -1)
    magnitude_mask = np.repeat(device_mask[..., None], 4, axis=-1).reshape(magnitude.shape)
    signals = np.concatenate((vector, magnitude), axis=-1)
    mask = np.concatenate((vector_mask, magnitude_mask), axis=-1)
    return FineFeatureBlock(
        "imu_arm_phase_direction",
        _temporal_summary(signals, mask, spectral=True),
        data.imu_available.astype(bool),
        {
            "evidence": "left/right arm raw and compensated direction, intensity, phase and frequency",
            "devices": ["left_arm", "right_arm"],
        },
    )


def _load_three_view(
    data: P100AData, path: Path, *, modality: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as archive:
        required = {"sample_ids", "users", "features", "action_logits"}
        if modality == "thermal":
            required |= {"modality", "modality_available"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"{modality} misses {sorted(missing)}")
        if modality == "thermal" and str(np.asarray(archive["modality"]).item()) != "thermal":
            raise RuntimeError("P106 thermal cache identity changed")
        ids = archive["sample_ids"].astype(str)
        lookup = {sample_id: index for index, sample_id in enumerate(ids)}
        order = np.asarray([lookup[value] for value in data.sample_ids], dtype=np.int64)
        users = archive["users"].astype(str)[order]
        if not np.array_equal(users, data.users) or set(users.tolist()) & set(H3_USERS):
            raise RuntimeError(f"{modality}/P106 alignment or H3 guard failed")
        features = np.asarray(archive["features"][order], dtype=np.float32)
        actions = np.asarray(archive["action_logits"][order], dtype=np.float32)
        available = (
            np.asarray(archive["modality_available"][order], dtype=bool)
            if modality == "thermal"
            else np.ones(len(order), dtype=bool)
        )
    if features.ndim != 3 or features.shape[1] != 3 or actions.shape[1] != 3:
        raise RuntimeError(f"unexpected {modality} three-view tensors")
    return features, actions, available


def _geometry_block(
    data: P100AData, path: Path, *, modality: str, name: str
) -> FineFeatureBlock:
    features, actions, available = _load_three_view(data, path, modality=modality)
    person = 1
    workspace = 2
    aligned = _flatten(
        features[:, person],
        features[:, workspace],
        features[:, workspace] - features[:, person],
        np.abs(features[:, workspace] - features[:, person]),
        actions[:, person],
        actions[:, workspace],
        actions[:, workspace] - actions[:, person],
        np.abs(actions[:, workspace] - actions[:, person]),
    )
    return FineFeatureBlock(
        name,
        aligned,
        available,
        {
            "evidence": "person/workspace descriptors plus signed and absolute geometry/interaction contrast",
            "source": str(path.resolve()),
            "historical_labels_requested": False,
            "h3_rows_selected": 0,
        },
    )


def load_forensic_blocks(data: P100AData) -> dict[str, FineFeatureBlock]:
    blocks = _local_blocks(data)
    blocks["global_visual_phase_delta"] = _global_phase(data)
    blocks["skeleton_wrist_arm_temporal"] = _skeleton_block(data)
    blocks["imu_arm_phase_direction"] = _imu_block(data)
    blocks["depth_person_workspace_geometry"] = _geometry_block(
        data,
        DEFAULT_DEPTH,
        modality="depth",
        name="depth_person_workspace_geometry",
    )
    blocks["thermal_person_workspace_interaction"] = _geometry_block(
        data,
        DEFAULT_THERMAL,
        modality="thermal",
        name="thermal_person_workspace_interaction",
    )
    if tuple(blocks) != FINE_BLOCKS:
        raise RuntimeError(f"P106 forensic block order changed: {tuple(blocks)}")
    for block in blocks.values():
        block.validate(len(data.sample_ids))
    return blocks
