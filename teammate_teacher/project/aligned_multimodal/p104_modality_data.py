"""Frozen representation loaders for P104 family-specific modality probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from p100a_global_teacher_data import H3_USERS, P100AData, load_p100a_data
from p103_local_feature_data import (
    VJEPA_VIEW_NAMES,
    VMAE_VIEW_NAMES,
    load_p103_local_data,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DEPTH = PROJECT / "runs/p91_videomaev2_depth_fold0_v1/complete_features.npz"
DEFAULT_THERMAL = PROJECT / "runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz"
MODALITIES = ("GlobalV", "LocalV", "Skeleton", "IMU", "Depth")
OPTIONAL_MODALITIES = ("Thermal",)
ALL_MODALITIES = MODALITIES + OPTIONAL_MODALITIES


@dataclass(frozen=True)
class ModalityFeatures:
    name: str
    aligned: np.ndarray
    interventions: dict[str, np.ndarray]
    available: np.ndarray
    audit: dict[str, Any]

    def validate(self, rows: int) -> None:
        if self.name not in ALL_MODALITIES:
            raise ValueError(f"unknown P104 modality: {self.name}")
        if self.aligned.shape[0] != rows or self.aligned.ndim != 2:
            raise ValueError(f"invalid {self.name} descriptor shape: {self.aligned.shape}")
        if self.available.shape != (rows,):
            raise ValueError(f"invalid {self.name} availability: {self.available.shape}")
        if not np.isfinite(self.aligned).all():
            raise ValueError(f"{self.name} aligned descriptor is non-finite")
        for name, values in self.interventions.items():
            if values.shape != self.aligned.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid {self.name}/{name}: {values.shape}")


def _flatten(*values: np.ndarray) -> np.ndarray:
    rows = int(values[0].shape[0])
    return np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(rows, -1) for value in values],
        axis=1,
    ).astype(np.float32, copy=False)


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.asarray(mask, dtype=np.float32)[..., None]


def _local_swap_indices(names: tuple[str, ...]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    output = []
    for name in names:
        if "_left" in name:
            swapped = name.replace("_left", "_right")
        elif "_right" in name:
            swapped = name.replace("_right", "_left")
        else:
            swapped = name
        output.append(lookup[swapped])
    return np.asarray(output, dtype=np.int64)


def _global_visual(data: P100AData) -> ModalityFeatures:
    aligned = _flatten(
        data.visual_vmae,
        data.visual_iv2,
        data.visual_vmae_action,
        data.visual_iv2_action,
    )
    result = ModalityFeatures(
        name="GlobalV",
        aligned=aligned,
        interventions={},
        available=np.ones(len(data.sample_ids), dtype=bool),
        audit={
            "sources": [
                "VideoMAEv2 distilled-base early/late scene/person/workspace feature+K710",
                "InternVideo2-L K400 early/late scene/person/workspace feature+K400",
            ],
            "descriptor_dim": int(aligned.shape[1]),
            "special_interventions": [],
        },
    )
    result.validate(len(data.sample_ids))
    return result


def _local_visual(data: P100AData) -> ModalityFeatures:
    local = load_p103_local_data(data.sample_ids, data.users)
    aligned = _flatten(
        local.vmae_features,
        local.vmae_actions,
        local.vjepa_features,
        local.vjepa_actions,
    )
    vmae_swap = _local_swap_indices(VMAE_VIEW_NAMES)
    vjepa_swap = _local_swap_indices(VJEPA_VIEW_NAMES)
    swapped = _flatten(
        local.vmae_features[:, vmae_swap],
        local.vmae_actions[:, vmae_swap],
        local.vjepa_features[:, vjepa_swap],
        local.vjepa_actions[:, vjepa_swap],
    )
    result = ModalityFeatures(
        name="LocalV",
        aligned=aligned,
        interventions={"left_right_swap": swapped},
        available=np.ones(len(data.sample_ids), dtype=bool),
        audit={
            "sources": ["P103-B3 VideoMAEv2 local", "P96 V-JEPA2 workspace/local"],
            "descriptor_dim": int(aligned.shape[1]),
            "special_interventions": ["left_right_swap"],
            "vmae_views": list(VMAE_VIEW_NAMES),
            "vjepa_views": list(VJEPA_VIEW_NAMES),
        },
    )
    result.validate(len(data.sample_ids))
    return result


def _skeleton(data: P100AData) -> ModalityFeatures:
    raw = _masked(data.skeleton_sequence, data.skeleton_mask)
    aligned = _flatten(
        data.skeleton_motionbert,
        data.skeleton_hdgcn,
        raw,
        data.skeleton_mask,
        data.skeleton_statistics,
    )
    reverse = _flatten(
        data.skeleton_motionbert[:, ::-1],
        data.skeleton_hdgcn[:, :, ::-1],
        raw[:, ::-1],
        data.skeleton_mask[:, ::-1],
        data.skeleton_statistics,
    )
    result = ModalityFeatures(
        name="Skeleton",
        aligned=aligned,
        interventions={"reverse_temporal": reverse},
        available=data.skeleton_available.astype(bool),
        audit={
            "sources": ["MotionBERT", "HD-GCN", "P86 body-local sequence/statistics"],
            "descriptor_dim": int(aligned.shape[1]),
            "special_interventions": ["reverse_temporal"],
            "available_rows": int(np.sum(data.skeleton_available > 0)),
        },
    )
    result.validate(len(data.sample_ids))
    return result


def _imu(data: P100AData) -> ModalityFeatures:
    raw = _masked(data.imu_sequence, data.imu_mask)
    aligned = _flatten(raw, data.imu_mask, data.imu_statistics)
    reverse = _flatten(
        raw[:, ::-1], data.imu_mask[:, ::-1], data.imu_statistics
    )
    result = ModalityFeatures(
        name="IMU",
        aligned=aligned,
        interventions={"reverse_temporal": reverse},
        available=data.imu_available.astype(bool),
        audit={
            "sources": ["P86 five-device sequence", "P89/P100 device/phase/statistics"],
            "descriptor_dim": int(aligned.shape[1]),
            "special_interventions": ["reverse_temporal"],
            "available_rows": int(np.sum(data.imu_available > 0)),
        },
    )
    result.validate(len(data.sample_ids))
    return result


def _depth(data: P100AData, path: Path) -> ModalityFeatures:
    # Explicit fields are intentional: historical labels/fold/head are not requested.
    with np.load(path.resolve(), allow_pickle=False) as archive:
        requested = {"sample_ids", "users", "features", "action_logits"}
        missing = requested - set(archive.files)
        if missing:
            raise KeyError(f"Depth archive misses {sorted(missing)}")
        source_ids = archive["sample_ids"].astype(str)
        source_users = archive["users"].astype(str)
        lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
        missing_ids = [sample_id for sample_id in data.sample_ids if sample_id not in lookup]
        if missing_ids:
            raise KeyError(f"Depth misses {len(missing_ids)} P104 rows")
        order = np.asarray([lookup[value] for value in data.sample_ids], dtype=np.int64)
        selected_users = source_users[order]
        if not np.array_equal(selected_users, data.users):
            raise RuntimeError("Depth/P104 user alignment differs")
        if set(selected_users.tolist()) & set(H3_USERS):
            raise RuntimeError("H3 Depth row selected by P104")
        features = np.asarray(archive["features"][order], dtype=np.float32)
        actions = np.asarray(archive["action_logits"][order], dtype=np.float32)
    aligned = _flatten(features, actions)
    result = ModalityFeatures(
        name="Depth",
        aligned=aligned,
        interventions={},
        available=np.ones(len(data.sample_ids), dtype=bool),
        audit={
            "source": str(path.resolve()),
            "requested_fields": sorted(requested),
            "historical_label_field_requested": False,
            "descriptor_dim": int(aligned.shape[1]),
            "special_interventions": [],
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
    )
    result.validate(len(data.sample_ids))
    return result


def _thermal(data: P100AData, path: Path) -> ModalityFeatures:
    # Explicit fields are intentional: historical labels/fold/head are not requested.
    with np.load(path.resolve(), allow_pickle=False) as archive:
        requested = {
            "sample_ids",
            "users",
            "features",
            "action_logits",
            "modality_available",
            "modality",
        }
        missing = requested - set(archive.files)
        if missing:
            raise KeyError(f"Thermal archive misses {sorted(missing)}")
        if str(np.asarray(archive["modality"]).item()) != "thermal":
            raise RuntimeError("P104 Thermal archive identity changed")
        source_ids = archive["sample_ids"].astype(str)
        source_users = archive["users"].astype(str)
        lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
        missing_ids = [sample_id for sample_id in data.sample_ids if sample_id not in lookup]
        if missing_ids:
            raise KeyError(f"Thermal misses {len(missing_ids)} P104 rows")
        order = np.asarray([lookup[value] for value in data.sample_ids], dtype=np.int64)
        selected_users = source_users[order]
        if not np.array_equal(selected_users, data.users):
            raise RuntimeError("Thermal/P104 user alignment differs")
        if set(selected_users.tolist()) & set(H3_USERS):
            raise RuntimeError("H3 Thermal row selected by P104")
        features = np.asarray(archive["features"][order], dtype=np.float32)
        actions = np.asarray(archive["action_logits"][order], dtype=np.float32)
        available = np.asarray(archive["modality_available"][order], dtype=bool)
    aligned = _flatten(features, actions)
    result = ModalityFeatures(
        name="Thermal",
        aligned=aligned,
        interventions={},
        available=available,
        audit={
            "source": str(path.resolve()),
            "requested_fields": sorted(requested),
            "historical_label_field_requested": False,
            "historical_fold_field_requested": False,
            "descriptor_dim": int(aligned.shape[1]),
            "special_interventions": [],
            "available_rows": int(available.sum()),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
    )
    result.validate(len(data.sample_ids))
    return result


def load_modality_features(
    name: str,
    *,
    data: P100AData | None = None,
    depth_path: Path = DEFAULT_DEPTH,
    thermal_path: Path = DEFAULT_THERMAL,
) -> ModalityFeatures:
    base = load_p100a_data() if data is None else data
    if name == "GlobalV":
        return _global_visual(base)
    if name == "LocalV":
        return _local_visual(base)
    if name == "Skeleton":
        return _skeleton(base)
    if name == "IMU":
        return _imu(base)
    if name == "Depth":
        return _depth(base, depth_path)
    if name == "Thermal":
        return _thermal(base, thermal_path)
    raise ValueError(f"unknown P104 modality: {name}")
