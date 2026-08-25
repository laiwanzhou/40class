from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np
import pandas as pd


MODALITY_COLUMNS = {
    "ir": "ir_path",
    "depth_color": "depth_color_path",
    "skeleton": "skeleton_path",
    "imu": "imu_path",
    "radar": "radar_path",
    "thermal": "thermal_path",
}
CORE_MODALITIES = ("ir", "depth_color", "skeleton", "imu")


@dataclass(frozen=True)
class CanonicalTrial:
    sample_id: str
    user_id: str
    class_id: int
    paths: Mapping[str, Path | None]
    availability: Mapping[str, bool]

    @property
    def core_available(self) -> bool:
        return any(bool(self.availability[name]) for name in CORE_MODALITIES)


def normalized_segment_bounds(length: int, segments: int = 8) -> np.ndarray:
    if length < 1:
        raise ValueError("sequence length must be positive")
    if segments < 1:
        raise ValueError("segment count must be positive")
    edges = np.rint(np.linspace(0, length, segments + 1)).astype(np.int64)
    edges[0], edges[-1] = 0, length
    return np.stack((edges[:-1], edges[1:]), axis=1)


def _resolve_optional_path(data_root: Path, value: object) -> Path | None:
    if pd.isna(value) or not str(value).strip():
        return None
    text = str(value)
    relative = PurePosixPath(text)
    if relative.is_absolute() or "\\" in text:
        raise ValueError(f"manifest path must be relative POSIX: {text}")
    return data_root.joinpath(*relative.parts)


def build_canonical_trials(
    manifest_path: Path,
    split_path: Path,
    data_root: Path,
    *,
    partition: str,
) -> list[CanonicalTrial]:
    if partition not in {"train", "validation"}:
        raise ValueError("partition must be train or validation")
    manifest = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        dtype={"sample_id": str, "user_id": str, "trial_id": str},
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    key = "train_user_ids" if partition == "train" else "validation_user_ids"
    users = {str(value) for value in split[key]}
    selected = manifest[manifest["user_id"].astype(str).isin(users)].copy()
    selected = selected.sort_values(["class_id", "sample_id"]).reset_index(drop=True)
    expected = 2039 if partition == "train" else 388
    if len(selected) != expected:
        raise ValueError(f"{partition} canonical sample count changed: {len(selected)}")
    if selected["sample_id"].astype(str).nunique() != expected:
        raise ValueError(f"{partition} canonical sample IDs are not unique")
    if set(selected["class_id"].astype(int)) != set(range(40)):
        raise ValueError(f"{partition} canonical class coverage changed")
    if set(selected["user_id"].astype(str)) != users:
        raise ValueError(f"{partition} canonical user coverage changed")

    trials: list[CanonicalTrial] = []
    for row in selected.itertuples(index=False):
        paths = {
            modality: _resolve_optional_path(data_root, getattr(row, column))
            for modality, column in MODALITY_COLUMNS.items()
        }
        availability = {name: path is not None for name, path in paths.items()}
        trials.append(
            CanonicalTrial(
                sample_id=str(row.sample_id),
                user_id=str(row.user_id),
                class_id=int(row.class_id),
                paths=paths,
                availability=availability,
            )
        )
    return trials

