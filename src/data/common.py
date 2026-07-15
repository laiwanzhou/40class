from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch


class NormalizableSequenceDataset(Protocol):
    def __len__(self) -> int: ...

    def load_tensor(self, index: int, apply_normalization: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]: ...

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None: ...


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def sorted_files(path: Path, suffixes: set[str], recursive: bool = False) -> list[Path]:
    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted(
        (item for item in iterator if item.is_file() and item.suffix.lower() in suffixes),
        key=lambda item: natural_key(item.relative_to(path).as_posix()),
    )


def resolve_manifest_path(data_root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or "\\" in value:
        raise ValueError(f"Manifest path must be relative POSIX path: {value}")
    return data_root.joinpath(*relative.parts)


def load_modality_frames(
    manifest_path: Path,
    fold_path: Path,
    data_root: Path,
    path_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    train_users = set(fold["train_users"])
    val_users = set(fold["val_users"])
    if not train_users or not val_users or train_users & val_users:
        raise ValueError("fold_0 contains invalid or overlapping user sets.")
    if path_column not in manifest.columns:
        raise ValueError(f"Manifest does not contain path column {path_column!r}.")

    present = manifest[path_column].fillna("").astype(str).str.strip().ne("")
    modality = manifest.loc[present].copy()
    unknown_users = set(modality["user_id"]) - train_users - val_users
    if unknown_users:
        raise ValueError(f"Users missing from fold_0: {sorted(unknown_users)}")
    modality["trial_path"] = modality[path_column].map(
        lambda value: resolve_manifest_path(data_root, str(value))
    )
    missing = modality.loc[~modality["trial_path"].map(Path.is_dir)]
    if not missing.empty:
        row = missing.iloc[0]
        raise FileNotFoundError(
            f"Missing trial for sample_id={row['sample_id']}: {row['trial_path']}"
        )
    train = modality[modality["user_id"].isin(train_users)].reset_index(drop=True)
    val = modality[modality["user_id"].isin(val_users)].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError(f"Empty split after filtering {path_column}.")
    return train, val


def resample_sequence(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"Expected non-empty [T,F] sequence, got {values.shape}")
    if len(values) == target_length:
        return values.astype(np.float32, copy=False)
    old_positions = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    new_positions = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    output = np.empty((target_length, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        output[:, channel] = np.interp(new_positions, old_positions, values[:, channel])
    return output


def resample_mask(mask: np.ndarray, target_length: int) -> np.ndarray:
    if len(mask) == target_length:
        return mask.astype(bool, copy=False)
    positions = np.linspace(0, len(mask) - 1, target_length).round().astype(int)
    return mask[positions].astype(bool)


def compute_sequence_normalization(
    dataset: NormalizableSequenceDataset,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None
    count = 0
    for index in range(sample_count):
        tensor, mask, _ = dataset.load_tensor(index, apply_normalization=False)
        values = tensor[mask].numpy().astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite value while computing normalization at index {index}")
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            total_sq = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    if total is None or total_sq is None or count == 0:
        raise ValueError("Could not compute sequence normalization statistics.")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class SequenceDatasetMixin:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    def normalize(self, values: np.ndarray, mask: np.ndarray, apply: bool) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        if apply and self.mean is not None and self.std is not None:
            values = (values - self.mean) / self.std
        values[~mask] = 0.0
        return values
