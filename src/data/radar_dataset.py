from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import SequenceDatasetMixin, resample_mask, resample_sequence, sorted_files


RADAR_COLUMNS = ("frame", "x", "y", "z", "v", "snr", "noise")


class RadarDataset(SequenceDatasetMixin, Dataset[dict[str, object]]):
    def __init__(self, frame: pd.DataFrame, sequence_length: int = 64) -> None:
        self.frame = frame.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.mean = None
        self.std = None

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _frame_features(group: pd.DataFrame) -> np.ndarray:
        result = [float(len(group))]
        for column in ("x", "y", "z", "v"):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float32)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                result.extend([0.0, 0.0, 0.0, 0.0])
            else:
                result.extend([float(values.mean()), float(values.std()), float(values.min()), float(values.max())])
        for column in ("snr", "noise"):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float32)
            values = values[np.isfinite(values)]
            result.extend([float(values.mean()) if len(values) else 0.0, float(values.std()) if len(values) else 0.0])
        return np.asarray(result, dtype=np.float32)

    def load_tensor(self, index: int, apply_normalization: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.frame.iloc[index]
        sample_id = str(row["sample_id"])
        trial_path = Path(row["trial_path"])
        csv_files = sorted_files(trial_path, {".csv"})
        if not csv_files:
            raise FileNotFoundError(f"No radar CSV for sample_id={sample_id}: {trial_path}")
        try:
            table = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)
        except Exception as exc:
            raise RuntimeError(f"Radar CSV read failed for sample_id={sample_id}: {trial_path}") from exc
        missing = set(RADAR_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"Missing Radar columns {sorted(missing)} for sample_id={sample_id}: {trial_path}")
        frame_ids = pd.to_numeric(table["frame"], errors="coerce").dropna().astype(int)
        if frame_ids.empty:
            values = np.zeros((self.sequence_length, 21), dtype=np.float32)
            mask = np.zeros(self.sequence_length, dtype=bool)
            values = self.normalize(values, mask, apply_normalization)
            return torch.from_numpy(values), torch.from_numpy(mask), 0
        table = table.assign(_frame=pd.to_numeric(table["frame"], errors="coerce")).dropna(subset=["_frame"])
        groups = {int(frame_id): group for frame_id, group in table.groupby("_frame", sort=True)}
        first, last = min(groups), max(groups)
        features: list[np.ndarray] = []
        mask: list[bool] = []
        zero = np.zeros(21, dtype=np.float32)
        for frame_id in range(first, last + 1):
            if frame_id in groups:
                features.append(self._frame_features(groups[frame_id]))
                mask.append(True)
            else:
                features.append(zero.copy())
                mask.append(False)
        values = resample_sequence(np.stack(features), self.sequence_length)
        sampled_mask = resample_mask(np.asarray(mask, dtype=bool), self.sequence_length)
        values = self.normalize(values, sampled_mask, apply_normalization)
        return torch.from_numpy(values), torch.from_numpy(sampled_mask), len(features)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        tensor, mask, original_length = self.load_tensor(index)
        return {
            "input": tensor,
            "temporal_mask": mask,
            "label": int(row["class_id"]),
            "sample_id": str(row["sample_id"]),
            "length": original_length,
        }
