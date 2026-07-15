from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import SequenceDatasetMixin, resample_sequence, sorted_files


DEVICE_ROLES = ("WTRA", "WTLA", "WTC", "WTRL", "WTLL")
SENSOR_COLUMNS = (
    "加速度X(g)", "加速度Y(g)", "加速度Z(g)",
    "角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)",
    "角度X(°)", "角度Y(°)", "角度Z(°)",
    "磁场X(uT)", "磁场Y(uT)", "磁场Z(uT)",
    "四元数0()", "四元数1()", "四元数2()", "四元数3()",
)


class IMUDataset(SequenceDatasetMixin, Dataset[dict[str, object]]):
    def __init__(self, frame: pd.DataFrame, sequence_length: int = 256) -> None:
        self.frame = frame.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.mean = None
        self.std = None

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _role(device_name: str) -> str:
        match = re.match(r"([A-Za-z]+)", device_name)
        return match.group(1).upper() if match else device_name.upper()

    def load_tensor(self, index: int, apply_normalization: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.frame.iloc[index]
        sample_id = str(row["sample_id"])
        trial_path = Path(row["trial_path"])
        csv_files = sorted_files(trial_path, {".csv"})
        if len(csv_files) != 2:
            raise ValueError(
                f"Expected two IMU CSV files for sample_id={sample_id}: {trial_path}; got {len(csv_files)}"
            )
        tables: list[pd.DataFrame] = []
        for csv_path in csv_files:
            try:
                table = pd.read_csv(csv_path)
            except Exception as exc:
                raise RuntimeError(f"IMU CSV read failed for sample_id={sample_id}: {csv_path}") from exc
            required = {"时间", "设备名称", *SENSOR_COLUMNS}
            missing = required - set(table.columns)
            if missing:
                raise ValueError(f"Missing IMU columns {sorted(missing)} for sample_id={sample_id}: {csv_path}")
            tables.append(table)
        non_empty_tables = [table.dropna(how="all") for table in tables if not table.dropna(how="all").empty]
        combined = (
            pd.concat(non_empty_tables, ignore_index=True)
            if non_empty_tables
            else pd.DataFrame(columns=tables[0].columns)
        )
        combined["_role"] = combined["设备名称"].astype(str).map(self._role)
        combined["_time"] = pd.to_datetime(combined["时间"], errors="coerce")

        role_arrays: list[np.ndarray] = []
        original_length = 0
        for role in DEVICE_ROLES:
            group = combined[combined["_role"] == role].sort_values("_time", kind="stable")
            original_length = max(original_length, len(group))
            if group.empty:
                role_arrays.append(np.zeros((self.sequence_length, len(SENSOR_COLUMNS)), dtype=np.float32))
                continue
            values = group.loc[:, SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            role_arrays.append(resample_sequence(values, self.sequence_length))
        values = np.concatenate(role_arrays, axis=1)
        mask = np.ones(self.sequence_length, dtype=bool)
        values = self.normalize(values, mask, apply_normalization)
        return torch.from_numpy(values), torch.from_numpy(mask), original_length

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
