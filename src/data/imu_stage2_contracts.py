from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


SENSOR_ORDER = ("LL", "RL", "LA", "RA", "C")
FEATURE_ORDER = (
    "acc_x_g",
    "acc_y_g",
    "acc_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "angle_x_deg",
    "angle_y_deg",
    "angle_z_deg",
    "mag_x_ut",
    "mag_y_ut",
    "mag_z_ut",
    "quat_0",
    "quat_1",
    "quat_2",
    "quat_3",
)


class DataStatus(str, Enum):
    SUCCESS = "success"
    SUCCESS_WITH_WARNINGS = "success_with_warnings"
    INCOMPLETE_SENSORS = "incomplete_sensors"
    NO_USABLE_GRID_CELLS = "no_usable_grid_cells"
    FAILED = "failed"


class WriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED_EXISTING = "skipped_existing"
    QC_ONLY = "qc_only"
    NOT_WRITTEN = "not_written"


@dataclass(frozen=True)
class TestSampleDescriptor:
    sample_id: str
    sample_directory: Path
    source_relative_path: Path


@dataclass(frozen=True)
class ImuActionSource:
    sample_id: str
    input_directory: Path
    input_csv_files: tuple[Path, ...]
    source_relative_path: Path
    class_id: int | None = None
    class_name: str | None = None
    user_id: str | None = None
    action_id: str | None = None

    def __post_init__(self) -> None:
        if not self.input_csv_files:
            raise ValueError("input_csv_files must not be empty")


@dataclass
class Stage1ActionData:
    sample_id: str
    dataframe: pd.DataFrame
    relative_time_ns: np.ndarray
    sensor_mask: np.ndarray
    source_metadata: dict[str, object]
    qc: dict[str, object]
    class_id: int | None = None
    class_name: str | None = None
    user_id: str | None = None
    action_id: str | None = None


@dataclass
class Stage2ActionResult:
    sample_id: str
    values: np.ndarray
    sensor_mask: np.ndarray
    valid_mask: np.ndarray
    timestamps_ms: np.ndarray
    qc: Mapping[str, object]
    status: DataStatus

    @property
    def usable_sensor_mask(self) -> np.ndarray:
        return self.valid_mask.any(axis=0)

    @property
    def imu_usable(self) -> bool:
        return bool(self.valid_mask.any())

    def validate(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise ValueError("values must be a NumPy array")
        if self.values.dtype != np.float32:
            raise ValueError("values must have dtype float32")
        if self.values.ndim != 3 or self.values.shape[1:] != (5, 16):
            raise ValueError("values must have shape (T, 5, 16)")

        time_steps = self.values.shape[0]
        if time_steps < 1:
            raise ValueError("T must be at least 1")

        if not isinstance(self.sensor_mask, np.ndarray):
            raise ValueError("sensor_mask must be a NumPy array")
        if self.sensor_mask.dtype != np.bool_:
            raise ValueError("sensor_mask must have dtype bool")
        if self.sensor_mask.shape != (5,):
            raise ValueError("sensor_mask must have shape (5,)")

        if not isinstance(self.valid_mask, np.ndarray):
            raise ValueError("valid_mask must be a NumPy array")
        if self.valid_mask.dtype != np.bool_:
            raise ValueError("valid_mask must have dtype bool")
        if self.valid_mask.shape != (time_steps, 5):
            raise ValueError("valid_mask must have shape (T, 5)")

        if not isinstance(self.timestamps_ms, np.ndarray):
            raise ValueError("timestamps_ms must be a NumPy array")
        if self.timestamps_ms.dtype != np.int64:
            raise ValueError("timestamps_ms must have dtype int64")
        if self.timestamps_ms.shape != (time_steps,):
            raise ValueError("timestamps_ms must have shape (T,)")

        if not isinstance(self.status, DataStatus):
            raise ValueError("status must be a DataStatus")
        if self.timestamps_ms[0] != 0:
            raise ValueError("timestamps_ms must start at 0")
        if time_steps > 1 and not np.all(np.diff(self.timestamps_ms) == 100):
            raise ValueError("timestamps_ms must increase by exactly 100")
        if not np.isfinite(self.values[self.valid_mask]).all():
            raise ValueError("valid cells must be finite")
        if not np.isnan(self.values[~self.valid_mask]).all():
            raise ValueError("invalid cells must be NaN")
        if np.any(self.valid_mask[:, ~self.sensor_mask]):
            raise ValueError("missing sensors cannot have valid cells")


@dataclass
class InferenceSample:
    sample_id: str
    imu_result: Stage2ActionResult | None
    imu_available: bool
    modality_mask: bool


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(contract)).hexdigest()


class MissingImuDirectoryError(Exception):
    error_code = "missing_imu_directory"
    failure_stage = "source_adapter"

    def __init__(self, sample_id: str, safe_message: str) -> None:
        self.sample_id = sample_id
        self.safe_message = safe_message
        super().__init__(f"{self.error_code}: {safe_message}")


class ImuPathNotDirectoryError(Exception):
    error_code = "imu_path_not_directory"
    failure_stage = "source_adapter"

    def __init__(self, sample_id: str, safe_message: str) -> None:
        self.sample_id = sample_id
        self.safe_message = safe_message
        super().__init__(f"{self.error_code}: {safe_message}")


class NoRecognizableImuCsvError(Exception):
    error_code = "no_recognizable_imu_csv"
    failure_stage = "source_adapter"

    def __init__(self, sample_id: str, safe_message: str) -> None:
        self.sample_id = sample_id
        self.safe_message = safe_message
        super().__init__(f"{self.error_code}: {safe_message}")


class NoValidStage1RecordsError(Exception):
    error_code = "no_valid_stage1_records"
    failure_stage = "stage1"

    def __init__(self, sample_id: str, safe_message: str) -> None:
        self.sample_id = sample_id
        self.safe_message = safe_message
        super().__init__(f"{self.error_code}: {safe_message}")


class Stage1DataValidationError(Exception):
    error_code = "stage1_data_validation_error"
    failure_stage = "stage1"

    def __init__(self, sample_id: str, safe_message: str) -> None:
        self.sample_id = sample_id
        self.safe_message = safe_message
        super().__init__(f"{self.error_code}: {safe_message}")


class NoUsableGridCellsError(Exception):
    error_code = "no_usable_grid_cells"
    failure_stage = "stage2"

    def __init__(self, sample_id: str, safe_message: str) -> None:
        self.sample_id = sample_id
        self.safe_message = safe_message
        super().__init__(f"{self.error_code}: {safe_message}")


class SequenceLengthSafetyError(Exception):
    error_code = "sequence_length_safety_error"
    failure_stage = "stage2"

    def __init__(self, sample_id: str, safe_message: str) -> None:
        self.sample_id = sample_id
        self.safe_message = safe_message
        super().__init__(f"{self.error_code}: {safe_message}")
