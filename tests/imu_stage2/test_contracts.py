from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.imu_stage2_contracts import (
    FEATURE_ORDER,
    SENSOR_ORDER,
    DataStatus,
    ImuActionSource,
    ImuPathNotDirectoryError,
    InferenceSample,
    MissingImuDirectoryError,
    NoRecognizableImuCsvError,
    NoUsableGridCellsError,
    NoValidStage1RecordsError,
    SequenceLengthSafetyError,
    Stage1ActionData,
    Stage1DataValidationError,
    Stage2ActionResult,
    TestSampleDescriptor as SampleDescriptor,
    WriteStatus,
    canonical_json_bytes,
    contract_sha256,
    sha256_file,
)


def make_minimal_result() -> Stage2ActionResult:
    return Stage2ActionResult(
        sample_id="sample",
        values=np.zeros((1, 5, 16), dtype=np.float32),
        sensor_mask=np.ones(5, dtype=bool),
        valid_mask=np.ones((1, 5), dtype=bool),
        timestamps_ms=np.array([0], dtype=np.int64),
        qc={},
        status=DataStatus.SUCCESS,
    )


def make_two_step_result() -> Stage2ActionResult:
    values = np.zeros((2, 5, 16), dtype=np.float32)
    valid_mask = np.ones((2, 5), dtype=bool)
    valid_mask[1, 4] = False
    values[1, 4, :] = np.nan
    return Stage2ActionResult(
        sample_id="sample",
        values=values,
        sensor_mask=np.ones(5, dtype=bool),
        valid_mask=valid_mask,
        timestamps_ms=np.array([0, 100], dtype=np.int64),
        qc={"warning_codes": []},
        status=DataStatus.SUCCESS,
    )


def test_fixed_orders_and_status_values_are_exact() -> None:
    assert SENSOR_ORDER == ("LL", "RL", "LA", "RA", "C")
    assert FEATURE_ORDER == (
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
    assert [status.value for status in DataStatus] == [
        "success",
        "success_with_warnings",
        "incomplete_sensors",
        "no_usable_grid_cells",
        "failed",
    ]
    assert [status.value for status in WriteStatus] == [
        "written",
        "skipped_existing",
        "qc_only",
        "not_written",
    ]


def test_canonical_json_and_contract_hash_are_order_independent() -> None:
    contract = {
        "schema_version": "imu-stage2-v1",
        "grid_step_ns": 100_000_000,
        "description": "传感器",
    }
    reversed_contract = dict(reversed(list(contract.items())))

    assert canonical_json_bytes(contract) == (
        b'{"description":"\xe4\xbc\xa0\xe6\x84\x9f\xe5\x99\xa8",'
        b'"grid_step_ns":100000000,"schema_version":"imu-stage2-v1"}'
    )
    assert contract_sha256(contract) == contract_sha256(reversed_contract)


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json_bytes({"bad": float("nan")})


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"stage-2\x00\xff")

    assert sha256_file(path) == (
        "99fdbc91469ac4a94451cb4ec95a82233f55c6c37447e7906327438b8e50c921"
    )


def test_source_descriptors_are_frozen_and_raw_source_is_nonempty(
    tmp_path: Path,
) -> None:
    descriptor = SampleDescriptor(
        sample_id="SM_test_0001",
        sample_directory=tmp_path / "SM_test_0001",
        source_relative_path=Path("SM_test_0001"),
    )
    with pytest.raises(FrozenInstanceError):
        descriptor.sample_id = "changed"  # type: ignore[misc]

    source = ImuActionSource(
        sample_id="SM_test_0001",
        input_directory=tmp_path / "SM_test_0001" / "IMU",
        input_csv_files=(tmp_path / "SM_test_0001" / "IMU" / "part1.csv",),
        source_relative_path=Path("SM_test_0001/IMU"),
    )
    assert source.input_csv_files[0].name == "part1.csv"
    with pytest.raises(ValueError, match="input_csv_files must not be empty"):
        ImuActionSource(
            sample_id="SM_test_0002",
            input_directory=tmp_path / "SM_test_0002" / "IMU",
            input_csv_files=(),
            source_relative_path=Path("SM_test_0002/IMU"),
        )


def test_stage1_and_inference_contracts_expose_approved_fields() -> None:
    stage1 = Stage1ActionData(
        sample_id="sample",
        dataframe=pd.DataFrame({"sensor_position": ["LL"]}),
        relative_time_ns=np.array([0], dtype=np.int64),
        sensor_mask=np.array([True, False, False, False, False], dtype=bool),
        source_metadata={"source": "artifact"},
        qc={"status": "success"},
        class_id=3,
        class_name="class-three",
        user_id="user1",
        action_id="1-1-1",
    )
    result = make_minimal_result()
    inference = InferenceSample(
        sample_id="sample",
        imu_result=result,
        imu_available=True,
        modality_mask=True,
    )

    assert stage1.relative_time_ns.dtype == np.int64
    assert stage1.sensor_mask.tolist() == [True, False, False, False, False]
    assert inference.imu_result is result
    assert inference.imu_available is inference.modality_mask is True


def test_valid_stage2_result_derives_usable_masks() -> None:
    result = make_two_step_result()

    result.validate()

    assert result.usable_sensor_mask.dtype == np.bool_
    assert result.usable_sensor_mask.tolist() == [True, True, True, True, True]
    assert result.imu_usable is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: setattr(r, "values", r.values.astype(np.float64)), "values must have dtype float32"),
        (lambda r: setattr(r, "values", np.zeros((1, 5, 15), dtype=np.float32)), "values must have shape"),
        (lambda r: setattr(r, "values", np.empty((0, 5, 16), dtype=np.float32)), "T must be at least 1"),
        (lambda r: setattr(r, "sensor_mask", r.sensor_mask.astype(np.uint8)), "sensor_mask must have dtype bool"),
        (lambda r: setattr(r, "sensor_mask", np.ones(4, dtype=bool)), "sensor_mask must have shape"),
        (lambda r: setattr(r, "valid_mask", r.valid_mask.astype(np.uint8)), "valid_mask must have dtype bool"),
        (lambda r: setattr(r, "valid_mask", np.ones((1, 4), dtype=bool)), "valid_mask must have shape"),
        (lambda r: setattr(r, "timestamps_ms", r.timestamps_ms.astype(np.int32)), "timestamps_ms must have dtype int64"),
        (lambda r: setattr(r, "timestamps_ms", np.array([0, 100], dtype=np.int64)), "timestamps_ms must have shape"),
        (lambda r: setattr(r, "status", "success"), "status must be a DataStatus"),
    ],
)
def test_stage2_result_rejects_wrong_types_and_shapes(mutate, message: str) -> None:
    result = make_minimal_result()
    mutate(result)

    with pytest.raises(ValueError, match=message):
        result.validate()


def test_stage2_result_requires_grid_start_and_step() -> None:
    result = make_two_step_result()
    result.timestamps_ms[0] = 1
    with pytest.raises(ValueError, match="timestamps_ms must start at 0"):
        result.validate()

    result = make_two_step_result()
    result.timestamps_ms[1] = 101
    with pytest.raises(ValueError, match="timestamps_ms must increase by exactly 100"):
        result.validate()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_stage2_result_requires_finite_values_at_valid_cells(bad_value: float) -> None:
    result = make_minimal_result()
    result.values[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="valid cells must be finite"):
        result.validate()


def test_stage2_result_requires_nan_at_invalid_cells() -> None:
    result = make_two_step_result()
    result.values[1, 4, :] = 0.0

    with pytest.raises(ValueError, match="invalid cells must be NaN"):
        result.validate()


def test_missing_sensor_cannot_have_valid_cells() -> None:
    result = make_minimal_result()
    result.sensor_mask[0] = False

    with pytest.raises(ValueError, match="missing sensors cannot have valid cells"):
        result.validate()


@pytest.mark.parametrize(
    ("error_type", "error_code", "failure_stage"),
    [
        (MissingImuDirectoryError, "missing_imu_directory", "source_adapter"),
        (ImuPathNotDirectoryError, "imu_path_not_directory", "source_adapter"),
        (NoRecognizableImuCsvError, "no_recognizable_imu_csv", "source_adapter"),
        (NoValidStage1RecordsError, "no_valid_stage1_records", "stage1"),
        (Stage1DataValidationError, "stage1_data_validation_error", "stage1"),
        (NoUsableGridCellsError, "no_usable_grid_cells", "stage2"),
        (SequenceLengthSafetyError, "sequence_length_safety_error", "stage2"),
    ],
)
def test_degradable_errors_carry_structured_fields(
    error_type: type[Exception],
    error_code: str,
    failure_stage: str,
) -> None:
    error = error_type("SM_test_0001", "safe detail")

    assert error.error_code == error_code
    assert error.failure_stage == failure_stage
    assert error.sample_id == "SM_test_0001"
    assert error.safe_message == "safe detail"
    assert str(error) == f"{error_code}: safe detail"
    assert error_type.__bases__ == (Exception,)


def test_generated_output_directories_are_ignored() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    rules = (repository_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    for expected in (
        "artifacts/",
        "stage2_audits/",
        "inference_audits/",
        "submissions/",
        "inference_bundle/",
    ):
        assert rules.count(expected) == 1
