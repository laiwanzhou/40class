from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import imu_stage2_core
from src.data.imu_stage2_contracts import (
    DataStatus,
    FEATURE_ORDER,
    SequenceLengthSafetyError,
    Stage1ActionData,
)
from src.data.imu_stage2_core import (
    AggregatedSensorSeries,
    build_action_grid,
    interpolate_sensor_on_grid,
    process_stage2_action,
    select_data_status,
)


Q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _values(
    ordinary: float = 0.0,
    *,
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: np.ndarray = Q,
) -> np.ndarray:
    return np.asarray(
        [ordinary] * 6 + list(angles) + [ordinary] * 3 + quaternion.tolist(),
        dtype=np.float64,
    )


def _series(
    times: list[int],
    values: list[np.ndarray],
) -> AggregatedSensorSeries:
    return AggregatedSensorSeries(
        sensor_position="LL",
        time_ns=np.asarray(times, dtype=np.int64),
        values=np.asarray(values, dtype=np.float64),
        qc={},
    )


def _action(end_ns: int) -> Stage1ActionData:
    record = {
        "sensor_position": "LL",
        "source_file": "source.csv",
        "source_row_index": 0,
        "_source_file_rank": 0,
        "_stage1_row_index": 0,
    }
    record.update(dict(zip(FEATURE_ORDER, _values(), strict=True)))
    return Stage1ActionData(
        sample_id="sample",
        dataframe=pd.DataFrame.from_records([record]),
        relative_time_ns=np.asarray([end_ns], dtype=np.int64),
        sensor_mask=np.asarray([True, False, False, False, False]),
        source_metadata={},
        qc={},
    )


def _multi_sensor_action(
    sensor_rows: dict[str, list[tuple[int, np.ndarray]]],
    *,
    sensor_mask: np.ndarray | None = None,
) -> Stage1ActionData:
    records: list[dict[str, object]] = []
    times: list[int] = []
    for sensor, rows in sensor_rows.items():
        for row_index, (time_ns, values) in enumerate(rows):
            record = {
                "sensor_position": sensor,
                "source_file": f"{sensor}.csv",
                "source_row_index": row_index,
                "_source_file_rank": 0,
                "_stage1_row_index": len(records),
            }
            record.update(dict(zip(FEATURE_ORDER, values, strict=True)))
            records.append(record)
            times.append(time_ns)
    return Stage1ActionData(
        sample_id="sample",
        dataframe=pd.DataFrame.from_records(records),
        relative_time_ns=np.asarray(times, dtype=np.int64),
        sensor_mask=(
            np.asarray([sensor in sensor_rows for sensor in ("LL", "RL", "LA", "RA", "C")], dtype=bool)
            if sensor_mask is None
            else sensor_mask
        ),
        source_metadata={},
        qc={},
    )


@pytest.mark.parametrize(
    ("end_ns", "expected_ms"),
    [
        (0, [0]),
        (99_999_999, [0]),
        (100_000_000, [0, 100]),
        (2_263_000_000, list(range(0, 2201, 100))),
    ],
)
def test_grid_uses_floor_and_includes_endpoint(end_ns: int, expected_ms: list[int]) -> None:
    assert (build_action_grid(end_ns) // 1_000_000).tolist() == expected_ms


def test_exact_hit_is_valid_without_neighbor_gap_check() -> None:
    result = interpolate_sensor_on_grid(
        _series([0, 400_000_000], [_values(1.0), _values(5.0)]),
        np.asarray([0, 100_000_000], dtype=np.int64),
    )

    assert result.valid_mask.tolist() == [True, False]
    assert result.exact_mask.tolist() == [True, False]
    assert result.interpolated_mask.tolist() == [False, False]
    assert np.allclose(result.values[0], _values(1.0))
    assert np.isnan(result.values[1]).all()


def test_interpolation_allows_300_ms_gap_inclusively() -> None:
    result = interpolate_sensor_on_grid(
        _series([0, 300_000_000], [_values(0.0), _values(3.0)]),
        build_action_grid(300_000_000),
    )

    assert result.valid_mask.tolist() == [True, True, True, True]
    assert result.exact_mask.tolist() == [True, False, False, True]
    assert result.interpolated_mask.tolist() == [False, True, True, False]
    assert np.allclose(result.values[1, 0:6], 1.0)
    assert np.allclose(result.values[2, 0:6], 2.0)


def test_interpolation_rejects_gap_over_300_ms() -> None:
    result = interpolate_sensor_on_grid(
        _series([0, 300_000_001], [_values(0.0), _values(3.0)]),
        build_action_grid(300_000_000),
    )

    assert result.valid_mask.tolist() == [True, False, False, False]
    assert result.exact_mask.tolist() == [True, False, False, False]
    assert not result.interpolated_mask.any()
    assert np.isnan(result.values[1:]).all()


def test_fragmented_interpolation_does_not_scan_full_grid_per_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(100, dtype=np.int64) * np.int64(300_000_001)
    series = _series(
        times.tolist(),
        [_values(float(index)) for index in range(len(times))],
    )
    grid_ns = build_action_grid(int(times[-1]) + 99_999_901)
    searchsorted_call_count = 0
    original_searchsorted = imu_stage2_core.np.searchsorted

    def counted_searchsorted(*args: object, **kwargs: object) -> object:
        nonlocal searchsorted_call_count
        searchsorted_call_count += 1
        return original_searchsorted(*args, **kwargs)

    monkeypatch.setattr(imu_stage2_core.np, "searchsorted", counted_searchsorted)

    result = interpolate_sensor_on_grid(series, grid_ns)

    assert result.valid_mask.sum() == 1
    assert result.exact_mask[0]
    assert not result.interpolated_mask.any()
    assert searchsorted_call_count <= len(series.time_ns) + len(grid_ns)


def test_interpolation_does_not_extrapolate_at_either_boundary() -> None:
    result = interpolate_sensor_on_grid(
        _series([100_000_000, 200_000_000], [_values(1.0), _values(2.0)]),
        build_action_grid(300_000_000),
    )

    assert result.valid_mask.tolist() == [False, True, True, False]
    assert np.isnan(result.values[[0, 3]]).all()


def test_interpolation_wraps_angles_after_unwrapped_linear_interpolation() -> None:
    result = interpolate_sensor_on_grid(
        _series(
            [0, 200_000_000],
            [_values(angles=(179.0, 179.0, 179.0)), _values(angles=(-179.0, -179.0, -179.0))],
        ),
        build_action_grid(200_000_000),
    )

    assert result.valid_mask.tolist() == [True, True, True]
    assert np.allclose(result.values[1, 6:9], [-180.0, -180.0, -180.0])


def test_interpolation_uses_normalized_shortest_sign_quaternion_nlerp() -> None:
    result = interpolate_sensor_on_grid(
        _series(
            [0, 200_000_000],
            [_values(quaternion=Q), _values(quaternion=np.asarray([0.0, -1.0, 0.0, 0.0]))],
        ),
        build_action_grid(200_000_000),
    )

    assert np.allclose(result.values[1, 12:16], [2**-0.5, -2**-0.5, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(result.values[1, 12:16]), 1.0)


def test_interpolation_uses_shared_endpoints_for_all_sixteen_features() -> None:
    left = _values(2.0, angles=(170.0, 10.0, -10.0), quaternion=Q)
    right = _values(
        6.0,
        angles=(-170.0, 30.0, -30.0),
        quaternion=np.asarray([0.0, 1.0, 0.0, 0.0]),
    )
    result = interpolate_sensor_on_grid(
        _series([0, 200_000_000], [left, right]),
        np.asarray([100_000_000], dtype=np.int64),
    )

    assert result.valid_mask.tolist() == [True]
    assert np.allclose(result.values[0, 0:6], 4.0)
    assert np.allclose(result.values[0, 6:9], [-180.0, 20.0, -20.0])
    assert np.allclose(result.values[0, 9:12], 4.0)
    assert np.allclose(result.values[0, 12:16], [2**-0.5, 2**-0.5, 0.0, 0.0])


def test_process_checks_hard_safety_limit_before_grid_allocation() -> None:
    with pytest.raises(SequenceLengthSafetyError, match="hard safety limit"):
        process_stage2_action(_action(1_000_000_000), hard_safety_limit_t=10)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_process_rejects_non_positive_or_non_integer_safety_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="hard_safety_limit_t"):
        process_stage2_action(_action(0), hard_safety_limit_t=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("failed", "no_usable", "incomplete", "warnings", "expected"),
    [
        (True, True, True, True, DataStatus.FAILED),
        (False, True, True, True, DataStatus.NO_USABLE_GRID_CELLS),
        (False, False, True, True, DataStatus.INCOMPLETE_SENSORS),
        (False, False, False, True, DataStatus.SUCCESS_WITH_WARNINGS),
        (False, False, False, False, DataStatus.SUCCESS),
    ],
)
def test_status_selection_uses_contract_precedence(
    failed: bool,
    no_usable: bool,
    incomplete: bool,
    warnings: bool,
    expected: DataStatus,
) -> None:
    assert select_data_status(
        failed=failed,
        no_usable_grid_cells=no_usable,
        incomplete_sensors=incomplete,
        has_warnings=warnings,
    ) is expected


def test_process_missing_sensor_columns_remain_all_invalid_and_nan() -> None:
    result = process_stage2_action(
        _multi_sensor_action({"LL": [(0, _values(2.0))]}),
    )

    assert result.status is DataStatus.INCOMPLETE_SENSORS
    assert result.sensor_mask.tolist() == [True, False, False, False, False]
    assert not result.valid_mask[:, 1:].any()
    assert np.isnan(result.values[:, 1:, :]).all()
    assert result.qc["warning_codes"] == ["incomplete_sensors"]


def test_process_no_usable_grid_cells_keeps_nonempty_all_invalid_result() -> None:
    result = process_stage2_action(
        _multi_sensor_action({"LL": [(50_000_000, _values(2.0))]}),
    )

    assert result.values.shape == (1, 5, 16)
    assert not result.valid_mask.any()
    assert np.isnan(result.values).all()
    assert not result.imu_usable
    assert result.status is DataStatus.NO_USABLE_GRID_CELLS
    assert result.qc["last_usable_timestamp_ns"] is None
    assert result.qc["valid_cell_count"] == 0
    assert result.qc["invalid_cell_count"] == 5
    assert result.qc["warning_codes"] == [
        "incomplete_sensors",
        "no_usable_grid_cells",
    ]


def test_process_duplicate_aggregation_is_a_warning_without_overriding_complete_status() -> None:
    action = _multi_sensor_action(
        {
            "LL": [(0, _values(1.0)), (0, _values(3.0))],
            "RL": [(0, _values())],
            "LA": [(0, _values())],
            "RA": [(0, _values())],
            "C": [(0, _values())],
        }
    )

    result = process_stage2_action(action)

    assert result.status is DataStatus.SUCCESS_WITH_WARNINGS
    assert result.qc["warning_codes"] == ["duplicate_timestamps_aggregated"]
    assert result.qc["duplicate_group_count"] == 1
    assert result.qc["valid_cell_count"] == 5
    assert result.qc["exact_hit_count"] == 5
    assert result.qc["interpolated_count"] == 0


def test_process_clean_complete_action_returns_validated_float32_result() -> None:
    result = process_stage2_action(
        _multi_sensor_action(
            {
                "LL": [(0, _values(1.0))],
                "RL": [(0, _values(2.0))],
                "LA": [(0, _values(3.0))],
                "RA": [(0, _values(4.0))],
                "C": [(0, _values(5.0))],
            }
        )
    )

    result.validate()
    assert result.status is DataStatus.SUCCESS
    assert result.values.dtype == np.float32
    assert result.valid_mask.tolist() == [[True, True, True, True, True]]
    assert result.timestamps_ms.tolist() == [0]
    assert result.qc["last_usable_timestamp_ns"] == 0
