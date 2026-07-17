from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.imu_stage2_contracts import FEATURE_ORDER, Stage1ActionData
from src.data.imu_stage2_core import (
    AggregatedSensorSeries,
    aggregate_sensor_timestamps,
    split_continuous_segments,
)


Q = np.asarray([0.5, -0.5, 0.5, -0.5], dtype=np.float64)


def _row(
    *,
    ordinary: float = 0.0,
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: np.ndarray = Q,
) -> np.ndarray:
    return np.asarray(
        [ordinary] * 6
        + list(angles)
        + [ordinary] * 3
        + quaternion.tolist(),
        dtype=np.float64,
    )


def make_stage1_group(
    rows: list[np.ndarray],
    *,
    times: list[int] | None = None,
    ranks: list[int] | None = None,
    source_rows: list[int] | None = None,
) -> Stage1ActionData:
    count = len(rows)
    times = [0] * count if times is None else times
    ranks = list(range(count)) if ranks is None else ranks
    source_rows = list(range(count)) if source_rows is None else source_rows
    records: list[dict[str, object]] = []
    for index, values in enumerate(rows):
        record: dict[str, object] = {
            "sensor_position": "LL",
            "source_file": f"source-{ranks[index]}.csv",
            "source_row_index": source_rows[index],
            "_source_file_rank": ranks[index],
            "_stage1_row_index": index,
        }
        record.update(dict(zip(FEATURE_ORDER, values, strict=True)))
        records.append(record)
    return Stage1ActionData(
        sample_id="sample",
        dataframe=pd.DataFrame.from_records(records),
        relative_time_ns=np.asarray(times, dtype=np.int64),
        sensor_mask=np.asarray([True, False, False, False, False]),
        source_metadata={},
        qc={},
    )


def test_duplicate_ordinary_features_use_float64_arithmetic_mean() -> None:
    series = aggregate_sensor_timestamps(
        make_stage1_group([_row(ordinary=1.0), _row(ordinary=3.0)])
    )

    assert series.values.dtype == np.float64
    assert np.array_equal(series.values[0, 0:6], np.full(6, 2.0))
    assert np.array_equal(series.values[0, 9:12], np.full(3, 2.0))


def test_duplicate_angles_use_wrapped_circular_mean() -> None:
    series = aggregate_sensor_timestamps(
        make_stage1_group(
            [
                _row(angles=(179.0, 10.0, -20.0)),
                _row(angles=(-179.0, 20.0, -40.0)),
            ]
        )
    )

    assert np.allclose(series.values[0, 6:9], [-180.0, 15.0, -30.0])


def test_duplicate_quaternions_align_sign_before_mean() -> None:
    series = aggregate_sensor_timestamps(make_stage1_group([_row(), _row(quaternion=-Q)]))

    assert np.allclose(np.linalg.norm(series.values[0, 12:16]), 1.0)
    assert np.allclose(series.values[0, 12:16], Q)


def test_partial_invalid_records_are_excluded_before_aggregation() -> None:
    invalid = _row(ordinary=99.0)
    invalid[0] = np.nan
    series = aggregate_sensor_timestamps(
        make_stage1_group([_row(ordinary=2.0), invalid])
    )

    assert series.time_ns.tolist() == [0]
    assert np.allclose(series.values[0, 0:6], 2.0)
    assert series.qc["excluded_record_count"] == 1
    assert series.qc["nonfinite_feature_record_count"] == 1


def test_single_valid_record_falls_back_with_normalized_quaternion() -> None:
    invalid = _row()
    invalid[12:16] = 0.0
    series = aggregate_sensor_timestamps(
        make_stage1_group([_row(quaternion=Q * 4.0), invalid])
    )

    assert series.time_ns.tolist() == [0]
    assert np.allclose(series.values[0, 12:16], Q)
    assert series.qc["invalid_quaternion_record_count"] == 1


def test_degenerate_circular_group_deletes_timestamp() -> None:
    series = aggregate_sensor_timestamps(
        make_stage1_group(
            [
                _row(angles=(0.0, 0.0, 0.0)),
                _row(angles=(180.0, 0.0, 0.0)),
            ]
        )
    )

    assert series.time_ns.size == 0
    assert series.values.shape == (0, len(FEATURE_ORDER))
    assert series.qc["aggregation_failed_timestamp_count"] == 1
    assert series.qc["degenerate_angle_group_count"] == 1


def test_group_with_only_degenerate_quaternions_deletes_timestamp() -> None:
    invalid = _row()
    invalid[12:16] = 0.0
    series = aggregate_sensor_timestamps(make_stage1_group([invalid, invalid.copy()]))

    assert series.time_ns.size == 0
    assert series.qc["aggregation_failed_timestamp_count"] == 1
    assert series.qc["invalid_quaternion_record_count"] == 2


def test_stable_rank_selects_first_quaternion_reference() -> None:
    series = aggregate_sensor_timestamps(
        make_stage1_group(
            [_row(quaternion=-Q), _row(quaternion=Q)],
            ranks=[9, 1],
            source_rows=[0, 0],
        )
    )

    assert np.allclose(series.values[0, 12:16], Q)


def test_duplicate_counts_describe_original_groups_before_exclusion() -> None:
    invalid = _row()
    invalid[0] = np.inf
    series = aggregate_sensor_timestamps(
        make_stage1_group(
            [_row(), invalid, _row()],
            times=[0, 0, 10],
        )
    )

    assert series.time_ns.tolist() == [0, 10]
    assert series.qc["duplicate_group_count"] == 1
    assert series.qc["duplicate_extra_record_count"] == 1
    assert series.qc["duplicate_max_group_size"] == 2
    assert series.qc["duplicate_excluded_record_count"] == 1


def test_exact_int64_timestamps_are_grouped_without_float_conversion() -> None:
    base = 2**53
    series = aggregate_sensor_timestamps(
        make_stage1_group(
            [_row(ordinary=1.0), _row(ordinary=3.0)],
            times=[base, base + 1],
        )
    )

    assert series.time_ns.tolist() == [base, base + 1]
    assert series.qc["duplicate_group_count"] == 0


def make_series(
    times: list[int],
    *,
    angles: list[float] | None = None,
    quaternions: list[np.ndarray] | None = None,
) -> AggregatedSensorSeries:
    count = len(times)
    values = np.zeros((count, len(FEATURE_ORDER)), dtype=np.float64)
    values[:, 6:9] = np.asarray(
        [0.0] * count if angles is None else angles,
        dtype=np.float64,
    )[:, None]
    values[:, 12:16] = np.asarray(
        [Q] * count if quaternions is None else quaternions,
        dtype=np.float64,
    )
    return AggregatedSensorSeries(
        sensor_position="LL",
        time_ns=np.asarray(times, dtype=np.int64),
        values=values,
        qc={},
    )


def test_continuity_restarts_after_gap_over_300_ms() -> None:
    series = make_series(times=[0, 300_000_000, 600_000_001])

    segments = split_continuous_segments(series, max_gap_ns=300_000_000)

    assert [segment.time_ns.tolist() for segment in segments] == [
        [0, 300_000_000],
        [600_000_001],
    ]


def test_each_segment_canonicalizes_its_first_quaternion() -> None:
    second_start = np.asarray([0.0, -0.6, 0.8, 0.0])
    series = make_series(
        times=[0, 100_000_000, 500_000_000],
        quaternions=[-Q, Q, second_start],
    )

    segments = split_continuous_segments(series)

    assert np.allclose(segments[0].values[0, 12:16], Q)
    assert np.allclose(segments[1].values[0, 12:16], -second_start)


def test_later_quaternion_signs_follow_adjacent_dot_products() -> None:
    q2 = np.asarray([0.6, -0.4, 0.6, -0.2])
    q2 /= np.linalg.norm(q2)
    series = make_series(
        times=[0, 100_000_000, 200_000_000],
        quaternions=[Q, -q2, -Q],
    )

    segment = split_continuous_segments(series)[0]
    quaternions = segment.values[:, 12:16]

    assert np.all(np.sum(quaternions[1:] * quaternions[:-1], axis=1) >= 0.0)


def test_angle_unwrap_does_not_cross_segment_boundary() -> None:
    series = make_series(
        times=[0, 100_000_000, 500_000_000, 600_000_000],
        angles=[179.0, -179.0, -179.0, 179.0],
    )

    segments = split_continuous_segments(series)

    assert np.allclose(segments[0].values[:, 6:9], [[179.0] * 3, [181.0] * 3])
    assert np.allclose(segments[1].values[:, 6:9], [[-179.0] * 3, [-181.0] * 3])
