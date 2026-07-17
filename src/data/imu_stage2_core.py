from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.imu_stage2_contracts import (
    FEATURE_ORDER,
    SENSOR_ORDER,
    Stage1ActionData,
)


QUATERNION_NORM_EPS = 1e-8
CIRCULAR_RESULTANT_EPS = 1e-8


def _feature_slice(first: str, last: str) -> slice:
    return slice(FEATURE_ORDER.index(first), FEATURE_ORDER.index(last) + 1)


ACCELERATION_AND_GYROSCOPE_SLICE = _feature_slice(
    "acc_x_g", "gyro_z_dps"
)
ANGLE_SLICE = _feature_slice("angle_x_deg", "angle_z_deg")
MAGNETOMETER_SLICE = _feature_slice("mag_x_ut", "mag_z_ut")
QUATERNION_SLICE = _feature_slice("quat_0", "quat_3")


@dataclass(frozen=True)
class ValidatedStage1Records:
    sensor_position: np.ndarray
    time_ns: np.ndarray
    values: np.ndarray
    source_file_rank: np.ndarray
    source_row_index: np.ndarray
    stage1_row_index: np.ndarray
    valid_record: np.ndarray
    nonfinite_feature_record: np.ndarray
    invalid_quaternion_record: np.ndarray
    action_end_ns: np.int64


@dataclass(frozen=True)
class AggregatedSensorSeries:
    sensor_position: str
    time_ns: np.ndarray
    values: np.ndarray
    qc: dict[str, int]

    def __post_init__(self) -> None:
        if self.sensor_position not in SENSOR_ORDER:
            raise ValueError("Unknown sensor position")
        if self.time_ns.dtype != np.int64 or self.time_ns.ndim != 1:
            raise ValueError("time_ns must be a one-dimensional int64 array")
        if self.values.dtype != np.float64:
            raise ValueError("values must have dtype float64")
        if self.values.shape != (len(self.time_ns), len(FEATURE_ORDER)):
            raise ValueError("values must have shape (T, 16)")
        if len(self.time_ns) > 1 and not np.all(np.diff(self.time_ns) > 0):
            raise ValueError("time_ns must be strictly increasing")
        if not np.isfinite(self.values).all():
            raise ValueError("aggregated values must be finite")


def _integer_column(frame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise ValueError(f"Stage 1 data is missing column: {name}")
    values = frame[name].to_numpy()
    if not all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        for value in values
    ):
        raise ValueError(f"Stage 1 column {name} must contain integers")
    return np.asarray(values, dtype=np.int64)


def validate_stage1_records(action: Stage1ActionData) -> ValidatedStage1Records:
    frame = action.dataframe
    if len(frame) == 0:
        raise ValueError("Stage 1 action must contain at least one record")
    if not isinstance(action.relative_time_ns, np.ndarray):
        raise ValueError("relative_time_ns must be a NumPy array")
    if action.relative_time_ns.dtype != np.int64 or action.relative_time_ns.ndim != 1:
        raise ValueError("relative_time_ns must be a one-dimensional int64 array")
    if len(action.relative_time_ns) != len(frame):
        raise ValueError("relative_time_ns length must match Stage 1 records")
    if np.any(action.relative_time_ns < 0):
        raise ValueError("relative_time_ns must be non-negative")

    missing = [name for name in ("sensor_position", *FEATURE_ORDER) if name not in frame]
    if missing:
        raise ValueError(f"Stage 1 data is missing columns: {missing}")
    sensor_position = frame["sensor_position"].astype(str).to_numpy()
    unknown = sorted(set(sensor_position) - set(SENSOR_ORDER))
    if unknown:
        raise ValueError(f"Unknown Stage 1 sensor positions: {unknown}")
    try:
        values = frame.loc[:, FEATURE_ORDER].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 1 features must be numeric") from error
    if values.shape != (len(frame), len(FEATURE_ORDER)):
        raise ValueError("Stage 1 feature matrix must have shape (N, 16)")

    source_file_rank = _integer_column(frame, "_source_file_rank")
    source_row_index = _integer_column(frame, "source_row_index")
    stage1_row_index = _integer_column(frame, "_stage1_row_index")
    nonfinite = ~np.isfinite(values).all(axis=1)
    quaternion_norm = np.linalg.norm(values[:, QUATERNION_SLICE], axis=1)
    invalid_quaternion = ~nonfinite & (
        ~np.isfinite(quaternion_norm) | (quaternion_norm < QUATERNION_NORM_EPS)
    )
    valid = ~(nonfinite | invalid_quaternion)
    return ValidatedStage1Records(
        sensor_position=sensor_position,
        time_ns=action.relative_time_ns.copy(),
        values=values,
        source_file_rank=source_file_rank,
        source_row_index=source_row_index,
        stage1_row_index=stage1_row_index,
        valid_record=valid,
        nonfinite_feature_record=nonfinite,
        invalid_quaternion_record=invalid_quaternion,
        action_end_ns=np.int64(action.relative_time_ns.max()),
    )


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def _aggregate_valid_group(values: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    output = np.empty(len(FEATURE_ORDER), dtype=np.float64)
    output[ACCELERATION_AND_GYROSCOPE_SLICE] = np.mean(
        values[:, ACCELERATION_AND_GYROSCOPE_SLICE], axis=0, dtype=np.float64
    )
    output[MAGNETOMETER_SLICE] = np.mean(
        values[:, MAGNETOMETER_SLICE], axis=0, dtype=np.float64
    )

    radians = np.deg2rad(values[:, ANGLE_SLICE])
    sine_mean = np.mean(np.sin(radians), axis=0, dtype=np.float64)
    cosine_mean = np.mean(np.cos(radians), axis=0, dtype=np.float64)
    resultant = np.hypot(sine_mean, cosine_mean)
    if not np.isfinite(resultant).all() or np.any(resultant < CIRCULAR_RESULTANT_EPS):
        return None, "angle"
    output[ANGLE_SLICE] = _wrap_degrees(
        np.rad2deg(np.arctan2(sine_mean, cosine_mean))
    )

    quaternions = values[:, QUATERNION_SLICE].copy()
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    reference = quaternions[0]
    flip = (quaternions @ reference) < 0.0
    quaternions[flip] *= -1.0
    quaternion_mean = np.mean(quaternions, axis=0, dtype=np.float64)
    quaternion_norm = np.linalg.norm(quaternion_mean)
    if not np.isfinite(quaternion_norm) or quaternion_norm < QUATERNION_NORM_EPS:
        return None, "quaternion"
    output[QUATERNION_SLICE] = quaternion_mean / quaternion_norm
    if not np.isfinite(output).all():
        return None, "quaternion"
    return output, None


def aggregate_sensor_timestamps(
    action: Stage1ActionData,
    sensor_position: str | None = None,
) -> AggregatedSensorSeries:
    records = validate_stage1_records(action)
    present = [sensor for sensor in SENSOR_ORDER if np.any(records.sensor_position == sensor)]
    if sensor_position is None:
        if len(present) != 1:
            raise ValueError("sensor_position is required when an action has multiple sensors")
        sensor_position = present[0]
    if sensor_position not in SENSOR_ORDER:
        raise ValueError(f"Unknown sensor position: {sensor_position}")

    selected = records.sensor_position == sensor_position
    selected_indices = np.flatnonzero(selected)
    order = np.lexsort(
        (
            records.stage1_row_index[selected],
            records.source_row_index[selected],
            records.source_file_rank[selected],
            records.time_ns[selected],
        )
    )
    ordered_indices = selected_indices[order]
    group_sizes: list[int] = []
    output_times: list[np.int64] = []
    output_values: list[np.ndarray] = []
    aggregation_failures = 0
    duplicate_aggregation_failures = 0
    degenerate_angles = 0
    degenerate_quaternions = 0
    duplicate_excluded = 0

    start = 0
    while start < len(ordered_indices):
        time_ns = records.time_ns[ordered_indices[start]]
        stop = start + 1
        while stop < len(ordered_indices) and records.time_ns[ordered_indices[stop]] == time_ns:
            stop += 1
        group = ordered_indices[start:stop]
        group_sizes.append(len(group))
        duplicate_excluded += int(np.count_nonzero(~records.valid_record[group])) if len(group) > 1 else 0
        valid_group = group[records.valid_record[group]]
        if len(valid_group) == 0:
            aggregated, failure = None, None
        else:
            aggregated, failure = _aggregate_valid_group(records.values[valid_group])
        if aggregated is None:
            aggregation_failures += 1
            duplicate_aggregation_failures += int(len(group) > 1)
            degenerate_angles += int(failure == "angle")
            degenerate_quaternions += int(failure == "quaternion")
        else:
            output_times.append(np.int64(time_ns))
            output_values.append(aggregated)
        start = stop

    duplicate_sizes = [size for size in group_sizes if size > 1]
    excluded = int(np.count_nonzero(selected & ~records.valid_record))
    qc = {
        "duplicate_group_count": len(duplicate_sizes),
        "duplicate_extra_record_count": sum(size - 1 for size in duplicate_sizes),
        "duplicate_max_group_size": max(group_sizes, default=0),
        "excluded_record_count": excluded,
        "aggregation_failed_timestamp_count": aggregation_failures,
        "duplicate_excluded_record_count": duplicate_excluded,
        "duplicate_aggregation_failed_timestamp_count": duplicate_aggregation_failures,
        "nonfinite_feature_record_count": int(
            np.count_nonzero(selected & records.nonfinite_feature_record)
        ),
        "invalid_quaternion_record_count": int(
            np.count_nonzero(selected & records.invalid_quaternion_record)
        ),
        "degenerate_angle_group_count": degenerate_angles,
        "degenerate_quaternion_group_count": degenerate_quaternions,
    }
    values_array = (
        np.vstack(output_values).astype(np.float64, copy=False)
        if output_values
        else np.empty((0, len(FEATURE_ORDER)), dtype=np.float64)
    )
    return AggregatedSensorSeries(
        sensor_position=sensor_position,
        time_ns=np.asarray(output_times, dtype=np.int64),
        values=values_array,
        qc=qc,
    )


def _canonicalize_segment_quaternions(values: np.ndarray) -> None:
    quaternions = values[:, QUATERNION_SLICE]
    first = quaternions[0]
    canonical_component = np.flatnonzero(np.abs(first) >= QUATERNION_NORM_EPS)
    if canonical_component.size == 0:
        raise ValueError("Segment starts with a degenerate quaternion")
    if first[canonical_component[0]] < 0.0:
        first *= -1.0
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index], quaternions[index - 1]) < 0.0:
            quaternions[index] *= -1.0


def split_continuous_segments(
    series: AggregatedSensorSeries,
    max_gap_ns: int = 300_000_000,
) -> list[AggregatedSensorSeries]:
    if isinstance(max_gap_ns, (bool, np.bool_)) or not isinstance(
        max_gap_ns, (int, np.integer)
    ):
        raise ValueError("max_gap_ns must be an integer")
    if max_gap_ns < 0:
        raise ValueError("max_gap_ns must be non-negative")
    if len(series.time_ns) == 0:
        return []

    boundaries = np.flatnonzero(np.diff(series.time_ns) > max_gap_ns) + 1
    segments: list[AggregatedSensorSeries] = []
    for indices in np.split(np.arange(len(series.time_ns)), boundaries):
        values = series.values[indices].copy()
        values[:, ANGLE_SLICE] = np.rad2deg(
            np.unwrap(np.deg2rad(values[:, ANGLE_SLICE]), axis=0)
        )
        _canonicalize_segment_quaternions(values)
        segments.append(
            AggregatedSensorSeries(
                sensor_position=series.sensor_position,
                time_ns=series.time_ns[indices].copy(),
                values=values,
                qc=series.qc.copy(),
            )
        )
    return segments
