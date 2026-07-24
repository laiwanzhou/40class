from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.preprocess_imu_stage1 as stage1
from src.data.imu_stage1_bridge import (
    discover_stage1_artifacts,
    load_stage1_action,
    process_raw_imu_source,
)
from src.data.imu_stage2_contracts import FEATURE_ORDER
from src.data.imu_stage2_core import process_stage2_action
from src.inference.imu_stage2_pipeline import (
    adapt_raw_imu_source,
    discover_test_samples,
)


def _feature_row(timestamp: str, base: float) -> list[object]:
    return [timestamp, "WTLL(device)", *[base + index for index in range(16)]]


def _write_raw_fixture(path: Path, rows: list[list[object]]) -> None:
    pd.DataFrame(rows, columns=stage1.REQUIRED_SOURCE_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def _write_stage1_artifact(
    root: Path,
    legacy: stage1.ActionResult,
    sample_id: str,
) -> None:
    relative = Path(sample_id)
    action = root / relative
    action.mkdir(parents=True)
    legacy.merged.to_csv(
        action / "imu_merged.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (action / "qc.json").write_text(
        json.dumps(legacy.qc, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    row = dict(legacy.manifest_row)
    row["sample_id"] = sample_id
    row["relative_action_path"] = relative.as_posix()
    row["output_csv"] = (relative / "imu_merged.csv").as_posix()
    with (root / "manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=stage1.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def test_raw_and_artifact_stage2_replay_are_exact_with_naturally_sorted_duplicates(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw-test"
    imu = raw_root / "SM_test_0001" / "IMU"
    imu.mkdir(parents=True)
    part10 = imu / "part10.csv"
    part2 = imu / "part2.csv"
    _write_raw_fixture(
        part10,
        [_feature_row("2025-01-01 00:00:00.000000000", 10.0)],
    )
    _write_raw_fixture(
        part2,
        [
            _feature_row("2025-01-01 00:00:00.000000000", 20.0),
            _feature_row("2025-01-01 00:00:00.200000000", 30.0),
        ],
    )

    descriptor = discover_test_samples(raw_root).samples[0]
    source = adapt_raw_imu_source(descriptor)
    assert [path.name for path in source.input_csv_files] == [
        "part2.csv",
        "part10.csv",
    ]
    raw_stage1 = process_raw_imu_source(source)
    assert raw_stage1.dataframe.loc[
        raw_stage1.dataframe["source_file"] == "part2.csv", "_source_file_rank"
    ].unique().tolist() == [0]
    assert raw_stage1.dataframe.loc[
        raw_stage1.dataframe["source_file"] == "part10.csv", "_source_file_rank"
    ].unique().tolist() == [1]
    raw_stage2 = process_stage2_action(raw_stage1)

    legacy = stage1.process_action(
        stage1.ActionDescriptor(
            class_id=0,
            class_name="",
            user_id="",
            action_id=descriptor.sample_id,
            input_directory=imu,
            relative_action_path=Path(descriptor.sample_id),
            input_csv_files=source.input_csv_files,
        )
    )
    artifact_root = tmp_path / "new_IMU"
    artifact_root.mkdir()
    _write_stage1_artifact(artifact_root, legacy, descriptor.sample_id)
    artifact_stage1 = load_stage1_action(
        discover_stage1_artifacts(artifact_root)[0]
    )
    artifact_stage2 = process_stage2_action(artifact_stage1)

    assert np.array_equal(
        raw_stage1.relative_time_ns,
        artifact_stage1.relative_time_ns,
    )
    assert np.array_equal(raw_stage2.timestamps_ms, artifact_stage2.timestamps_ms)
    assert np.array_equal(raw_stage2.sensor_mask, artifact_stage2.sensor_mask)
    assert np.array_equal(raw_stage2.valid_mask, artifact_stage2.valid_mask)
    assert np.array_equal(raw_stage2.values, artifact_stage2.values, equal_nan=True)
    assert raw_stage2.values.dtype == artifact_stage2.values.dtype == np.float32
    assert raw_stage2.status is artifact_stage2.status

    exact_qc_fields = (
        "warning_codes",
        "usable_sensors",
        "usable_sensor_mask",
        "grid_length",
        "duplicate_group_count",
        "duplicate_extra_record_count",
        "duplicate_max_group_size",
        "excluded_record_count",
        "aggregation_failed_timestamp_count",
        "exact_hit_count",
        "interpolated_count",
        "invalid_count",
        "per_sensor_valid_count",
    )
    for field in exact_qc_fields:
        assert raw_stage2.qc[field] == artifact_stage2.qc[field]
    assert raw_stage2.qc["duplicate_group_count"] == 1
    assert raw_stage2.qc["duplicate_extra_record_count"] == 1
    assert raw_stage2.qc["interpolated_count"] == 1
    assert tuple(raw_stage1.dataframe.columns) == (
        "sensor_position",
        *FEATURE_ORDER,
        "source_file",
        "source_row_index",
        "_source_file_rank",
        "_stage1_row_index",
    )


def test_raw_and_artifact_replay_are_exact_with_empty_validated_file(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw-test"
    imu = raw_root / "SM_test_0001" / "IMU"
    imu.mkdir(parents=True)
    part2 = imu / "part2.csv"
    part10 = imu / "part10.csv"
    _write_raw_fixture(
        part2,
        [
            _feature_row("2025-01-01 00:00:00.000000000", 10.0),
            _feature_row("2025-01-01 00:00:00.200000000", 20.0),
        ],
    )
    _write_raw_fixture(part10, [])

    descriptor = discover_test_samples(raw_root).samples[0]
    source = adapt_raw_imu_source(descriptor)
    assert [path.name for path in source.input_csv_files] == [
        "part2.csv",
        "part10.csv",
    ]
    raw_stage1 = process_raw_imu_source(source)
    raw_stage2 = process_stage2_action(raw_stage1)

    legacy = stage1.process_action(
        stage1.ActionDescriptor(
            class_id=0,
            class_name="",
            user_id="",
            action_id=descriptor.sample_id,
            input_directory=imu,
            relative_action_path=Path(descriptor.sample_id),
            input_csv_files=source.input_csv_files,
        )
    )
    artifact_root = tmp_path / "new_IMU"
    artifact_root.mkdir()
    _write_stage1_artifact(artifact_root, legacy, descriptor.sample_id)
    artifact_stage1 = load_stage1_action(
        discover_stage1_artifacts(artifact_root)[0]
    )
    artifact_stage2 = process_stage2_action(artifact_stage1)

    assert raw_stage1.dataframe["source_row_index"].dtype == np.dtype(np.int64)
    assert artifact_stage1.dataframe["source_row_index"].dtype == np.dtype(
        np.int64
    )
    assert raw_stage1.dataframe["_source_file_rank"].dtype == np.dtype(np.int64)
    assert artifact_stage1.dataframe["_source_file_rank"].dtype == np.dtype(
        np.int64
    )
    assert raw_stage1.dataframe["_stage1_row_index"].dtype == np.dtype(np.int64)
    assert artifact_stage1.dataframe["_stage1_row_index"].dtype == np.dtype(
        np.int64
    )
    assert raw_stage1.relative_time_ns.dtype == np.dtype(np.int64)
    assert artifact_stage1.relative_time_ns.dtype == np.dtype(np.int64)
    assert np.array_equal(
        raw_stage1.relative_time_ns,
        artifact_stage1.relative_time_ns,
    )
    assert np.array_equal(raw_stage1.sensor_mask, artifact_stage1.sensor_mask)
    pd.testing.assert_frame_equal(
        raw_stage1.dataframe,
        artifact_stage1.dataframe,
        check_exact=True,
        check_dtype=True,
    )
    stage1_qc_fields = (
        "status",
        "input_csv_files",
        "total_input_rows",
        "valid_output_rows",
        "rejected_rows",
        "unknown_sensor_rows",
        "present_sensors",
        "missing_sensors",
    )
    for field in stage1_qc_fields:
        assert raw_stage1.qc[field] == artifact_stage1.qc[field]
    assert np.array_equal(raw_stage2.timestamps_ms, artifact_stage2.timestamps_ms)
    assert np.array_equal(raw_stage2.sensor_mask, artifact_stage2.sensor_mask)
    assert np.array_equal(
        raw_stage2.usable_sensor_mask,
        artifact_stage2.usable_sensor_mask,
    )
    assert np.array_equal(raw_stage2.valid_mask, artifact_stage2.valid_mask)
    raw_nan = np.isnan(raw_stage2.values)
    artifact_nan = np.isnan(artifact_stage2.values)
    assert np.array_equal(raw_nan, artifact_nan)
    assert np.array_equal(
        raw_stage2.values.view(np.uint32)[~raw_nan],
        artifact_stage2.values.view(np.uint32)[~artifact_nan],
    )
    assert raw_stage2.status is artifact_stage2.status

    exact_qc_fields = (
        "warning_codes",
        "usable_sensors",
        "usable_sensor_mask",
        "grid_length",
        "duplicate_group_count",
        "duplicate_extra_record_count",
        "duplicate_max_group_size",
        "excluded_record_count",
        "aggregation_failed_timestamp_count",
        "exact_hit_count",
        "interpolated_count",
        "invalid_count",
        "per_sensor_valid_count",
    )
    for field in exact_qc_fields:
        assert raw_stage2.qc[field] == artifact_stage2.qc[field]
    assert raw_stage1.qc["total_input_rows"] == 2
    assert artifact_stage1.qc["total_input_rows"] == 2
