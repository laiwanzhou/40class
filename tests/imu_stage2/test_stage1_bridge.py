from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.preprocess_imu_stage1 as stage1
from src.data.imu_stage1_bridge import (
    decimal_seconds_to_ns,
    discover_stage1_artifacts,
    load_stage1_action,
    process_raw_imu_source,
    stage1_manifest_row_sha256,
)
from src.data.imu_stage2_contracts import FEATURE_ORDER, ImuActionSource, sha256_file


def make_manifest_row(*, output_csv: str) -> dict[str, str]:
    row = {column: "" for column in stage1.MANIFEST_COLUMNS}
    row.update(
        {
            "sample_id": "7__user1__1-1-1",
            "class_id": "007",
            "class_name": "Class_seven",
            "user_id": "user1",
            "action_id": "1-1-1",
            "relative_action_path": "7_Class_seven/user1/1-1-1",
            "output_csv": output_csv,
            "status": "incomplete_sensors",
            "csv_file_count": "2",
            "total_input_rows": "2",
            "valid_output_rows": "2",
            "rejected_rows": "0",
            "unknown_sensor_rows": "0",
            "present_sensors": "LL;C",
            "missing_sensors": "RL;LA;RA",
            "ll_rows": "1",
            "c_rows": "1",
            "duration_s": "0.091",
            "warning_count": "0",
        }
    )
    return row


def write_manifest(root: Path, row: dict[str, object]) -> None:
    with (root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stage1.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def write_artifact(
    root: Path,
    *,
    manifest_row: dict[str, str] | None = None,
    merged: pd.DataFrame | None = None,
    qc: dict[str, object] | None = None,
) -> tuple[Path, dict[str, str]]:
    relative = Path("7_Class_seven") / "user1" / "1-1-1"
    action_directory = root / relative
    action_directory.mkdir(parents=True)
    output_relpath = (relative / "imu_merged.csv").as_posix()
    row = manifest_row or make_manifest_row(output_csv=output_relpath)
    write_manifest(root, row)

    if merged is None:
        records: list[dict[str, object]] = []
        for relative_time_s, relative_time_ms, sensor, source_file, source_row, base in (
            ("0.000000001", 0, "LL", "part10.csv", 7, 10.0),
            ("0.091", 91, "C", "part2.csv", 3, 20.0),
        ):
            record: dict[str, object] = {
                "relative_time_s": relative_time_s,
                "relative_time_ms": relative_time_ms,
                "sensor_position": sensor,
                "source_file": source_file,
                "source_row_index": source_row,
            }
            record.update(
                {feature: base + index for index, feature in enumerate(FEATURE_ORDER)}
            )
            records.append(record)
        merged = pd.DataFrame(records, columns=stage1.OUTPUT_COLUMNS)
    merged.to_csv(action_directory / "imu_merged.csv", index=False, encoding="utf-8-sig")

    payload = qc or {
        "status": row["status"],
        "input_csv_files": ["part2.csv", "part10.csv"],
        "present_sensors": ["LL", "C"],
        "missing_sensors": ["RL", "LA", "RA"],
    }
    (action_directory / "qc.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return action_directory, row


def write_raw_csv(path: Path, timestamp: str, device: str, base: float) -> None:
    row = [timestamp, device, *[base + index for index in range(16)]]
    pd.DataFrame([row], columns=stage1.REQUIRED_SOURCE_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0.091", 91_000_000), ("0.000001", 1_000), ("0.000000001", 1)],
)
def test_decimal_seconds_to_ns_is_exact(text: str, expected: int) -> None:
    actual = decimal_seconds_to_ns(text)

    assert isinstance(actual, np.int64)
    assert actual == expected


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-0.001"])
def test_decimal_seconds_to_ns_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ValueError, match="Invalid relative time"):
        decimal_seconds_to_ns(text)


def test_decimal_seconds_to_ns_rejects_subnanosecond() -> None:
    with pytest.raises(ValueError, match="represented exactly"):
        decimal_seconds_to_ns("0.0000000001")


def test_decimal_seconds_to_ns_rejects_int64_overflow() -> None:
    with pytest.raises(OverflowError, match="outside int64 range"):
        decimal_seconds_to_ns("9223372036.854775808")


def test_discovery_preserves_manifest_text_and_hashes_every_column(
    tmp_path: Path,
) -> None:
    root = tmp_path / "new_IMU"
    root.mkdir()
    action_directory, original_row = write_artifact(root)

    descriptors = discover_stage1_artifacts(root)

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor.sample_id == "7__user1__1-1-1"
    assert descriptor.manifest_row["class_id"] == "007"
    assert list(descriptor.manifest_row) == stage1.MANIFEST_COLUMNS
    assert descriptor.output_csv_path == action_directory / "imu_merged.csv"
    assert descriptor.qc_path == action_directory / "qc.json"
    assert descriptor.manifest_row_sha256 == stage1_manifest_row_sha256(
        original_row
    )

    for column in stage1.MANIFEST_COLUMNS:
        changed = dict(original_row)
        changed[column] += "changed"
        assert stage1_manifest_row_sha256(changed) != descriptor.manifest_row_sha256


def test_discovery_rejects_manifest_column_drift(tmp_path: Path) -> None:
    root = tmp_path / "new_IMU"
    root.mkdir()
    with (root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*reversed(stage1.MANIFEST_COLUMNS)])

    with pytest.raises(ValueError, match="Stage 1 manifest columns"):
        discover_stage1_artifacts(root)


def test_discovery_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    root = tmp_path / "new_IMU"
    root.mkdir()
    _, row = write_artifact(root)
    with (root / "manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=stage1.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)

    with pytest.raises(ValueError, match="Duplicate Stage 1 sample_id"):
        discover_stage1_artifacts(root)


def test_offline_loader_uses_exact_text_fixed_features_and_file_ranks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "new_IMU"
    root.mkdir()
    action_directory, _ = write_artifact(root)
    descriptor = discover_stage1_artifacts(root)[0]

    data = load_stage1_action(descriptor)

    assert data.relative_time_ns.dtype == np.int64
    assert data.relative_time_ns.tolist() == [1, 91_000_000]
    assert data.dataframe.columns.tolist() == [
        "sensor_position",
        *FEATURE_ORDER,
        "source_file",
        "source_row_index",
        "_source_file_rank",
        "_stage1_row_index",
    ]
    assert data.dataframe["_source_file_rank"].tolist() == [1, 0]
    assert data.dataframe["_stage1_row_index"].tolist() == [0, 1]
    assert data.sensor_mask.tolist() == [True, False, False, False, True]
    assert data.class_id == 7
    assert data.class_name == "Class_seven"
    assert data.source_metadata["stage1_output_csv_sha256"] == sha256_file(
        action_directory / "imu_merged.csv"
    )
    assert data.source_metadata["stage1_qc_sha256"] == sha256_file(
        action_directory / "qc.json"
    )
    assert data.source_metadata["stage1_manifest_row_sha256"] == (
        descriptor.manifest_row_sha256
    )


def test_raw_bridge_uses_exact_absolute_deltas_and_writes_nothing(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "SM_test_0001" / "IMU"
    input_directory.mkdir(parents=True)
    part2 = input_directory / "part2.csv"
    part10 = input_directory / "part10.csv"
    write_raw_csv(part2, "2025-01-01 00:00:00.091000000", "WTC(device)", 20.0)
    write_raw_csv(part10, "2025-01-01 00:00:00.000000001", "WTLL(device)", 10.0)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    source = ImuActionSource(
        sample_id="SM_test_0001",
        input_directory=input_directory,
        input_csv_files=(part2, part10),
        source_relative_path=Path("SM_test_0001/IMU"),
    )

    data = process_raw_imu_source(source)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert data.relative_time_ns.tolist() == [0, 90_999_999]
    assert data.dataframe["sensor_position"].tolist() == ["LL", "C"]
    assert data.dataframe["_source_file_rank"].tolist() == [1, 0]
    assert data.dataframe["_stage1_row_index"].tolist() == [0, 1]
    assert data.sensor_mask.tolist() == [True, False, False, False, True]
    assert data.class_id is None
    assert data.qc["status"] == "incomplete_sensors"


def test_offline_and_raw_bridges_produce_identical_numerical_inputs(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw" / "7_Class_seven" / "user1" / "1-1-1"
    raw_directory.mkdir(parents=True)
    part2 = raw_directory / "part2.csv"
    part10 = raw_directory / "part10.csv"
    write_raw_csv(part2, "2025-01-01 00:00:00.091000000", "WTC(device)", 20.0)
    write_raw_csv(part10, "2025-01-01 00:00:00.000000001", "WTLL(device)", 10.0)
    action_descriptor = stage1.ActionDescriptor(
        class_id=7,
        class_name="Class_seven",
        user_id="user1",
        action_id="1-1-1",
        input_directory=raw_directory,
        relative_action_path=Path("7_Class_seven/user1/1-1-1"),
        input_csv_files=(part2, part10),
    )
    legacy = stage1.process_action(action_descriptor)
    artifact_root = tmp_path / "new_IMU"
    artifact_root.mkdir()
    action_directory = artifact_root / action_descriptor.relative_action_path
    action_directory.mkdir(parents=True)
    legacy.merged.to_csv(
        action_directory / "imu_merged.csv", index=False, encoding="utf-8-sig"
    )
    (action_directory / "qc.json").write_text(
        json.dumps(legacy.qc, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    write_manifest(artifact_root, legacy.manifest_row)
    offline = load_stage1_action(discover_stage1_artifacts(artifact_root)[0])
    raw = process_raw_imu_source(
        ImuActionSource(
            sample_id=legacy.manifest_row["sample_id"],
            input_directory=raw_directory,
            input_csv_files=(part2, part10),
            source_relative_path=action_descriptor.relative_action_path,
            class_id=7,
            class_name="Class_seven",
            user_id="user1",
            action_id="1-1-1",
        )
    )

    assert np.array_equal(offline.relative_time_ns, raw.relative_time_ns)
    assert np.array_equal(offline.sensor_mask, raw.sensor_mask)
    pd.testing.assert_frame_equal(offline.dataframe, raw.dataframe)
