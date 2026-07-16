from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
import pandas as pd

import scripts.preprocess_imu_stage1 as imu


RAW_COLUMNS = [
    "时间", "设备名称", "加速度X(g)", "加速度Y(g)", "加速度Z(g)",
    "角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)",
    "角度X(°)", "角度Y(°)", "角度Z(°)",
    "磁场X(uT)", "磁场Y(uT)", "磁场Z(uT)",
    "四元数0()", "四元数1()", "四元数2()", "四元数3()",
    "温度(°C)", "版本号()", "电量(%)",
]

QC_KEYS = [
    "class_id", "class_name", "user_id", "action_id", "input_directory",
    "input_csv_files", "csv_file_count", "status", "total_input_rows",
    "valid_output_rows", "rejected_rows", "unknown_sensor_rows",
    "present_sensors", "missing_sensors", "rows_per_sensor",
    "duplicate_timestamp_count_per_sensor", "min_relative_time_ms",
    "max_relative_time_ms", "duration_s", "columns_written", "warnings",
    "error_message", "action_start_time", "action_end_time",
    "structural_rejected_rows", "content_rejected_rows", "file_errors",
]
MANIFEST_KEYS = [
    "sample_id", "class_id", "class_name", "user_id", "action_id",
    "relative_action_path", "output_csv", "status", "csv_file_count",
    "total_input_rows", "valid_output_rows", "rejected_rows",
    "unknown_sensor_rows", "present_sensors", "missing_sensors", "ll_rows",
    "rl_rows", "la_rows", "ra_rows", "c_rows", "ll_duplicate_timestamps",
    "rl_duplicate_timestamps", "la_duplicate_timestamps",
    "ra_duplicate_timestamps", "c_duplicate_timestamps", "duration_s",
    "warning_count", "error_message",
]


def write_imu_csv(path, rows: list[tuple[str, str]]) -> None:
    records = []
    for timestamp, device in rows:
        records.append([timestamp, device, *([1.0] * 17), "v1", 80])
    pd.DataFrame(records, columns=RAW_COLUMNS).to_csv(path, index=False, encoding="utf-8")


def test_structural_inspection_rejects_rows_and_preserves_indices(
    tmp_path: Path,
) -> None:
    source = tmp_path / "structural.csv"
    valid_row = ["2025-01-01 00:00:00", "WTLL(device)", *(["1"] * 17), "v1", "80"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_COLUMNS)
        writer.writerow(valid_row)
        writer.writerow([])
        writer.writerow([f" \ufeff{column} " for column in RAW_COLUMNS])
        writer.writerow(valid_row[:-1])
        writer.writerow(valid_row)

    result = imu.read_csv_robust(source)

    assert result.total_input_rows == 5
    assert result.dataframe["source_row_index"].tolist() == [1, 5]
    assert [row.reject_reason for row in result.rejected_rows] == [
        "blank_row",
        "repeated_header",
        "field_count_mismatch",
    ]
    assert [row.source_row_index for row in result.rejected_rows] == [2, 3, 4]


def test_multiline_record_tracks_first_physical_line(tmp_path: Path) -> None:
    source = tmp_path / "multiline.csv"
    multiline_row = ["2025-01-01\n00:00:00", "WTLL(device)", *(["1"] * 17), "v1", "80"]
    following_row = ["2025-01-02", "WTLL(device)", *(["2"] * 17), "v1", "80"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_COLUMNS)
        writer.writerow(multiline_row)
        writer.writerow(following_row)

    result = imu.read_csv_robust(source)

    assert result.dataframe["source_line_number"].tolist() == [2, 4]


def test_cr_only_utf8_sig_tracks_physical_record_start_lines(tmp_path: Path) -> None:
    source = tmp_path / "cr-only.csv"
    multiline_row = [
        "2025-01-01\r00:00:00",
        "WTLL(device)",
        *(["1"] * 17),
        "v1",
        "80",
    ]
    following_row = ["2025-01-02", "WTLL(device)", *(["2"] * 17), "v1", "80"]
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\r")
    writer.writerow(RAW_COLUMNS)
    writer.writerow(multiline_row)
    writer.writerow(following_row)
    source.write_bytes(text.getvalue().encode("utf-8-sig"))

    result = imu.read_csv_robust(source)

    assert result.file_errors == []
    assert result.total_input_rows == 2
    assert result.dataframe["source_row_index"].tolist() == [1, 2]
    assert result.dataframe["source_line_number"].tolist() == [2, 4]


def test_unclosed_quoted_record_is_fatal_csv_syntax_error(tmp_path: Path) -> None:
    source = tmp_path / "unclosed.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_COLUMNS)
        handle.write('"unclosed record')

    result = imu.read_csv_robust(source)

    assert len(result.file_errors) == 1
    assert result.file_errors[0].error_type == "csv_syntax_error"
    assert result.file_errors[0].source_line_number is not None


@pytest.mark.parametrize(
    "duplicate_column",
    [RAW_COLUMNS[2], RAW_COLUMNS[11].replace("uT", "μT")],
    ids=["exact", "unicode_variant"],
)
def test_header_collisions_are_fatal_schema_errors(
    tmp_path: Path, duplicate_column: str
) -> None:
    source = tmp_path / "collision.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([*RAW_COLUMNS, duplicate_column])

    result = imu.read_csv_robust(source)

    assert [error.error_type for error in result.file_errors] == [
        "duplicate_normalized_columns"
    ]
    assert result.file_errors[0].source_line_number == 1
    assert result.dataframe.empty
    assert result.total_input_rows == 0


def test_invalid_utf8_in_header_is_fatal_without_leaking(tmp_path: Path) -> None:
    source = tmp_path / "invalid-header.csv"
    source.write_bytes(b"\xff,invalid\n")

    result = imu.read_csv_robust(source)

    assert [error.error_type for error in result.file_errors] == [
        "utf8_decode_error"
    ]
    assert result.file_errors[0].source_line_number == 1
    assert result.dataframe.empty
    assert result.total_input_rows == 0
    assert result.rejected_rows == []


def test_invalid_utf8_in_data_preserves_completed_record_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-data.csv"
    valid_row = ["2025-01-01", "WTLL(device)", *(["1"] * 17), "v1", "80"]
    valid_prefix = (
        ",".join(RAW_COLUMNS) + "\n" + ",".join(valid_row) + "\n\n"
    ).encode("utf-8")
    source.write_bytes(valid_prefix + b"\xff,invalid\n")

    result = imu.read_csv_robust(source)

    assert [error.error_type for error in result.file_errors] == [
        "utf8_decode_error"
    ]
    assert result.file_errors[0].source_line_number == 4
    assert result.dataframe["source_row_index"].tolist() == [1]
    assert [row.source_row_index for row in result.rejected_rows] == [2]
    assert [row.reject_reason for row in result.rejected_rows] == ["blank_row"]
    assert result.total_input_rows == 2


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("WTLL(DC:E1:B0:1F:67:6E)", "LL"),
        ("wtrl(E5:9E:9B:1F:CE:48)", "RL"),
        (" WTLA(device-3) ", "LA"),
        ("WTRA(04)", "RA"),
        ("WTC(C4:80:F9:46:D1:94)", "C"),
        ("unknown", None),
    ],
)
def test_device_names_map_to_sensor_positions(
    device_name: str, expected: str | None
) -> None:
    assert imu.parse_sensor_position(device_name) == expected


def test_content_validation_combines_reasons_in_deterministic_order() -> None:
    records = [
        [
            "not-a-time",
            "mystery-device",
            "bad-acc-x",
            *(["1"] * 15),
            "20",
            "v1",
            "80",
        ],
        ["2025-01-01", "also-unknown", *(["1"] * 16), "20", "v1", "80"],
    ]
    frame = pd.DataFrame(records, columns=RAW_COLUMNS)
    frame["source_file"] = "content.csv"
    frame["source_line_number"] = [2, 3]
    frame["source_row_index"] = [1, 2]
    read_result = imu.CsvReadResult(
        source_file="content.csv",
        dataframe=frame,
        total_input_rows=2,
    )

    result = imu.validate_dataframe(read_result)

    assert result.dataframe.empty
    assert result.unknown_sensor_rows == 2
    assert [row.reject_stage for row in result.rejected_rows] == [
        "content",
        "content",
    ]
    assert [row.reject_reason for row in result.rejected_rows] == [
        "invalid_time;unknown_sensor;non_numeric_acc_x_g",
        "unknown_sensor",
    ]
    assert json.loads(result.rejected_rows[0].raw_row) == records[0]


def test_invalid_optional_metadata_warns_without_rejecting_valid_row() -> None:
    record = [
        "2025-01-01 00:00:00",
        "WTLL(device)",
        *(["1"] * 16),
        "not-a-temperature",
        "v1",
        "not-a-battery",
    ]
    frame = pd.DataFrame([record], columns=RAW_COLUMNS)
    frame["source_file"] = "metadata.csv"
    frame["source_line_number"] = [2]
    frame["source_row_index"] = [1]
    read_result = imu.CsvReadResult(
        source_file="metadata.csv",
        dataframe=frame,
        warnings=["upstream warning"],
        total_input_rows=1,
    )

    result = imu.validate_dataframe(read_result)

    assert len(result.dataframe) == 1
    assert result.rejected_rows == []
    assert result.warnings == [
        "upstream warning",
        f"Invalid optional metadata values in {RAW_COLUMNS[18]}: 1",
        f"Invalid optional metadata values in {RAW_COLUMNS[20]}: 1",
    ]
    assert not set(RAW_COLUMNS[18:]).intersection(result.dataframe.columns)


def test_action_discovery_sorts_class_ids_as_integers(tmp_path: Path) -> None:
    input_root = tmp_path / "IMU"
    for class_name in ("10_Ten", "2_Two", "1_One"):
        action = input_root / class_name / "user1" / "1-1-1"
        action.mkdir(parents=True)
        (action / "data.csv").write_text("时间,设备名称\n", encoding="utf-8")

    actions = imu.discover_action_directories(input_root)

    assert [action.class_id for action in actions] == [1, 2, 10]


def test_all_sensors_share_the_actions_earliest_time_zero(tmp_path) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    write_imu_csv(
        action_dir / "part.csv",
        [
            ("2025-01-01 00:00:00.250", "WTRL(device)"),
            ("2025-01-01 00:00:00.100", "WTLL(device)"),
        ],
    )

    result = imu.process_action_directory(
        action_dir, input_root, tmp_path / "new_IMU"
    )

    output = result.merged
    assert output["relative_time_ms"].tolist() == [0, 150]
    assert output["sensor_position"].tolist() == ["LL", "RL"]
    assert not (tmp_path / "new_IMU").exists()


def test_exact_submillisecond_time_controls_sort_and_duplicates(tmp_path: Path) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    write_imu_csv(
        action_dir / "part.csv",
        [
            ("2025-01-01 00:00:00.1004", "WTLL(device)"),
            ("2025-01-01 00:00:00.1002", "WTLL(device)"),
            ("2025-01-01 00:00:00.1002", "WTLL(device)"),
        ],
    )

    result = imu.process_action_directory(action_dir, input_root, tmp_path / "out")

    assert result.merged["source_row_index"].tolist() == [2, 3, 1]
    assert result.merged["relative_time_ms"].tolist() == [0, 0, 0]
    assert result.qc["duplicate_timestamp_count_per_sensor"]["LL"] == 1


def test_equal_cross_sensor_timestamps_are_not_duplicates(tmp_path: Path) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    write_imu_csv(
        action_dir / "part.csv",
        [
            ("2025-01-01 00:00:00.1002", "WTRL(device)"),
            ("2025-01-01 00:00:00.1002", "WTLL(device)"),
        ],
    )

    result = imu.process_action_directory(action_dir, input_root, tmp_path / "out")

    assert result.merged["sensor_position"].tolist() == ["LL", "RL"]
    counts = result.qc["duplicate_timestamp_count_per_sensor"]
    assert counts["LL"] == 0
    assert counts["RL"] == 0


def test_incomplete_sensors_and_output_schema(tmp_path: Path) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    write_imu_csv(
        action_dir / "part.csv",
        [
            ("2025-01-01 00:00:00.100", "WTLL(device)"),
            ("2025-01-01 00:00:00.200", "WTRL(device)"),
        ],
    )

    result = imu.process_action_directory(action_dir, input_root, tmp_path / "out")

    assert result.status == "incomplete_sensors"
    assert result.qc["missing_sensors"] == ["LA", "RA", "C"]
    assert result.merged.columns.tolist() == imu.OUTPUT_COLUMNS


def test_qc_and_manifest_have_complete_ordered_typed_metadata(tmp_path: Path) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    write_imu_csv(
        action_dir / "part.csv",
        [
            ("2025-01-01 00:00:00.100", "WTLL(device)"),
            ("2025-01-01 00:00:00.100", "WTLL(device)"),
            ("2025-01-01 00:00:00.200", "WTRL(device)"),
            ("2025-01-01 00:00:00.300", "WTLA(device)"),
            ("2025-01-01 00:00:00.400", "WTRA(device)"),
            ("2025-01-01 00:00:00.500", "WTC(device)"),
        ],
    )

    result = imu.process_action_directory(action_dir, input_root, tmp_path / "out")

    assert list(result.qc) == QC_KEYS
    assert list(result.manifest_row) == MANIFEST_KEYS
    assert result.status == "success_with_warnings"
    assert result.qc["input_directory"] == str(action_dir.resolve())
    assert result.qc["input_csv_files"] == ["part.csv"]
    assert result.qc["rows_per_sensor"] == {
        "LL": 2, "RL": 1, "LA": 1, "RA": 1, "C": 1
    }
    assert result.qc["duplicate_timestamp_count_per_sensor"] == {
        "LL": 1, "RL": 0, "LA": 0, "RA": 0, "C": 0
    }
    assert result.qc["min_relative_time_ms"] == 0
    assert result.qc["max_relative_time_ms"] == 400
    assert result.qc["duration_s"] == pytest.approx(0.4)
    assert result.qc["action_start_time"] == "2025-01-01T00:00:00.100000"
    assert result.qc["action_end_time"] == "2025-01-01T00:00:00.500000"
    assert result.qc["columns_written"] == imu.OUTPUT_COLUMNS
    assert result.manifest_row["sample_id"] == "0__user1__1-1-1"
    assert result.manifest_row["relative_action_path"] == "0_Wash_face/user1/1-1-1"
    assert result.manifest_row["output_csv"] == (
        "0_Wash_face/user1/1-1-1/imu_merged.csv"
    )
    assert isinstance(result.manifest_row["warning_count"], int)


def test_fatal_error_in_any_direct_csv_fails_whole_action(tmp_path: Path) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    write_imu_csv(
        action_dir / "good.csv",
        [("2025-01-01 00:00:00.100", "WTLL(device)")],
    )
    (action_dir / "bad.csv").write_text(
        f"{RAW_COLUMNS[0]},{RAW_COLUMNS[1]}\n", encoding="utf-8"
    )

    result = imu.process_action_directory(action_dir, input_root, tmp_path / "out")

    assert result.status == "failed"
    assert result.merged.empty
    assert result.merged.columns.tolist() == imu.OUTPUT_COLUMNS
    assert result.qc["valid_output_rows"] == 0
    assert result.qc["file_errors"][0]["source_file"] == "bad.csv"
    assert result.qc["file_errors"][0]["error_type"] == "missing_required_columns"
    assert result.manifest_row["output_csv"] == ""


def test_action_processing_preserves_input_bytes_paths_and_entries(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    write_imu_csv(
        action_dir / "part-2.csv",
        [("2025-01-01 00:00:00.200", "WTRL(device)")],
    )
    write_imu_csv(
        action_dir / "part-1.csv",
        [("2025-01-01 00:00:00.100", "WTLL(device)")],
    )

    def snapshot() -> tuple[list[str], dict[str, str]]:
        paths = sorted(
            path.relative_to(input_root).as_posix()
            for path in input_root.rglob("*")
        )
        hashes = {
            path.relative_to(input_root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in input_root.rglob("*.csv")
        }
        return paths, hashes

    before = snapshot()
    result = imu.process_action_directory(action_dir, input_root, tmp_path / "out")
    after = snapshot()

    assert result.merged["source_file"].tolist() == ["part-1.csv", "part-2.csv"]
    assert after == before
    assert not (tmp_path / "out").exists()
