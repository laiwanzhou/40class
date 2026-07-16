from __future__ import annotations

import csv
import io
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

    imu.process_action_directory(action_dir, input_root, tmp_path / "new_IMU")

    output = pd.read_csv(
        tmp_path / "new_IMU" / "0_Wash_face" / "user1" / "1-1-1" / "imu_merged.csv"
    )
    assert output["relative_time_ms"].tolist() == [0, 150]
    assert output["sensor_position"].tolist() == ["LL", "RL"]
