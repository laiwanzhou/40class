from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
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


def make_action_result(
    tmp_path: Path,
    *,
    status: str = "success",
    rejected: bool = False,
) -> imu.ActionResult:
    relative_path = Path("0_Wash_face") / "user1" / "1-1-1"
    descriptor = imu.ActionDescriptor(
        class_id=0,
        class_name="Wash_face",
        user_id="user1",
        action_id="1-1-1",
        input_directory=tmp_path / "input" / relative_path,
        relative_action_path=relative_path,
        input_csv_files=(),
    )
    merged = pd.DataFrame(
        [[0.0, 0, "LL", *([1.0] * 16), "part.csv", 1]],
        columns=imu.OUTPUT_COLUMNS,
    )
    rejected_frame = pd.DataFrame(
        (
            [["part.csv", 2, 1, "content", "invalid_time", '["bad"]']]
            if rejected
            else []
        ),
        columns=imu.REJECTED_COLUMNS,
    )
    if status == "failed":
        merged = pd.DataFrame(columns=imu.OUTPUT_COLUMNS)
    qc = {
        "status": status,
        "valid_output_rows": len(merged),
        "rejected_rows": len(rejected_frame),
    }
    return imu.ActionResult(
        descriptor=descriptor,
        status=status,
        merged=merged,
        rejected=rejected_frame,
        qc=qc,
        manifest_row={"status": status},
    )


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


@pytest.mark.parametrize("has_rejected_rows", [False, True])
def test_write_action_result_stages_exact_utf8_sig_artifact_set(
    tmp_path: Path, has_rejected_rows: bool
) -> None:
    output_root = tmp_path / "new_IMU"
    result = make_action_result(tmp_path, rejected=has_rejected_rows)

    write_result = imu.write_action_result(result, output_root, overwrite=False)

    output_directory = output_root / result.descriptor.relative_action_path
    expected_names = {"imu_merged.csv", "qc.json"}
    if has_rejected_rows:
        expected_names.add("rejected_rows.csv")
    assert write_result == imu.WriteResult(True, output_directory)
    assert {path.name for path in output_directory.iterdir()} == expected_names
    merged_path = output_directory / "imu_merged.csv"
    assert merged_path.read_bytes()[:3] == b"\xef\xbb\xbf"
    actual_merged = pd.read_csv(merged_path, encoding="utf-8-sig")
    assert actual_merged.columns.tolist() == imu.OUTPUT_COLUMNS
    pd.testing.assert_frame_equal(actual_merged, result.merged)
    assert json.loads((output_directory / "qc.json").read_text("utf-8")) == result.qc
    if has_rejected_rows:
        actual_rejected = pd.read_csv(
            output_directory / "rejected_rows.csv", encoding="utf-8-sig"
        )
        assert actual_rejected.columns.tolist() == imu.REJECTED_COLUMNS
        pd.testing.assert_frame_equal(actual_rejected, result.rejected)
    assert not any(".staging-" in path.name for path in output_root.rglob("*"))
    assert not any(".backup-" in path.name for path in output_root.rglob("*"))


def test_overwrite_success_with_failed_removes_stale_merged_csv(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "new_IMU"
    successful = make_action_result(tmp_path)
    failed = make_action_result(tmp_path, status="failed", rejected=True)
    assert imu.write_action_result(successful, output_root, overwrite=False).written

    write_result = imu.write_action_result(failed, output_root, overwrite=True)

    output_directory = output_root / failed.descriptor.relative_action_path
    assert write_result == imu.WriteResult(True, output_directory)
    assert {path.name for path in output_directory.iterdir()} == {
        "qc.json",
        "rejected_rows.csv",
    }
    assert not (output_directory / "imu_merged.csv").exists()
    assert json.loads((output_directory / "qc.json").read_text("utf-8")) == failed.qc
    assert not any(".staging-" in path.name for path in output_root.rglob("*"))
    assert not any(".backup-" in path.name for path in output_root.rglob("*"))


def test_overwrite_rejected_with_clean_success_removes_stale_rejection_csv(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "new_IMU"
    rejected = make_action_result(tmp_path, rejected=True)
    clean = make_action_result(tmp_path)
    assert imu.write_action_result(rejected, output_root, overwrite=False).written

    write_result = imu.write_action_result(clean, output_root, overwrite=True)

    output_directory = output_root / clean.descriptor.relative_action_path
    assert write_result == imu.WriteResult(True, output_directory)
    assert {path.name for path in output_directory.iterdir()} == {
        "imu_merged.csv",
        "qc.json",
    }
    assert not (output_directory / "rejected_rows.csv").exists()
    actual_merged = pd.read_csv(
        output_directory / "imu_merged.csv", encoding="utf-8-sig"
    )
    pd.testing.assert_frame_equal(actual_merged, clean.merged)
    assert not any(".staging-" in path.name for path in output_root.rglob("*"))
    assert not any(".backup-" in path.name for path in output_root.rglob("*"))


def test_existing_action_skip_is_byte_for_byte_immutable(tmp_path: Path) -> None:
    output_root = tmp_path / "new_IMU"
    original = make_action_result(tmp_path, rejected=True)
    assert imu.write_action_result(original, output_root, overwrite=False).written
    output_directory = output_root / original.descriptor.relative_action_path
    (output_directory / "preserve.bin").write_bytes(b"\x00manual\xff")

    def snapshot() -> dict[str, bytes]:
        return {
            path.relative_to(output_directory).as_posix(): path.read_bytes()
            for path in output_directory.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    replacement = make_action_result(tmp_path)
    replacement.qc["status"] = "changed"

    write_result = imu.write_action_result(
        replacement, output_root, overwrite=False
    )

    assert write_result == imu.WriteResult(False, output_directory)
    assert snapshot() == before
    assert not any(".staging-" in path.name for path in output_root.rglob("*"))
    assert not any(".backup-" in path.name for path in output_root.rglob("*"))


def test_writer_rejects_destination_overlapping_input_action(tmp_path: Path) -> None:
    result = make_action_result(tmp_path)
    result.descriptor.input_directory.mkdir(parents=True)
    sentinel = result.descriptor.input_directory / "source.csv"
    sentinel.write_bytes(b"original input bytes")

    write_result = imu.write_action_result(
        result, tmp_path / "input", overwrite=True
    )

    assert not write_result.written
    assert "overlaps input action" in write_result.error_message
    assert sentinel.read_bytes() == b"original input bytes"
    assert {path.name for path in result.descriptor.input_directory.iterdir()} == {
        "source.csv"
    }


def test_overwrite_install_failure_restores_original_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "new_IMU"
    original = make_action_result(tmp_path, rejected=True)
    assert imu.write_action_result(original, output_root, overwrite=False).written
    output_directory = output_root / original.descriptor.relative_action_path
    before = {
        path.name: path.read_bytes()
        for path in output_directory.iterdir()
        if path.is_file()
    }
    real_rename = Path.rename

    def fail_staging_install(path: Path, target: Path) -> Path:
        if ".staging-" in path.name:
            raise OSError("injected Windows rename failure")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_install)

    write_result = imu.write_action_result(
        make_action_result(tmp_path), output_root, overwrite=True
    )

    assert not write_result.written
    assert "injected Windows rename failure" in write_result.error_message
    assert {
        path.name: path.read_bytes()
        for path in output_directory.iterdir()
        if path.is_file()
    } == before
    assert not any(".staging-" in path.name for path in output_root.rglob("*"))
    assert not any(".backup-" in path.name for path in output_root.rglob("*"))


def test_partial_backup_cleanup_failure_preserves_committed_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "new_IMU"
    original = make_action_result(tmp_path, rejected=True)
    assert imu.write_action_result(original, output_root, overwrite=False).written
    output_directory = output_root / original.descriptor.relative_action_path
    (output_directory / "unmanaged.bin").write_bytes(b"old unmanaged bytes")
    replacement = make_action_result(tmp_path)
    replacement.qc["generation"] = "replacement"
    real_rmtree = shutil.rmtree

    def partially_delete_backup_then_fail(path: Path, *args, **kwargs) -> None:
        candidate = Path(path)
        if ".backup-" in candidate.name:
            (candidate / "unmanaged.bin").unlink()
            raise OSError("injected partial backup cleanup failure")
        real_rmtree(candidate, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", partially_delete_backup_then_fail)

    write_result = imu.write_action_result(
        replacement, output_root, overwrite=True
    )

    assert write_result.written
    assert write_result.output_directory == output_directory
    assert "injected partial backup cleanup failure" in write_result.error_message
    assert {path.name for path in output_directory.iterdir()} == {
        "imu_merged.csv",
        "qc.json",
    }
    assert json.loads((output_directory / "qc.json").read_text("utf-8")) == (
        replacement.qc
    )
    backups = [
        path
        for path in output_directory.parent.iterdir()
        if ".backup-" in path.name
    ]
    assert len(backups) == 1
    assert not (backups[0] / "unmanaged.bin").exists()


def test_writer_rejects_windows_junction_in_mirrored_destination(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "new_IMU"
    result = make_action_result(tmp_path)
    requested = output_root / result.descriptor.relative_action_path
    requested.parent.mkdir(parents=True)
    victim = output_root / "different-action"
    victim.mkdir(parents=True)
    sentinel = victim / "victim.bin"
    sentinel.write_bytes(b"must survive")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(requested), str(victim)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"Windows junction creation unavailable: {completed.stderr}")

    write_result = imu.write_action_result(result, output_root, overwrite=True)

    assert not write_result.written
    assert "reparse" in write_result.error_message.casefold()
    assert write_result.output_directory == requested
    assert sentinel.read_bytes() == b"must survive"
    assert requested.is_junction()


def test_writer_rejects_supported_symlink_in_mirrored_destination(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "new_IMU"
    result = make_action_result(tmp_path)
    requested = output_root / result.descriptor.relative_action_path
    requested.parent.mkdir(parents=True)
    victim = output_root / "different-action"
    victim.mkdir(parents=True)
    sentinel = victim / "victim.bin"
    sentinel.write_bytes(b"must survive")
    try:
        requested.symlink_to(victim, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlink creation unavailable: {error}")

    write_result = imu.write_action_result(result, output_root, overwrite=True)

    assert not write_result.written
    assert "reparse" in write_result.error_message.casefold()
    assert write_result.output_directory == requested
    assert sentinel.read_bytes() == b"must survive"
    assert requested.is_symlink()


@pytest.mark.parametrize("overwrite", [False, True])
def test_writer_rejects_regular_file_at_action_destination(
    tmp_path: Path, overwrite: bool
) -> None:
    output_root = tmp_path / "new_IMU"
    result = make_action_result(tmp_path)
    destination = output_root / result.descriptor.relative_action_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"conflicting regular file")

    write_result = imu.write_action_result(result, output_root, overwrite=overwrite)

    assert not write_result.written
    assert "not a directory" in write_result.error_message
    assert write_result.output_directory == destination
    assert destination.read_bytes() == b"conflicting regular file"


def test_staging_uuid_collision_preserves_preexisting_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "new_IMU"
    result = make_action_result(tmp_path)
    destination = output_root / result.descriptor.relative_action_path
    destination.parent.mkdir(parents=True)
    staging = destination.parent / f".{destination.name}.staging-collision"
    staging.mkdir()
    sentinel = staging / "preexisting.bin"
    sentinel.write_bytes(b"not owned by writer")

    class FixedUuid:
        hex = "collision"

    monkeypatch.setattr(imu.uuid, "uuid4", lambda: FixedUuid())

    write_result = imu.write_action_result(result, output_root, overwrite=False)

    assert not write_result.written
    assert write_result.error_message
    assert sentinel.read_bytes() == b"not owned by writer"
    assert {path.name for path in staging.iterdir()} == {"preexisting.bin"}
    assert not destination.exists()
