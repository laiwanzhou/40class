from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
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


def test_in_memory_core_never_validates_a_fatal_read_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    bad_csv = action_dir / "bad.csv"
    bad_csv.write_text(f"{RAW_COLUMNS[0]},{RAW_COLUMNS[1]}\n", encoding="utf-8")
    descriptor = imu.ActionDescriptor(
        class_id=0,
        class_name="Wash_face",
        user_id="user1",
        action_id="1-1-1",
        input_directory=action_dir,
        relative_action_path=Path("0_Wash_face/user1/1-1-1"),
        input_csv_files=(bad_csv,),
    )
    original_validate = imu.validate_dataframe

    def guarded_validate(result: imu.CsvReadResult) -> imu.ValidatedCsvResult:
        assert not result.file_errors
        return original_validate(result)

    monkeypatch.setattr(imu, "validate_dataframe", guarded_validate)

    memory = imu.process_action_in_memory(descriptor)
    legacy = imu.build_legacy_action_result(memory)

    assert memory.validated_results == (None,)
    assert [error.error_type for error in memory.file_errors] == [
        "missing_required_columns"
    ]
    assert legacy.status == "failed"
    assert legacy.manifest_row["output_csv"] == ""


def test_in_memory_core_round_trips_through_legacy_adapter(tmp_path: Path) -> None:
    input_root = tmp_path / "IMU"
    action_dir = input_root / "0_Wash_face" / "user1" / "1-1-1"
    action_dir.mkdir(parents=True)
    part2 = action_dir / "part2.csv"
    part10 = action_dir / "part10.csv"
    write_imu_csv(part2, [("2025-01-01 00:00:00.200", "WTRL(device)")])
    write_imu_csv(part10, [("2025-01-01 00:00:00.100", "WTLL(device)")])
    descriptor = imu.ActionDescriptor(
        class_id=0,
        class_name="Wash_face",
        user_id="user1",
        action_id="1-1-1",
        input_directory=action_dir,
        relative_action_path=Path("0_Wash_face/user1/1-1-1"),
        input_csv_files=(part2, part10),
    )

    memory = imu.process_action_in_memory(descriptor)
    adapted = imu.build_legacy_action_result(memory)
    public = imu.process_action(descriptor)

    assert memory.exact_rows["absolute_time"].dtype == "datetime64[ns]"
    assert memory.exact_rows["_source_file_rank"].tolist() == [0, 1]
    pd.testing.assert_frame_equal(adapted.merged, public.merged)
    pd.testing.assert_frame_equal(adapted.rejected, public.rejected)
    assert adapted.status == public.status
    assert adapted.qc == public.qc
    assert adapted.manifest_row == public.manifest_row


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


@pytest.mark.parametrize("relationship", ["equal", "output_below", "input_below"])
def test_cli_rejects_overlapping_roots_before_any_write(
    tmp_path: Path, relationship: str
) -> None:
    if relationship == "equal":
        input_root = output_root = tmp_path / "shared"
        input_root.mkdir()
    elif relationship == "output_below":
        input_root = tmp_path / "input"
        input_root.mkdir()
        output_root = input_root / "generated"
    else:
        output_root = tmp_path / "outer"
        input_root = output_root / "input"
        input_root.mkdir(parents=True)
    sentinel = input_root / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    before = {
        path.relative_to(tmp_path).as_posix(): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in tmp_path.rglob("*")
    }

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    after = {
        path.relative_to(tmp_path).as_posix(): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in tmp_path.rglob("*")
    }
    assert exit_code == 2
    assert after == before
    assert not (output_root / "processing.log").exists()
    assert not (output_root / "manifest.csv").exists()
    assert not any(".staging-" in path.name for path in tmp_path.rglob("*"))
    assert not any(".backup-" in path.name for path in tmp_path.rglob("*"))


def _write_action(
    input_root: Path,
    class_name: str,
    action_id: str,
    sensors: tuple[str, ...] = ("WTLL", "WTRL", "WTLA", "WTRA", "WTC"),
) -> Path:
    action = input_root / class_name / "user1" / action_id
    action.mkdir(parents=True)
    write_imu_csv(
        action / "part.csv",
        [
            (f"2025-01-01 00:00:00.{index + 1:03d}", f"{sensor}(device)")
            for index, sensor in enumerate(sensors)
        ],
    )
    return action


def test_cli_dry_run_reports_chinese_summary_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "1_One", "1-1-1")

    exit_code = imu.main(
        [
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--dry-run",
        ]
    )

    summary = capsys.readouterr().err
    assert exit_code == 0
    assert not output_root.exists()
    for label in (
        "类别数", "用户数", "动作总数", "成功数", "成功但有警告数",
        "传感器不完整数", "已跳过现有输出数", "失败数", "输入总行数",
        "有效输出行数", "拒绝行数", "LL 缺失动作数", "RL 缺失动作数",
        "LA 缺失动作数", "RA 缺失动作数", "C 缺失动作数",
    ):
        assert label in summary


def test_cli_dry_run_scans_after_failure_returns_one_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    failed = _write_action(input_root, "1_One", "1-1-1")
    (failed / "part.csv").write_text(
        f"{RAW_COLUMNS[0]},{RAW_COLUMNS[1]}\n", encoding="utf-8"
    )
    _write_action(input_root, "2_Two", "1-1-1")

    exit_code = imu.main(
        [
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--dry-run",
        ]
    )

    summary = capsys.readouterr().err
    assert exit_code == 1
    assert "动作总数: 2" in summary
    assert "失败数: 1" in summary
    assert not output_root.exists()


def test_cli_normal_run_writes_sorted_complete_manifest_log_qc_and_skips(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "10_Ten", "1-1-1")
    incomplete = _write_action(
        input_root, "2_Two", "1-1-2", sensors=("WTLL", "WTRL")
    )
    failed = _write_action(input_root, "2_Two", "1-1-3")
    (failed / "part.csv").write_text(
        f"{RAW_COLUMNS[0]},{RAW_COLUMNS[1]}\n", encoding="utf-8"
    )
    skipped = _write_action(input_root, "1_One", "1-1-1")
    prior = imu.process_action_directory(skipped, input_root, output_root)
    assert imu.write_action_result(prior, output_root, overwrite=False).written
    skipped_dir = output_root / "1_One" / "user1" / "1-1-1"
    skipped_before = {
        path.name: path.read_bytes() for path in skipped_dir.iterdir()
    }

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 1
    assert (output_root / "processing.log").exists()
    manifest_path = output_root / "manifest.csv"
    assert manifest_path.read_bytes()[:3] == b"\xef\xbb\xbf"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    assert manifest.columns.tolist() == MANIFEST_KEYS
    assert manifest["class_id"].tolist() == [1, 2, 2, 10]
    assert set(manifest["status"]) == {
        "success", "incomplete_sensors", "failed", "skipped_existing"
    }
    failed_dir = output_root / "2_Two" / "user1" / "1-1-3"
    assert (failed_dir / "qc.json").exists()
    assert not (failed_dir / "imu_merged.csv").exists()
    assert (output_root / incomplete.relative_to(input_root) / "imu_merged.csv").exists()
    assert {
        path.name: path.read_bytes() for path in skipped_dir.iterdir()
    } == skipped_before
    assert "two directory renames" in (output_root / "processing.log").read_text(
        encoding="utf-8"
    )


def test_build_manifest_natural_sorts_users_and_actions(tmp_path: Path) -> None:
    results = []
    for class_id, user_id, action_id in (
        (10, "user2", "1-1-10"),
        (2, "user10", "1-1-2"),
        (2, "user2", "1-1-10"),
        (2, "user2", "1-1-2"),
    ):
        result = make_action_result(tmp_path)
        result.descriptor = imu.ActionDescriptor(
            class_id=class_id,
            class_name=str(class_id),
            user_id=user_id,
            action_id=action_id,
            input_directory=tmp_path / "input",
            relative_action_path=Path(str(class_id)) / user_id / action_id,
            input_csv_files=(),
        )
        result.manifest_row = dict.fromkeys(MANIFEST_KEYS, "")
        for column in (
            "class_id", "csv_file_count", "total_input_rows", "valid_output_rows",
            "rejected_rows", "unknown_sensor_rows", "ll_rows", "rl_rows",
            "la_rows", "ra_rows", "c_rows", "ll_duplicate_timestamps",
            "rl_duplicate_timestamps", "la_duplicate_timestamps",
            "ra_duplicate_timestamps", "c_duplicate_timestamps", "warning_count",
        ):
            result.manifest_row[column] = 0
        result.manifest_row["duration_s"] = None
        result.manifest_row.update(
            class_id=class_id, user_id=user_id, action_id=action_id
        )
        results.append(result)

    manifest = imu.build_manifest(results)

    assert list(zip(manifest.class_id, manifest.user_id, manifest.action_id)) == [
        (2, "user2", "1-1-2"),
        (2, "user2", "1-1-10"),
        (2, "user10", "1-1-2"),
        (10, "user2", "1-1-10"),
    ]
    assert manifest.columns.tolist() == MANIFEST_KEYS


def test_cli_global_discovery_error_returns_two_without_action_writes(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    malformed = input_root / "not-an-action"
    malformed.mkdir(parents=True)
    (malformed / "data.csv").write_text("x\n", encoding="utf-8")

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 2
    assert not output_root.exists()


def test_cli_help_lists_exact_four_options() -> None:
    completed = subprocess.run(
        ["python", "scripts/preprocess_imu_stage1.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    for option in ("--input-root", "--output-root", "--overwrite", "--dry-run"):
        assert option in completed.stdout


def test_cli_writer_exception_becomes_failed_qc_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "1_One", "1-1-1")
    _write_action(input_root, "2_Two", "1-1-1")
    real_writer = imu.write_action_result
    calls = 0

    def fail_first_writer_setup(result, root, overwrite):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected writer setup failure")
        return real_writer(result, root, overwrite)

    monkeypatch.setattr(imu, "write_action_result", fail_first_writer_setup)

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 1
    manifest = pd.read_csv(output_root / "manifest.csv", encoding="utf-8-sig")
    assert manifest["status"].tolist() == ["failed", "success"]
    assert "injected writer setup failure" in manifest.loc[0, "error_message"]
    first_output = output_root / "1_One" / "user1" / "1-1-1"
    assert (first_output / "qc.json").exists()
    assert not (first_output / "imu_merged.csv").exists()
    assert (output_root / "2_Two" / "user1" / "1-1-1" / "imu_merged.csv").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_malformed_json_object_qc_skips_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    action = _write_action(input_root, "1_One", "1-1-1")
    prior = imu.process_action_directory(action, input_root, output_root)
    assert imu.write_action_result(prior, output_root, overwrite=False).written
    output_directory = output_root / "1_One" / "user1" / "1-1-1"
    (output_directory / "qc.json").write_text(
        json.dumps(
            {
                "total_input_rows": {"invalid": "mapping"},
                "valid_output_rows": "not-an-integer",
                "present_sensors": 42,
                "missing_sensors": None,
                "rows_per_sensor": [],
                "warnings": None,
            }
        ),
        encoding="utf-8",
    )
    before = {
        path.relative_to(output_directory).as_posix(): path.read_bytes()
        for path in output_directory.rglob("*") if path.is_file()
    }
    argv = ["--input-root", str(input_root), "--output-root", str(output_root)]
    if dry_run:
        argv.append("--dry-run")

    exit_code = imu.main(argv)

    assert exit_code == 0
    assert {
        path.relative_to(output_directory).as_posix(): path.read_bytes()
        for path in output_directory.rglob("*") if path.is_file()
    } == before
    if dry_run:
        assert not (output_root / "processing.log").exists()
        assert not (output_root / "manifest.csv").exists()
        assert "无法读取现有 QC" in capsys.readouterr().err
    else:
        manifest = pd.read_csv(output_root / "manifest.csv", encoding="utf-8-sig")
        assert manifest.loc[0, "status"] == "skipped_existing"
        assert manifest.loc[0, "total_input_rows"] == 0
        assert "无法读取现有 QC" in (output_root / "processing.log").read_text(
            encoding="utf-8"
        )


def test_dry_run_summary_failure_returns_two_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "1_One", "1-1-1")

    def fail_summary(results):
        raise OSError("injected console reporting failure")

    monkeypatch.setattr(imu, "_summarize", fail_summary)

    exit_code = imu.main(
        [
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert not output_root.exists()


def test_skipped_qc_error_message_strips_carriage_return_and_line_feed(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    action = _write_action(input_root, "1_One", "1-1-1")
    prior = imu.process_action_directory(action, input_root, output_root)
    assert imu.write_action_result(prior, output_root, overwrite=False).written
    qc_path = output_root / "1_One" / "user1" / "1-1-1" / "qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["error_message"] = "line one\rline two\nline three\r\nline four"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")

    assert imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    ) == 0

    manifest = pd.read_csv(output_root / "manifest.csv", encoding="utf-8-sig")
    assert manifest.loc[0, "error_message"] == (
        "line one line two line three  line four"
    )
    assert "\r" not in manifest.loc[0, "error_message"]
    assert "\n" not in manifest.loc[0, "error_message"]


def test_cli_normal_clean_run_returns_zero(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "1_One", "1-1-1")

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 0
    manifest = pd.read_csv(output_root / "manifest.csv", encoding="utf-8-sig")
    assert manifest["status"].tolist() == ["success"]


def test_cli_returned_csv_write_failure_gets_failed_qc_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "1_One", "1-1-1")
    _write_action(input_root, "2_Two", "1-1-1")
    real_to_csv = pd.DataFrame.to_csv
    injected = False

    def fail_first_merged_csv(frame, path_or_buf=None, *args, **kwargs):
        nonlocal injected
        if (
            not injected
            and Path(path_or_buf).name == "imu_merged.csv"
            and "1_One" in Path(path_or_buf).parts
        ):
            injected = True
            raise OSError("injected merged CSV serialization failure")
        return real_to_csv(frame, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_first_merged_csv)

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 1
    first_output = output_root / "1_One" / "user1" / "1-1-1"
    failed_qc = json.loads((first_output / "qc.json").read_text(encoding="utf-8"))
    assert failed_qc["status"] == "failed"
    assert "injected merged CSV serialization failure" in failed_qc["error_message"]
    assert not (first_output / "imu_merged.csv").exists()
    manifest = pd.read_csv(output_root / "manifest.csv", encoding="utf-8-sig")
    assert manifest["status"].tolist() == ["failed", "success"]
    assert "injected merged CSV serialization failure" in manifest.loc[0, "error_message"]
    assert (output_root / "2_Two" / "user1" / "1-1-1" / "imu_merged.csv").exists()


def test_cli_qc_only_destination_is_immutable_conflict_and_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    first_action = _write_action(input_root, "1_One", "1-1-1")
    _write_action(input_root, "2_Two", "1-1-1")
    successful = imu.process_action_directory(first_action, input_root, output_root)
    prior_failure = imu._failed_action_result(
        successful.descriptor, OSError("prior action failure")
    )
    assert imu.write_action_result(
        prior_failure, output_root, overwrite=False
    ).written
    first_output = output_root / "1_One" / "user1" / "1-1-1"
    qc_path = first_output / "qc.json"
    old_qc_bytes = qc_path.read_bytes()

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    summary = capsys.readouterr().err
    assert exit_code == 1
    assert qc_path.read_bytes() == old_qc_bytes
    assert {path.name for path in first_output.iterdir()} == {"qc.json"}
    manifest = pd.read_csv(output_root / "manifest.csv", encoding="utf-8-sig")
    assert manifest["status"].tolist() == ["failed", "success"]
    assert pd.isna(manifest.loc[0, "output_csv"])
    assert "Existing action output conflict" in manifest.loc[0, "error_message"]
    assert "失败数: 1" in summary
    assert "成功数: 1" in summary
    assert (output_root / "2_Two" / "user1" / "1-1-1" / "imu_merged.csv").exists()

    overwrite_exit = imu.main(
        [
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--overwrite",
        ]
    )

    assert overwrite_exit == 0
    assert qc_path.read_bytes() != old_qc_bytes
    assert (first_output / "imu_merged.csv").exists()
    overwritten_manifest = pd.read_csv(
        output_root / "manifest.csv", encoding="utf-8-sig"
    )
    assert overwritten_manifest["status"].tolist() == ["success", "success"]


class _TrackingHandler(logging.Handler):
    def __init__(self, *, fail_flush: bool = False, fail_close: bool = False) -> None:
        super().__init__()
        self.fail_flush = fail_flush
        self.fail_close = fail_close
        self.flush_attempts = 0
        self.close_attempts = 0

    def emit(self, record: logging.LogRecord) -> None:
        pass

    def flush(self) -> None:
        self.flush_attempts += 1
        if self.fail_flush:
            self.fail_flush = False
            raise OSError("injected handler flush failure")

    def close(self) -> None:
        self.close_attempts += 1
        if self.fail_close:
            self.fail_close = False
            raise OSError("injected handler close failure")
        super().close()


def test_cli_logging_setup_failure_cleans_created_handlers_and_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "1_One", "1-1-1")
    created = _TrackingHandler()
    monkeypatch.setattr(imu.logging, "StreamHandler", lambda: created)

    def fail_file_handler(*args, **kwargs):
        raise OSError("injected file-handler setup failure")

    monkeypatch.setattr(imu.logging, "FileHandler", fail_file_handler)

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 2
    assert created.close_attempts >= 1
    assert created not in logging.getLogger("preprocess_imu_stage1").handlers


def test_cli_manifest_write_failure_returns_two_and_detaches_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "1_One", "1-1-1")

    real_to_csv = pd.DataFrame.to_csv

    def fail_manifest_write(frame, path_or_buf=None, *args, **kwargs):
        if Path(path_or_buf).name == "manifest.csv":
            raise OSError("injected manifest write failure")
        return real_to_csv(frame, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_manifest_write)

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 2
    assert logging.getLogger("preprocess_imu_stage1").handlers == []


def test_cli_teardown_failures_attempt_every_resource_and_return_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_action(input_root, "1_One", "1-1-1")
    console = _TrackingHandler(fail_flush=True)
    file_handler = _TrackingHandler(fail_close=True)
    monkeypatch.setattr(imu.logging, "StreamHandler", lambda: console)
    monkeypatch.setattr(imu.logging, "FileHandler", lambda *args, **kwargs: file_handler)

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 2
    assert console.flush_attempts >= 1
    assert console.close_attempts >= 1
    assert file_handler.flush_attempts >= 1
    assert file_handler.close_attempts >= 1
    logger = logging.getLogger("preprocess_imu_stage1")
    assert console not in logger.handlers
    assert file_handler not in logger.handlers


@pytest.mark.parametrize(
    "mutate",
    [
        lambda qc: qc.update(total_input_rows=-1),
        lambda qc: qc["rows_per_sensor"].update(LL=-1),
        lambda qc: qc["duplicate_timestamp_count_per_sensor"].update(RL=-1),
        lambda qc: qc.update(duration_s=float("nan")),
        lambda qc: qc.update(duration_s=-0.1),
        lambda qc: qc.update(present_sensors=["RL", "LL"]),
        lambda qc: qc.update(present_sensors=["LL", "LL"]),
        lambda qc: qc.update(
            present_sensors=["LL"], missing_sensors=["LL", "RL", "LA", "RA", "C"]
        ),
    ],
    ids=[
        "negative-count", "negative-sensor-count", "negative-duplicate-count",
        "nonfinite-duration", "negative-duration", "sensor-order",
        "duplicate-sensor", "overlapping-sensors",
    ],
)
def test_semantically_malformed_existing_qc_is_immutable_skipped_fallback(
    tmp_path: Path, mutate
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    action = _write_action(input_root, "1_One", "1-1-1")
    prior = imu.process_action_directory(action, input_root, output_root)
    assert imu.write_action_result(prior, output_root, overwrite=False).written
    output_directory = output_root / "1_One" / "user1" / "1-1-1"
    qc_path = output_directory / "qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    mutate(qc)
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    before = {
        path.relative_to(output_directory).as_posix(): path.read_bytes()
        for path in output_directory.rglob("*") if path.is_file()
    }

    exit_code = imu.main(
        ["--input-root", str(input_root), "--output-root", str(output_root)]
    )

    assert exit_code == 0
    assert {
        path.relative_to(output_directory).as_posix(): path.read_bytes()
        for path in output_directory.rglob("*") if path.is_file()
    } == before
    manifest = pd.read_csv(output_root / "manifest.csv", encoding="utf-8-sig")
    assert manifest.loc[0, "status"] == "skipped_existing"
    assert manifest.loc[0, "total_input_rows"] == 0
    assert pd.isna(manifest.loc[0, "present_sensors"])
    assert pd.isna(manifest.loc[0, "missing_sensors"])
    assert "QC" in (output_root / "processing.log").read_text(encoding="utf-8")


def test_build_manifest_preserves_representative_python_types_and_blanks(
    tmp_path: Path,
) -> None:
    action = _write_action(tmp_path / "input", "1_One", "1-1-1")
    result = imu.process_action_directory(action, tmp_path / "input", tmp_path / "out")
    failed = imu._failed_action_result(result.descriptor, OSError("failed"))

    manifest = imu.build_manifest([result, failed])

    assert pd.api.types.is_integer_dtype(manifest["class_id"])
    assert pd.api.types.is_integer_dtype(manifest["total_input_rows"])
    assert pd.api.types.is_float_dtype(manifest["duration_s"])
    assert type(manifest.at[0, "status"]) is str
    assert manifest.at[1, "output_csv"] == ""
    assert pd.isna(manifest.at[1, "duration_s"])
    assert math.isfinite(manifest.at[0, "duration_s"])


def test_build_manifest_rejects_nonfinite_duration(tmp_path: Path) -> None:
    action = _write_action(tmp_path / "input", "1_One", "1-1-1")
    result = imu.process_action_directory(action, tmp_path / "input", tmp_path / "out")
    result.manifest_row["duration_s"] = float("inf")

    with pytest.raises(ValueError, match="duration_s"):
        imu.build_manifest([result])
