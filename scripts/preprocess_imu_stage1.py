from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_ROOT = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\IMU"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU"
)


@dataclass(frozen=True)
class ActionDescriptor:
    class_id: int
    class_name: str
    user_id: str
    action_id: str
    input_directory: Path
    relative_action_path: Path
    input_csv_files: tuple[Path, ...]


@dataclass(frozen=True)
class RejectedRow:
    source_file: str
    source_line_number: int | None
    source_row_index: int | None
    reject_stage: str
    reject_reason: str
    raw_row: str


@dataclass(frozen=True)
class FileError:
    source_file: str
    error_type: str
    source_line_number: int | None
    message: str


@dataclass
class CsvReadResult:
    source_file: str
    dataframe: pd.DataFrame
    rejected_rows: list[RejectedRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    file_errors: list[FileError] = field(default_factory=list)
    total_input_rows: int = 0


@dataclass
class ValidatedCsvResult:
    dataframe: pd.DataFrame
    rejected_rows: list[RejectedRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_sensor_rows: int = 0


@dataclass
class ActionResult:
    descriptor: ActionDescriptor
    status: str
    merged: pd.DataFrame
    rejected: pd.DataFrame
    qc: dict[str, Any]
    manifest_row: dict[str, Any]


@dataclass
class InMemoryActionResult:
    descriptor: ActionDescriptor
    validated_results: tuple[ValidatedCsvResult | None, ...]
    read_results: tuple[CsvReadResult, ...]
    exact_rows: pd.DataFrame
    rejected_rows: list[RejectedRow]
    warnings: list[str]
    file_errors: list[FileError]
    total_input_rows: int
    unknown_sensor_rows: int


@dataclass(frozen=True)
class WriteResult:
    written: bool
    output_directory: Path
    error_message: str = ""


def _verified_action_path(output_root: Path, path: Path) -> Path:
    resolved_root = output_root.resolve()
    lexical_path = Path(os.path.abspath(path))
    if lexical_path == resolved_root or not lexical_path.is_relative_to(
        resolved_root
    ):
        raise ValueError(f"Managed path is outside output root: {lexical_path}")
    current = resolved_root
    for component in lexical_path.relative_to(resolved_root).parts:
        current /= component
        if not os.path.lexists(current):
            continue
        attributes = getattr(current.lstat(), "st_file_attributes", 0)
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        is_junction = getattr(current, "is_junction", lambda: False)()
        if (
            (reparse_attribute and attributes & reparse_attribute)
            or current.is_symlink()
            or is_junction
        ):
            raise ValueError(f"Managed path contains reparse point: {current}")
    resolved_path = lexical_path.resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(
        resolved_root
    ):
        raise ValueError(f"Managed path is outside output root: {resolved_path}")
    return lexical_path


def _verified_sibling(
    output_root: Path, destination: Path, candidate: Path
) -> Path:
    resolved_destination = _verified_action_path(output_root, destination)
    resolved_candidate = _verified_action_path(output_root, candidate)
    if resolved_candidate.parent != resolved_destination.parent:
        raise ValueError(
            f"Managed temporary path is not beside destination: {resolved_candidate}"
        )
    return resolved_candidate


def _remove_managed_tree(output_root: Path, path: Path) -> None:
    resolved_path = _verified_action_path(output_root, path)
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


SENSOR_ORDER = {"LL": 0, "RL": 1, "LA": 2, "RA": 3, "C": 4}
MANIFEST_COLUMNS = [
    "sample_id", "class_id", "class_name", "user_id", "action_id",
    "relative_action_path", "output_csv", "status", "csv_file_count",
    "total_input_rows", "valid_output_rows", "rejected_rows",
    "unknown_sensor_rows", "present_sensors", "missing_sensors", "ll_rows",
    "rl_rows", "la_rows", "ra_rows", "c_rows", "ll_duplicate_timestamps",
    "rl_duplicate_timestamps", "la_duplicate_timestamps",
    "ra_duplicate_timestamps", "c_duplicate_timestamps", "duration_s",
    "warning_count", "error_message",
]
QC_FIELDS = [
    "class_id", "class_name", "user_id", "action_id", "input_directory",
    "input_csv_files", "csv_file_count", "status", "total_input_rows",
    "valid_output_rows", "rejected_rows", "unknown_sensor_rows",
    "present_sensors", "missing_sensors", "rows_per_sensor",
    "duplicate_timestamp_count_per_sensor", "min_relative_time_ms",
    "max_relative_time_ms", "duration_s", "columns_written", "warnings",
    "error_message", "action_start_time", "action_end_time",
    "structural_rejected_rows", "content_rejected_rows", "file_errors",
]
OUTPUT_COLUMNS = [
    "relative_time_s",
    "relative_time_ms",
    "sensor_position",
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
    "source_file",
    "source_row_index",
]
REJECTED_COLUMNS = [
    "source_file",
    "source_line_number",
    "source_row_index",
    "reject_stage",
    "reject_reason",
    "raw_row",
]
DEVICE_PREFIX_TO_SENSOR = {
    "WTLL": "LL",
    "WTRL": "RL",
    "WTLA": "LA",
    "WTRA": "RA",
    "WTC": "C",
}

TIME_COLUMN = "时间"
DEVICE_COLUMN = "设备名称"
FEATURE_COLUMNS = (
    "加速度X(g)",
    "加速度Y(g)",
    "加速度Z(g)",
    "角速度X(°/s)",
    "角速度Y(°/s)",
    "角速度Z(°/s)",
    "角度X(°)",
    "角度Y(°)",
    "角度Z(°)",
    "磁场X(uT)",
    "磁场Y(uT)",
    "磁场Z(uT)",
    "四元数0()",
    "四元数1()",
    "四元数2()",
    "四元数3()",
)
OPTIONAL_METADATA_COLUMNS = ("温度(°C)", "版本号()", "电量(%)")
REQUIRED_SOURCE_COLUMNS = (TIME_COLUMN, DEVICE_COLUMN, *FEATURE_COLUMNS)
FEATURE_COLUMN_MAP = {
    "加速度X(g)": "acc_x_g",
    "加速度Y(g)": "acc_y_g",
    "加速度Z(g)": "acc_z_g",
    "角速度X(°/s)": "gyro_x_dps",
    "角速度Y(°/s)": "gyro_y_dps",
    "角速度Z(°/s)": "gyro_z_dps",
    "角度X(°)": "angle_x_deg",
    "角度Y(°)": "angle_y_deg",
    "角度Z(°)": "angle_z_deg",
    "磁场X(uT)": "mag_x_ut",
    "磁场Y(uT)": "mag_y_ut",
    "磁场Z(uT)": "mag_z_ut",
    "四元数0()": "quat_0",
    "四元数1()": "quat_1",
    "四元数2()": "quat_2",
    "四元数3()": "quat_3",
}


def normalize_column_name(name: object) -> str:
    """Return a presentation-independent header fingerprint."""
    text = str(name).translate(
        str.maketrans({"（": "(", "）": ")", "μ": "u", "µ": "u"})
    )
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) not in {"Cc", "Cf"}
    )
    return text.strip().casefold()


CANONICAL_SOURCE_COLUMN_MAP = {
    normalize_column_name(column): column
    for column in (*REQUIRED_SOURCE_COLUMNS, *OPTIONAL_METADATA_COLUMNS)
}


def read_csv_robust(path: Path) -> CsvReadResult:
    rejected_rows: list[RejectedRow] = []
    warnings: list[str] = []
    file_errors: list[FileError] = []
    valid_records: list[list[str]] = []
    valid_line_numbers: list[int] = []
    valid_data_indices: list[int] = []
    normalized_header: list[str] = []
    total_input_rows = 0
    reader: Any = None
    decode_line_number: int | None = None

    try:
        with path.open("rb") as handle:
            def decoded_lines() -> Any:
                nonlocal decode_line_number
                buffer = bytearray()
                physical_line_number = 0
                while True:
                    chunk = handle.read(8192)
                    end_of_file = not chunk
                    buffer.extend(chunk)
                    separator_index = 0
                    while separator_index < len(buffer):
                        byte = buffer[separator_index]
                        if byte == 10:
                            line_end = separator_index + 1
                        elif byte == 13:
                            if separator_index + 1 == len(buffer) and not end_of_file:
                                break
                            line_end = separator_index + 1
                            if (
                                separator_index + 1 < len(buffer)
                                and buffer[separator_index + 1] == 10
                            ):
                                line_end += 1
                        else:
                            separator_index += 1
                            continue

                        raw_line = bytes(buffer[:line_end])
                        del buffer[:line_end]
                        separator_index = 0
                        physical_line_number += 1
                        decode_line_number = physical_line_number
                        encoding = (
                            "utf-8-sig" if physical_line_number == 1 else "utf-8"
                        )
                        yield raw_line.decode(encoding)

                    if end_of_file:
                        if buffer:
                            physical_line_number += 1
                            decode_line_number = physical_line_number
                            encoding = (
                                "utf-8-sig"
                                if physical_line_number == 1
                                else "utf-8"
                            )
                            yield bytes(buffer).decode(encoding)
                        break

            reader = csv.reader(decoded_lines(), strict=True)
            try:
                header = next(reader)
            except StopIteration:
                header = []

            header_fingerprint = [normalize_column_name(cell) for cell in header]
            if not header or not any(header_fingerprint):
                file_errors.append(
                    FileError(
                        source_file=path.name,
                        error_type="empty_or_missing_header",
                        source_line_number=1,
                        message="CSV header is empty or missing",
                    )
                )
            else:
                normalized_header = [
                    CANONICAL_SOURCE_COLUMN_MAP.get(fingerprint, fingerprint)
                    for fingerprint in header_fingerprint
                ]
                duplicate_columns = list(
                    dict.fromkeys(
                        column
                        for column in normalized_header
                        if normalized_header.count(column) > 1
                    )
                )
                if duplicate_columns:
                    file_errors.append(
                        FileError(
                            source_file=path.name,
                            error_type="duplicate_normalized_columns",
                            source_line_number=1,
                            message="Duplicate normalized columns: "
                            + ", ".join(duplicate_columns),
                        )
                    )
                    normalized_header = []
                else:
                    missing_required = [
                        column
                        for column in REQUIRED_SOURCE_COLUMNS
                        if column not in normalized_header
                    ]
                    if missing_required:
                        file_errors.append(
                            FileError(
                                source_file=path.name,
                                error_type="missing_required_columns",
                                source_line_number=1,
                                message="Missing required columns: "
                                + ", ".join(missing_required),
                            )
                        )
                    else:
                        for column in OPTIONAL_METADATA_COLUMNS:
                            if column not in normalized_header:
                                warnings.append(
                                    f"Missing optional metadata column: {column}"
                                )

                        data_row_index = 0
                        while True:
                            record_start_line = reader.line_num + 1
                            try:
                                row = next(reader)
                            except StopIteration:
                                break
                            data_row_index += 1
                            total_input_rows = data_row_index
                            raw_row = json.dumps(row, ensure_ascii=False)

                            if not row or not any(cell.strip() for cell in row):
                                reject_reason = "blank_row"
                            elif [
                                normalize_column_name(cell) for cell in row
                            ] == header_fingerprint:
                                reject_reason = "repeated_header"
                            elif len(row) != len(normalized_header):
                                reject_reason = "field_count_mismatch"
                            else:
                                valid_records.append(row)
                                valid_line_numbers.append(record_start_line)
                                valid_data_indices.append(data_row_index)
                                continue

                            rejected_rows.append(
                                RejectedRow(
                                    source_file=path.name,
                                    source_line_number=record_start_line,
                                    source_row_index=data_row_index,
                                    reject_stage="structural",
                                    reject_reason=reject_reason,
                                    raw_row=raw_row,
                                )
                            )
    except UnicodeDecodeError as error:
        file_errors.append(
            FileError(
                source_file=path.name,
                error_type="utf8_decode_error",
                source_line_number=decode_line_number,
                message=str(error),
            )
        )
    except csv.Error as error:
        file_errors.append(
            FileError(
                source_file=path.name,
                error_type="csv_syntax_error",
                source_line_number=getattr(reader, "line_num", None),
                message=str(error),
            )
        )
    frame = pd.DataFrame(valid_records, columns=normalized_header)
    frame["source_file"] = path.name
    frame["source_line_number"] = valid_line_numbers
    frame["source_row_index"] = valid_data_indices
    return CsvReadResult(
        source_file=path.name,
        dataframe=frame,
        rejected_rows=rejected_rows,
        warnings=warnings,
        file_errors=file_errors,
        total_input_rows=total_input_rows,
    )


def parse_sensor_position(device_name: object) -> str | None:
    """Return the sensor position encoded by a complete device prefix."""
    normalized = str(device_name).strip().upper()
    match = re.match(r"^(WTLL|WTRL|WTLA|WTRA|WTC)(?:\s*\(|$)", normalized)
    if match is None:
        return None
    return DEVICE_PREFIX_TO_SENSOR[match.group(1)]


def validate_dataframe(result: CsvReadResult) -> ValidatedCsvResult:
    raw_frame = result.dataframe.copy()
    absolute_time = pd.to_datetime(
        raw_frame[TIME_COLUMN], errors="coerce", format="mixed"
    )
    sensor_position = raw_frame[DEVICE_COLUMN].map(parse_sensor_position)
    converted_features = pd.DataFrame(index=raw_frame.index)
    rejection_reasons: list[list[str]] = [[] for _ in range(len(raw_frame))]

    for position, invalid in enumerate(absolute_time.isna().to_numpy()):
        if invalid:
            rejection_reasons[position].append("invalid_time")
    for position, invalid in enumerate(sensor_position.isna().to_numpy()):
        if invalid:
            rejection_reasons[position].append("unknown_sensor")

    for source_column, output_column in FEATURE_COLUMN_MAP.items():
        converted = pd.to_numeric(raw_frame[source_column], errors="coerce")
        converted_features[output_column] = converted
        for position, invalid in enumerate(converted.isna().to_numpy()):
            if invalid:
                rejection_reasons[position].append(f"non_numeric_{output_column}")

    warnings = list(result.warnings)
    for column in (OPTIONAL_METADATA_COLUMNS[0], OPTIONAL_METADATA_COLUMNS[2]):
        if column not in raw_frame.columns:
            continue
        original = raw_frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_count = int((original.notna() & converted.isna()).sum())
        if invalid_count:
            warnings.append(
                f"Invalid optional metadata values in {column}: {invalid_count}"
            )

    rejected_rows = list(result.rejected_rows)
    source_columns = [
        column
        for column in raw_frame.columns
        if column
        not in {"source_file", "source_line_number", "source_row_index"}
    ]
    for position, reasons in enumerate(rejection_reasons):
        if not reasons:
            continue
        row = raw_frame.iloc[position]

        def trace_index(column: str) -> int | None:
            value = row.get(column)
            return None if value is None or pd.isna(value) else int(value)

        raw_values: list[Any] = []
        for column in source_columns:
            value = row[column]
            if pd.isna(value):
                raw_values.append(None)
            elif hasattr(value, "item"):
                raw_values.append(value.item())
            else:
                raw_values.append(value)
        rejected_rows.append(
            RejectedRow(
                source_file=str(row.get("source_file", result.source_file)),
                source_line_number=trace_index("source_line_number"),
                source_row_index=trace_index("source_row_index"),
                reject_stage="content",
                reject_reason=";".join(reasons),
                raw_row=json.dumps(raw_values, ensure_ascii=False),
            )
        )

    validated = pd.DataFrame(
        {
            "absolute_time": absolute_time,
            "sensor_position": sensor_position,
        },
        index=raw_frame.index,
    )
    validated = pd.concat([validated, converted_features], axis=1)
    for column in ("source_file", "source_line_number", "source_row_index"):
        validated[column] = raw_frame[column]
    accepted_mask = pd.Series(
        [not reasons for reasons in rejection_reasons], index=raw_frame.index
    )

    return ValidatedCsvResult(
        dataframe=validated.loc[accepted_mask].reset_index(drop=True),
        rejected_rows=rejected_rows,
        warnings=warnings,
        unknown_sensor_rows=int(sensor_position.isna().sum()),
    )


def natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def discover_action_directories(input_root: Path) -> list[ActionDescriptor]:
    actions: list[ActionDescriptor] = []
    for directory in input_root.rglob("*"):
        if not directory.is_dir():
            continue
        csv_files = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".csv"
                ),
                key=lambda path: natural_key(path.name),
            )
        )
        if not csv_files:
            continue
        relative = directory.relative_to(input_root)
        if len(relative.parts) < 3:
            raise ValueError(f"Invalid action path: {relative.as_posix()}")
        class_text, class_name = relative.parts[0].split("_", 1)
        actions.append(
            ActionDescriptor(
                class_id=int(class_text),
                class_name=class_name,
                user_id=relative.parts[1],
                action_id=relative.parts[-1],
                input_directory=directory,
                relative_action_path=relative,
                input_csv_files=csv_files,
            )
        )
    return sorted(
        actions,
        key=lambda item: (
            item.class_id,
            natural_key(item.user_id),
            natural_key(item.action_id),
            item.relative_action_path.as_posix().casefold(),
        ),
    )


def process_action_directory(
    action_dir: Path, input_root: Path, output_root: Path
) -> ActionResult:
    del output_root
    relative = action_dir.relative_to(input_root)
    if len(relative.parts) < 3:
        raise ValueError(f"Invalid action path: {relative.as_posix()}")
    class_text, class_name = relative.parts[0].split("_", 1)
    csv_files = tuple(
        sorted(
            (
                path
                for path in action_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".csv"
            ),
            key=lambda path: natural_key(path.name),
        )
    )
    descriptor = ActionDescriptor(
        class_id=int(class_text),
        class_name=class_name,
        user_id=relative.parts[1],
        action_id=relative.parts[-1],
        input_directory=action_dir,
        relative_action_path=relative,
        input_csv_files=csv_files,
    )
    return process_action(descriptor)


def build_in_memory_action_result(
    descriptor: ActionDescriptor,
    read_results: tuple[CsvReadResult, ...],
    validated_results: tuple[ValidatedCsvResult | None, ...],
) -> InMemoryActionResult:
    if len(read_results) != len(descriptor.input_csv_files):
        raise ValueError("read result count does not match input files")
    if len(validated_results) != len(read_results):
        raise ValueError("validated result count does not match read results")

    frames: list[pd.DataFrame] = []
    rejected_rows: list[RejectedRow] = []
    warnings: list[str] = []
    file_errors: list[FileError] = []
    total_input_rows = 0
    unknown_sensor_rows = 0

    for source_file_rank, (read_result, validated) in enumerate(
        zip(read_results, validated_results, strict=True)
    ):
        total_input_rows += read_result.total_input_rows
        if read_result.file_errors:
            if validated is not None:
                raise ValueError("fatal read result must not be validated")
            rejected_rows.extend(read_result.rejected_rows)
            warnings.extend(read_result.warnings)
            file_errors.extend(read_result.file_errors)
            continue
        if validated is None:
            raise ValueError("successful read result requires validation")
        exact_frame = validated.dataframe.copy()
        exact_frame["_source_file_rank"] = source_file_rank
        frames.append(exact_frame)
        rejected_rows.extend(validated.rejected_rows)
        warnings.extend(validated.warnings)
        unknown_sensor_rows += validated.unknown_sensor_rows
        file_errors.extend(read_result.file_errors)

    exact_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return InMemoryActionResult(
        descriptor=descriptor,
        validated_results=validated_results,
        read_results=read_results,
        exact_rows=exact_rows,
        rejected_rows=rejected_rows,
        warnings=warnings,
        file_errors=file_errors,
        total_input_rows=total_input_rows,
        unknown_sensor_rows=unknown_sensor_rows,
    )


def process_action_in_memory(descriptor: ActionDescriptor) -> InMemoryActionResult:
    """Return validated exact-time rows without writing Stage 1 artifacts."""
    read_results = tuple(
        read_csv_robust(path) for path in descriptor.input_csv_files
    )
    validated_results = tuple(
        None if result.file_errors else validate_dataframe(result)
        for result in read_results
    )
    return build_in_memory_action_result(
        descriptor,
        read_results,
        validated_results,
    )


def build_legacy_action_result(memory: InMemoryActionResult) -> ActionResult:
    descriptor = memory.descriptor
    rejected_rows = list(memory.rejected_rows)
    warnings = list(memory.warnings)
    file_errors = list(memory.file_errors)
    total_input_rows = memory.total_input_rows
    unknown_sensor_rows = memory.unknown_sensor_rows
    rejected = pd.DataFrame(
        [asdict(row) for row in rejected_rows], columns=REJECTED_COLUMNS
    )
    candidate = memory.exact_rows.copy()
    duplicate_counts = {sensor: 0 for sensor in SENSOR_ORDER}
    rows_per_sensor = {sensor: 0 for sensor in SENSOR_ORDER}
    action_start_time: pd.Timestamp | None = None
    action_end_time: pd.Timestamp | None = None
    merged = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if not file_errors and not candidate.empty:
        action_start_time = candidate["absolute_time"].min()
        action_end_time = candidate["absolute_time"].max()
        candidate["relative_time_s"] = (
            candidate["absolute_time"] - action_start_time
        ).dt.total_seconds()
        candidate["relative_time_ms"] = (
            candidate["relative_time_s"] * 1000
        ).round().astype("int64")
        candidate["sensor_order"] = candidate["sensor_position"].map(SENSOR_ORDER)
        candidate = candidate.sort_values(
            ["absolute_time", "sensor_order", "source_file", "source_row_index"],
            kind="mergesort",
        ).reset_index(drop=True)
        duplicate_mask = candidate.duplicated(
            ["sensor_position", "absolute_time"], keep="first"
        )
        for sensor, count in (
            candidate.loc[duplicate_mask, "sensor_position"].value_counts().items()
        ):
            duplicate_counts[str(sensor)] = int(count)
        counts = candidate["sensor_position"].value_counts()
        rows_per_sensor = {
            sensor: int(counts.get(sensor, 0)) for sensor in SENSOR_ORDER
        }
        merged = candidate.loc[:, OUTPUT_COLUMNS]

    present_sensors = [
        sensor for sensor in SENSOR_ORDER if rows_per_sensor[sensor] > 0
    ]
    missing_sensors = [
        sensor for sensor in SENSOR_ORDER if rows_per_sensor[sensor] == 0
    ]
    duplicate_total = sum(duplicate_counts.values())
    structural_rejected_rows = sum(
        row.reject_stage == "structural" for row in rejected_rows
    )
    content_rejected_rows = sum(
        row.reject_stage == "content" for row in rejected_rows
    )
    if file_errors or merged.empty:
        status = "failed"
    elif missing_sensors:
        status = "incomplete_sensors"
    elif rejected_rows or duplicate_total or warnings:
        status = "success_with_warnings"
    else:
        status = "success"

    error_parts = [
        f"{error.source_file}: {error.error_type}: {error.message}"
        for error in file_errors
    ]
    if status == "failed" and not error_parts:
        error_parts.append("No valid output rows")
    error_message = "; ".join(error_parts).replace("\r", " ").replace("\n", " ")
    duration_s = (
        None
        if action_start_time is None or action_end_time is None
        else float((action_end_time - action_start_time).total_seconds())
    )
    min_relative_time_ms = (
        None if merged.empty else int(merged["relative_time_ms"].min())
    )
    max_relative_time_ms = (
        None if merged.empty else int(merged["relative_time_ms"].max())
    )
    qc = {
        "class_id": int(descriptor.class_id),
        "class_name": descriptor.class_name,
        "user_id": descriptor.user_id,
        "action_id": descriptor.action_id,
        "input_directory": str(descriptor.input_directory.resolve()),
        "input_csv_files": [path.name for path in descriptor.input_csv_files],
        "csv_file_count": int(len(descriptor.input_csv_files)),
        "status": status,
        "total_input_rows": int(total_input_rows),
        "valid_output_rows": int(len(merged)),
        "rejected_rows": int(len(rejected_rows)),
        "unknown_sensor_rows": int(unknown_sensor_rows),
        "present_sensors": present_sensors,
        "missing_sensors": missing_sensors,
        "rows_per_sensor": rows_per_sensor,
        "duplicate_timestamp_count_per_sensor": duplicate_counts,
        "min_relative_time_ms": min_relative_time_ms,
        "max_relative_time_ms": max_relative_time_ms,
        "duration_s": duration_s,
        "columns_written": list(OUTPUT_COLUMNS) if not merged.empty else [],
        "warnings": warnings,
        "error_message": error_message,
        "action_start_time": (
            None if action_start_time is None else action_start_time.isoformat()
        ),
        "action_end_time": (
            None if action_end_time is None else action_end_time.isoformat()
        ),
        "structural_rejected_rows": int(structural_rejected_rows),
        "content_rejected_rows": int(content_rejected_rows),
        "file_errors": [asdict(error) for error in file_errors],
    }
    output_csv = (
        ""
        if status == "failed"
        else (descriptor.relative_action_path / "imu_merged.csv").as_posix()
    )
    manifest_row = {
        "sample_id": (
            f"{descriptor.class_id}__{descriptor.user_id}__{descriptor.action_id}"
        ),
        "class_id": int(descriptor.class_id),
        "class_name": descriptor.class_name,
        "user_id": descriptor.user_id,
        "action_id": descriptor.action_id,
        "relative_action_path": descriptor.relative_action_path.as_posix(),
        "output_csv": output_csv,
        "status": status,
        "csv_file_count": int(len(descriptor.input_csv_files)),
        "total_input_rows": int(total_input_rows),
        "valid_output_rows": int(len(merged)),
        "rejected_rows": int(len(rejected_rows)),
        "unknown_sensor_rows": int(unknown_sensor_rows),
        "present_sensors": ";".join(present_sensors),
        "missing_sensors": ";".join(missing_sensors),
        "ll_rows": rows_per_sensor["LL"],
        "rl_rows": rows_per_sensor["RL"],
        "la_rows": rows_per_sensor["LA"],
        "ra_rows": rows_per_sensor["RA"],
        "c_rows": rows_per_sensor["C"],
        "ll_duplicate_timestamps": duplicate_counts["LL"],
        "rl_duplicate_timestamps": duplicate_counts["RL"],
        "la_duplicate_timestamps": duplicate_counts["LA"],
        "ra_duplicate_timestamps": duplicate_counts["RA"],
        "c_duplicate_timestamps": duplicate_counts["C"],
        "duration_s": duration_s,
        "warning_count": int(len(warnings)),
        "error_message": error_message,
    }
    return ActionResult(
        descriptor=descriptor,
        status=status,
        merged=merged,
        rejected=rejected,
        qc=qc,
        manifest_row=manifest_row,
    )


def process_action(descriptor: ActionDescriptor) -> ActionResult:
    return build_legacy_action_result(process_action_in_memory(descriptor))


def write_action_result(
    result: ActionResult, output_root: Path, overwrite: bool
) -> WriteResult:
    resolved_root = output_root.resolve()
    requested_destination = Path(
        os.path.abspath(resolved_root / result.descriptor.relative_action_path)
    )
    try:
        destination = _verified_action_path(resolved_root, requested_destination)
    except Exception as error:
        return WriteResult(
            written=False,
            output_directory=requested_destination,
            error_message=str(error),
        )
    input_root = result.descriptor.input_directory.resolve()
    for _ in result.descriptor.relative_action_path.parts:
        input_root = input_root.parent
    if destination.is_relative_to(input_root) or input_root.is_relative_to(
        destination
    ):
        return WriteResult(
            written=False,
            output_directory=destination,
            error_message="Output destination overlaps input action tree",
        )
    if destination.exists() and not destination.is_dir():
        return WriteResult(
            written=False,
            output_directory=destination,
            error_message="Existing action output is not a directory",
        )
    if destination.exists() and not overwrite:
        if (destination / "imu_merged.csv").is_file():
            return WriteResult(written=False, output_directory=destination)
        return WriteResult(
            written=False,
            output_directory=destination,
            error_message=(
                "Existing action output conflict: destination exists without "
                "imu_merged.csv"
            ),
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = _verified_sibling(
        resolved_root,
        destination,
        destination.parent / f".{destination.name}.staging-{token}",
    )
    backup = _verified_sibling(
        resolved_root,
        destination,
        destination.parent / f".{destination.name}.backup-{token}",
    )
    backup_created = False
    staging_created = False
    try:
        staging.mkdir()
        staging_created = True
        successful = result.status != "failed"
        if successful:
            if result.merged.columns.tolist() != OUTPUT_COLUMNS:
                raise ValueError("Merged output columns do not match OUTPUT_COLUMNS")
            result.merged.to_csv(
                staging / "imu_merged.csv",
                index=False,
                encoding="utf-8-sig",
            )
        (staging / "qc.json").write_text(
            json.dumps(result.qc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not result.rejected.empty:
            if result.rejected.columns.tolist() != REJECTED_COLUMNS:
                raise ValueError(
                    "Rejected output columns do not match REJECTED_COLUMNS"
                )
            result.rejected.to_csv(
                staging / "rejected_rows.csv",
                index=False,
                encoding="utf-8-sig",
            )

        expected_names = {"qc.json"}
        if successful:
            expected_names.add("imu_merged.csv")
        if not result.rejected.empty:
            expected_names.add("rejected_rows.csv")
        actual_names = {path.name for path in staging.iterdir()}
        if actual_names != expected_names or not all(
            path.is_file() for path in staging.iterdir()
        ):
            raise RuntimeError(
                "Staged managed artifact set mismatch: "
                f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
            )

        if destination.exists():
            destination.rename(backup)
            backup_created = True
        staging.rename(destination)
        staging_created = False
    except Exception as error:
        error_messages = [str(error)]
        if backup_created and backup.exists():
            if not destination.exists():
                try:
                    backup.rename(destination)
                    backup_created = False
                except Exception as restore_error:
                    error_messages.append(f"backup restore failed: {restore_error}")
        if staging_created:
            try:
                _remove_managed_tree(resolved_root, staging)
            except Exception as cleanup_error:
                error_messages.append(f"staging cleanup failed: {cleanup_error}")
        return WriteResult(
            written=False,
            output_directory=destination,
            error_message="; ".join(error_messages),
        )

    if backup_created:
        try:
            _remove_managed_tree(resolved_root, backup)
        except Exception as cleanup_error:
            return WriteResult(
                written=True,
                output_directory=destination,
                error_message=(
                    f"backup cleanup failed; backup retained at {backup}: "
                    f"{cleanup_error}"
                ),
            )
    return WriteResult(written=True, output_directory=destination)


def validate_roots(input_root: Path, output_root: Path) -> None:
    resolved_input = input_root.resolve(strict=False)
    resolved_output = output_root.resolve(strict=False)
    if (
        resolved_input == resolved_output
        or resolved_input in resolved_output.parents
        or resolved_output in resolved_input.parents
    ):
        raise ValueError("Input and output roots must not overlap")
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input root does not exist: {resolved_input}")
    if not resolved_input.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {resolved_input}")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess stage-1 IMU actions")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _failed_action_result(
    descriptor: ActionDescriptor, error: BaseException
) -> ActionResult:
    message = str(error).replace("\r", " ").replace("\n", " ")
    rows_per_sensor = {sensor: 0 for sensor in SENSOR_ORDER}
    duplicate_counts = {sensor: 0 for sensor in SENSOR_ORDER}
    qc = {
        "class_id": descriptor.class_id,
        "class_name": descriptor.class_name,
        "user_id": descriptor.user_id,
        "action_id": descriptor.action_id,
        "input_directory": str(descriptor.input_directory.resolve()),
        "input_csv_files": [path.name for path in descriptor.input_csv_files],
        "csv_file_count": len(descriptor.input_csv_files),
        "status": "failed",
        "total_input_rows": 0,
        "valid_output_rows": 0,
        "rejected_rows": 0,
        "unknown_sensor_rows": 0,
        "present_sensors": [],
        "missing_sensors": list(SENSOR_ORDER),
        "rows_per_sensor": rows_per_sensor,
        "duplicate_timestamp_count_per_sensor": duplicate_counts,
        "min_relative_time_ms": None,
        "max_relative_time_ms": None,
        "duration_s": None,
        "columns_written": [],
        "warnings": [],
        "error_message": message,
        "action_start_time": None,
        "action_end_time": None,
        "structural_rejected_rows": 0,
        "content_rejected_rows": 0,
        "file_errors": [],
    }
    row = {
        "sample_id": (
            f"{descriptor.class_id}__{descriptor.user_id}__{descriptor.action_id}"
        ),
        "class_id": descriptor.class_id,
        "class_name": descriptor.class_name,
        "user_id": descriptor.user_id,
        "action_id": descriptor.action_id,
        "relative_action_path": descriptor.relative_action_path.as_posix(),
        "output_csv": "",
        "status": "failed",
        "csv_file_count": len(descriptor.input_csv_files),
        "total_input_rows": 0,
        "valid_output_rows": 0,
        "rejected_rows": 0,
        "unknown_sensor_rows": 0,
        "present_sensors": "",
        "missing_sensors": ";".join(SENSOR_ORDER),
        "ll_rows": 0,
        "rl_rows": 0,
        "la_rows": 0,
        "ra_rows": 0,
        "c_rows": 0,
        "ll_duplicate_timestamps": 0,
        "rl_duplicate_timestamps": 0,
        "la_duplicate_timestamps": 0,
        "ra_duplicate_timestamps": 0,
        "c_duplicate_timestamps": 0,
        "duration_s": None,
        "warning_count": 0,
        "error_message": message,
    }
    return ActionResult(
        descriptor=descriptor,
        status="failed",
        merged=pd.DataFrame(columns=OUTPUT_COLUMNS),
        rejected=pd.DataFrame(columns=REJECTED_COLUMNS),
        qc=qc,
        manifest_row=row,
    )


def _existing_action_result(
    descriptor: ActionDescriptor, output_root: Path, logger: logging.Logger | None
) -> ActionResult:
    output_directory = output_root / descriptor.relative_action_path
    qc_path = output_directory / "qc.json"
    qc: dict[str, Any] = {}
    try:
        loaded = json.loads(qc_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("QC root is not an object")
        _validate_existing_qc(loaded)
        qc = loaded
    except Exception as error:
        if logger is not None:
            logger.warning("无法读取现有 QC %s: %s", qc_path, error)
        else:
            print(f"无法读取现有 QC {qc_path}: {error}", file=sys.stderr)
    rows = qc.get("rows_per_sensor", {})
    duplicates = qc.get("duplicate_timestamp_count_per_sensor", {})
    present = [sensor for sensor in SENSOR_ORDER if sensor in qc.get("present_sensors", [])]
    missing = [sensor for sensor in SENSOR_ORDER if sensor in qc.get("missing_sensors", [])]
    row = {
        "sample_id": (
            f"{descriptor.class_id}__{descriptor.user_id}__{descriptor.action_id}"
        ),
        "class_id": descriptor.class_id,
        "class_name": descriptor.class_name,
        "user_id": descriptor.user_id,
        "action_id": descriptor.action_id,
        "relative_action_path": descriptor.relative_action_path.as_posix(),
        "output_csv": (descriptor.relative_action_path / "imu_merged.csv").as_posix(),
        "status": "skipped_existing",
        "csv_file_count": int(qc.get("csv_file_count", len(descriptor.input_csv_files))),
        "total_input_rows": int(qc.get("total_input_rows", 0)),
        "valid_output_rows": int(qc.get("valid_output_rows", 0)),
        "rejected_rows": int(qc.get("rejected_rows", 0)),
        "unknown_sensor_rows": int(qc.get("unknown_sensor_rows", 0)),
        "present_sensors": ";".join(present),
        "missing_sensors": ";".join(missing),
        "ll_rows": int(rows.get("LL", 0)),
        "rl_rows": int(rows.get("RL", 0)),
        "la_rows": int(rows.get("LA", 0)),
        "ra_rows": int(rows.get("RA", 0)),
        "c_rows": int(rows.get("C", 0)),
        "ll_duplicate_timestamps": int(duplicates.get("LL", 0)),
        "rl_duplicate_timestamps": int(duplicates.get("RL", 0)),
        "la_duplicate_timestamps": int(duplicates.get("LA", 0)),
        "ra_duplicate_timestamps": int(duplicates.get("RA", 0)),
        "c_duplicate_timestamps": int(duplicates.get("C", 0)),
        "duration_s": qc.get("duration_s"),
        "warning_count": len(qc.get("warnings", [])),
        "error_message": str(qc.get("error_message", "")).replace(
            "\r", " "
        ).replace("\n", " "),
    }
    qc_for_run = dict(qc)
    qc_for_run["status"] = "skipped_existing"
    return ActionResult(
        descriptor=descriptor,
        status="skipped_existing",
        merged=pd.DataFrame(columns=OUTPUT_COLUMNS),
        rejected=pd.DataFrame(columns=REJECTED_COLUMNS),
        qc=qc_for_run,
        manifest_row=row,
    )


def _validate_existing_qc(qc: dict[str, Any]) -> None:
    if set(qc) != set(QC_FIELDS):
        missing = sorted(set(QC_FIELDS) - set(qc))
        extra = sorted(set(qc) - set(QC_FIELDS))
        raise ValueError(f"QC fields mismatch; missing={missing}, extra={extra}")

    def require_int(name: str, *, nullable: bool = False) -> None:
        value = qc[name]
        if nullable and value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"QC field {name} must be an integer")

    for name in (
        "class_id", "csv_file_count", "total_input_rows", "valid_output_rows",
        "rejected_rows", "unknown_sensor_rows", "structural_rejected_rows",
        "content_rejected_rows",
    ):
        require_int(name)
    for name in (
        "class_id", "csv_file_count", "total_input_rows", "valid_output_rows",
        "rejected_rows", "unknown_sensor_rows", "structural_rejected_rows",
        "content_rejected_rows",
    ):
        if qc[name] < 0:
            raise ValueError(f"QC field {name} must be non-negative")
    require_int("min_relative_time_ms", nullable=True)
    require_int("max_relative_time_ms", nullable=True)

    for name in (
        "class_name", "user_id", "action_id", "input_directory", "status",
        "error_message",
    ):
        if not isinstance(qc[name], str):
            raise TypeError(f"QC field {name} must be a string")
    if qc["status"] not in {
        "success", "success_with_warnings", "incomplete_sensors", "failed"
    }:
        raise ValueError(f"Unknown QC status: {qc['status']}")

    for name in (
        "input_csv_files", "present_sensors", "missing_sensors",
        "columns_written", "warnings",
    ):
        value = qc[name]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise TypeError(f"QC field {name} must be a string list")
    for name in ("present_sensors", "missing_sensors"):
        if any(sensor not in SENSOR_ORDER for sensor in qc[name]):
            raise ValueError(f"QC field {name} contains an unknown sensor")
        expected_order = [sensor for sensor in SENSOR_ORDER if sensor in qc[name]]
        if qc[name] != expected_order:
            raise ValueError(f"QC field {name} must be unique and in sensor order")
    if set(qc["present_sensors"]) & set(qc["missing_sensors"]):
        raise ValueError("QC present_sensors and missing_sensors must be disjoint")

    for name in ("rows_per_sensor", "duplicate_timestamp_count_per_sensor"):
        value = qc[name]
        if not isinstance(value, dict) or set(value) != set(SENSOR_ORDER):
            raise TypeError(f"QC field {name} must contain all sensor keys")
        if any(isinstance(count, bool) or not isinstance(count, int) for count in value.values()):
            raise TypeError(f"QC field {name} values must be integers")
        if any(count < 0 for count in value.values()):
            raise ValueError(f"QC field {name} values must be non-negative")

    duration = qc["duration_s"]
    if duration is not None and (
        isinstance(duration, bool) or not isinstance(duration, (int, float))
    ):
        raise TypeError("QC field duration_s must be numeric or null")
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise ValueError("QC field duration_s must be finite and non-negative")
    for name in ("action_start_time", "action_end_time"):
        if qc[name] is not None and not isinstance(qc[name], str):
            raise TypeError(f"QC field {name} must be a string or null")

    file_errors = qc["file_errors"]
    if not isinstance(file_errors, list):
        raise TypeError("QC field file_errors must be a list")
    expected_error_fields = {
        "source_file", "error_type", "source_line_number", "message"
    }
    for file_error in file_errors:
        if not isinstance(file_error, dict) or set(file_error) != expected_error_fields:
            raise TypeError("QC file_errors entry has invalid fields")
        if not all(
            isinstance(file_error[name], str)
            for name in ("source_file", "error_type", "message")
        ):
            raise TypeError("QC file_errors text fields must be strings")
        line = file_error["source_line_number"]
        if line is not None and (isinstance(line, bool) or not isinstance(line, int)):
            raise TypeError("QC file error line must be an integer or null")


def build_manifest(results: list[ActionResult]) -> pd.DataFrame:
    ordered = sorted(
        results,
        key=lambda result: (
            int(result.descriptor.class_id),
            natural_key(result.descriptor.user_id),
            natural_key(result.descriptor.action_id),
            result.descriptor.relative_action_path.as_posix().casefold(),
        ),
    )
    rows = [result.manifest_row for result in ordered]
    integer_columns = (
        "class_id", "csv_file_count", "total_input_rows", "valid_output_rows",
        "rejected_rows", "unknown_sensor_rows", "ll_rows", "rl_rows",
        "la_rows", "ra_rows", "c_rows", "ll_duplicate_timestamps",
        "rl_duplicate_timestamps", "la_duplicate_timestamps",
        "ra_duplicate_timestamps", "c_duplicate_timestamps", "warning_count",
    )
    string_columns = tuple(
        column
        for column in MANIFEST_COLUMNS
        if column not in integer_columns and column != "duration_s"
    )
    for row in rows:
        if set(row) != set(MANIFEST_COLUMNS):
            raise ValueError("Manifest row fields do not match MANIFEST_COLUMNS")
        for column in integer_columns:
            value = row[column]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Manifest field {column} must be a non-negative integer"
                )
        for column in string_columns:
            if not isinstance(row[column], str):
                raise TypeError(f"Manifest field {column} must be a string")
        duration = row["duration_s"]
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError(
                "Manifest field duration_s must be finite, non-negative, or null"
            )
        if row["status"] == "failed" and row["output_csv"] != "":
            raise ValueError("Failed manifest row output_csv must be blank")
    return pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS)


def _cleanup_logger_handlers(
    logger: logging.Logger, handlers: list[logging.Handler]
) -> bool:
    failed = False
    for handler in handlers:
        try:
            if handler in logger.handlers:
                logger.removeHandler(handler)
        except Exception:
            failed = True
        try:
            handler.flush()
        except Exception:
            failed = True
        try:
            handler.close()
        except Exception:
            failed = True
    return failed


def _summarize(results: list[ActionResult]) -> None:
    statuses = {
        status: sum(result.status == status for result in results)
        for status in (
            "success", "success_with_warnings", "incomplete_sensors",
            "skipped_existing", "failed",
        )
    }
    lines = [
        f"类别数: {len({result.descriptor.class_id for result in results})}",
        f"用户数: {len({result.descriptor.user_id for result in results})}",
        f"动作总数: {len(results)}",
        f"成功数: {statuses['success']}",
        f"成功但有警告数: {statuses['success_with_warnings']}",
        f"传感器不完整数: {statuses['incomplete_sensors']}",
        f"已跳过现有输出数: {statuses['skipped_existing']}",
        f"失败数: {statuses['failed']}",
        "输入总行数: "
        f"{sum(int(result.manifest_row.get('total_input_rows', 0)) for result in results)}",
        "有效输出行数: "
        f"{sum(int(result.manifest_row.get('valid_output_rows', 0)) for result in results)}",
        "拒绝行数: "
        f"{sum(int(result.manifest_row.get('rejected_rows', 0)) for result in results)}",
    ]
    for sensor in SENSOR_ORDER:
        missing_count = sum(
            sensor in str(result.manifest_row.get("missing_sensors", "")).split(";")
            for result in results
        )
        lines.append(f"{sensor} 缺失动作数: {missing_count}")
    for line in lines:
        print(line, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        validate_roots(args.input_root, args.output_root)
    except Exception as error:
        print(f"全局错误: {error}", file=sys.stderr)
        return 2
    try:
        actions = discover_action_directories(args.input_root.resolve())
    except Exception as error:
        print(f"全局错误: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        try:
            results: list[ActionResult] = []
            for descriptor in actions:
                existing_csv = (
                    args.output_root
                    / descriptor.relative_action_path
                    / "imu_merged.csv"
                )
                if existing_csv.is_file() and not args.overwrite:
                    result = _existing_action_result(
                        descriptor, args.output_root, None
                    )
                else:
                    try:
                        result = process_action(descriptor)
                    except Exception as error:
                        result = _failed_action_result(descriptor, error)
                result.merged = pd.DataFrame(columns=OUTPUT_COLUMNS)
                result.rejected = pd.DataFrame(columns=REJECTED_COLUMNS)
                results.append(result)
            _summarize(results)
            return 1 if any(
                result.status == "failed" for result in results
            ) else 0
        except Exception as error:
            try:
                print(f"全局错误: {error}", file=sys.stderr)
            except Exception:
                pass
            return 2

    logger = logging.getLogger("preprocess_imu_stage1")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if _cleanup_logger_handlers(logger, list(logger.handlers)):
        try:
            print("全局错误: 无法清理既有日志处理器", file=sys.stderr)
        except Exception:
            pass
        return 2
    invocation_handlers: list[logging.Handler] = []
    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        invocation_handlers.append(console_handler)
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(
            args.output_root / "processing.log", encoding="utf-8"
        )
        invocation_handlers.append(file_handler)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    except Exception as error:
        _cleanup_logger_handlers(logger, invocation_handlers)
        try:
            print(f"全局错误: 日志初始化失败: {error}", file=sys.stderr)
        except Exception:
            pass
        return 2

    results = []
    exit_code = 0
    try:
        logger.info(
            "Task 5 writer note: overwrite installation uses two directory renames; "
            "a process crash between them can leave a recoverable backup sibling."
        )
        for descriptor in actions:
            existing_csv = (
                args.output_root / descriptor.relative_action_path / "imu_merged.csv"
            )
            if existing_csv.is_file() and not args.overwrite:
                result = _existing_action_result(descriptor, args.output_root, logger)
                logger.info("跳过现有输出: %s", descriptor.relative_action_path)
            else:
                try:
                    result = process_action(descriptor)
                except Exception as error:
                    logger.exception("动作处理异常: %s", descriptor.input_directory)
                    result = _failed_action_result(descriptor, error)
                try:
                    write_result = write_action_result(
                        result, args.output_root, overwrite=args.overwrite
                    )
                except Exception as error:
                    logger.exception(
                        "动作写入异常 %s", descriptor.relative_action_path
                    )
                    write_result = WriteResult(
                        written=False,
                        output_directory=(
                            args.output_root / descriptor.relative_action_path
                        ),
                        error_message=str(error),
                    )
                if not write_result.written and write_result.error_message:
                    failure_message = write_result.error_message
                    logger.error(
                        "动作写入失败 %s: %s",
                        descriptor.relative_action_path,
                        failure_message,
                    )
                    result = _failed_action_result(
                        descriptor, OSError(failure_message)
                    )
                    try:
                        write_result = write_action_result(
                            result, args.output_root, overwrite=args.overwrite
                        )
                    except Exception as qc_error:
                        logger.exception(
                            "失败动作 QC 写入异常 %s",
                            descriptor.relative_action_path,
                        )
                        write_result = WriteResult(
                            written=False,
                            output_directory=(
                                args.output_root
                                / descriptor.relative_action_path
                            ),
                            error_message=(
                                f"{failure_message}; failed QC write: {qc_error}"
                            ),
                        )
                    if not write_result.written:
                        qc_error = write_result.error_message or "unknown QC write failure"
                        combined_error = (
                            failure_message
                            if qc_error == failure_message
                            else f"{failure_message}; failed QC write: {qc_error}"
                        )
                        result.qc["error_message"] = combined_error
                        result.manifest_row["error_message"] = (
                            combined_error.replace("\r", " ").replace("\n", " ")
                        )
                if not write_result.written:
                    if not write_result.error_message and existing_csv.is_file():
                        result = _existing_action_result(
                            descriptor, args.output_root, logger
                        )
                    elif write_result.error_message:
                        result.status = "failed"
                        result.qc["status"] = "failed"
                        result.manifest_row["status"] = "failed"
                        result.manifest_row["output_csv"] = ""
                if write_result.error_message and write_result.written:
                    logger.warning("动作写入警告: %s", write_result.error_message)
            logger.info("动作状态 %s: %s", result.status, descriptor.relative_action_path)
            result.merged = pd.DataFrame(columns=OUTPUT_COLUMNS)
            result.rejected = pd.DataFrame(columns=REJECTED_COLUMNS)
            results.append(result)

        manifest = build_manifest(results)
        manifest.to_csv(
            args.output_root / "manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _summarize(results)
        exit_code = 1 if any(result.status == "failed" for result in results) else 0
    except Exception as error:
        logger.exception("全局错误: %s", error)
        exit_code = 2
    finally:
        if _cleanup_logger_handlers(logger, invocation_handlers):
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
