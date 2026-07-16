from __future__ import annotations

import csv
import json
import os
import re
import shutil
import stat
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


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


def process_action(descriptor: ActionDescriptor) -> ActionResult:
    frames: list[pd.DataFrame] = []
    rejected_rows: list[RejectedRow] = []
    warnings: list[str] = []
    file_errors: list[FileError] = []
    total_input_rows = 0
    unknown_sensor_rows = 0

    for path in descriptor.input_csv_files:
        read_result = read_csv_robust(path)
        total_input_rows += read_result.total_input_rows
        if read_result.file_errors:
            rejected_rows.extend(read_result.rejected_rows)
            warnings.extend(read_result.warnings)
            file_errors.extend(read_result.file_errors)
            continue
        validated = validate_dataframe(read_result)
        frames.append(validated.dataframe)
        rejected_rows.extend(validated.rejected_rows)
        warnings.extend(validated.warnings)
        unknown_sensor_rows += validated.unknown_sensor_rows
        file_errors.extend(read_result.file_errors)

    rejected = pd.DataFrame(
        [asdict(row) for row in rejected_rows], columns=REJECTED_COLUMNS
    )
    candidate = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
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
        return WriteResult(written=False, output_directory=destination)

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
