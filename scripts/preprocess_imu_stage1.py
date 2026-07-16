from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
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


SENSOR_ORDER = {"LL": 0, "RL": 1, "LA": 2, "RA": 3, "C": 4}
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
