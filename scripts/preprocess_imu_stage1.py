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
                for decode_line_number, raw_line in enumerate(handle, start=1):
                    encoding = "utf-8-sig" if decode_line_number == 1 else "utf-8"
                    yield raw_line.decode(encoding)

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
