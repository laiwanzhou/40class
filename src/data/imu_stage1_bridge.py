from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.preprocess_imu_stage1 as stage1
from src.data.imu_stage2_contracts import (
    FEATURE_ORDER,
    SENSOR_ORDER,
    ImuActionSource,
    NoValidStage1RecordsError,
    Stage1ActionData,
    Stage1DataValidationError,
    canonical_json_bytes,
    sha256_file,
)


@dataclass(frozen=True)
class Stage1ArtifactDescriptor:
    root: Path
    sample_id: str
    action_relative_path: Path
    output_csv_path: Path
    qc_path: Path
    manifest_row: dict[str, str]
    manifest_row_sha256: str


def decimal_seconds_to_ns(text: str) -> np.int64:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError("Invalid relative time") from error
    if not value.is_finite() or value < 0:
        raise ValueError("Invalid relative time")
    nanoseconds = value * Decimal(1_000_000_000)
    if nanoseconds != nanoseconds.to_integral_value():
        raise ValueError(
            "Relative time cannot be represented exactly in nanoseconds"
        )
    integer = int(nanoseconds)
    if not 0 <= integer <= np.iinfo(np.int64).max:
        raise OverflowError("Relative time is outside int64 range")
    return np.int64(integer)


def stage1_manifest_row_sha256(row: Mapping[str, str]) -> str:
    if list(row) != stage1.MANIFEST_COLUMNS:
        raise ValueError("Stage 1 manifest row columns do not match contract")
    payload = {column: row[column] for column in stage1.MANIFEST_COLUMNS}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _resolve_managed_path(root: Path, relative_text: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / Path(relative_text)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("Stage 1 artifact path escapes input root") from error
    return candidate


def discover_stage1_artifacts(root: Path) -> list[Stage1ArtifactDescriptor]:
    root = root.resolve()
    manifest_path = root / "manifest.csv"
    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if manifest.columns.tolist() != stage1.MANIFEST_COLUMNS:
        raise ValueError("Stage 1 manifest columns do not match contract")

    descriptors: list[Stage1ArtifactDescriptor] = []
    seen_sample_ids: set[str] = set()
    for record in manifest.to_dict(orient="records"):
        row = {column: str(record[column]) for column in stage1.MANIFEST_COLUMNS}
        if not row["sample_id"]:
            raise ValueError("Stage 1 manifest sample_id must not be empty")
        if row["sample_id"] in seen_sample_ids:
            raise ValueError(f"Duplicate Stage 1 sample_id: {row['sample_id']}")
        seen_sample_ids.add(row["sample_id"])
        if not row["output_csv"]:
            raise ValueError(
                f"Stage 1 action is not loadable: {row['sample_id']}"
            )
        action_relative_path = Path(row["relative_action_path"])
        output_csv_path = _resolve_managed_path(root, row["output_csv"])
        qc_path = _resolve_managed_path(
            root, (action_relative_path / "qc.json").as_posix()
        )
        if not output_csv_path.is_file():
            raise FileNotFoundError(output_csv_path)
        if not qc_path.is_file():
            raise FileNotFoundError(qc_path)
        descriptors.append(
            Stage1ArtifactDescriptor(
                root=root,
                sample_id=row["sample_id"],
                action_relative_path=action_relative_path,
                output_csv_path=output_csv_path,
                qc_path=qc_path,
                manifest_row=row,
                manifest_row_sha256=stage1_manifest_row_sha256(row),
            )
        )
    return descriptors


def _optional_int(text: str) -> int | None:
    return None if text == "" else int(text)


def _sensor_mask(frame: pd.DataFrame) -> np.ndarray:
    unknown = set(frame["sensor_position"].astype(str)) - set(SENSOR_ORDER)
    if unknown:
        raise ValueError(f"Unknown Stage 1 sensor positions: {sorted(unknown)}")
    present = set(frame["sensor_position"].astype(str))
    return np.asarray([sensor in present for sensor in SENSOR_ORDER], dtype=bool)


def _model_input_frame(
    frame: pd.DataFrame,
    source_file_ranks: Mapping[str, int] | None,
) -> pd.DataFrame:
    required = [
        "sensor_position",
        *FEATURE_ORDER,
        "source_file",
        "source_row_index",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Stage 1 data is missing columns: {missing}")
    selected = frame.loc[:, required].copy()
    if source_file_ranks is None:
        if "_source_file_rank" not in frame.columns:
            raise ValueError("Raw Stage 1 data is missing source file ranks")
        selected["_source_file_rank"] = frame["_source_file_rank"].to_numpy(
            dtype=np.int64
        )
    else:
        ranks = selected["source_file"].map(source_file_ranks)
        if ranks.isna().any():
            missing_files = sorted(
                set(selected.loc[ranks.isna(), "source_file"].astype(str))
            )
            raise ValueError(
                f"Stage 1 source files are absent from QC: {missing_files}"
            )
        selected["_source_file_rank"] = ranks.to_numpy(dtype=np.int64)
    selected["_stage1_row_index"] = np.arange(len(selected), dtype=np.int64)
    return selected


def load_stage1_action(
    descriptor: Stage1ArtifactDescriptor,
) -> Stage1ActionData:
    frame = pd.read_csv(
        descriptor.output_csv_path,
        dtype={"relative_time_s": "string"},
        encoding="utf-8-sig",
    )
    if "relative_time_s" not in frame.columns:
        raise ValueError("Stage 1 data is missing relative_time_s")
    relative_time_ns = np.asarray(
        [decimal_seconds_to_ns(str(value)) for value in frame["relative_time_s"]],
        dtype=np.int64,
    )
    qc_payload = json.loads(descriptor.qc_path.read_text(encoding="utf-8"))
    if not isinstance(qc_payload, dict):
        raise ValueError("Stage 1 QC must be a JSON object")
    input_csv_files = qc_payload.get("input_csv_files")
    if not isinstance(input_csv_files, list) or not all(
        isinstance(value, str) for value in input_csv_files
    ):
        raise ValueError("Stage 1 QC input_csv_files must be a string list")
    file_ranks = {name: rank for rank, name in enumerate(input_csv_files)}
    selected = _model_input_frame(frame, file_ranks)
    row = descriptor.manifest_row
    root = descriptor.root
    source_metadata: dict[str, object] = {
        "entry_path": "stage1_artifact",
        "source_stage1_manifest": "manifest.csv",
        "source_stage1_manifest_sha256": sha256_file(root / "manifest.csv"),
        "stage1_output_csv_relpath": descriptor.output_csv_path.relative_to(
            root
        ).as_posix(),
        "stage1_qc_relpath": descriptor.qc_path.relative_to(root).as_posix(),
        "stage1_output_csv_sha256": sha256_file(descriptor.output_csv_path),
        "stage1_qc_sha256": sha256_file(descriptor.qc_path),
        "stage1_manifest_row_sha256": descriptor.manifest_row_sha256,
    }
    return Stage1ActionData(
        sample_id=descriptor.sample_id,
        dataframe=selected,
        relative_time_ns=relative_time_ns,
        sensor_mask=_sensor_mask(selected),
        source_metadata=source_metadata,
        qc=qc_payload,
        class_id=_optional_int(row["class_id"]),
        class_name=row["class_name"] or None,
        user_id=row["user_id"] or None,
        action_id=row["action_id"] or None,
    )


def process_raw_imu_source(source: ImuActionSource) -> Stage1ActionData:
    descriptor = stage1.ActionDescriptor(
        class_id=0 if source.class_id is None else source.class_id,
        class_name="" if source.class_name is None else source.class_name,
        user_id="" if source.user_id is None else source.user_id,
        action_id=source.sample_id if source.action_id is None else source.action_id,
        input_directory=source.input_directory,
        relative_action_path=source.source_relative_path,
        input_csv_files=source.input_csv_files,
    )
    memory = stage1.process_action_in_memory(descriptor)
    if memory.file_errors:
        error_types = ", ".join(error.error_type for error in memory.file_errors)
        raise Stage1DataValidationError(
            source.sample_id,
            f"Stage 1 file validation failed: {error_types}",
        )
    if memory.exact_rows.empty:
        raise NoValidStage1RecordsError(
            source.sample_id,
            "No valid Stage 1 records",
        )

    frame = memory.exact_rows.copy()
    frame["_sensor_order"] = frame["sensor_position"].map(stage1.SENSOR_ORDER)
    frame = frame.sort_values(
        ["absolute_time", "_sensor_order", "source_file", "source_row_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    action_start_time = frame["absolute_time"].min()
    relative_time_ns = (
        (frame["absolute_time"] - action_start_time)
        .to_numpy(dtype="timedelta64[ns]")
        .astype(np.int64)
    )
    selected = _model_input_frame(frame, source_file_ranks=None)
    legacy = stage1.build_legacy_action_result(memory)
    source_metadata: dict[str, object] = {
        "entry_path": "raw_stage1_core",
        "source_relative_path": source.source_relative_path.as_posix(),
        "input_csv_files": [path.name for path in source.input_csv_files],
    }
    return Stage1ActionData(
        sample_id=source.sample_id,
        dataframe=selected,
        relative_time_ns=relative_time_ns,
        sensor_mask=_sensor_mask(selected),
        source_metadata=source_metadata,
        qc=legacy.qc,
        class_id=source.class_id,
        class_name=source.class_name,
        user_id=source.user_id,
        action_id=source.action_id,
    )
