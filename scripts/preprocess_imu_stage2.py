#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.imu_stage1_bridge import (
    Stage1ArtifactDescriptor,
    discover_stage1_artifacts,
    load_stage1_action,
)
from src.data.imu_stage2_contracts import (
    SENSOR_ORDER,
    DataStatus,
    NoValidStage1RecordsError,
    SequenceLengthSafetyError,
    Stage1DataValidationError,
    Stage2ActionResult,
    WriteStatus,
    sha256_file,
)
from src.data.imu_stage2_core import process_stage2_action
from src.data.imu_stage2_io import (
    FINGERPRINT_KEYS,
    Stage2AtomicWriteError,
    _absolute_lexical,
    _assert_no_reparse_components,
    _checked_action_path,
    _load_json_strict,
    _remove_tree_checked,
    _replace_checked,
    _validate_fingerprints,
    _write_staged_action,
    build_stage2_schema,
    load_stage2_schema,
    validate_existing_action,
    write_action_atomic,
    write_json_atomic,
)


HARD_SAFETY_LIMIT_T = 10_000
ROOT_FILES = frozenset({"schema.json", "manifest.csv", "processing.log"})
WARNING_REGISTRY = (
    "incomplete_sensors",
    "duplicate_timestamps_aggregated",
    "records_excluded",
    "angle_aggregation_degenerate",
    "quaternion_aggregation_degenerate",
    "no_usable_grid_cells",
)
SUMMARY_KEYS = (
    "summary_version",
    "source_stage1_manifest_sha256",
    "stage2_contract_sha256",
    "action_count",
    "data_status_counts",
    "imu_usable_action_count",
    "strict_5sensor_candidate_count",
    "total_grid_length",
    "valid_cell_count",
    "invalid_cell_count",
    "exact_hit_count",
    "interpolated_count",
    "all_sensor_valid_timestep_count",
    "all_sensor_invalid_timestep_count",
    "duplicate_group_count",
    "duplicate_extra_record_count",
    "duplicate_max_group_size",
    "excluded_record_count",
    "aggregation_failed_timestamp_count",
)
MANIFEST_COLUMNS = (
    "sample_id",
    "class_id",
    "class_name",
    "user_id",
    "action_id",
    "relative_action_path",
    "stage1_output_csv_relpath",
    "stage1_qc_relpath",
    "stage2_npz_relpath",
    "stage2_qc_relpath",
    "status",
    "write_status",
    "imu_usable",
    "error_message",
    "warning_codes",
    "sensor_mask",
    "usable_sensor_mask",
    "missing_sensors",
    "usable_sensors",
    "grid_length",
    "duration_ms",
    "stage1_action_end_ns",
    "grid_end_ns",
    "unrepresented_tail_ns",
    "first_usable_timestamp_ns",
    "last_usable_timestamp_ns",
    "valid_cell_count",
    "invalid_cell_count",
    "valid_cell_ratio",
    "all_sensor_valid_timestep_count",
    "all_sensor_invalid_timestep_count",
    "exact_hit_count",
    "interpolated_count",
    "invalid_count",
    "ll_valid_count",
    "rl_valid_count",
    "la_valid_count",
    "ra_valid_count",
    "c_valid_count",
    "duplicate_group_count",
    "duplicate_extra_record_count",
    "duplicate_max_group_size",
    "excluded_record_count",
    "aggregation_failed_timestamp_count",
    "stage1_output_csv_sha256",
    "stage1_qc_sha256",
    "stage1_manifest_row_sha256",
    "stage2_contract_sha256",
)
ACTION_DATA_ERRORS = (
    FileNotFoundError,
    UnicodeError,
    ValueError,
    OverflowError,
    pd.errors.ParserError,
    NoValidStage1RecordsError,
    Stage1DataValidationError,
    SequenceLengthSafetyError,
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_roots(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    input_lexical = _absolute_lexical(Path(input_root))
    output_lexical = _absolute_lexical(Path(output_root))
    _assert_no_reparse_components(input_lexical)
    _assert_no_reparse_components(output_lexical)
    if not input_lexical.is_dir():
        raise ValueError("Stage 1 input root must be an existing real directory")
    input_resolved = input_lexical.resolve(strict=True)
    output_resolved = output_lexical.resolve(strict=False)
    if _is_relative_to(input_resolved, output_resolved) or _is_relative_to(
        output_resolved, input_resolved
    ):
        raise ValueError("Stage 1 input and Stage 2 output roots overlap")
    return input_lexical, output_lexical


def _fingerprints(
    descriptor: Stage1ArtifactDescriptor,
    stage2_contract_sha256: str,
) -> dict[str, str]:
    return {
        "stage1_output_csv_sha256": sha256_file(descriptor.output_csv_path),
        "stage1_qc_sha256": sha256_file(descriptor.qc_path),
        "stage1_manifest_row_sha256": descriptor.manifest_row_sha256,
        "stage2_contract_sha256": stage2_contract_sha256,
    }


def _expected_action_directories(
    output_root: Path,
    descriptors: Sequence[Stage1ArtifactDescriptor],
) -> dict[Path, Stage1ArtifactDescriptor]:
    return {
        _absolute_lexical(output_root / descriptor.action_relative_path): descriptor
        for descriptor in descriptors
    }


def _validate_action_paths(
    output_root: Path,
    descriptors: Sequence[Stage1ArtifactDescriptor],
) -> None:
    action_paths: list[Path] = []
    for descriptor in descriptors:
        relative = descriptor.action_relative_path
        if (
            relative.is_absolute()
            or not relative.parts
            or relative == Path(".")
            or ".." in relative.parts
        ):
            raise ValueError(
                f"Invalid Stage 2 action path: {descriptor.sample_id}"
            )
        action_paths.append(_absolute_lexical(output_root / relative))
    if len(set(action_paths)) != len(action_paths):
        raise ValueError("Stage 2 action paths must be unique")
    action_set = set(action_paths)
    for action_path in action_paths:
        for parent in action_path.parents:
            if parent == output_root:
                break
            if parent in action_set:
                raise ValueError("Stage 2 action paths must not be nested")


def _validate_managed_tree(
    output_root: Path,
    descriptors: Sequence[Stage1ArtifactDescriptor],
) -> None:
    expected = _expected_action_directories(output_root, descriptors)
    allowed_directories = {output_root}
    for action_directory in expected:
        current = action_directory
        while current != output_root:
            allowed_directories.add(current)
            current = current.parent
    for path in output_root.rglob("*"):
        _assert_no_reparse_components(_absolute_lexical(path))
        if path.name.startswith((".staging-", ".backup-", ".tmp-")):
            raise ValueError(f"Stage 2 output contains transaction residue: {path}")
        if path.is_dir():
            if path not in allowed_directories:
                raise ValueError(f"Stage 2 output contains unknown directory: {path}")
            continue
        if path.parent == output_root and path.name in ROOT_FILES:
            continue
        if path.parent in expected and path.name in {"imu_stage2.npz", "qc.json"}:
            continue
        raise ValueError(f"Stage 2 output contains unknown managed file: {path}")


def preflight_run_mode(
    output_root: Path,
    *,
    resume: bool,
    overwrite: bool,
    dry_run: bool,
    hard_safety_limit_t: int,
    schema: Mapping[str, object],
    descriptors: Sequence[Stage1ArtifactDescriptor],
    source_stage1_manifest_sha256: str,
) -> dict[str, Stage2ActionResult | None]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if hard_safety_limit_t <= 0:
        raise ValueError("--hard-safety-limit-t must be positive")
    if hard_safety_limit_t != HARD_SAFETY_LIMIT_T:
        raise ValueError("hard_safety_limit_t is incompatible with imu-stage2-v1")
    output_root = _absolute_lexical(output_root)
    _validate_action_paths(output_root, descriptors)
    if not resume and not overwrite:
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError("Fresh Stage 2 output root must be missing or empty")
        return {}
    if not output_root.is_dir():
        raise ValueError("Resume/overwrite requires an existing Stage 2 output root")
    _assert_no_reparse_components(output_root)
    existing_schema = load_stage2_schema(output_root / "schema.json")
    if existing_schema["contract_sha256"] != schema["contract_sha256"]:
        raise ValueError("Existing Stage 2 contract is incompatible")
    if existing_schema["contract"]["hard_safety_limit_t"] != hard_safety_limit_t:
        raise ValueError("Existing Stage 2 safety limit is incompatible")
    if resume and (
        existing_schema["provenance"]["source_stage1_manifest_sha256"]
        != source_stage1_manifest_sha256
    ):
        raise ValueError("Resume source Stage 1 manifest fingerprint mismatch")
    _validate_managed_tree(output_root, descriptors)
    verified: dict[str, Stage2ActionResult | None] = {}
    for action_directory, descriptor in _expected_action_directories(
        output_root, descriptors
    ).items():
        if not action_directory.exists():
            continue
        managed_files = {path.name for path in action_directory.iterdir()}
        expected_fingerprints = _fingerprints(
            descriptor, str(schema["contract_sha256"])
        )
        if managed_files == {"qc.json"}:
            _validate_qc_only(
                action_directory,
                descriptor,
                expected_fingerprints,
                require_current_fingerprints=resume,
            )
            verified[descriptor.sample_id] = None
            continue
        if managed_files != {"imu_stage2.npz", "qc.json"}:
            raise ValueError(
                f"Existing Stage 2 action is not a complete artifact: {descriptor.sample_id}"
            )
        if resume:
            verified[descriptor.sample_id] = validate_existing_action(
                action_directory, expected_fingerprints
            )
        else:
            qc = json.loads((action_directory / "qc.json").read_text(encoding="utf-8"))
            old_fingerprints = {
                key: str(qc.get(key, "")) for key in FINGERPRINT_KEYS
            }
            validate_existing_action(action_directory, old_fingerprints)
            if (
                _validate_fingerprints(old_fingerprints)[
                    "stage2_contract_sha256"
                ]
                != schema["contract_sha256"]
            ):
                raise ValueError("Existing Stage 2 action contract is incompatible")
    return verified


def _ordered_warning_codes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    known = [code for code in WARNING_REGISTRY if code in value]
    unknown = sorted(
        code
        for code in value
        if isinstance(code, str) and code not in WARNING_REGISTRY
    )
    return [*known, *unknown]


def _sensor_names(mask: object) -> str:
    if not isinstance(mask, list) or len(mask) != len(SENSOR_ORDER):
        return ""
    return ";".join(
        sensor for sensor, enabled in zip(SENSOR_ORDER, mask, strict=True) if enabled
    )


def _manifest_row(
    descriptor: Stage1ArtifactDescriptor,
    *,
    status: DataStatus,
    write_status: WriteStatus,
    fingerprints: Mapping[str, str],
    result: Stage2ActionResult | None,
    error_message: str = "",
    qc_published: bool = True,
) -> dict[str, object]:
    source = descriptor.manifest_row
    relative = descriptor.action_relative_path
    row: dict[str, object] = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "sample_id": descriptor.sample_id,
            "class_id": source["class_id"],
            "class_name": source["class_name"],
            "user_id": source["user_id"],
            "action_id": source["action_id"],
            "relative_action_path": relative.as_posix(),
            "stage1_output_csv_relpath": source["output_csv"],
            "stage1_qc_relpath": (relative / "qc.json").as_posix(),
            "stage2_npz_relpath": (
                (relative / "imu_stage2.npz").as_posix() if result is not None else ""
            ),
            "stage2_qc_relpath": (
                (relative / "qc.json").as_posix() if qc_published else ""
            ),
            "status": status.value,
            "write_status": write_status.value,
            "imu_usable": "" if result is None else str(result.imu_usable).lower(),
            "error_message": error_message,
            **fingerprints,
        }
    )
    if result is None:
        return row
    qc = result.qc
    sensor_mask = result.sensor_mask.tolist()
    usable_sensor_mask = result.usable_sensor_mask.tolist()
    per_sensor = qc.get("per_sensor_valid_count", {})
    row.update(
        {
            "warning_codes": ";".join(_ordered_warning_codes(qc.get("warning_codes"))),
            "sensor_mask": _sensor_names(sensor_mask),
            "usable_sensor_mask": _sensor_names(usable_sensor_mask),
            "missing_sensors": ";".join(
                sensor
                for sensor, present in zip(SENSOR_ORDER, sensor_mask, strict=True)
                if not present
            ),
            "usable_sensors": _sensor_names(usable_sensor_mask),
            "duration_ms": int(result.timestamps_ms[-1]),
        }
    )
    for key in (
        "grid_length",
        "stage1_action_end_ns",
        "grid_end_ns",
        "unrepresented_tail_ns",
        "first_usable_timestamp_ns",
        "last_usable_timestamp_ns",
        "valid_cell_count",
        "invalid_cell_count",
        "valid_cell_ratio",
        "all_sensor_valid_timestep_count",
        "all_sensor_invalid_timestep_count",
        "exact_hit_count",
        "interpolated_count",
        "invalid_count",
        "duplicate_group_count",
        "duplicate_extra_record_count",
        "duplicate_max_group_size",
        "excluded_record_count",
        "aggregation_failed_timestamp_count",
    ):
        row[key] = qc.get(key, "")
    if isinstance(per_sensor, Mapping):
        for sensor in SENSOR_ORDER:
            row[f"{sensor.lower()}_valid_count"] = per_sensor.get(sensor, "")
    return row


def build_manifest(rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=MANIFEST_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(MANIFEST_COLUMNS):
            raise ValueError("Stage 2 manifest row columns do not match contract")
        writer.writerow(row)
    return stream.getvalue()


def _normalized_manifest_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    return [
        {
            column: "" if row[column] is None else str(row[column])
            for column in MANIFEST_COLUMNS
        }
        for row in rows
    ]


def _write_manifest_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    text = build_manifest(rows)
    temporary = path.parent / f".tmp-{path.name}-{uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8-sig", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(MANIFEST_COLUMNS):
                raise ValueError("Published Stage 2 manifest columns are invalid")
            reopened = list(reader)
        if len(reopened) != len(rows):
            raise ValueError("Published Stage 2 manifest row count is invalid")
        if reopened != _normalized_manifest_rows(rows):
            raise ValueError("Published Stage 2 manifest content is invalid")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _qc_metadata(
    descriptor: Stage1ArtifactDescriptor,
) -> dict[str, object]:
    row = descriptor.manifest_row
    relative = descriptor.action_relative_path
    return {
        "class_id": int(row["class_id"]) if row["class_id"] else None,
        "class_name": row["class_name"] or None,
        "user_id": row["user_id"] or None,
        "action_id": row["action_id"] or None,
        "relative_action_path": relative.as_posix(),
        "stage1_output_csv_relpath": row["output_csv"],
        "stage1_qc_relpath": (relative / "qc.json").as_posix(),
        "stage2_npz_relpath": (relative / "imu_stage2.npz").as_posix(),
        "stage2_qc_relpath": (relative / "qc.json").as_posix(),
    }


def _failed_qc(
    descriptor: Stage1ArtifactDescriptor,
    fingerprints: Mapping[str, str],
    error: BaseException,
) -> dict[str, object]:
    return {
        **_qc_metadata(descriptor),
        "sample_id": descriptor.sample_id,
        "status": DataStatus.FAILED.value,
        "write_status": WriteStatus.QC_ONLY.value,
        "imu_usable": False,
        "warning_codes": [],
        "error_type": type(error).__name__,
        "error_message": str(error),
        **fingerprints,
    }


def _validate_qc_only(
    action_directory: Path,
    descriptor: Stage1ArtifactDescriptor,
    expected_fingerprints: Mapping[str, str],
    *,
    require_current_fingerprints: bool,
) -> dict[str, object]:
    if {path.name for path in action_directory.iterdir()} != {"qc.json"}:
        raise ValueError("Failed Stage 2 action must contain only qc.json")
    qc = _load_json_strict(action_directory / "qc.json")
    if qc.get("sample_id") != descriptor.sample_id:
        raise ValueError("Failed Stage 2 QC sample identity mismatch")
    if qc.get("status") != DataStatus.FAILED.value:
        raise ValueError("QC-only Stage 2 action must have status=failed")
    if qc.get("write_status") != WriteStatus.QC_ONLY.value:
        raise ValueError("QC-only Stage 2 action must have write_status=qc_only")
    if qc.get("imu_usable") is not False:
        raise ValueError("QC-only Stage 2 action must have imu_usable=false")
    actual_fingerprints = _validate_fingerprints(
        {key: qc.get(key) for key in FINGERPRINT_KEYS}
    )
    expected = _validate_fingerprints(expected_fingerprints)
    if actual_fingerprints["stage2_contract_sha256"] != expected[
        "stage2_contract_sha256"
    ]:
        raise ValueError("QC-only Stage 2 contract fingerprint mismatch")
    if require_current_fingerprints and actual_fingerprints != expected:
        raise ValueError("QC-only Stage 2 source fingerprint mismatch")
    return qc


def _transaction_error(
    message: str,
    final: Path,
    backup: Path,
    *errors: BaseException,
) -> Stage2AtomicWriteError:
    details = "; ".join(str(error) for error in errors)
    return Stage2AtomicWriteError(
        f"{message}; final={final}; backup={backup}; errors={details}"
    )


def _install_staged_directory(
    root: Path,
    final: Path,
    staging: Path,
    backup: Path,
    validate_original: Callable[[Path], None] | None,
) -> None:
    if not final.exists():
        try:
            _replace_checked(root, staging, final)
        except BaseException as install_error:
            try:
                _remove_tree_checked(root, staging, ".staging-")
            except BaseException as cleanup_error:
                raise _transaction_error(
                    "Stage 2 install failed and staging cleanup failed",
                    final,
                    backup,
                    install_error,
                    cleanup_error,
                ) from install_error
            raise
        return
    if validate_original is None:
        raise AssertionError("Existing Stage 2 action validator is required")
    validate_original(final)
    _replace_checked(root, final, backup)
    try:
        _replace_checked(root, staging, final)
    except BaseException as install_error:
        try:
            _replace_checked(root, backup, final)
            _remove_tree_checked(root, staging, ".staging-")
        except BaseException as restore_error:
            raise _transaction_error(
                "Stage 2 replacement failed and original restore failed",
                final,
                backup,
                install_error,
                restore_error,
            ) from install_error
        raise
    try:
        _remove_tree_checked(root, backup, ".backup-")
    except BaseException as cleanup_error:
        try:
            validate_original(backup)
        except BaseException as validation_error:
            raise _transaction_error(
                "Stage 2 backup cleanup failed and backup validation failed",
                final,
                backup,
                cleanup_error,
                validation_error,
            ) from cleanup_error
        try:
            _replace_checked(root, final, staging)
            _replace_checked(root, backup, final)
            validate_original(final)
            _remove_tree_checked(root, staging, ".staging-")
        except BaseException as restore_error:
            raise _transaction_error(
                "Stage 2 backup cleanup failed and original restore failed",
                final,
                backup,
                cleanup_error,
                restore_error,
            ) from cleanup_error
        raise _transaction_error(
            "Stage 2 backup cleanup failed; original action restored",
            final,
            backup,
            cleanup_error,
        ) from cleanup_error


def _staged_paths(
    output_root: Path,
    action_relative_path: Path,
) -> tuple[Path, Path, Path, Path]:
    root, final = _checked_action_path(output_root, action_relative_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(final.parent)
    token = uuid4().hex
    staging = final.parent / f".staging-{final.name}-{token}"
    backup = final.parent / f".backup-{final.name}-{token}"
    _assert_no_reparse_components(staging)
    _assert_no_reparse_components(backup)
    return root, final, staging, backup


def _original_action_validator(
    action_directory: Path,
    descriptor: Stage1ArtifactDescriptor,
) -> Callable[[Path], None]:
    managed_files = {path.name for path in action_directory.iterdir()}
    qc = _load_json_strict(action_directory / "qc.json")
    fingerprints = _validate_fingerprints(
        {key: qc.get(key) for key in FINGERPRINT_KEYS}
    )
    if managed_files == {"qc.json"}:
        def validate_qc_only(path: Path) -> None:
            _validate_qc_only(
                path,
                descriptor,
                fingerprints,
                require_current_fingerprints=True,
            )

        validate_qc_only(action_directory)
        return validate_qc_only
    if managed_files == {"imu_stage2.npz", "qc.json"}:
        def validate_tensor(path: Path) -> None:
            validate_existing_action(path, fingerprints)

        validate_tensor(action_directory)
        return validate_tensor
    raise ValueError("Existing Stage 2 action has an invalid file set")


def _publish_failed_qc(
    output_root: Path,
    descriptor: Stage1ArtifactDescriptor,
    payload: Mapping[str, object],
) -> None:
    root, final, staging, backup = _staged_paths(
        output_root, descriptor.action_relative_path
    )
    try:
        staging.mkdir()
        write_json_atomic(staging / "qc.json", payload)
        _validate_qc_only(
            staging,
            descriptor,
            {key: str(payload[key]) for key in FINGERPRINT_KEYS},
            require_current_fingerprints=True,
        )
        validate_original = (
            _original_action_validator(final, descriptor)
            if final.exists()
            else None
        )
        _install_staged_directory(
            root, final, staging, backup, validate_original
        )
    finally:
        if staging.exists():
            _remove_tree_checked(root, staging, ".staging-")


def _publish_result_replacing_qc_only(
    output_root: Path,
    descriptor: Stage1ArtifactDescriptor,
    result: Stage2ActionResult,
    fingerprints: Mapping[str, str],
) -> None:
    root, final, staging, backup = _staged_paths(
        output_root, descriptor.action_relative_path
    )
    try:
        staging.mkdir()
        _write_staged_action(
            staging,
            result,
            fingerprints,
            _qc_metadata(descriptor),
        )
        validate_original = _original_action_validator(final, descriptor)
        _install_staged_directory(
            root, final, staging, backup, validate_original
        )
    finally:
        if staging.exists():
            _remove_tree_checked(root, staging, ".staging-")


def _summary(
    rows: Sequence[Mapping[str, object]],
    *,
    source_stage1_manifest_sha256: str,
    stage2_contract_sha256: str,
) -> dict[str, object]:
    status_counts = {
        status.value: sum(row["status"] == status.value for row in rows)
        for status in DataStatus
    }

    def integer_sum(key: str) -> int:
        return sum(int(row[key]) for row in rows if row[key] != "")

    summary: dict[str, object] = {
        "summary_version": "imu-stage2-summary-v1",
        "source_stage1_manifest_sha256": source_stage1_manifest_sha256,
        "stage2_contract_sha256": stage2_contract_sha256,
        "action_count": len(rows),
        "data_status_counts": status_counts,
        "imu_usable_action_count": sum(row["imu_usable"] == "true" for row in rows),
        "strict_5sensor_candidate_count": sum(
            row["sensor_mask"] == ";".join(SENSOR_ORDER)
            and row["usable_sensor_mask"] == ";".join(SENSOR_ORDER)
            for row in rows
        ),
        "total_grid_length": integer_sum("grid_length"),
        "valid_cell_count": integer_sum("valid_cell_count"),
        "invalid_cell_count": integer_sum("invalid_cell_count"),
        "exact_hit_count": integer_sum("exact_hit_count"),
        "interpolated_count": integer_sum("interpolated_count"),
        "all_sensor_valid_timestep_count": integer_sum(
            "all_sensor_valid_timestep_count"
        ),
        "all_sensor_invalid_timestep_count": integer_sum(
            "all_sensor_invalid_timestep_count"
        ),
        "duplicate_group_count": integer_sum("duplicate_group_count"),
        "duplicate_extra_record_count": integer_sum(
            "duplicate_extra_record_count"
        ),
        "duplicate_max_group_size": max(
            (int(row["duplicate_max_group_size"]) for row in rows if row["duplicate_max_group_size"] != ""),
            default=0,
        ),
        "excluded_record_count": integer_sum("excluded_record_count"),
        "aggregation_failed_timestamp_count": integer_sum(
            "aggregation_failed_timestamp_count"
        ),
    }
    if tuple(summary) != SUMMARY_KEYS:
        raise AssertionError("Stage 2 summary keys are not in contract order")
    return summary


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build IMU Stage 2 v1 artifacts")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--hard-safety-limit-t", type=int, default=HARD_SAFETY_LIMIT_T
    )
    parser.add_argument(
        "--summary-format", choices=("human", "json"), default="human"
    )
    return parser


def _emit_summary(summary: Mapping[str, object], summary_format: str) -> None:
    if summary_format == "json":
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    else:
        counts = summary["data_status_counts"]
        print(
            f"Stage 2 actions={summary['action_count']} "
            f"failed={counts[DataStatus.FAILED.value]} "
            f"imu_usable={summary['imu_usable_action_count']}"
        )


def _process_actions(
    descriptors: Sequence[Stage1ArtifactDescriptor],
    *,
    output_root: Path,
    schema: Mapping[str, object],
    verified_resume: Mapping[str, Stage2ActionResult | None],
    dry_run: bool,
    overwrite: bool,
    hard_safety_limit_t: int,
    log_handle,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    contract_hash = str(schema["contract_sha256"])
    for descriptor in descriptors:
        fingerprints = _fingerprints(descriptor, contract_hash)
        if (
            descriptor.sample_id in verified_resume
            and verified_resume[descriptor.sample_id] is not None
        ):
            result = verified_resume[descriptor.sample_id]
            if result is None:
                raise AssertionError("Verified Stage 2 result unexpectedly missing")
            rows.append(
                _manifest_row(
                    descriptor,
                    status=result.status,
                    write_status=WriteStatus.SKIPPED_EXISTING,
                    fingerprints=fingerprints,
                    result=result,
                )
            )
            continue
        try:
            stage1_action = load_stage1_action(descriptor)
            result = process_stage2_action(
                stage1_action,
                hard_safety_limit_t=hard_safety_limit_t,
            )
        except ACTION_DATA_ERRORS as error:
            if not dry_run:
                _publish_failed_qc(
                    output_root,
                    descriptor,
                    _failed_qc(descriptor, fingerprints, error),
                )
            rows.append(
                _manifest_row(
                    descriptor,
                    status=DataStatus.FAILED,
                    write_status=(
                        WriteStatus.NOT_WRITTEN
                        if dry_run
                        else WriteStatus.QC_ONLY
                    ),
                    fingerprints=fingerprints,
                    result=None,
                    error_message=str(error),
                    qc_published=not dry_run,
                )
            )
            if log_handle is not None:
                log_handle.write(
                    f"sample_id={descriptor.sample_id} status=failed "
                    f"error_type={type(error).__name__}\n"
                )
                log_handle.flush()
            continue
        if not dry_run:
            if (
                descriptor.sample_id in verified_resume
                and verified_resume[descriptor.sample_id] is None
            ):
                _publish_result_replacing_qc_only(
                    output_root,
                    descriptor,
                    result,
                    fingerprints,
                )
            else:
                write_action_atomic(
                    output_root,
                    descriptor.action_relative_path,
                    result,
                    fingerprints,
                    overwrite=overwrite,
                    qc_metadata=_qc_metadata(descriptor),
                )
        rows.append(
            _manifest_row(
                descriptor,
                status=result.status,
                write_status=(
                    WriteStatus.NOT_WRITTEN if dry_run else WriteStatus.WRITTEN
                ),
                fingerprints=fingerprints,
                result=result,
                qc_published=not dry_run,
            )
        )
        if log_handle is not None:
            log_handle.write(
                f"sample_id={descriptor.sample_id} status={result.status.value}\n"
            )
            log_handle.flush()
    return rows


def _validate_final_output(
    output_root: Path,
    descriptors: Sequence[Stage1ArtifactDescriptor],
    rows: Sequence[Mapping[str, object]],
    schema: Mapping[str, object],
) -> None:
    loaded_schema = load_stage2_schema(output_root / "schema.json")
    if loaded_schema != schema:
        raise ValueError("Published Stage 2 schema does not match this run")
    _validate_managed_tree(output_root, descriptors)
    with (output_root / "manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(MANIFEST_COLUMNS):
            raise ValueError("Published Stage 2 manifest columns are invalid")
        published_rows = list(reader)
    if published_rows != _normalized_manifest_rows(rows):
        raise ValueError("Published Stage 2 manifest content is invalid")
    if len(rows) != len(descriptors):
        raise ValueError("Published Stage 2 action count is invalid")
    descriptor_by_id = {descriptor.sample_id: descriptor for descriptor in descriptors}
    if len(descriptor_by_id) != len(descriptors):
        raise ValueError("Stage 2 descriptors contain duplicate sample IDs")
    for row in published_rows:
        sample_id = row["sample_id"]
        if sample_id not in descriptor_by_id:
            raise ValueError("Published Stage 2 manifest has unknown sample ID")
        descriptor = descriptor_by_id[sample_id]
        action_directory = output_root / descriptor.action_relative_path
        fingerprints = _fingerprints(
            descriptor, str(schema["contract_sha256"])
        )
        if row["status"] == DataStatus.FAILED.value:
            if row["write_status"] != WriteStatus.QC_ONLY.value:
                raise ValueError("Failed Stage 2 manifest row is not qc_only")
            _validate_qc_only(
                action_directory,
                descriptor,
                fingerprints,
                require_current_fingerprints=True,
            )
            if row["stage2_npz_relpath"] or not row["stage2_qc_relpath"]:
                raise ValueError("Failed Stage 2 manifest paths are inconsistent")
            continue
        if row["write_status"] not in {
            WriteStatus.WRITTEN.value,
            WriteStatus.SKIPPED_EXISTING.value,
        }:
            raise ValueError("Tensor Stage 2 manifest write status is invalid")
        result = validate_existing_action(action_directory, fingerprints)
        if row["status"] != result.status.value:
            raise ValueError("Stage 2 manifest status disagrees with action QC")
        if not row["stage2_npz_relpath"] or not row["stage2_qc_relpath"]:
            raise ValueError("Tensor Stage 2 manifest paths are missing")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        input_root, output_root = validate_roots(args.input_root, args.output_root)
        descriptors = discover_stage1_artifacts(input_root)
        source_manifest_hash = sha256_file(input_root / "manifest.csv")
        schema = build_stage2_schema(
            {
                "implementation_version": "imu-stage2-v1",
                "generator_script": "scripts/preprocess_imu_stage2.py",
                "git_commit": _git_commit(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_stage1_manifest": "manifest.csv",
                "source_stage1_manifest_sha256": source_manifest_hash,
            }
        )
        verified_resume = preflight_run_mode(
            output_root,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            hard_safety_limit_t=args.hard_safety_limit_t,
            schema=schema,
            descriptors=descriptors,
            source_stage1_manifest_sha256=source_manifest_hash,
        )
    except SystemExit as error:
        return int(error.code)
    except (OSError, ValueError, zipfile.BadZipFile, pd.errors.ParserError) as error:
        print(f"Stage 2 global preflight error: {error}", file=sys.stderr)
        return 2

    log_handle = None
    try:
        if not args.dry_run:
            output_root.mkdir(parents=True, exist_ok=True)
            _assert_no_reparse_components(output_root)
            write_json_atomic(output_root / "schema.json", schema)
            load_stage2_schema(output_root / "schema.json")
            log_handle = (output_root / "processing.log").open(
                "a", encoding="utf-8", newline="\n"
            )
            log_handle.write("run_started=true\n")
            log_handle.flush()
        rows = _process_actions(
            descriptors,
            output_root=output_root,
            schema=schema,
            verified_resume=verified_resume,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            hard_safety_limit_t=args.hard_safety_limit_t,
            log_handle=log_handle,
        )
        summary = _summary(
            rows,
            source_stage1_manifest_sha256=source_manifest_hash,
            stage2_contract_sha256=str(schema["contract_sha256"]),
        )
        if not args.dry_run:
            _write_manifest_atomic(output_root / "manifest.csv", rows)
            _validate_final_output(output_root, descriptors, rows, schema)
            if log_handle is not None:
                log_handle.write("closed_normally=true\n")
                log_handle.flush()
        _emit_summary(summary, args.summary_format)
        return 1 if summary["data_status_counts"][DataStatus.FAILED.value] else 0
    except KeyboardInterrupt:
        print("Stage 2 interrupted", file=sys.stderr)
        return 2
    except BaseException as error:
        print(f"Stage 2 global processing error: {error}", file=sys.stderr)
        return 2
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
