from __future__ import annotations

import json
import os
import re
import shutil
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.data.imu_stage2_contracts import (
    FEATURE_ORDER,
    SENSOR_ORDER,
    DataStatus,
    Stage2ActionResult,
    WriteStatus,
    contract_sha256,
)


NPZ_KEYS = frozenset(
    {"values", "sensor_mask", "valid_mask", "timestamps_ms"}
)
PROVENANCE_KEYS = frozenset(
    {
        "implementation_version",
        "generator_script",
        "git_commit",
        "created_at",
        "source_stage1_manifest",
        "source_stage1_manifest_sha256",
    }
)
FINGERPRINT_KEYS = frozenset(
    {
        "stage1_output_csv_sha256",
        "stage1_qc_sha256",
        "stage1_manifest_row_sha256",
        "stage2_contract_sha256",
    }
)
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
TENSOR_DATA_STATUSES = frozenset(
    {
        DataStatus.SUCCESS,
        DataStatus.SUCCESS_WITH_WARNINGS,
        DataStatus.INCOMPLETE_SENSORS,
        DataStatus.NO_USABLE_GRID_CELLS,
    }
)


class Stage2AtomicWriteError(RuntimeError):
    """A staged action write could not be restored to a clean state."""


def _normalize_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _validate_provenance(provenance: Mapping[str, object]) -> dict[str, object]:
    if set(provenance) != PROVENANCE_KEYS:
        raise ValueError("Stage 2 provenance keys do not match contract")
    if provenance["source_stage1_manifest"] != "manifest.csv":
        raise ValueError("source_stage1_manifest must be manifest.csv")
    normalized = dict(provenance)
    normalized["source_stage1_manifest_sha256"] = _normalize_sha256(
        provenance["source_stage1_manifest_sha256"],
        "source_stage1_manifest_sha256",
    )
    return normalized


def _stage2_contract() -> dict[str, object]:
    return {
        "schema_version": "imu-stage2-v1",
        "stage1_contract_version": "imu-stage1-v1",
        "grid_frequency_hz": 10,
        "grid_step_ns": 100_000_000,
        "max_interpolation_gap_ns": 300_000_000,
        "hard_safety_limit_t": 10_000,
        "sensor_order": list(SENSOR_ORDER),
        "feature_order": list(FEATURE_ORDER),
        "values_dtype": "float32",
        "sensor_mask_dtype": "bool",
        "valid_mask_dtype": "bool",
        "timestamps_dtype": "int64",
        "invalid_value": "NaN",
        "standardized": False,
        "angle_range": "[-180, 180)",
        "time_key": "relative_time_ns",
        "duplicate_timestamp_policy": "feature_aware_aggregation",
        "interpolation_policy": "feature_aware",
        "boundary_extrapolation": False,
        "sequence_storage": "variable_length_unpadded",
        "container": "uncompressed_npz",
    }


def build_stage2_schema(
    provenance: Mapping[str, object],
) -> dict[str, object]:
    normalized_provenance = _validate_provenance(provenance)
    contract = _stage2_contract()
    return {
        "contract": contract,
        "provenance": normalized_provenance,
        "contract_sha256": contract_sha256(contract),
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    temporary = path.parent / f".tmp-{path.name}-{uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("r", encoding="utf-8") as handle:
            json.load(handle)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_stage2_schema(path: Path) -> dict[str, object]:
    payload = _load_json_strict(Path(path))
    if not isinstance(payload, dict) or set(payload) != {
        "contract",
        "provenance",
        "contract_sha256",
    }:
        raise ValueError("Stage 2 schema top-level keys do not match contract")
    contract = payload["contract"]
    provenance = payload["provenance"]
    if not isinstance(contract, dict):
        raise ValueError("Stage 2 schema contract must be an object")
    if not isinstance(provenance, dict):
        raise ValueError("Stage 2 provenance must be an object")
    normalized_provenance = _validate_provenance(provenance)
    expected_hash = contract_sha256(contract)
    actual_hash = _normalize_sha256(
        payload["contract_sha256"], "contract_sha256"
    )
    if actual_hash != expected_hash:
        raise ValueError("Stage 2 contract_sha256 does not match contract")
    if contract != _stage2_contract():
        raise ValueError("Stage 2 schema contract is incompatible")
    payload["provenance"] = normalized_provenance
    payload["contract_sha256"] = actual_hash
    return payload


def load_and_validate_npz(
    path: Path,
    *,
    sample_id: str,
    status: DataStatus,
    qc: Mapping[str, object],
) -> Stage2ActionResult:
    path = Path(path)
    with zipfile.ZipFile(path) as container:
        members = container.infolist()
        expected_members = {f"{key}.npy" for key in NPZ_KEYS}
        member_names = [member.filename for member in members]
        if (
            len(member_names) != len(expected_members)
            or set(member_names) != expected_members
        ):
            raise ValueError("Stage 2 NPZ members do not match exact key contract")
        if any(member.compress_type != zipfile.ZIP_STORED for member in members):
            raise ValueError("Stage 2 NPZ must use an uncompressed ZIP container")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != NPZ_KEYS:
            raise ValueError("Stage 2 NPZ keys do not match contract")
        try:
            values = archive["values"].copy()
            sensor_mask = archive["sensor_mask"].copy()
            valid_mask = archive["valid_mask"].copy()
            timestamps_ms = archive["timestamps_ms"].copy()
        except ValueError as error:
            raise ValueError("Object arrays are forbidden in Stage 2 NPZ") from error
    result = Stage2ActionResult(
        sample_id=sample_id,
        values=values,
        sensor_mask=sensor_mask,
        valid_mask=valid_mask,
        timestamps_ms=timestamps_ms,
        qc=dict(qc),
        status=status,
    )
    result.validate()
    return result


def _load_json_strict(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("Stage 2 QC must be a JSON object")
    return payload


def _validate_fingerprints(fingerprints: Mapping[str, str]) -> dict[str, str]:
    if set(fingerprints) != FINGERPRINT_KEYS:
        raise ValueError("Stage 2 action fingerprint keys do not match contract")
    return {
        key: _normalize_sha256(fingerprints[key], key)
        for key in FINGERPRINT_KEYS
    }


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Stage 2 QC {name} must be a non-negative integer")
    return value


def _action_qc_payload(
    result: Stage2ActionResult,
    fingerprints: Mapping[str, str],
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    result.validate()
    payload = dict(metadata or {})
    payload.update(result.qc)
    valid_cell_count = int(result.valid_mask.sum())
    invalid_cell_count = int(result.valid_mask.size - valid_cell_count)
    payload.update(
        {
            "sample_id": result.sample_id,
            "status": result.status.value,
            "write_status": WriteStatus.WRITTEN.value,
            "imu_usable": result.imu_usable,
            "sensor_mask": result.sensor_mask.tolist(),
            "usable_sensor_mask": result.usable_sensor_mask.tolist(),
            "grid_length": len(result.timestamps_ms),
            "valid_cell_count": valid_cell_count,
            "invalid_cell_count": invalid_cell_count,
            "invalid_count": invalid_cell_count,
            **_validate_fingerprints(fingerprints),
        }
    )
    exact_hit_count = _nonnegative_int(
        payload.get("exact_hit_count"), "exact_hit_count"
    )
    interpolated_count = _nonnegative_int(
        payload.get("interpolated_count"), "interpolated_count"
    )
    if exact_hit_count + interpolated_count != valid_cell_count:
        raise ValueError("Stage 2 QC count mismatch before publication")
    _validate_qc_against_result(payload, result, fingerprints)
    return payload


def _expected_tensor_qc(result: Stage2ActionResult) -> dict[str, object]:
    valid_mask = result.valid_mask
    usable_sensor_mask = result.usable_sensor_mask
    temporal_valid_mask = valid_mask.any(axis=1)
    usable_timestamps = result.timestamps_ms[temporal_valid_mask]
    valid_cell_count = int(valid_mask.sum())
    invalid_cell_count = int(valid_mask.size - valid_cell_count)
    expected: dict[str, object] = {
        "sample_id": result.sample_id,
        "status": result.status.value,
        "write_status": WriteStatus.WRITTEN.value,
        "imu_usable": result.imu_usable,
        "sensor_mask": result.sensor_mask.tolist(),
        "usable_sensor_mask": usable_sensor_mask.tolist(),
        "missing_sensors": [
            sensor
            for sensor, present in zip(
                SENSOR_ORDER, result.sensor_mask, strict=True
            )
            if not present
        ],
        "usable_sensors": [
            sensor
            for sensor, usable in zip(
                SENSOR_ORDER, usable_sensor_mask, strict=True
            )
            if usable
        ],
        "grid_length": len(result.timestamps_ms),
        "grid_end_ns": int(result.timestamps_ms[-1]) * 1_000_000,
        "first_usable_timestamp_ns": (
            int(usable_timestamps[0]) * 1_000_000
            if len(usable_timestamps)
            else None
        ),
        "last_usable_timestamp_ns": (
            int(usable_timestamps[-1]) * 1_000_000
            if len(usable_timestamps)
            else None
        ),
        "valid_cell_count": valid_cell_count,
        "invalid_cell_count": invalid_cell_count,
        "valid_cell_ratio": valid_cell_count / valid_mask.size,
        "all_sensor_valid_timestep_count": int(valid_mask.all(axis=1).sum()),
        "all_sensor_invalid_timestep_count": int(
            (~valid_mask.any(axis=1)).sum()
        ),
        "invalid_count": invalid_cell_count,
        "per_sensor_valid_count": {
            sensor: int(valid_mask[:, index].sum())
            for index, sensor in enumerate(SENSOR_ORDER)
        },
    }
    return expected


def _expected_tensor_status(
    result: Stage2ActionResult,
    warning_codes: object,
) -> DataStatus:
    if not isinstance(warning_codes, list) or not all(
        isinstance(code, str) and code for code in warning_codes
    ):
        raise ValueError("Stage 2 QC warning_codes must be a list of strings")
    if not result.imu_usable:
        return DataStatus.NO_USABLE_GRID_CELLS
    if not bool(result.sensor_mask.all()):
        return DataStatus.INCOMPLETE_SENSORS
    if warning_codes:
        return DataStatus.SUCCESS_WITH_WARNINGS
    return DataStatus.SUCCESS


def _validate_qc_against_result(
    qc: Mapping[str, object],
    result: Stage2ActionResult,
    expected_fingerprints: Mapping[str, str],
) -> None:
    normalized_expected = _validate_fingerprints(expected_fingerprints)
    actual_fingerprints = _validate_fingerprints(
        {key: qc.get(key) for key in FINGERPRINT_KEYS}
    )
    for key, expected in normalized_expected.items():
        if actual_fingerprints[key] != expected:
            raise ValueError(f"Stage 2 source fingerprint mismatch: {key}")
    if result.status not in TENSOR_DATA_STATUSES:
        raise ValueError("Stage 2 tensor-bearing action has invalid data status")
    if qc.get("write_status") != WriteStatus.WRITTEN.value:
        raise ValueError("Stage 2 tensor-bearing action must have write_status=written")
    expected_status = _expected_tensor_status(result, qc.get("warning_codes"))
    if result.status is not expected_status:
        raise ValueError("Stage 2 tensor-bearing action status contradicts masks")
    expected_values = _expected_tensor_qc(result)
    for key, expected in expected_values.items():
        if qc.get(key) != expected:
            raise ValueError(f"Stage 2 QC count or tensor mismatch: {key}")
    exact_hit_count = _nonnegative_int(
        qc.get("exact_hit_count"), "exact_hit_count"
    )
    interpolated_count = _nonnegative_int(
        qc.get("interpolated_count"), "interpolated_count"
    )
    if exact_hit_count + interpolated_count != expected_values["valid_cell_count"]:
        raise ValueError("Stage 2 QC count mismatch: exact/interpolated")
    stage1_action_end_ns = _nonnegative_int(
        qc.get("stage1_action_end_ns"), "stage1_action_end_ns"
    )
    unrepresented_tail_ns = _nonnegative_int(
        qc.get("unrepresented_tail_ns"), "unrepresented_tail_ns"
    )
    if unrepresented_tail_ns >= 100_000_000:
        raise ValueError("Stage 2 QC unrepresented_tail_ns is outside grid step")
    if (
        expected_values["grid_end_ns"] + unrepresented_tail_ns
        != stage1_action_end_ns
    ):
        raise ValueError("Stage 2 QC action extent does not match grid")
    if "usable_sensor_count" in qc and qc["usable_sensor_count"] != int(
        result.usable_sensor_mask.sum()
    ):
        raise ValueError("Stage 2 QC usable_sensor_count mismatch")
    if "duration_ms" in qc and qc["duration_ms"] != int(result.timestamps_ms[-1]):
        raise ValueError("Stage 2 QC duration_ms mismatch")


def validate_existing_action(
    action_directory: Path,
    expected_fingerprints: Mapping[str, str],
) -> Stage2ActionResult:
    action_directory = Path(action_directory)
    _assert_no_reparse_components(_absolute_lexical(action_directory))
    if not action_directory.is_dir():
        raise ValueError("Stage 2 action path must be a real directory")
    managed_files = {path.name for path in action_directory.iterdir()}
    if managed_files != {"imu_stage2.npz", "qc.json"}:
        raise ValueError("Stage 2 action managed files do not match contract")
    qc = _load_json_strict(action_directory / "qc.json")
    try:
        status = DataStatus(str(qc["status"]))
        sample_id = str(qc["sample_id"])
    except (KeyError, ValueError) as error:
        raise ValueError("Stage 2 QC identity or status is invalid") from error
    result = load_and_validate_npz(
        action_directory / "imu_stage2.npz",
        sample_id=sample_id,
        status=status,
        qc=qc,
    )
    _validate_qc_against_result(qc, result, expected_fingerprints)
    return result


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse_point(path: Path) -> bool:
    if not os.path.lexists(path):
        return False
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    is_junction = getattr(path, "is_junction", lambda: False)()
    return bool(
        path.is_symlink()
        or is_junction
        or (reparse_flag and attributes & reparse_flag)
    )


def _assert_no_reparse_components(path: Path) -> None:
    path = _absolute_lexical(path)
    current = Path(path.anchor)
    if current and _is_reparse_point(current):
        raise ValueError(f"Stage 2 managed path contains reparse point: {current}")
    for component in path.parts[1:]:
        current /= component
        if _is_reparse_point(current):
            raise ValueError(
                f"Stage 2 managed path contains reparse point: {current}"
            )


def _verified_under(
    root: Path,
    candidate: Path,
    *,
    allow_root: bool = False,
) -> Path:
    root_lexical = _absolute_lexical(root)
    candidate_lexical = _absolute_lexical(candidate)
    _assert_no_reparse_components(root_lexical)
    _assert_no_reparse_components(candidate_lexical)
    if not root_lexical.is_dir():
        raise ValueError("Stage 2 output root must be a real directory")
    try:
        candidate_lexical.relative_to(root_lexical)
    except ValueError as error:
        raise ValueError("Stage 2 path escapes the lexical output root") from error
    if candidate_lexical == root_lexical and not allow_root:
        raise ValueError("Stage 2 managed path must be below output root")
    root_resolved = root_lexical.resolve(strict=True)
    candidate_resolved = candidate_lexical.resolve(strict=False)
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("Stage 2 path escapes the resolved output root") from error
    return candidate_lexical


def _checked_action_path(output_root: Path, relative_path: Path) -> tuple[Path, Path]:
    output_root = _absolute_lexical(Path(output_root))
    relative_path = Path(relative_path)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or relative_path == Path(".")
        or ".." in relative_path.parts
    ):
        raise ValueError("Stage 2 action path must be relative to output root")
    _verified_under(output_root, output_root, allow_root=True)
    final = _verified_under(output_root, output_root / relative_path)
    return output_root, final


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _replace_checked(root: Path, source: Path, destination: Path) -> None:
    checked_source = _verified_under(root, source)
    checked_destination = _verified_under(root, destination)
    if not os.path.lexists(checked_source):
        raise FileNotFoundError(checked_source)
    if os.path.lexists(checked_destination):
        raise FileExistsError(checked_destination)
    if checked_source.parent != checked_destination.parent:
        raise ValueError("Stage 2 atomic paths must be siblings")
    devices = {
        os.stat(root).st_dev,
        os.stat(checked_source.parent).st_dev,
        os.stat(checked_destination.parent).st_dev,
    }
    if len(devices) != 1:
        raise ValueError("Stage 2 atomic paths must share the output filesystem")
    _replace_path(checked_source, checked_destination)


def _remove_tree_checked(root: Path, path: Path, prefix: str) -> None:
    checked = _verified_under(root, path)
    if not checked.name.startswith(prefix):
        raise ValueError("Refusing to remove an unmanaged Stage 2 path")
    if os.path.lexists(checked):
        shutil.rmtree(checked)


def _write_staged_action(
    staging: Path,
    result: Stage2ActionResult,
    fingerprints: Mapping[str, str],
    qc_metadata: Mapping[str, object] | None,
) -> None:
    np.savez(
        staging / "imu_stage2.npz",
        values=result.values,
        sensor_mask=result.sensor_mask,
        valid_mask=result.valid_mask,
        timestamps_ms=result.timestamps_ms,
    )
    qc = _action_qc_payload(result, fingerprints, qc_metadata)
    write_json_atomic(staging / "qc.json", qc)
    validate_existing_action(staging, fingerprints)


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


def _restore_after_install_failure(
    root: Path,
    final: Path,
    staging: Path,
    backup: Path,
    install_error: BaseException,
) -> None:
    try:
        _replace_checked(root, backup, final)
    except BaseException as restore_error:
        raise _transaction_error(
            "Stage 2 install failed and backup restore failed",
            final,
            backup,
            install_error,
            restore_error,
        ) from install_error
    try:
        _remove_tree_checked(root, staging, ".staging-")
    except BaseException as cleanup_error:
        raise _transaction_error(
            "Stage 2 install failed; original restored but staging cleanup failed",
            final,
            backup,
            install_error,
            cleanup_error,
        ) from install_error


def _rollback_after_backup_cleanup_failure(
    root: Path,
    final: Path,
    staging: Path,
    backup: Path,
    existing_fingerprints: Mapping[str, str],
    cleanup_error: BaseException,
) -> None:
    try:
        _replace_checked(root, final, staging)
    except BaseException as unpublish_error:
        raise _transaction_error(
            "Stage 2 backup cleanup failed and new action unpublish failed",
            final,
            backup,
            cleanup_error,
            unpublish_error,
        ) from cleanup_error
    try:
        validate_existing_action(backup, existing_fingerprints)
        _replace_checked(root, backup, final)
        validate_existing_action(final, existing_fingerprints)
    except BaseException as restore_error:
        raise _transaction_error(
            "Stage 2 backup cleanup failed and original restore failed",
            final,
            backup,
            cleanup_error,
            restore_error,
        ) from cleanup_error
    try:
        _remove_tree_checked(root, staging, ".staging-")
    except BaseException as cleanup_new_error:
        raise _transaction_error(
            "Stage 2 original restored but temporary new action cleanup failed",
            final,
            backup,
            cleanup_error,
            cleanup_new_error,
        ) from cleanup_error
    raise _transaction_error(
        "Stage 2 backup cleanup failed; original action restored",
        final,
        backup,
        cleanup_error,
    ) from cleanup_error


def write_action_atomic(
    output_root: Path,
    action_relative_path: Path,
    result: Stage2ActionResult,
    fingerprints: Mapping[str, str],
    *,
    overwrite: bool = False,
    qc_metadata: Mapping[str, object] | None = None,
) -> WriteStatus:
    root, final = _checked_action_path(output_root, action_relative_path)
    fingerprints = _validate_fingerprints(fingerprints)
    final.parent.mkdir(parents=True, exist_ok=True)
    final = _verified_under(root, final)
    final_exists = os.path.lexists(final)
    if final_exists and not overwrite:
        raise FileExistsError(final)
    existing_fingerprints: dict[str, str] | None = None
    if final_exists:
        existing_qc = _load_json_strict(final / "qc.json")
        existing_fingerprints = {
            key: str(existing_qc.get(key, "")) for key in FINGERPRINT_KEYS
        }
        validate_existing_action(final, existing_fingerprints)
        existing_fingerprints = _validate_fingerprints(existing_fingerprints)
        if (
            existing_fingerprints["stage2_contract_sha256"]
            != fingerprints["stage2_contract_sha256"]
        ):
            raise ValueError("Cannot overwrite an incompatible Stage 2 contract")

    token = uuid4().hex
    staging = final.parent / f".staging-{final.name}-{token}"
    backup = final.parent / f".backup-{final.name}-{token}"
    staging = _verified_under(root, staging)
    backup = _verified_under(root, backup)
    staging_created = False
    try:
        staging.mkdir()
        staging_created = True
        _write_staged_action(staging, result, fingerprints, qc_metadata)
    except BaseException:
        if staging_created and os.path.lexists(staging):
            _remove_tree_checked(root, staging, ".staging-")
        raise

    if not final_exists:
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
        return WriteStatus.WRITTEN

    if existing_fingerprints is None:
        raise AssertionError("Existing Stage 2 action fingerprints are unavailable")
    try:
        _replace_checked(root, final, backup)
    except BaseException:
        _remove_tree_checked(root, staging, ".staging-")
        raise
    try:
        _replace_checked(root, staging, final)
    except BaseException as install_error:
        _restore_after_install_failure(
            root,
            final,
            staging,
            backup,
            install_error,
        )
        raise
    try:
        _remove_tree_checked(root, backup, ".backup-")
    except BaseException as cleanup_error:
        _rollback_after_backup_cleanup_failure(
            root,
            final,
            staging,
            backup,
            existing_fingerprints,
            cleanup_error,
        )
    return WriteStatus.WRITTEN
