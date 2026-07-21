from __future__ import annotations

import json
import os
import shutil
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
    if set(provenance) != PROVENANCE_KEYS:
        raise ValueError("Stage 2 provenance keys do not match contract")
    if provenance["source_stage1_manifest"] != "manifest.csv":
        raise ValueError("source_stage1_manifest must be manifest.csv")
    contract = _stage2_contract()
    return {
        "contract": contract,
        "provenance": dict(provenance),
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
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise ValueError("Stage 2 provenance keys do not match contract")
    expected_hash = contract_sha256(contract)
    if payload["contract_sha256"] != expected_hash:
        raise ValueError("Stage 2 contract_sha256 does not match contract")
    if contract != _stage2_contract():
        raise ValueError("Stage 2 schema contract is incompatible")
    return payload


def load_and_validate_npz(
    path: Path,
    *,
    sample_id: str,
    status: DataStatus,
    qc: Mapping[str, object],
) -> Stage2ActionResult:
    path = Path(path)
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
    normalized = {key: str(fingerprints[key]) for key in FINGERPRINT_KEYS}
    if any(not value for value in normalized.values()):
        raise ValueError("Stage 2 action fingerprints must not be empty")
    return normalized


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
    return payload


def _validate_qc_against_result(
    qc: Mapping[str, object],
    result: Stage2ActionResult,
    expected_fingerprints: Mapping[str, str],
) -> None:
    for key, expected in _validate_fingerprints(expected_fingerprints).items():
        if qc.get(key) != expected:
            raise ValueError(f"Stage 2 source fingerprint mismatch: {key}")
    expected_values: dict[str, object] = {
        "sample_id": result.sample_id,
        "status": result.status.value,
        "imu_usable": result.imu_usable,
        "sensor_mask": result.sensor_mask.tolist(),
        "usable_sensor_mask": result.usable_sensor_mask.tolist(),
        "grid_length": len(result.timestamps_ms),
        "valid_cell_count": int(result.valid_mask.sum()),
        "invalid_cell_count": int(result.valid_mask.size - result.valid_mask.sum()),
        "invalid_count": int(result.valid_mask.size - result.valid_mask.sum()),
    }
    for key, expected in expected_values.items():
        if qc.get(key) != expected:
            raise ValueError(f"Stage 2 QC count or tensor mismatch: {key}")
    try:
        WriteStatus(str(qc["write_status"]))
    except (KeyError, ValueError) as error:
        raise ValueError("Stage 2 QC write_status is invalid") from error
    exact_hit_count = _nonnegative_int(
        qc.get("exact_hit_count"), "exact_hit_count"
    )
    interpolated_count = _nonnegative_int(
        qc.get("interpolated_count"), "interpolated_count"
    )
    if exact_hit_count + interpolated_count != expected_values["valid_cell_count"]:
        raise ValueError("Stage 2 QC count mismatch: exact/interpolated")


def validate_existing_action(
    action_directory: Path,
    expected_fingerprints: Mapping[str, str],
) -> Stage2ActionResult:
    action_directory = Path(action_directory)
    if action_directory.is_symlink() or not action_directory.is_dir():
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


def _resolved_under(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate_resolved = candidate.resolve(strict=False)
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("Stage 2 path escapes the resolved output root") from error
    return candidate_resolved


def _checked_action_path(output_root: Path, relative_path: Path) -> tuple[Path, Path]:
    output_root = Path(output_root)
    relative_path = Path(relative_path)
    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("Stage 2 output root must be a real directory")
    if relative_path.is_absolute() or not relative_path.parts or relative_path == Path("."):
        raise ValueError("Stage 2 action path must be relative to output root")
    root_resolved = output_root.resolve(strict=True)
    final = _resolved_under(root_resolved, root_resolved / relative_path)
    if final == root_resolved:
        raise ValueError("Stage 2 action path must be below output root")
    return root_resolved, final


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _replace_checked(root: Path, source: Path, destination: Path) -> None:
    checked_source = _resolved_under(root, source)
    checked_destination = _resolved_under(root, destination)
    if checked_source.parent != checked_destination.parent:
        raise ValueError("Stage 2 atomic paths must be siblings")
    if os.stat(root).st_dev != os.stat(checked_source.parent).st_dev:
        raise ValueError("Stage 2 atomic paths must share the output filesystem")
    _replace_path(checked_source, checked_destination)


def _remove_tree_checked(root: Path, path: Path, prefix: str) -> None:
    checked = _resolved_under(root, path)
    if not checked.name.startswith(prefix):
        raise ValueError("Refusing to remove an unmanaged Stage 2 path")
    if checked.exists():
        shutil.rmtree(checked)


def _write_staged_action(
    staging: Path,
    result: Stage2ActionResult,
    fingerprints: Mapping[str, str],
    qc_metadata: Mapping[str, object] | None,
) -> None:
    staging.mkdir()
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
    final = _resolved_under(root, final)
    if final.exists() and not overwrite:
        raise FileExistsError(final)
    if final.exists():
        existing_qc = _load_json_strict(final / "qc.json")
        existing_fingerprints = {
            key: str(existing_qc.get(key, "")) for key in FINGERPRINT_KEYS
        }
        validate_existing_action(final, existing_fingerprints)
        if (
            existing_fingerprints["stage2_contract_sha256"]
            != fingerprints["stage2_contract_sha256"]
        ):
            raise ValueError("Cannot overwrite an incompatible Stage 2 contract")

    token = uuid4().hex
    staging = final.parent / f".staging-{final.name}-{token}"
    backup = final.parent / f".backup-{final.name}-{token}"
    staging = _resolved_under(root, staging)
    backup = _resolved_under(root, backup)
    moved_existing = False
    try:
        _write_staged_action(staging, result, fingerprints, qc_metadata)
        if final.exists():
            _replace_checked(root, final, backup)
            moved_existing = True
        try:
            _replace_checked(root, staging, final)
        except BaseException:
            if moved_existing and not final.exists() and backup.exists():
                _replace_checked(root, backup, final)
                moved_existing = False
            raise
        if moved_existing:
            _remove_tree_checked(root, backup, ".backup-")
            moved_existing = False
    finally:
        if staging.exists():
            _remove_tree_checked(root, staging, ".staging-")
        if moved_existing and not final.exists() and backup.exists():
            _replace_checked(root, backup, final)
    return WriteStatus.WRITTEN
