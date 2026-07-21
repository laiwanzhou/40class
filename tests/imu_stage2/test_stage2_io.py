from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import src.data.imu_stage2_io as io_module
from src.data.imu_stage2_contracts import (
    DataStatus,
    Stage2ActionResult,
    WriteStatus,
)
from src.data.imu_stage2_io import (
    build_stage2_schema,
    load_and_validate_npz,
    load_stage2_schema,
    validate_existing_action,
    write_action_atomic,
    write_json_atomic,
)


def make_provenance(**overrides: object) -> dict[str, object]:
    provenance: dict[str, object] = {
        "implementation_version": "test-implementation",
        "generator_script": "scripts/preprocess_imu_stage2.py",
        "git_commit": "abc123",
        "created_at": "2026-07-21T00:00:00+08:00",
        "source_stage1_manifest": "manifest.csv",
        "source_stage1_manifest_sha256": "1" * 64,
    }
    provenance.update(overrides)
    return provenance


def make_arrays() -> dict[str, np.ndarray]:
    values = np.zeros((2, 5, 16), dtype=np.float32)
    valid_mask = np.ones((2, 5), dtype=bool)
    valid_mask[1, 4] = False
    values[1, 4, :] = np.nan
    return {
        "values": values,
        "sensor_mask": np.ones(5, dtype=bool),
        "valid_mask": valid_mask,
        "timestamps_ms": np.array([0, 100], dtype=np.int64),
    }


def make_result(fill: float = 0.0) -> Stage2ActionResult:
    values = np.full((1, 5, 16), fill, dtype=np.float32)
    return Stage2ActionResult(
        sample_id="sample",
        values=values,
        sensor_mask=np.ones(5, dtype=bool),
        valid_mask=np.ones((1, 5), dtype=bool),
        timestamps_ms=np.array([0], dtype=np.int64),
        qc={
            "grid_length": 1,
            "valid_cell_count": 5,
            "invalid_cell_count": 0,
            "exact_hit_count": 5,
            "interpolated_count": 0,
            "invalid_count": 0,
            "warning_codes": [],
        },
        status=DataStatus.SUCCESS,
    )


def make_fingerprints(**overrides: str) -> dict[str, str]:
    fingerprints = {
        "stage1_output_csv_sha256": "1" * 64,
        "stage1_qc_sha256": "2" * 64,
        "stage1_manifest_row_sha256": "3" * 64,
        "stage2_contract_sha256": "4" * 64,
    }
    fingerprints.update(overrides)
    return fingerprints


def test_schema_hash_depends_only_on_contract() -> None:
    first = build_stage2_schema(make_provenance(created_at="first"))
    second = build_stage2_schema(make_provenance(created_at="second"))

    assert first["contract"] == second["contract"]
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["provenance"] != second["provenance"]
    assert first["contract"]["hard_safety_limit_t"] == 10_000


def test_schema_round_trip_validates_contract_hash(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    schema = build_stage2_schema(make_provenance())
    write_json_atomic(path, schema)

    assert load_stage2_schema(path) == schema
    assert not list(tmp_path.glob(".tmp-*"))

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract"]["grid_frequency_hz"] = 20
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract_sha256"):
        load_stage2_schema(path)


def test_write_json_atomic_rejects_nonfinite_without_publication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qc.json"

    with pytest.raises(ValueError):
        write_json_atomic(path, {"bad": float("nan")})

    assert not path.exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_load_schema_rejects_nonfinite_provenance(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    schema = build_stage2_schema(make_provenance())
    text = json.dumps(schema).replace(
        '"implementation_version": "test-implementation"',
        '"implementation_version": NaN',
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Non-finite JSON"):
        load_stage2_schema(path)


def test_load_npz_requires_exact_keys_dtypes_and_preserves_nan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "imu_stage2.npz"
    arrays = make_arrays()
    np.savez(path, **arrays)

    result = load_and_validate_npz(
        path,
        sample_id="sample",
        status=DataStatus.SUCCESS_WITH_WARNINGS,
        qc={"warning_codes": ["records_excluded"]},
    )

    assert result.values.dtype == np.float32
    assert result.sensor_mask.dtype == np.bool_
    assert result.valid_mask.dtype == np.bool_
    assert result.timestamps_ms.dtype == np.int64
    assert np.isnan(result.values[~result.valid_mask]).all()
    assert result.status is DataStatus.SUCCESS_WITH_WARNINGS


@pytest.mark.parametrize(
    "mutate",
    [
        lambda arrays: arrays.pop("valid_mask"),
        lambda arrays: arrays.update(extra=np.array([1], dtype=np.int64)),
        lambda arrays: arrays.update(values=arrays["values"].astype(np.float64)),
        lambda arrays: arrays.update(timestamps_ms=np.array([0, 101], dtype=np.int64)),
    ],
)
def test_load_npz_rejects_malformed_contract(
    tmp_path: Path,
    mutate,
) -> None:
    path = tmp_path / "bad.npz"
    arrays = make_arrays()
    mutate(arrays)
    np.savez(path, **arrays)

    with pytest.raises(ValueError):
        load_and_validate_npz(
            path,
            sample_id="sample",
            status=DataStatus.SUCCESS,
            qc={},
        )


def test_load_npz_disables_pickle(tmp_path: Path) -> None:
    path = tmp_path / "object.npz"
    arrays = make_arrays()
    arrays["values"] = np.array([object()], dtype=object)
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="Object arrays"):
        load_and_validate_npz(
            path,
            sample_id="sample",
            status=DataStatus.SUCCESS,
            qc={},
        )


def test_write_action_atomic_publishes_validated_uncompressed_action(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    action_relative_path = Path("class/user/action")

    write_status = write_action_atomic(
        output_root,
        action_relative_path,
        make_result(),
        make_fingerprints(),
    )

    action_directory = output_root / action_relative_path
    assert write_status is WriteStatus.WRITTEN
    assert sorted(path.name for path in action_directory.iterdir()) == [
        "imu_stage2.npz",
        "qc.json",
    ]
    with zipfile.ZipFile(action_directory / "imu_stage2.npz") as archive:
        assert all(
            info.compress_type == zipfile.ZIP_STORED
            for info in archive.infolist()
        )
    qc = json.loads((action_directory / "qc.json").read_text(encoding="utf-8"))
    assert qc["status"] == "success"
    assert qc["write_status"] == "written"
    assert qc["stage2_contract_sha256"] == "4" * 64
    reopened = validate_existing_action(
        action_directory,
        make_fingerprints(),
    )
    assert reopened.status is DataStatus.SUCCESS
    assert not list(output_root.rglob(".staging-*"))
    assert not list(output_root.rglob(".backup-*"))


def test_validate_existing_action_rejects_source_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    action_relative_path = Path("class/user/action")
    write_action_atomic(
        output_root,
        action_relative_path,
        make_result(),
        make_fingerprints(),
    )

    with pytest.raises(ValueError, match="fingerprint"):
        validate_existing_action(
            output_root / action_relative_path,
            make_fingerprints(stage1_qc_sha256="9" * 64),
        )


def test_validate_existing_action_rejects_qc_count_mismatch(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    action_relative_path = Path("class/user/action")
    write_action_atomic(
        output_root,
        action_relative_path,
        make_result(),
        make_fingerprints(),
    )
    qc_path = output_root / action_relative_path / "qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["valid_cell_count"] = 4
    qc_path.write_text(json.dumps(qc), encoding="utf-8")

    with pytest.raises(ValueError, match="QC count"):
        validate_existing_action(
            output_root / action_relative_path,
            make_fingerprints(),
        )


def test_validate_existing_action_rejects_negative_counts_with_valid_sum(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    action_relative_path = Path("class/user/action")
    write_action_atomic(
        output_root,
        action_relative_path,
        make_result(),
        make_fingerprints(),
    )
    qc_path = output_root / action_relative_path / "qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["exact_hit_count"] = -1
    qc["interpolated_count"] = 6
    qc_path.write_text(json.dumps(qc), encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative integer"):
        validate_existing_action(
            output_root / action_relative_path,
            make_fingerprints(),
        )


def test_validate_existing_action_rejects_unknown_file(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    action_relative_path = Path("class/user/action")
    write_action_atomic(
        output_root,
        action_relative_path,
        make_result(),
        make_fingerprints(),
    )
    (output_root / action_relative_path / "unknown.txt").write_text(
        "unexpected",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="managed files"):
        validate_existing_action(
            output_root / action_relative_path,
            make_fingerprints(),
        )


def test_overwrite_replaces_action_without_residue(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    action_relative_path = Path("class/user/action")
    write_action_atomic(
        output_root,
        action_relative_path,
        make_result(fill=0.0),
        make_fingerprints(),
    )

    write_action_atomic(
        output_root,
        action_relative_path,
        make_result(fill=2.0),
        make_fingerprints(),
        overwrite=True,
    )

    reopened = validate_existing_action(
        output_root / action_relative_path,
        make_fingerprints(),
    )
    assert np.all(reopened.values == 2.0)
    assert not list(output_root.rglob(".staging-*"))
    assert not list(output_root.rglob(".backup-*"))


def test_overwrite_restores_original_after_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    action_relative_path = Path("class/user/action")
    action_directory = output_root / action_relative_path
    write_action_atomic(
        output_root,
        action_relative_path,
        make_result(fill=0.0),
        make_fingerprints(),
    )
    real_replace = io_module._replace_path
    failed = False

    def fail_install_once(source: Path, destination: Path) -> None:
        nonlocal failed
        if source.name.startswith(".staging-") and destination == action_directory:
            failed = True
            raise OSError("injected install failure")
        real_replace(source, destination)

    monkeypatch.setattr(io_module, "_replace_path", fail_install_once)

    with pytest.raises(OSError, match="injected install failure"):
        write_action_atomic(
            output_root,
            action_relative_path,
            make_result(fill=3.0),
            make_fingerprints(),
            overwrite=True,
        )

    assert failed
    reopened = validate_existing_action(action_directory, make_fingerprints())
    assert np.all(reopened.values == 0.0)
    assert not list(output_root.rglob(".staging-*"))
    assert not list(output_root.rglob(".backup-*"))


def test_write_action_rejects_existing_without_overwrite(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    action_relative_path = Path("class/user/action")
    write_action_atomic(
        output_root,
        action_relative_path,
        make_result(),
        make_fingerprints(),
    )

    with pytest.raises(FileExistsError):
        write_action_atomic(
            output_root,
            action_relative_path,
            make_result(fill=1.0),
            make_fingerprints(),
        )


def test_write_action_rejects_path_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()

    with pytest.raises(ValueError, match="output root"):
        write_action_atomic(
            output_root,
            Path("../escape"),
            make_result(),
            make_fingerprints(),
        )

    assert not (tmp_path / "escape").exists()
