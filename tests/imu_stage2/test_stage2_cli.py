from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.preprocess_imu_stage1 as stage1
import scripts.preprocess_imu_stage2 as stage2_cli
from src.data.imu_stage2_contracts import DataStatus, FEATURE_ORDER


SUMMARY_KEYS = {
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
}


def _manifest_row(action_index: int, relative: Path) -> dict[str, str]:
    row = {column: "" for column in stage1.MANIFEST_COLUMNS}
    action_id = f"1-1-{action_index}"
    sample_id = f"1__user1__{action_id}"
    row.update(
        {
            "sample_id": sample_id,
            "class_id": "1",
            "class_name": "Class_one",
            "user_id": "user1",
            "action_id": action_id,
            "relative_action_path": relative.as_posix(),
            "output_csv": (relative / "imu_merged.csv").as_posix(),
            "status": "success",
            "csv_file_count": "1",
            "total_input_rows": "10",
            "valid_output_rows": "10",
            "rejected_rows": "0",
            "unknown_sensor_rows": "0",
            "present_sensors": "LL;RL;LA;RA;C",
            "missing_sensors": "",
            "ll_rows": "2",
            "rl_rows": "2",
            "la_rows": "2",
            "ra_rows": "2",
            "c_rows": "2",
            "duration_s": "0.1",
            "warning_count": "0",
        }
    )
    return row


def _feature_values(base: float, *, usable: bool) -> dict[str, float]:
    values = {feature: base + index / 10 for index, feature in enumerate(FEATURE_ORDER)}
    values.update({"quat_0": 1.0 if usable else 0.0, "quat_1": 0.0, "quat_2": 0.0, "quat_3": 0.0})
    return values


def make_stage1_root(tmp_path: Path, kinds: tuple[str, ...]) -> Path:
    root = tmp_path / "new_IMU"
    root.mkdir()
    rows: list[dict[str, str]] = []
    sensors = ("LL", "RL", "LA", "RA", "C")
    for action_index, kind in enumerate(kinds, start=1):
        relative = Path("1_Class_one") / "user1" / f"1-1-{action_index}"
        action_directory = root / relative
        action_directory.mkdir(parents=True)
        row = _manifest_row(action_index, relative)
        selected_sensors = sensors if kind in {"success", "failed"} else ("LL",)
        records: list[dict[str, object]] = []
        for time_index, relative_time in enumerate(("0.0", "0.1")):
            for sensor_index, sensor in enumerate(selected_sensors):
                record: dict[str, object] = {
                    "relative_time_s": "not-a-time" if kind == "failed" and time_index == 0 else relative_time,
                    "relative_time_ms": time_index * 100,
                    "sensor_position": sensor,
                    "source_file": "part1.csv",
                    "source_row_index": time_index * len(selected_sensors) + sensor_index,
                }
                record.update(
                    _feature_values(
                        10.0 * action_index + sensor_index + time_index,
                        usable=kind != "no_usable",
                    )
                )
                records.append(record)
        pd.DataFrame(records, columns=stage1.OUTPUT_COLUMNS).to_csv(
            action_directory / "imu_merged.csv",
            index=False,
            encoding="utf-8-sig",
        )
        present = list(selected_sensors)
        qc = {
            "status": "success" if len(present) == 5 else "incomplete_sensors",
            "input_csv_files": ["part1.csv"],
            "present_sensors": present,
            "missing_sensors": [sensor for sensor in sensors if sensor not in present],
        }
        (action_directory / "qc.json").write_text(
            json.dumps(qc, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        if len(present) != 5:
            row["status"] = "incomplete_sensors"
            row["present_sensors"] = ";".join(present)
            row["missing_sensors"] = ";".join(sensor for sensor in sensors if sensor not in present)
            for sensor in ("rl", "la", "ra", "c"):
                row[f"{sensor}_rows"] = "0"
        rows.append(row)
    with (root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stage1.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return root


def run_main(
    input_root: Path,
    output_root: Path,
    *extra: str,
) -> int:
    return stage2_cli.main(
        [
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            *extra,
        ]
    )


@pytest.mark.parametrize("relationship", ["equal", "output_parent", "output_child"])
def test_validate_roots_rejects_equal_or_overlapping_paths(
    tmp_path: Path,
    relationship: str,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    if relationship == "equal":
        output_root = input_root
    elif relationship == "output_parent":
        output_root = tmp_path
    else:
        output_root = input_root / "stage2"

    with pytest.raises(ValueError, match="overlap"):
        stage2_cli.validate_roots(input_root, output_root)


def test_validate_roots_rejects_resolved_link_overlap(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    alias = tmp_path / "alias"
    try:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(alias), str(input_root)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            alias.symlink_to(input_root, target_is_directory=True)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.fail(f"Could not create required directory alias: {error}")

    with pytest.raises(ValueError, match="reparse|overlap"):
        stage2_cli.validate_roots(input_root, alias / "stage2")


def test_fresh_nonempty_output_fails_before_any_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "stage2"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    exit_code = run_main(input_root, output_root, "--summary-format", "json")

    assert exit_code == 2
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in output_root.iterdir()) == ["sentinel.txt"]
    assert capsys.readouterr().out == ""


def test_run_modes_and_safety_limit_are_mutually_validated(tmp_path: Path) -> None:
    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "stage2"

    assert run_main(input_root, output_root, "--resume", "--overwrite") == 2
    assert run_main(input_root, output_root, "--hard-safety-limit-t", "0") == 2
    assert run_main(input_root, output_root, "--hard-safety-limit-t", "9999") == 2
    assert not output_root.exists()


def test_dry_run_json_is_one_fixed_summary_and_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = make_stage1_root(
        tmp_path,
        ("success", "incomplete", "no_usable", "failed"),
    )
    output_root = tmp_path / "stage2"

    exit_code = run_main(
        input_root,
        output_root,
        "--dry-run",
        "--summary-format",
        "json",
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 1
    assert set(summary) == SUMMARY_KEYS
    assert summary["action_count"] == 4
    assert list(summary["data_status_counts"]) == [status.value for status in DataStatus]
    assert summary["data_status_counts"] == {
        "success": 1,
        "success_with_warnings": 0,
        "incomplete_sensors": 1,
        "no_usable_grid_cells": 1,
        "failed": 1,
    }
    assert summary["imu_usable_action_count"] == 2
    assert summary["strict_5sensor_candidate_count"] == 1
    assert not output_root.exists()
    assert "processing" not in captured.out.lower()


def test_formal_run_writes_atomic_root_artifacts_and_failed_qc(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = make_stage1_root(tmp_path, ("success", "failed"))
    output_root = tmp_path / "stage2"

    exit_code = run_main(input_root, output_root, "--summary-format", "json")

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert summary["data_status_counts"]["failed"] == 1
    assert (output_root / "schema.json").is_file()
    assert (output_root / "manifest.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert "closed_normally=true" in (output_root / "processing.log").read_text(encoding="utf-8")
    manifest = pd.read_csv(
        output_root / "manifest.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    assert len(manifest) == 2
    failed = manifest.loc[manifest["status"] == "failed"].iloc[0]
    assert failed["write_status"] == "qc_only"
    assert failed["grid_length"] == ""
    assert failed["valid_cell_count"] == ""
    failed_directory = output_root / Path(failed["relative_action_path"])
    assert {path.name for path in failed_directory.iterdir()} == {"qc.json"}
    failed_qc = json.loads((failed_directory / "qc.json").read_text(encoding="utf-8"))
    assert failed_qc["status"] == "failed"
    assert failed_qc["write_status"] == "qc_only"
    assert not list(output_root.rglob(".staging-*"))
    assert not list(output_root.rglob(".backup-*"))
    assert not list(output_root.glob(".tmp-*"))


def test_resume_skips_verified_action_and_preserves_data_status(
    tmp_path: Path,
) -> None:
    input_root = make_stage1_root(tmp_path, ("incomplete",))
    output_root = tmp_path / "stage2"
    assert run_main(input_root, output_root) == 0
    action_directory = output_root / "1_Class_one" / "user1" / "1-1-1"
    npz_path = action_directory / "imu_stage2.npz"
    before = npz_path.stat().st_mtime_ns

    assert run_main(input_root, output_root, "--resume") == 0

    assert npz_path.stat().st_mtime_ns == before
    manifest = pd.read_csv(
        output_root / "manifest.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    assert manifest.loc[0, "status"] == "incomplete_sensors"
    assert manifest.loc[0, "write_status"] == "skipped_existing"


@pytest.mark.parametrize("corruption", ["npz", "source", "unknown", "schema"])
def test_resume_preflight_rejects_untrusted_existing_tree(
    tmp_path: Path,
    corruption: str,
) -> None:
    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "stage2"
    assert run_main(input_root, output_root) == 0
    action_directory = output_root / "1_Class_one" / "user1" / "1-1-1"
    if corruption == "npz":
        (action_directory / "imu_stage2.npz").write_bytes(b"corrupt")
    elif corruption == "source":
        with (input_root / "1_Class_one" / "user1" / "1-1-1" / "imu_merged.csv").open("ab") as handle:
            handle.write(b"\n")
    elif corruption == "unknown":
        (action_directory / "unknown.txt").write_text("unknown", encoding="utf-8")
    else:
        schema_path = output_root / "schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["contract"]["grid_frequency_hz"] = 20
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
    before = sorted(path.relative_to(output_root).as_posix() for path in output_root.rglob("*"))

    assert run_main(input_root, output_root, "--resume") == 2

    after = sorted(path.relative_to(output_root).as_posix() for path in output_root.rglob("*"))
    assert after == before


def test_overwrite_replaces_action_under_same_contract_without_residue(
    tmp_path: Path,
) -> None:
    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "stage2"
    assert run_main(input_root, output_root) == 0
    action_directory = output_root / "1_Class_one" / "user1" / "1-1-1"
    with np.load(action_directory / "imu_stage2.npz", allow_pickle=False) as archive:
        before = archive["values"].copy()
    source_csv = input_root / "1_Class_one" / "user1" / "1-1-1" / "imu_merged.csv"
    frame = pd.read_csv(source_csv, encoding="utf-8-sig")
    frame["acc_x_g"] += 100.0
    frame.to_csv(source_csv, index=False, encoding="utf-8-sig")

    assert run_main(input_root, output_root, "--overwrite") == 0

    with np.load(action_directory / "imu_stage2.npz", allow_pickle=False) as archive:
        after = archive["values"].copy()
    assert not np.array_equal(before, after)
    assert not list(output_root.rglob(".staging-*"))
    assert not list(output_root.rglob(".backup-*"))


def test_overwrite_rejects_unknown_managed_file_before_action_processing(
    tmp_path: Path,
) -> None:
    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "stage2"
    assert run_main(input_root, output_root) == 0
    unknown = output_root / "unknown.bin"
    unknown.write_bytes(b"keep")

    assert run_main(input_root, output_root, "--overwrite") == 2

    assert unknown.read_bytes() == b"keep"


def test_overwrite_rejects_action_contract_mismatch_before_root_writes(
    tmp_path: Path,
) -> None:
    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "stage2"
    assert run_main(input_root, output_root) == 0
    action_qc_path = output_root / "1_Class_one" / "user1" / "1-1-1" / "qc.json"
    action_qc = json.loads(action_qc_path.read_text(encoding="utf-8"))
    action_qc["stage2_contract_sha256"] = "f" * 64
    action_qc_path.write_text(json.dumps(action_qc), encoding="utf-8")
    schema_before = (output_root / "schema.json").read_bytes()
    log_before = (output_root / "processing.log").read_bytes()

    assert run_main(input_root, output_root, "--overwrite") == 2

    assert (output_root / "schema.json").read_bytes() == schema_before
    assert (output_root / "processing.log").read_bytes() == log_before


def test_resume_reprocesses_verified_failed_qc_only_without_residue(
    tmp_path: Path,
) -> None:
    input_root = make_stage1_root(tmp_path, ("failed",))
    output_root = tmp_path / "stage2"
    assert run_main(input_root, output_root) == 1
    action_directory = output_root / "1_Class_one" / "user1" / "1-1-1"
    assert {path.name for path in action_directory.iterdir()} == {"qc.json"}

    assert run_main(input_root, output_root, "--resume") == 1

    assert {path.name for path in action_directory.iterdir()} == {"qc.json"}
    qc = json.loads((action_directory / "qc.json").read_text(encoding="utf-8"))
    assert qc["status"] == "failed"
    assert qc["write_status"] == "qc_only"
    manifest = pd.read_csv(
        output_root / "manifest.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    assert manifest.loc[0, "status"] == "failed"
    assert manifest.loc[0, "write_status"] == "qc_only"
    assert manifest.loc[0, "stage2_npz_relpath"] == ""
    assert manifest.loc[0, "stage2_qc_relpath"].endswith("qc.json")
    assert not list(output_root.rglob(".staging-*"))
    assert not list(output_root.rglob(".backup-*"))


def test_overwrite_atomically_replaces_old_tensor_with_failed_qc_only(
    tmp_path: Path,
) -> None:
    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "stage2"
    assert run_main(input_root, output_root) == 0
    action_directory = output_root / "1_Class_one" / "user1" / "1-1-1"
    source_csv = input_root / "1_Class_one" / "user1" / "1-1-1" / "imu_merged.csv"
    frame = pd.read_csv(source_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.loc[0, "relative_time_s"] = "not-a-time"
    frame.to_csv(source_csv, index=False, encoding="utf-8-sig")

    assert run_main(input_root, output_root, "--overwrite") == 1

    assert {path.name for path in action_directory.iterdir()} == {"qc.json"}
    qc = json.loads((action_directory / "qc.json").read_text(encoding="utf-8"))
    assert qc["status"] == "failed"
    assert qc["write_status"] == "qc_only"
    manifest = pd.read_csv(
        output_root / "manifest.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    assert manifest.loc[0, "status"] == "failed"
    assert manifest.loc[0, "write_status"] == "qc_only"
    assert manifest.loc[0, "stage2_npz_relpath"] == ""
    assert manifest.loc[0, "stage2_qc_relpath"].endswith("qc.json")
    assert not list(output_root.rglob(".staging-*"))
    assert not list(output_root.rglob(".backup-*"))


def test_qc_only_backup_partial_delete_is_not_restored_as_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = make_stage1_root(tmp_path, ("failed",))
    output_root = tmp_path / "stage2"
    assert run_main(input_root, output_root) == 1
    action_directory = output_root / "1_Class_one" / "user1" / "1-1-1"
    original_remove = stage2_cli._remove_tree_checked
    injected = False

    def partially_remove_backup(root: Path, path: Path, prefix: str) -> None:
        nonlocal injected
        if prefix == ".backup-" and not injected:
            injected = True
            (path / "qc.json").unlink()
            raise OSError("injected partial backup deletion")
        original_remove(root, path, prefix)

    monkeypatch.setattr(stage2_cli, "_remove_tree_checked", partially_remove_backup)

    assert run_main(input_root, output_root, "--resume") == 2

    assert {path.name for path in action_directory.iterdir()} == {"qc.json"}
    qc = json.loads((action_directory / "qc.json").read_text(encoding="utf-8"))
    assert qc["status"] == "failed"
    assert qc["write_status"] == "qc_only"
    backups = list(output_root.rglob(".backup-*"))
    assert len(backups) == 1
    assert not any(backups[0].iterdir())
    assert not list(output_root.rglob(".staging-*"))
    assert "backup validation failed" in capsys.readouterr().err


@pytest.mark.parametrize("relationship", ["duplicate", "nested"])
def test_fresh_rejects_duplicate_or_nested_action_paths_before_writes(
    tmp_path: Path,
    relationship: str,
) -> None:
    input_root = make_stage1_root(tmp_path, ("success", "success"))
    manifest_path = input_root / "manifest.csv"
    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    first_relative = Path(manifest.loc[0, "relative_action_path"])
    if relationship == "duplicate":
        manifest.loc[1, "relative_action_path"] = first_relative.as_posix()
        manifest.loc[1, "output_csv"] = manifest.loc[0, "output_csv"]
    else:
        second_relative = Path(manifest.loc[1, "relative_action_path"])
        nested_relative = first_relative / "nested"
        (input_root / second_relative).rename(input_root / nested_relative)
        manifest.loc[1, "relative_action_path"] = nested_relative.as_posix()
        manifest.loc[1, "output_csv"] = (nested_relative / "imu_merged.csv").as_posix()
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    output_root = tmp_path / "stage2"

    assert run_main(input_root, output_root) == 2

    assert not output_root.exists()


def test_manifest_atomic_reread_rejects_content_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.csv"
    row: dict[str, object] = {column: "" for column in stage2_cli.MANIFEST_COLUMNS}
    row["sample_id"] = "original"
    original_fsync = stage2_cli.os.fsync

    def mutate_after_fsync(file_descriptor: int) -> None:
        original_fsync(file_descriptor)
        temporary = next(tmp_path.glob(".tmp-manifest.csv-*"))
        text = temporary.read_text(encoding="utf-8-sig")
        temporary.write_text(
            text.replace("original", "tampered"),
            encoding="utf-8-sig",
        )

    monkeypatch.setattr(stage2_cli.os, "fsync", mutate_after_fsync)

    with pytest.raises(ValueError, match="content"):
        stage2_cli._write_manifest_atomic(path, [row])

    assert not path.exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_final_validation_rejects_artifact_corrupted_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "stage2"
    original_write_manifest = stage2_cli._write_manifest_atomic

    def write_manifest_then_corrupt(path: Path, rows) -> None:
        original_write_manifest(path, rows)
        npz_path = output_root / "1_Class_one" / "user1" / "1-1-1" / "imu_stage2.npz"
        npz_path.write_bytes(b"corrupt-after-manifest")

    monkeypatch.setattr(stage2_cli, "_write_manifest_atomic", write_manifest_then_corrupt)

    assert run_main(input_root, output_root) == 2

    log_text = (output_root / "processing.log").read_text(encoding="utf-8")
    assert "closed_normally=true" not in log_text


def test_script_imports_src_from_external_cwd_with_cleared_pythonpath(
    tmp_path: Path,
) -> None:
    script = Path(stage2_cli.__file__).resolve()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    for option in (
        "--input-root",
        "--output-root",
        "--dry-run",
        "--resume",
        "--overwrite",
        "--hard-safety-limit-t",
        "--summary-format",
    ):
        assert option in help_result.stdout

    input_root = make_stage1_root(tmp_path, ("success",))
    output_root = tmp_path / "external-stage2"
    dry_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--dry-run",
            "--summary-format",
            "json",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_result.returncode == 0, dry_result.stderr
    assert set(json.loads(dry_result.stdout)) == SUMMARY_KEYS
    assert not output_root.exists()
