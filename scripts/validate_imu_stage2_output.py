#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts import preprocess_imu_stage2 as stage2_cli
from src.data.imu_stage1_bridge import discover_stage1_artifacts
from src.data.imu_stage2_contracts import DataStatus, WriteStatus, sha256_file
from src.data.imu_stage2_io import (
    _absolute_lexical,
    _assert_no_reparse_components,
    _load_json_strict,
    load_stage2_schema,
    validate_existing_action,
    write_json_atomic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation for an IMU Stage 2 v1 output root"
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-summary", type=Path)
    parser.add_argument("--audit-output", type=Path)
    return parser


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_audit_output(
    audit_output: Path | None,
    input_root: Path,
    output_root: Path,
) -> Path | None:
    if audit_output is None:
        return None
    candidate = _absolute_lexical(audit_output)
    _assert_no_reparse_components(candidate)
    if not candidate.parent.is_dir() or candidate.is_dir():
        raise ValueError("Audit output parent must be an existing real directory")
    candidate_resolved = candidate.resolve(strict=False)
    for data_root in (input_root, output_root):
        root_resolved = data_root.resolve(strict=True)
        if _is_relative_to(candidate_resolved, root_resolved):
            raise ValueError("Audit output must be outside both data roots")
    return candidate


def _read_manifest(output_root: Path) -> list[dict[str, str]]:
    path = output_root / "manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(stage2_cli.MANIFEST_COLUMNS):
            raise ValueError("Stage 2 manifest columns do not match contract")
        rows = list(reader)
    if any(set(row) != set(stage2_cli.MANIFEST_COLUMNS) for row in rows):
        raise ValueError("Stage 2 manifest row columns do not match contract")
    return rows


def _expected_manifest_row(
    row: Mapping[str, str],
    descriptor,
    schema: Mapping[str, object],
    output_root: Path,
) -> dict[str, str]:
    fingerprints = stage2_cli._fingerprints(
        descriptor, str(schema["contract_sha256"])
    )
    try:
        status = DataStatus(row["status"])
        write_status = WriteStatus(row["write_status"])
    except ValueError as error:
        raise ValueError("Stage 2 manifest status or write_status is invalid") from error
    action_directory = output_root / descriptor.action_relative_path
    if status is DataStatus.FAILED:
        if write_status is not WriteStatus.QC_ONLY:
            raise ValueError("Failed Stage 2 manifest row must be qc_only")
        qc = stage2_cli._validate_qc_only(
            action_directory,
            descriptor,
            fingerprints,
            require_current_fingerprints=True,
        )
        expected = stage2_cli._manifest_row(
            descriptor,
            status=status,
            write_status=write_status,
            fingerprints=fingerprints,
            result=None,
            error_message=str(qc.get("error_message", "")),
        )
    else:
        if write_status not in {WriteStatus.WRITTEN, WriteStatus.SKIPPED_EXISTING}:
            raise ValueError("Tensor Stage 2 manifest write_status is invalid")
        result = validate_existing_action(action_directory, fingerprints)
        if result.sample_id != descriptor.sample_id or result.status is not status:
            raise ValueError("Stage 2 manifest identity or status disagrees with action")
        expected = stage2_cli._manifest_row(
            descriptor,
            status=status,
            write_status=write_status,
            fingerprints=fingerprints,
            result=result,
        )
    return stage2_cli._normalized_manifest_rows([expected])[0]


def validate_output(
    input_root: Path,
    output_root: Path,
    *,
    expected_summary: Path | None = None,
) -> dict[str, object]:
    input_root, output_root = stage2_cli.validate_roots(input_root, output_root)
    if not output_root.is_dir():
        raise ValueError("Stage 2 output root must be an existing real directory")
    descriptors = discover_stage1_artifacts(input_root)
    stage2_cli._validate_action_paths(output_root, descriptors)
    stage2_cli._validate_managed_tree(output_root, descriptors)

    schema = load_stage2_schema(output_root / "schema.json")
    source_manifest_sha256 = sha256_file(input_root / "manifest.csv")
    if (
        schema["provenance"]["source_stage1_manifest_sha256"]
        != source_manifest_sha256
    ):
        raise ValueError("Stage 2 schema source manifest fingerprint mismatch")

    rows = _read_manifest(output_root)
    if len(rows) != len(descriptors):
        raise ValueError("Stage 2 manifest action count does not match Stage 1")
    descriptor_by_id = {descriptor.sample_id: descriptor for descriptor in descriptors}
    if len(descriptor_by_id) != len(descriptors):
        raise ValueError("Stage 1 descriptors contain duplicate sample IDs")
    if [row["sample_id"] for row in rows] != [
        descriptor.sample_id for descriptor in descriptors
    ]:
        raise ValueError("Stage 2 manifest sample order or identity is invalid")

    for row in rows:
        descriptor = descriptor_by_id.get(row["sample_id"])
        if descriptor is None:
            raise ValueError("Stage 2 manifest contains an unknown sample ID")
        expected_row = _expected_manifest_row(row, descriptor, schema, output_root)
        if dict(row) != expected_row:
            differing = sorted(
                key for key in stage2_cli.MANIFEST_COLUMNS if row[key] != expected_row[key]
            )
            raise ValueError(
                "Stage 2 manifest disagrees with validated action: "
                + ",".join(differing)
            )

    summary = stage2_cli._summary(
        rows,
        source_stage1_manifest_sha256=source_manifest_sha256,
        stage2_contract_sha256=str(schema["contract_sha256"]),
    )
    if expected_summary is not None:
        expected = _load_json_strict(expected_summary)
        if tuple(expected) != stage2_cli.SUMMARY_KEYS or expected != summary:
            raise ValueError("Stage 2 summary does not match expected summary")
    return summary


def _emit_summary(summary: Mapping[str, object]) -> None:
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        input_root, output_root = stage2_cli.validate_roots(
            args.input_root, args.output_root
        )
        audit_output = _validate_audit_output(
            args.audit_output, input_root, output_root
        )
        summary = validate_output(
            input_root,
            output_root,
            expected_summary=args.expected_summary,
        )
        if audit_output is not None:
            write_json_atomic(audit_output, summary)
        _emit_summary(summary)
        return 1 if summary["data_status_counts"][DataStatus.FAILED.value] else 0
    except SystemExit as error:
        return int(error.code)
    except (
        OSError,
        ValueError,
        KeyError,
        csv.Error,
        zipfile.BadZipFile,
        pd.errors.ParserError,
    ) as error:
        print(f"Stage 2 validation error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Stage 2 validation interrupted", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
