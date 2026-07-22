#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.imu_stage2_pipeline import (
    BUNDLE_MANIFEST_NAME,
    build_inference_bundle_manifest,
    derive_submission_contract,
    load_inference_bundle,
    validate_inference_config,
)


ROLE_FILENAMES = {
    "checkpoint": "checkpoint.pt",
    "model_config": "model_config.yaml",
    "stage2_schema": "schema.json",
    "normalization_npz": "imu_normalization.npz",
    "normalization_json": "imu_normalization.json",
    "class_order": "class_order.json",
    "inference_config": "inference_config.yaml",
}


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a validated IMU inference bundle")
    for role in ROLE_FILENAMES:
        parser.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def build_bundle(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir).absolute()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    sources = {role: Path(getattr(args, role)).resolve(strict=True) for role in ROLE_FILENAMES}
    if any(not path.is_file() for path in sources.values()):
        raise ValueError("Every bundle source artifact must be a regular file")
    if len(set(sources.values())) != len(sources):
        raise ValueError("Bundle source artifacts must be distinct files")
    submission_contract = derive_submission_contract(Path(args.sample_submission))
    staging = output.parent / f".{output.name}.staging-{uuid4().hex}"
    try:
        staging.mkdir()
        managed: dict[str, Path] = {}
        for role, source in sources.items():
            destination = staging / ROLE_FILENAMES[role]
            shutil.copyfile(source, destination)
            managed[role] = destination
        submission_path = staging / "submission_contract.json"
        _write_json(submission_path, submission_contract)
        managed["submission_contract"] = submission_path
        manifest = build_inference_bundle_manifest(staging, managed)
        _write_json(staging / BUNDLE_MANIFEST_NAME, manifest)
        loaded = load_inference_bundle(staging)
        validate_inference_config(loaded.inference_config)
        os.replace(staging, output)
        load_inference_bundle(output)
        return output
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        output = build_bundle(args)
        print(json.dumps({"status": "success", "bundle_root": str(output)}, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
