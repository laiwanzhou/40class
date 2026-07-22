#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess_imu_stage1 import MANIFEST_COLUMNS, OUTPUT_COLUMNS
from scripts import preprocess_imu_stage2 as stage2_cli
from src.data.imu_stage1_bridge import (
    Stage1ArtifactDescriptor,
    discover_stage1_artifacts,
    stage1_manifest_row_sha256,
)
from src.data.imu_stage2_contracts import sha256_file
from src.data.imu_stage2_io import (
    build_stage2_schema,
    write_action_atomic,
    write_json_atomic,
)
from src.inference.imu_stage2_pipeline import (
    InferenceBundle,
    collate_inference_samples,
    decode_predictions,
    discover_test_samples,
    load_inference_bundle,
    preprocess_inference_sample_with_diagnostics,
    validate_inference_config,
    validate_logits,
    write_submission_atomic,
)
from src.models import build_imu_stage2_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic raw-test IMU Stage 2 inference")
    parser.add_argument("--raw-test-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def _validate_config(bundle: InferenceBundle) -> dict[str, object]:
    return validate_inference_config(bundle.inference_config)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _validate_runtime_paths(
    raw_test_root: Path,
    bundle_root: Path,
    output_csv: Path,
    audit_dir: Path | None,
) -> tuple[Path, Path, Path, Path | None]:
    raw = Path(raw_test_root).resolve(strict=True)
    bundle = Path(bundle_root).resolve(strict=True)
    if not raw.is_dir():
        raise NotADirectoryError(raw)
    if not bundle.is_dir():
        raise NotADirectoryError(bundle)
    output = Path(output_csv).resolve(strict=False)
    audit = None if audit_dir is None else Path(audit_dir).resolve(strict=False)
    named_paths = [("raw_test_root", raw), ("bundle_root", bundle), ("output_csv", output)]
    if audit is not None:
        named_paths.append(("audit_dir", audit))
    for index, (first_name, first) in enumerate(named_paths):
        for second_name, second in named_paths[index + 1 :]:
            if _paths_overlap(first, second):
                raise ValueError(
                    f"Inference paths overlap: {first_name}={first} and "
                    f"{second_name}={second}"
                )
    return raw, bundle, output, audit


def _create_audit_staging(audit_root: Path | None) -> Path | None:
    if audit_root is None:
        return None
    root = Path(audit_root)
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ValueError("Audit root must be missing or empty")
    else:
        root.mkdir()
    staging = root / f".inference.staging-{uuid4().hex}"
    staging.mkdir()
    return staging


def _streaming_batches(
    samples: Iterable[object], config: dict[str, object]
) -> Iterator[list[object]]:
    maximum = int(config["maximum_batch_size"])
    budget = int(config["batch_feature_budget"])
    current: list[object] = []
    current_max_t = 1
    for sample in samples:
        imu_result = getattr(sample, "imu_result")
        length = 0 if imu_result is None else int(imu_result.values.shape[0])
        candidate_t = max(current_max_t, length, 1)
        if current and (len(current) >= maximum or candidate_t * 5 * 16 * (len(current) + 1) > budget):
            yield current
            current = []
            current_max_t = 1
        current.append(sample)
        current_max_t = max(current_max_t, length, 1)
        if len(current) >= maximum or current_max_t * 5 * 16 * len(current) >= budget:
            yield current
            current = []
            current_max_t = 1
    if current:
        yield current


def _normalize_batch(batch: dict[str, object], bundle: InferenceBundle, device: torch.device) -> dict[str, object]:
    values = torch.as_tensor(batch["values"], dtype=torch.float32)
    valid = torch.as_tensor(batch["valid_mask"], dtype=torch.bool)
    mean = torch.from_numpy(bundle.normalization_arrays["mean"]).to(dtype=torch.float64)
    scale = torch.from_numpy(bundle.normalization_arrays["applied_scale"]).to(dtype=torch.float64)
    centered = (values.to(torch.float64) - mean.unsqueeze(0).unsqueeze(0)) / scale.unsqueeze(0).unsqueeze(0)
    standardized = torch.where(valid.unsqueeze(-1), centered, torch.zeros_like(centered)).to(torch.float32)
    if not torch.isfinite(standardized).all():
        raise ValueError("Normalized inference batch is non-finite")
    normalized = dict(batch)
    normalized["values"] = standardized
    for key, value in list(normalized.items()):
        if isinstance(value, torch.Tensor):
            normalized[key] = value.to(device)
    return normalized


def _decimal_seconds(nanoseconds: int) -> str:
    whole, remainder = divmod(int(nanoseconds), 1_000_000_000)
    return str(whole) if remainder == 0 else f"{whole}.{remainder:09d}".rstrip("0")


def _write_stage1_intermediate(
    stage1_root: Path,
    sample_id: str,
    stage1: object,
) -> tuple[dict[str, object], Stage1ArtifactDescriptor]:
    relative_action = Path(sample_id)
    stage1_action = stage1_root / relative_action
    stage1_action.mkdir(parents=True)
    frame = stage1.dataframe.copy()
    frame.insert(0, "relative_time_ms", stage1.relative_time_ns.astype(np.float64) / 1_000_000.0)
    frame.insert(0, "relative_time_s", [_decimal_seconds(value) for value in stage1.relative_time_ns])
    frame = frame.loc[:, OUTPUT_COLUMNS]
    stage1_csv = stage1_action / "imu_merged.csv"
    stage1_qc = stage1_action / "qc.json"
    frame.to_csv(stage1_csv, index=False, encoding="utf-8-sig")
    stage1_qc.write_text(
        json.dumps(stage1.qc, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    qc = stage1.qc
    rows_per_sensor = qc.get("rows_per_sensor", {})
    duplicate_counts = qc.get("duplicate_timestamp_count_per_sensor", {})
    present = qc.get("present_sensors", [])
    missing = qc.get("missing_sensors", [])
    warnings = qc.get("warnings", [])
    manifest_row: dict[str, object] = {
        "sample_id": sample_id,
        "class_id": "" if stage1.class_id is None else stage1.class_id,
        "class_name": "" if stage1.class_name is None else stage1.class_name,
        "user_id": "" if stage1.user_id is None else stage1.user_id,
        "action_id": "" if stage1.action_id is None else stage1.action_id,
        "relative_action_path": relative_action.as_posix(),
        "output_csv": (relative_action / "imu_merged.csv").as_posix(),
        "status": qc.get("status", "success"),
        "csv_file_count": qc.get("csv_file_count", 0),
        "total_input_rows": qc.get("total_input_rows", len(frame)),
        "valid_output_rows": qc.get("valid_output_rows", len(frame)),
        "rejected_rows": qc.get("rejected_rows", 0),
        "unknown_sensor_rows": qc.get("unknown_sensor_rows", 0),
        "present_sensors": ";".join(str(value) for value in present),
        "missing_sensors": ";".join(str(value) for value in missing),
        "ll_rows": rows_per_sensor.get("LL", 0),
        "rl_rows": rows_per_sensor.get("RL", 0),
        "la_rows": rows_per_sensor.get("LA", 0),
        "ra_rows": rows_per_sensor.get("RA", 0),
        "c_rows": rows_per_sensor.get("C", 0),
        "ll_duplicate_timestamps": duplicate_counts.get("LL", 0),
        "rl_duplicate_timestamps": duplicate_counts.get("RL", 0),
        "la_duplicate_timestamps": duplicate_counts.get("LA", 0),
        "ra_duplicate_timestamps": duplicate_counts.get("RA", 0),
        "c_duplicate_timestamps": duplicate_counts.get("C", 0),
        "duration_s": qc.get("duration_s", 0),
        "warning_count": len(warnings),
        "error_message": qc.get("error_message", ""),
    }
    normalized_row = {
        column: "" if manifest_row[column] is None else str(manifest_row[column])
        for column in MANIFEST_COLUMNS
    }
    descriptor = Stage1ArtifactDescriptor(
        root=stage1_root,
        sample_id=sample_id,
        action_relative_path=relative_action,
        output_csv_path=stage1_csv,
        qc_path=stage1_qc,
        manifest_row=normalized_row,
        manifest_row_sha256=stage1_manifest_row_sha256(normalized_row),
    )
    return manifest_row, descriptor


def _write_stage2_intermediate(
    stage2_root: Path,
    descriptor: Stage1ArtifactDescriptor,
    result: object,
    stage2_contract_sha256: str,
) -> dict[str, object]:
    fingerprints = {
        "stage1_output_csv_sha256": sha256_file(descriptor.output_csv_path),
        "stage1_qc_sha256": sha256_file(descriptor.qc_path),
        "stage1_manifest_row_sha256": descriptor.manifest_row_sha256,
        "stage2_contract_sha256": stage2_contract_sha256,
    }
    write_action_atomic(
        stage2_root,
        descriptor.action_relative_path,
        result,
        fingerprints,
        qc_metadata=stage2_cli._qc_metadata(descriptor),
    )
    return stage2_cli._manifest_row(
        descriptor,
        status=result.status,
        write_status=stage2_cli.WriteStatus.WRITTEN,
        fingerprints=fingerprints,
        result=result,
    )


def _write_stage1_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _finalize_online_intermediates(
    run: Path,
    stage1_rows: list[dict[str, object]],
    stage2_rows: list[dict[str, object]],
    bundle: InferenceBundle,
) -> None:
    stage1_root = run / "intermediates" / "stage1"
    stage2_root = run / "intermediates" / "stage2"
    _write_stage1_manifest(stage1_root / "manifest.csv", stage1_rows)
    source_hash = sha256_file(stage1_root / "manifest.csv")
    training_provenance = bundle.stage2_schema["provenance"]
    if not isinstance(training_provenance, dict):
        raise ValueError("Training Stage 2 provenance is invalid")
    provenance = dict(training_provenance)
    provenance.update(
        {
            "generator_script": "scripts/infer_imu_stage2.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_stage1_manifest": "manifest.csv",
            "source_stage1_manifest_sha256": source_hash,
        }
    )
    schema = build_stage2_schema(provenance)
    if schema["contract_sha256"] != bundle.stage2_schema["contract_sha256"]:
        raise ValueError("Online Stage 2 contract differs from inference bundle")
    write_json_atomic(stage2_root / "schema.json", schema)
    stage2_cli._write_manifest_atomic(stage2_root / "manifest.csv", stage2_rows)
    (stage2_root / "processing.log").write_text(
        "online_inference_intermediates=true\nclosed_normally=true\n", encoding="utf-8"
    )
    descriptors = discover_stage1_artifacts(stage1_root)
    by_id = {descriptor.sample_id: descriptor for descriptor in descriptors}
    successful_descriptors = [by_id[str(row["sample_id"])] for row in stage2_rows]
    stage2_cli._validate_final_output(
        stage2_root, successful_descriptors, stage2_rows, schema
    )


def _write_audit(
    run: Path,
    manifest_rows: list[dict[str, object]],
    problems: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    columns = [
        "sample_id", "source_status", "stage1_status", "stage2_status", "imu_available",
        "modality_mask", "failure_reason", "warning_codes", "sequence_length", "prediction",
    ]
    with (run / "inference_manifest.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    (run / "problematic_sample_qc.json").write_text(
        json.dumps(problems, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (run / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (run / "processing.log").open("x", encoding="utf-8", newline="\n") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _publish_success_transaction(
    output: Path,
    rows: list[tuple[str, object]],
    contract: dict[str, object],
    audit_root: Path | None,
    manifest_rows: list[dict[str, object]],
    problems: list[dict[str, object]],
    summary: dict[str, object],
    *,
    overwrite: bool,
    prepared_audit: Path | None = None,
) -> Path | None:
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    audit = None if audit_root is None else Path(audit_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:12]
    audit_staging: Path | None = prepared_audit
    audit_final: Path | None = None
    output_staging = output.parent / f".{output.name}.staging-{uuid4().hex}"
    output_backup = output.parent / f".{output.name}.backup-{uuid4().hex}"
    output_existed = output.exists()
    output_published = False
    audit_published = False
    preserve_backup = False
    try:
        write_submission_atomic(output_staging, rows, contract, overwrite=False)
        if audit is not None:
            if audit.exists():
                if not audit.is_dir():
                    raise NotADirectoryError(audit)
                visible = [path for path in audit.iterdir() if path != audit_staging]
                if visible:
                    raise ValueError("Audit root must be missing or empty")
            else:
                audit.mkdir()
            if audit_staging is None:
                audit_staging = audit / f".{run_id}.staging-{uuid4().hex}"
                audit_staging.mkdir()
            elif audit_staging.parent != audit or not audit_staging.is_dir():
                raise ValueError("Prepared audit staging directory is invalid")
            audit_final = audit / run_id
            _write_audit(audit_staging, manifest_rows, problems, summary)
            loaded_summary = json.loads(
                (audit_staging / "inference_summary.json").read_text(encoding="utf-8")
            )
            if loaded_summary != summary or summary.get("status") != "success" or summary.get("exit_code") != 0:
                raise ValueError("Success audit payload failed validation")
        if output_existed:
            shutil.copy2(output, output_backup)
        os.replace(output_staging, output)
        output_published = True
        if audit_staging is not None:
            assert audit_final is not None
            os.replace(audit_staging, audit_final)
            audit_published = True
        if output_backup.exists():
            output_backup.unlink()
        return audit_final
    except BaseException as publish_error:
        rollback_errors: list[BaseException] = []
        if audit_published and audit_final is not None and audit_final.exists():
            try:
                shutil.rmtree(audit_final)
            except BaseException as error:
                rollback_errors.append(error)
        if output_published and output.exists():
            try:
                output.unlink()
            except BaseException as error:
                rollback_errors.append(error)
        if output_existed and output_published and output_backup.exists():
            try:
                os.replace(output_backup, output)
            except BaseException as error:
                preserve_backup = True
                rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                "Inference publication failed and rollback was incomplete; "
                f"output={output}; backup={output_backup}; "
                f"rollback_errors={[str(error) for error in rollback_errors]}"
            ) from publish_error
        raise
    finally:
        if output_staging.exists():
            output_staging.unlink()
        if output_backup.exists() and not preserve_backup:
            output_backup.unlink()
        if audit_staging is not None and audit_staging.exists():
            shutil.rmtree(audit_staging)


def run(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    raw_test_root, bundle_root, output, audit_root = _validate_runtime_paths(
        args.raw_test_root,
        args.bundle_root,
        args.output_csv,
        args.audit_dir,
    )
    if output.exists() and not args.overwrite_output:
        raise FileExistsError(output)
    if not output.parent.exists() or not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if args.save_intermediates and args.audit_dir is None:
        raise ValueError("--save-intermediates requires --audit-dir")
    bundle = load_inference_bundle(bundle_root)
    config = _validate_config(bundle)
    discovery = discover_test_samples(raw_test_root)
    contract = bundle.submission_contract["contract"]
    if not isinstance(contract, dict):
        raise ValueError("Submission contract payload is invalid")
    expected_ids = list(contract["sample_ids"])
    discovered_ids = [descriptor.sample_id for descriptor in discovery.samples]
    if discovered_ids != expected_ids:
        raise ValueError("Discovered test IDs do not match submission contract")

    manifest_rows: list[dict[str, object]] = []
    manifest_by_id: dict[str, dict[str, object]] = {}
    problems: list[dict[str, object]] = []
    seed = int(config["inference_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(config["deterministic_algorithms"]))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(config["deterministic_algorithms"])
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model = build_imu_stage2_model(bundle.model_config, num_classes=bundle.class_order.num_classes).to(device)
    checkpoint = torch.load(bundle.paths["checkpoint"], map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint model_state_dict is missing")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    decoded: list[object] = []
    batch_sizes: list[int] = []
    unavailable_count = 0
    stage1_rows: list[dict[str, object]] = []
    stage2_rows: list[dict[str, object]] = []
    audit_staging = _create_audit_staging(audit_root)
    try:
        if args.save_intermediates:
            assert audit_staging is not None
            (audit_staging / "intermediates" / "stage1").mkdir(parents=True)
            (audit_staging / "intermediates" / "stage2").mkdir()

        def prepared_samples() -> Iterator[object]:
            nonlocal unavailable_count
            for descriptor in discovery.samples:
                diagnostics = preprocess_inference_sample_with_diagnostics(
                    descriptor,
                    hard_safety_limit_t=int(config["hard_safety_limit_t"]),
                )
                sample = diagnostics.sample
                error = diagnostics.degradation_error
                if not sample.imu_available:
                    unavailable_count += 1
                if args.save_intermediates and diagnostics.stage1_result is not None:
                    assert audit_staging is not None
                    stage1_root = audit_staging / "intermediates" / "stage1"
                    stage2_root = audit_staging / "intermediates" / "stage2"
                    stage1_row, stage1_descriptor = _write_stage1_intermediate(
                        stage1_root, descriptor.sample_id, diagnostics.stage1_result
                    )
                    stage1_rows.append(stage1_row)
                    if diagnostics.stage2_result is not None:
                        stage2_rows.append(
                            _write_stage2_intermediate(
                                stage2_root,
                                stage1_descriptor,
                                diagnostics.stage2_result,
                                str(bundle.stage2_schema["contract_sha256"]),
                            )
                        )
                failure_reason = "" if error is None else str(
                    getattr(error, "error_code", type(error).__name__)
                )
                if error is not None:
                    problems.append(
                        {
                            "sample_id": descriptor.sample_id,
                            "error_code": failure_reason,
                            "failure_stage": str(
                                getattr(error, "failure_stage", "unknown")
                            ),
                            "safe_message": str(
                                getattr(error, "safe_message", str(error))
                            ),
                        }
                    )
                imu_result = sample.imu_result
                warnings = [] if imu_result is None else list(
                    imu_result.qc.get("warning_codes", [])
                )
                row = {
                    "sample_id": descriptor.sample_id,
                    "source_status": diagnostics.source_status,
                    "stage1_status": diagnostics.stage1_status,
                    "stage2_status": diagnostics.stage2_status,
                    "imu_available": sample.imu_available,
                    "modality_mask": sample.modality_mask,
                    "failure_reason": failure_reason,
                    "warning_codes": json.dumps(warnings, separators=(",", ":")),
                    "sequence_length": 0 if imu_result is None else len(imu_result.values),
                    "prediction": "",
                }
                manifest_rows.append(row)
                manifest_by_id[descriptor.sample_id] = row
                yield sample

        with torch.inference_mode():
            for group in _streaming_batches(prepared_samples(), config):
                batch_sizes.append(len(group))
                batch = _normalize_batch(
                    collate_inference_samples(group), bundle, device
                )
                output_mapping = model(batch)
                logits = output_mapping.get("logits")
                validated = validate_logits(
                    logits,
                    batch_size=len(group),
                    num_classes=bundle.class_order.num_classes,
                )
                group_predictions = decode_predictions(
                    validated, bundle.class_order, contract
                )
                for sample, prediction in zip(group, group_predictions, strict=True):
                    manifest_by_id[sample.sample_id]["prediction"] = prediction
                    decoded.append(prediction)

        if unavailable_count and config["imu_unavailable_policy"] != "packaged_null_embedding":
            return 1, {
                "exit_code": 1,
                "status": "incomplete",
                "sample_count": len(manifest_rows),
                "predicted_sample_count": 0,
                "reason": "imu_unavailable_policy_cannot_predict",
            }
        if len(decoded) != len(discovery.samples):
            return 1, {
                "exit_code": 1,
                "status": "incomplete",
                "predicted_sample_count": len(decoded),
            }
        if args.save_intermediates:
            assert audit_staging is not None
            _finalize_online_intermediates(
                audit_staging, stage1_rows, stage2_rows, bundle
            )
        ignored_root = list(discovery.ignored_entries)
        ignored_samples = list(discovery.sample_ignored_entries)
        summary: dict[str, object] = {
            "exit_code": 0,
            "status": "success",
            "sample_count": len(manifest_rows),
            "predicted_sample_count": len(decoded),
            "unavailable_imu_count": unavailable_count,
            "ignored_entry_count": len(ignored_root) + len(ignored_samples),
            "ignored_root_entries": ignored_root,
            "ignored_sample_entries": ignored_samples,
            "seed": seed,
            "framework": f"torch-{torch.__version__}",
            "device": str(device),
            "eval_mode": not model.training,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "batch_feature_budget": config["batch_feature_budget"],
            "maximum_batch_size": config["maximum_batch_size"],
            "batch_sizes": batch_sizes,
        }
        _publish_success_transaction(
            output,
            list(zip(discovered_ids, decoded, strict=True)),
            contract,
            audit_root,
            manifest_rows,
            problems,
            summary,
            overwrite=bool(args.overwrite_output),
            prepared_audit=audit_staging,
        )
        audit_staging = None
        return 0, summary
    finally:
        if audit_staging is not None and audit_staging.exists():
            shutil.rmtree(audit_staging)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        code, summary = run(args)
    except Exception as error:
        code = 2
        summary = {"exit_code": 2, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
    except KeyboardInterrupt:
        code = 2
        summary = {"exit_code": 2, "status": "interrupted"}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
