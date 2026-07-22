#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess_imu_stage1 import MANIFEST_COLUMNS, OUTPUT_COLUMNS
from src.data.imu_stage1_bridge import stage1_manifest_row_sha256
from src.data.imu_stage2_contracts import sha256_file
from src.data.imu_stage2_io import write_action_atomic
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


def _create_audit_run(audit_root: Path | None) -> Path | None:
    if audit_root is None:
        return None
    root = Path(audit_root)
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ValueError("Audit root must be missing or empty")
    else:
        root.mkdir()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:12]
    run = root / run_id
    run.mkdir()
    return run


def _batches(samples: list[object], config: dict[str, object]) -> list[list[object]]:
    maximum = int(config["maximum_batch_size"])
    budget = int(config["batch_feature_budget"])
    result: list[list[object]] = []
    current: list[object] = []
    current_max_t = 1
    for sample in samples:
        imu_result = getattr(sample, "imu_result")
        length = 0 if imu_result is None else int(imu_result.values.shape[0])
        candidate_t = max(current_max_t, length, 1)
        if current and (len(current) >= maximum or candidate_t * 5 * 16 * (len(current) + 1) > budget):
            result.append(current)
            current = []
            current_max_t = 1
        current.append(sample)
        current_max_t = max(current_max_t, length, 1)
    if current:
        result.append(current)
    return result


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


def _save_real_intermediate(
    run: Path,
    sample: object,
    stage1: object | None,
    bundle: InferenceBundle,
) -> None:
    result = getattr(sample, "imu_result")
    if result is None or stage1 is None:
        return
    root = run / "intermediates" / str(getattr(sample, "sample_id"))
    stage1_root = root / "stage1"
    stage1_action = stage1_root / "action"
    stage2_root = root / "stage2"
    stage1_action.mkdir(parents=True)
    stage2_root.mkdir()
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
        "sample_id": str(getattr(sample, "sample_id")),
        "class_id": "" if stage1.class_id is None else stage1.class_id,
        "class_name": "" if stage1.class_name is None else stage1.class_name,
        "user_id": "" if stage1.user_id is None else stage1.user_id,
        "action_id": "" if stage1.action_id is None else stage1.action_id,
        "relative_action_path": "action",
        "output_csv": "action/imu_merged.csv",
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
    manifest_path = stage1_root / "manifest.csv"
    with manifest_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(manifest_row)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        loaded_row = next(csv.DictReader(handle))
    shutil.copyfile(bundle.paths["stage2_schema"], stage2_root / "schema.json")
    write_action_atomic(
        stage2_root,
        Path("action"),
        result,
        {
            "stage1_output_csv_sha256": sha256_file(stage1_csv),
            "stage1_qc_sha256": sha256_file(stage1_qc),
            "stage1_manifest_row_sha256": stage1_manifest_row_sha256(loaded_row),
            "stage2_contract_sha256": str(bundle.stage2_schema["contract_sha256"]),
        },
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


def run(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    output = Path(args.output_csv)
    if output.exists() and not args.overwrite_output:
        raise FileExistsError(output)
    if not output.parent.exists() or not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if args.save_intermediates and args.audit_dir is None:
        raise ValueError("--save-intermediates requires --audit-dir")
    bundle = load_inference_bundle(args.bundle_root)
    config = _validate_config(bundle)
    audit_run = _create_audit_run(args.audit_dir)
    discovery = discover_test_samples(args.raw_test_root)
    contract = bundle.submission_contract["contract"]
    if not isinstance(contract, dict):
        raise ValueError("Submission contract payload is invalid")
    expected_ids = list(contract["sample_ids"])
    discovered_ids = [descriptor.sample_id for descriptor in discovery.samples]
    if discovered_ids != expected_ids:
        raise ValueError("Discovered test IDs do not match submission contract")

    samples: list[object] = []
    manifest_rows: list[dict[str, object]] = []
    problems: list[dict[str, object]] = []
    for descriptor in discovery.samples:
        sample, error, stage1 = preprocess_inference_sample_with_diagnostics(
            descriptor,
            hard_safety_limit_t=int(config["hard_safety_limit_t"]),
        )
        samples.append(sample)
        if args.save_intermediates and audit_run is not None:
            _save_real_intermediate(audit_run, sample, stage1, bundle)
        failure_reason = "" if error is None else str(getattr(error, "error_code", type(error).__name__))
        if error is not None:
            problems.append(
                {
                    "sample_id": descriptor.sample_id,
                    "error_code": failure_reason,
                    "failure_stage": str(getattr(error, "failure_stage", "unknown")),
                    "safe_message": str(getattr(error, "safe_message", str(error))),
                }
            )
        imu_result = sample.imu_result
        warnings = [] if imu_result is None else list(imu_result.qc.get("warning_codes", []))
        manifest_rows.append(
            {
                "sample_id": descriptor.sample_id,
                "source_status": "available" if error is None else "unavailable",
                "stage1_status": "success" if stage1 is not None else "unavailable",
                "stage2_status": "success" if imu_result is not None else "unavailable",
                "imu_available": sample.imu_available,
                "modality_mask": sample.modality_mask,
                "failure_reason": failure_reason,
                "warning_codes": json.dumps(warnings, separators=(",", ":")),
                "sequence_length": 0 if imu_result is None else len(imu_result.values),
                "prediction": "",
            }
        )

    if any(not sample.imu_available for sample in samples) and config[
        "imu_unavailable_policy"
    ] != "packaged_null_embedding":
        summary = {
            "exit_code": 1,
            "status": "incomplete",
            "sample_count": len(samples),
            "predicted_sample_count": 0,
            "reason": "imu_unavailable_policy_cannot_predict",
        }
        if audit_run is not None:
            _write_audit(audit_run, manifest_rows, problems, summary)
        return 1, summary

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
    with torch.inference_mode():
        for group in _batches(samples, config):
            batch_sizes.append(len(group))
            batch = _normalize_batch(collate_inference_samples(group), bundle, device)
            output_mapping = model(batch)
            logits = output_mapping.get("logits")
            validated = validate_logits(logits, batch_size=len(group), num_classes=bundle.class_order.num_classes)
            decoded.extend(decode_predictions(validated, bundle.class_order, contract))
    if len(decoded) != len(samples):
        summary = {"exit_code": 1, "status": "incomplete", "predicted_sample_count": len(decoded)}
        if audit_run is not None:
            _write_audit(audit_run, manifest_rows, problems, summary)
        return 1, summary
    for row, prediction in zip(manifest_rows, decoded, strict=True):
        row["prediction"] = prediction
    summary: dict[str, object] = {
        "exit_code": 0,
        "status": "success",
        "sample_count": len(samples),
        "predicted_sample_count": len(decoded),
        "unavailable_imu_count": sum(not sample.imu_available for sample in samples),
        "ignored_entry_count": len(discovery.ignored_entries),
        "seed": seed,
        "framework": f"torch-{torch.__version__}",
        "device": str(device),
        "eval_mode": not model.training,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "batch_feature_budget": config["batch_feature_budget"],
        "maximum_batch_size": config["maximum_batch_size"],
        "batch_sizes": batch_sizes,
    }
    if audit_run is not None:
        _write_audit(audit_run, manifest_rows, problems, summary)
    write_submission_atomic(
        output,
        list(zip(discovered_ids, decoded, strict=True)),
        contract,
        overwrite=bool(args.overwrite_output),
    )
    return 0, summary


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
