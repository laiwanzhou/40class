from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import torch

from scripts.audit_thermal_v2_inputs import (
    YoloPosePredictor,
    build_pose_lookup,
    build_trial_tensors,
    load_jsonl,
)
from src.train_thermal_generation2 import (
    build_generation2_student,
    load_generation2_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_CONTEXT = PROJECT_ROOT / "metadata/thermal/thermal_v2_trial_context.jsonl"
DEFAULT_NORMALIZATION = PROJECT_ROOT / "metadata/thermal/thermal_v2_train12_normalization.json"
DEFAULT_YOLO = Path(r"D:\work\2026.7.14_kaggle\40class\yolo11n-pose.pt")
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/thermal_generation2_environment_probe.json"
VRAM_LIMIT_MIB = 7300.0
PACKAGE_LIMIT_BYTES = 95_000_000


class ProvenancePolicyError(ValueError):
    """Raised when a deployable student manifest contains forbidden provenance."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_ledger(paths: Sequence[Path]) -> dict[str, Any]:
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    unique_bytes = 0
    for path in paths:
        resolved = path.resolve()
        digest = sha256_file(resolved)
        size = resolved.stat().st_size
        counted = digest not in seen
        if counted:
            unique_bytes += size
            seen.add(digest)
        items.append(
            {
                "path": str(resolved),
                "sha256": digest,
                "bytes": size,
                "counted": counted,
            }
        )
    return {
        "items": items,
        "unique_sha256_count": len(seen),
        "unique_bytes": unique_bytes,
    }


def _has_pretrained_true(value: Any, *, key_path: str = "") -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{key_path}.{key}" if key_path else str(key)
            if "pretrained" in str(key).casefold() and nested is True:
                return path
            found = _has_pretrained_true(nested, key_path=path)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found = _has_pretrained_true(nested, key_path=f"{key_path}[{index}]")
            if found:
                return found
    return None


def scan_student_provenance(model: torch.nn.Module, deployment_files: Sequence[Path]) -> None:
    provenance = getattr(model, "initialization_provenance", None)
    if not isinstance(provenance, dict):
        raise ProvenancePolicyError("student initialization provenance is missing")
    forbidden = _has_pretrained_true(provenance)
    if forbidden:
        raise ProvenancePolicyError(f"pretrained student provenance is forbidden: {forbidden}")
    for path in deployment_files:
        if "teacher" in str(path).casefold():
            raise ProvenancePolicyError(f"teacher asset is forbidden from deployment: {path}")


def state_dict_proxy(model: torch.nn.Module) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    content = buffer.getvalue()
    return {"bytes": len(content), "sha256": sha256_bytes(content)}


def _model_inputs(batch: dict[str, torch.Tensor], route: str) -> dict[str, torch.Tensor]:
    if route == "b_x3d_xs":
        return {
            "full_rgb": batch["full_rgb"],
            "window_mask": batch["window_mask"],
            "availability": batch["availability"],
            "quality": batch["quality"],
        }
    return batch


def _forward(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], route: str
) -> dict[str, torch.Tensor]:
    values = _model_inputs(batch, route)
    if route == "b_x3d_xs":
        return model(
            values["full_rgb"],
            window_mask=values["window_mask"],
            availability=values["availability"],
            quality=values["quality"],
        )
    return model(**values)


def _slice_to_device(
    batch: dict[str, torch.Tensor], *, count: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value[:count].to(device) for key, value in batch.items()}


def canonicalize_model_batch(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output = dict(batch)
    for key in ("full_rgb", "crop_rgb"):
        value = output[key]
        if value.ndim != 6 or value.shape[2:4] != (16, 3):
            raise ValueError(f"{key} audit tensor must have shape [B,W,16,3,H,W]")
        output[key] = value.permute(0, 1, 3, 2, 4, 5).contiguous()
    return output


def _latency_summary(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    route: str,
    *,
    iterations: int,
) -> dict[str, float]:
    for _ in range(2):
        _forward(model, batch, route)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _forward(model, batch, route)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / batch["full_rgb"].shape[0])
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms_per_trial": statistics.median(samples),
        "p95_ms_per_trial": ordered[p95_index],
        "iterations": iterations,
    }


def probe_model(
    *,
    config: dict[str, Any],
    cpu_batch: dict[str, torch.Tensor],
    device: torch.device,
    physical_batch: int,
    latency_iterations: int,
    deployment_files: Sequence[Path],
) -> dict[str, Any]:
    model = build_generation2_student(config)
    scan_student_provenance(model, deployment_files)
    proxy = state_dict_proxy(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    batch = _slice_to_device(cpu_batch, count=physical_batch, device=device)
    model = model.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.inference_mode():
        output = _forward(model, batch, config["route"])
        finite_inference = bool(torch.isfinite(output["logits"]).all())
        output_shape = list(output["logits"].shape)
        latency = _latency_summary(
            model,
            batch,
            config["route"],
            iterations=latency_iterations,
        )
    inference_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    inference_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    accumulation_steps = 8 // physical_batch
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    finite_train = True
    last_loss = None
    for _ in range(accumulation_steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            train_output = _forward(model, batch, config["route"])
            loss = train_output["logits"].float().square().mean() / accumulation_steps
        finite_train = finite_train and bool(torch.isfinite(loss))
        loss.backward()
        last_loss = loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    train_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    train_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    finite_train = finite_train and last_loss is not None and all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )

    file_ledger = deployment_ledger(deployment_files)
    complete_bytes = proxy["bytes"] + file_ledger["unique_bytes"]
    result = {
        "parameters": parameters,
        "state_dict_proxy": proxy,
        "physical_batch_trials": physical_batch,
        "fp32_inference": {
            "output_shape": output_shape,
            "finite": finite_inference,
            "peak_allocated_mib": inference_allocated,
            "peak_reserved_mib": inference_reserved,
            "latency": latency,
        },
        "bfloat16_train_smoke": {
            "finite": bool(finite_train),
            "optimizer_steps": 1,
            "accumulation_steps": accumulation_steps,
            "effective_batch_trials": physical_batch * accumulation_steps,
            "seconds_per_optimizer_step": train_seconds,
            "peak_allocated_mib": train_allocated,
            "peak_reserved_mib": train_reserved,
            "persistent_checkpoint_written": False,
        },
        "deployment": {
            "state_dict_bytes": proxy["bytes"],
            "file_ledger": file_ledger,
            "complete_package_bytes": complete_bytes,
            "limit_bytes_exclusive": PACKAGE_LIMIT_BYTES,
            "pretrained_student_weights": False,
            "teacher_assets": [],
        },
    }
    result["gates"] = {
        "finite": finite_inference and bool(finite_train),
        "vram": train_allocated < VRAM_LIMIT_MIB,
        "deployment": complete_bytes < PACKAGE_LIMIT_BYTES,
    }
    del optimizer, model, batch, output
    torch.cuda.empty_cache()
    return result


def build_real_batch(
    *,
    data_root: Path,
    context_path: Path,
    normalization_path: Path,
    yolo_weights: Path,
    count: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    records = load_jsonl(context_path)
    candidates = sorted(
        (
            row
            for row in records
            if row["development_split"] == "train12"
            and row["usable"]
            and row["context_available"]
        ),
        key=lambda row: str(row["sample_id"]),
    )
    selected = candidates[:count]
    if len(selected) < count:
        raise ValueError("not enough real train12 Thermal trials for the probe")
    pose_lookup, pose_jobs = build_pose_lookup(
        selected,
        data_root=data_root,
        predictor=YoloPosePredictor(yolo_weights, device=device),
        batch_size=32,
    )
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    mean = torch.tensor(normalization["rgb_mean"], dtype=torch.float32)
    std = torch.tensor(normalization["rgb_std"], dtype=torch.float32)
    rows = [
        build_trial_tensors(
            record,
            data_root=data_root,
            pose_lookup=pose_lookup,
            mean=mean,
            std=std,
        )
        for record in selected
    ]
    batch = {
        key: torch.stack([row[key] for row in rows])
        for key in (
            "full_rgb",
            "crop_rgb",
            "motion",
            "pose",
            "pose_mask",
            "availability",
            "quality",
        )
    }
    batch["window_mask"] = torch.ones(count, 3, dtype=torch.bool)
    return canonicalize_model_batch(batch), {
        "real_thermal_trials": count,
        "sample_ids": [str(row["sample_id"]) for row in selected],
        "pose_frame_jobs": pose_jobs,
        "pose_available_trials": int(batch["availability"][:, 3].sum().item()),
        "labels_or_metrics_used_for_selection": False,
        "source_partition": "train12",
        "forbidden_evidence_read": False,
    }


def _driver_version() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        return output.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return None


def _repository_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def projected_times(models: dict[str, Any], *, train_trials: int = 1922) -> dict[str, Any]:
    steps = math.ceil(train_trials / 8)
    b_seconds = models["b_x3d_xs"]["bfloat16_train_smoke"]["seconds_per_optimizer_step"]
    a_seconds = models["a_multistream"]["bfloat16_train_smoke"]["seconds_per_optimizer_step"]

    def estimate(seconds: float, epochs: int, basis: str, note: str) -> dict[str, Any]:
        return {
            "estimated_hours": seconds * steps * epochs / 3600,
            "steps_per_epoch": steps,
            "epochs": epochs,
            "basis": basis,
            "note": note,
        }

    return {
        "b_x3d_xs": estimate(b_seconds, 50, "measured", "single A2 optimizer step"),
        "a_direct": estimate(a_seconds, 50, "measured", "single A2 optimizer step"),
        "a_kd": estimate(a_seconds * 1.02, 50, "heuristic", "A step plus small KL overhead"),
        "c1_r2plus1d18": estimate(
            a_seconds * 5.0,
            30,
            "heuristic",
            "unmeasured 5x A-step multiplier; Route C must probe independently",
        ),
        "c2_videomae_s": estimate(
            a_seconds * 10.0,
            30,
            "heuristic",
            "conditional unmeasured 10x A-step multiplier; Route C must probe independently",
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Thermal generation-2 runtime without training.")
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--physical-batch", type=int, default=2)
    parser.add_argument("--latency-iterations", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("A2 requires the reference CUDA GPU")
    configs = [load_generation2_config(path.resolve()) for path in args.config]
    by_route = {config["route"]: config for config in configs}
    if set(by_route) != {"b_x3d_xs", "a_multistream"}:
        raise ValueError("A2 requires exactly B-X3D-XS and A-multistream configs")
    cpu_batch, input_contract = build_real_batch(
        data_root=args.data_root.resolve(),
        context_path=args.context.resolve(),
        normalization_path=args.normalization.resolve(),
        yolo_weights=args.yolo_weights.resolve(),
        count=args.physical_batch,
        device=args.yolo_device,
    )
    device = torch.device("cuda")
    model_files = {
        "b_x3d_xs": [
            PROJECT_ROOT / "src/models/thermal_x3d_xs.py",
            PROJECT_ROOT / "src/data/thermal_v2_sampling.py",
            args.normalization.resolve(),
        ],
        "a_multistream": [
            PROJECT_ROOT / "src/models/thermal_x3d_xs.py",
            PROJECT_ROOT / "src/models/thermal_multistream.py",
            PROJECT_ROOT / "src/data/thermal_v2_features.py",
            PROJECT_ROOT / "src/data/thermal_v2_sampling.py",
            PROJECT_ROOT / "src/roi/thermal_trial_context.py",
            args.normalization.resolve(),
            args.yolo_weights.resolve(),
        ],
    }
    models: dict[str, Any] = {}
    for route in ("b_x3d_xs", "a_multistream"):
        try:
            models[route] = probe_model(
                config=by_route[route],
                cpu_batch=cpu_batch,
                device=device,
                physical_batch=args.physical_batch,
                latency_iterations=args.latency_iterations,
                deployment_files=model_files[route],
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if args.physical_batch == 1:
                raise
            models[route] = probe_model(
                config=by_route[route],
                cpu_batch=cpu_batch,
                device=device,
                physical_batch=1,
                latency_iterations=args.latency_iterations,
                deployment_files=model_files[route],
            )
            models[route]["fallback_reason"] = "physical_batch_2_cuda_oom"
    passed = all(all(result["gates"].values()) for result in models.values())
    properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "stage": "A2",
        "status": "passed" if passed else "failed",
        "zero_formal_training": True,
        "throwaway_optimizer_steps": len(models),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "total_vram_mib": properties.total_memory / (1024**2),
            "driver_version": _driver_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "provenance": {
            "repository_head_at_probe": _repository_head(),
            "probe_script_sha256": sha256_file(Path(__file__).resolve()),
            "config_sha256": {
                config["experiment_id"]: sha256_file(Path(config["config_path"]))
                for config in configs
            },
            "context_sha256": sha256_file(args.context.resolve()),
            "normalization_sha256": sha256_file(args.normalization.resolve()),
            "yolo_weights_sha256": sha256_file(args.yolo_weights.resolve()),
        },
        "measurement_scope": {
            "latency": "model_only_preprocessed_tensors",
            "online_yolo_latency_measured": False,
            "train_smoke_effective_batch": "repeated_real_probe_trials_for_memory_only",
        },
        "input_contract": input_contract,
        "models": models,
        "projected_training_time": projected_times(models),
        "gates": {
            "peak_allocated_mib_max_exclusive": VRAM_LIMIT_MIB,
            "complete_package_bytes_max_exclusive": PACKAGE_LIMIT_BYTES,
            "all_models_passed": passed,
        },
        "evidence_boundary": {
            "heldout4_labels_read": False,
            "competition_test_read": False,
            "quarantined_evidence_read": False,
            "ir_depth_inputs_read": False,
            "ir_x3d_modified": False,
            "persistent_learned_weights_written": False,
        },
        "a3_prerequisites": {
            "route_b_report_verified": False,
            "full_pose_cache_present": False,
            "a_direct_training_authorized": False,
        },
        "next_action": (
            "route_b_and_pose_cache_before_a3_authorization"
            if passed
            else "stop_runtime_gate_failed"
        ),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
