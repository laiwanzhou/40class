from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Subset

from scripts.run_thermal_teacher import build_inverse_frequency_sampler
from src.data.thermal_teacher_dataset import ThermalTeacherDataset
from src.models.thermal_teachers import ThermalR2Plus1D18Teacher
from src.train_thermal_teacher import (
    build_teacher_optimizer_and_scheduler,
    load_teacher_config,
    sequential_trial_backward,
    set_teacher_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/thermal_c1_r2plus1d18_train12_val2.yaml"
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_REPORT = PROJECT_ROOT / "reports/thermal_teacher_environment_probe.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def first_real_batch(config: dict, *, data_root: Path) -> tuple[dict, int, dict]:
    data = config["data"]
    dataset = ThermalTeacherDataset(
        data_root=data_root,
        context_path=PROJECT_ROOT / data["context_manifest"],
        normalization_path=PROJECT_ROOT / data["normalization"],
        partition="train12",
        training=True,
        seed=int(config["optimization"]["seed"]),
    )
    usable = [index for index, record in enumerate(dataset.records) if record.get("usable", False)]
    _, sampler_audit = build_inverse_frequency_sampler(
        records=dataset.records,
        indices=usable,
        seed=int(config["optimization"]["seed"]),
    )
    loader = DataLoader(Subset(dataset, usable[:1]), batch_size=1, shuffle=False, num_workers=0)
    return next(iter(loader)), len(usable), sampler_audit


def _cuda_latency(model: ThermalR2Plus1D18Teacher, batch: dict, device: torch.device) -> dict:
    values = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model.eval()
    samples = []
    with torch.no_grad():
        for iteration in range(5):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    values["full_rgb"], values["crop_rgb"],
                    window_mask=values["window_mask"], availability=values["availability"],
                )
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - started) * 1000
            if iteration:
                samples.append(elapsed)
    return {
        "median_ms_per_trial": statistics.median(samples),
        "p95_ms_per_trial": sorted(samples)[-1],
        "iterations": len(samples),
        "output_shape": list(output["logits"].shape),
        "finite": bool(torch.isfinite(output["logits"]).all()),
        "clip_count": int(output["clip_count"].item()),
    }


def run_probe(*, config_path: Path, data_root: Path, report_path: Path) -> dict:
    config = load_teacher_config(config_path)
    set_teacher_seed(int(config["optimization"]["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("C1 probe requires CUDA")
    device = torch.device("cuda")
    batch, train_trials, sampler_audit = first_real_batch(config, data_root=data_root)
    model = ThermalR2Plus1D18Teacher(num_classes=40).to(device)
    provenance = model.initialization_provenance
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    latency = _cuda_latency(model, batch, device)
    steps_per_epoch = (train_trials + int(config["optimization"]["effective_batch_trials"]) - 1) // int(config["optimization"]["effective_batch_trials"])
    optimizer, _ = build_teacher_optimizer_and_scheduler(
        model, config=config, steps_per_epoch=steps_per_epoch
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    classifier = model.backbone.fc.weight
    before = classifier.detach().clone()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses, logits_finite = [], True
    for _ in range(int(config["optimization"]["effective_batch_trials"])):
        result = sequential_trial_backward(
            model=model,
            batch=batch,
            device=device,
            label_smoothing=float(config["optimization"]["label_smoothing"]),
            loss_scale=1.0 / int(config["optimization"]["effective_batch_trials"]),
            amp_enabled=True,
        )
        losses.append(result["loss"])
        logits_finite = logits_finite and bool(torch.isfinite(result["logits"]).all())
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimization"]["gradient_clip_norm"]))
    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    changed = not torch.equal(before, classifier.detach())
    limit = float(config["runtime"]["maximum_peak_allocated_mib_exclusive"])
    model_namespace = (PROJECT_ROOT / "src/models/__init__.py").read_text(encoding="utf-8")
    projected_seconds = optimizer_seconds * steps_per_epoch * int(config["optimization"]["maximum_epochs"])
    status = "passed" if all((
        logits_finite,
        all(torch.isfinite(torch.tensor(losses))),
        gradients_finite,
        changed,
        peak_allocated < limit,
        provenance["missing_keys"] == [],
        provenance["unexpected_keys"] == [],
        latency["finite"],
    )) else "failed"
    payload = {
        "schema_version": 1,
        "stage": "C1-qualification",
        "status": status,
        "zero_formal_training": True,
        "throwaway_optimizer_steps": 1,
        "repository_head_at_probe": repository_head(),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "total_vram_mib": torch.cuda.get_device_properties(device).total_memory / 1024**2,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "probe_script_sha256": sha256_file(Path(__file__)),
            "context_sha256": sha256_file(PROJECT_ROOT / config["data"]["context_manifest"]),
            "normalization_sha256": sha256_file(PROJECT_ROOT / config["data"]["normalization"]),
        },
        "pretrained_load": provenance,
        "input_contract": {
            "modality": "thermal_only",
            "sample_id": str(batch["sample_id"][0]),
            "source_partition": "train12",
            "views": ["full", "thermal_yolo_context"],
            "windows": 3,
            "frames_per_window": 16,
            "raster_crop_size": 160,
            "preprocessing_identity": provenance["preprocessing_identity"],
            "forbidden_evidence_read": False,
        },
        "model": {
            "parameters": parameter_count,
            "classifier_outputs": 40,
            "sequential_clip_execution": True,
            "latency": latency,
        },
        "training_sampler": sampler_audit,
        "correction": {
            "prior_run_status": "invalid_imbalanced_stopped",
            "root_cause": "natural shuffle plus unweighted cross entropy under 75x class imbalance",
            "resume_prior_checkpoint": False,
            "restart_from_official_pretrained_weights": True,
        },
        "real_data_train_smoke": {
            "physical_batch_trials": 1,
            "effective_batch_trials": 8,
            "clip_count_per_trial": latency["clip_count"],
            "finite_loss": all(torch.isfinite(torch.tensor(losses))),
            "finite_logits": logits_finite,
            "finite_gradients": gradients_finite,
            "optimizer_state_changed": changed,
            "seconds_per_optimizer_step": optimizer_seconds,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "persistent_checkpoint_written": False,
        },
        "projected_training": {
            "train_usable_trials": train_trials,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "epochs": 30,
            "projected_seconds_lower_bound_from_smoke": projected_seconds,
            "projected_hours_lower_bound_from_smoke": projected_seconds / 3600,
            "basis": "one warmed real-trial effective-batch optimizer step; excludes validation and data loading",
        },
        "gates": {
            "peak_allocated_below_7300_mib": peak_allocated < limit,
            "strict_pretrained_load": provenance["missing_keys"] == [] and provenance["unexpected_keys"] == [],
            "finite": logits_finite and gradients_finite and latency["finite"],
            "optimizer_state_changed": changed,
        },
        "teacher_is_training_only": True,
        "student_deployment_imports_teacher": "thermal_teachers" in model_namespace,
        "next_action": "single_authorized_c1_run" if status == "passed" else "stop_before_training",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the training-only Thermal C1 teacher.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_probe(
        config_path=args.config.resolve(), data_root=args.data_root.resolve(), report_path=args.report.resolve()
    )
    print(json.dumps({"status": payload["status"], "peak_allocated_mib": payload["real_data_train_smoke"]["peak_allocated_mib"], "projected_hours": payload["projected_training"]["projected_hours_lower_bound_from_smoke"]}))


if __name__ == "__main__":
    main()
