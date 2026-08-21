from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.data.thermal_teacher_dataset import ThermalTeacherDataset
from src.models.thermal_teachers import ThermalR2Plus1D18Teacher
from src.train_thermal_generation2 import checkpoint_rank, fixed_label_metrics
from src.train_thermal_teacher import (
    build_teacher_optimizer_and_scheduler,
    collect_teacher_predictions,
    load_teacher_config,
    require_teacher_training_authorization,
    sequential_trial_backward,
    set_teacher_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def prepare_output_root(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("output root is non-empty; resume and overwrite are forbidden")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_inverse_frequency_sampler(
    *, records: list[dict[str, Any]], indices: list[int], seed: int
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    if not indices:
        raise ValueError("class-balanced sampler requires at least one training sample")
    labels = [int(records[index]["class_id"]) for index in indices]
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )
    total_mass = float(weights.sum())
    class_mass = {
        str(class_id): float(weights[torch.tensor(labels) == class_id].sum()) / total_mass
        for class_id in sorted(counts)
    }
    audit = {
        "policy": "inverse_frequency_weighted_random_replacement",
        "basis": "train12_usable_class_id",
        "replacement": True,
        "epoch_samples": len(indices),
        "class_counts": {str(class_id): counts[class_id] for class_id in sorted(counts)},
        "class_probability_mass": class_mass,
        "validation_sampling": "natural_once",
    }
    return sampler, audit


def build_loaders(
    config: dict, *, data_root: Path, num_workers: int
) -> tuple[DataLoader, DataLoader, ThermalTeacherDataset, dict[str, Any]]:
    data = config["data"]
    common = {
        "data_root": data_root.resolve(),
        "context_path": (PROJECT_ROOT / data["context_manifest"]).resolve(),
        "normalization_path": (PROJECT_ROOT / data["normalization"]).resolve(),
        "seed": int(config["optimization"]["seed"]),
    }
    train_dataset = ThermalTeacherDataset(**common, partition="train12", training=True)
    validation_dataset = ThermalTeacherDataset(**common, partition="val_user6_user7", training=False)
    train_indices = [index for index, record in enumerate(train_dataset.records) if record.get("usable", False)]
    validation_indices = [index for index, record in enumerate(validation_dataset.records) if record.get("usable", False)]
    sampler, sampler_audit = build_inverse_frequency_sampler(
        records=train_dataset.records,
        indices=train_indices,
        seed=int(config["optimization"]["seed"]),
    )
    options = {
        "batch_size": 1,
        "num_workers": num_workers,
        "pin_memory": True,
        # Recreate workers each epoch so set_epoch changes clip-consistent augmentation.
        "persistent_workers": False,
    }
    train = DataLoader(
        Subset(train_dataset, train_indices), sampler=sampler, shuffle=False, **options
    )
    validation = DataLoader(Subset(validation_dataset, validation_indices), shuffle=False, **options)
    return train, validation, train_dataset, sampler_audit


def run_train_epoch(*, model: ThermalR2Plus1D18Teacher, loader: DataLoader, config: dict, device: torch.device, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int) -> dict[str, Any]:
    model.train()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(config["optimization"]["effective_batch_trials"])
    logits, labels, users = [], [], []
    losses, optimizer_steps = [], 0
    for index, batch in enumerate(loader, start=1):
        result = sequential_trial_backward(
            model=model,
            batch=batch,
            device=device,
            label_smoothing=float(config["optimization"]["label_smoothing"]),
            loss_scale=1.0 / accumulation,
            amp_enabled=True,
        )
        logits.append(result["logits"].cpu().numpy())
        labels.append(batch["label"].numpy())
        users.extend(str(value) for value in batch["user_id"])
        losses.append(result["loss"])
        if index % accumulation == 0 or index == len(loader):
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise FloatingPointError("non-finite teacher gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimization"]["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if index % 100 == 0 or index == len(loader):
            print(json.dumps({
                "event": "train_progress",
                "epoch": epoch,
                "trials_completed": index,
                "trials_total": len(loader),
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)
    merged_logits = np.concatenate(logits).astype(np.float32)
    merged_labels = np.concatenate(labels).astype(np.int64)
    metrics = fixed_label_metrics(labels=merged_labels, logits=merged_logits, users=np.asarray(users))
    metrics.update({
        "loss": float(np.mean(losses)),
        "eligible_samples": len(merged_labels),
        "optimizer_steps": optimizer_steps,
        "metric_semantics": "online_logits_from_changing_model_states_not_epoch_end",
        "sampled_class_counts": np.bincount(merged_labels, minlength=40).tolist(),
        "predicted_class_counts": np.bincount(
            merged_logits.argmax(axis=1), minlength=40
        ).tolist(),
    })
    return metrics


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"confusion_matrix", "per_class_recall"}}


def build_epoch_history_row(
    *,
    epoch: int,
    selected: bool,
    seconds: float,
    learning_rates: list[float],
    online_train: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "selected": selected,
        "seconds": seconds,
        "learning_rates": learning_rates,
        "online_train": online_train,
        "validation": validation,
    }


def write_reports(manifest: dict[str, Any]) -> None:
    json_path = PROJECT_ROOT / "reports/thermal_c1_r2plus1d18_train12_val2.json"
    md_path = PROJECT_ROOT / "reports/thermal_c1_r2plus1d18_train12_val2.md"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metrics = manifest["selected_metrics"]
    gate = manifest["teacher_gate"]
    lines = [
        "# Thermal C1 R(2+1)D-18 train12 / user6-user7 report",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Repository commit at start: `{manifest['repository_head_at_start']}`",
        f"- Selected epoch: `{manifest['selected_epoch']}` of `30`",
        f"- Accuracy: `{metrics['accuracy']:.5f}`",
        f"- Macro-F1: `{metrics['macro_f1']:.5f}`",
        f"- Worst-user Accuracy: `{metrics['worst_user_accuracy']:.5f}`",
        f"- user6/user7 Accuracy: `{metrics['user_accuracy'].get('user6', 0):.5f}` / `{metrics['user_accuracy'].get('user7', 0):.5f}`",
        f"- NLL: `{metrics['nll']:.5f}`",
        f"- Zero-recall classes: `{metrics['zero_recall_classes']}/40`",
        f"- Teacher gate passed: `{gate['passed']}`",
        "- Training sampler: `inverse_frequency_weighted_random_replacement` over usable train12 class IDs",
        "- Validation sampling: `natural_once`",
        f"- Checkpoint: `{manifest['checkpoint_bytes']}` bytes, SHA256 `{manifest['checkpoint_sha256']}`",
        f"- CUDA peak allocated/reserved: `{manifest['cuda_peak_allocated_mib']:.2f}` / `{manifest['cuda_peak_reserved_mib']:.2f}` MiB",
        f"- Total runtime: `{manifest['total_seconds'] / 3600:.2f}` hours",
        "",
        "The model consumed Thermal full-frame and Thermal YOLO context views only. It did not read heldout labels, competition test, quarantined evidence, IR, or Depth. The checkpoint and teacher runtime are training-only and are excluded from the student deployment package.",
        "",
        "C2 and teacher-logit export remain separately gated. This run does not authorize either action.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_formal_training(config: dict, *, output_root: Path, data_root: Path, num_workers: int) -> dict[str, Any]:
    output_root = prepare_output_root(output_root)
    launch_manifest_path = output_root / "launch_manifest.json"
    launch_manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "running",
        "experiment_id": config["experiment_id"],
        "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_start": repository_head(),
        "automatic_resume": False,
        "automatic_extension": False,
        "maximum_epochs": 30,
    }, indent=2) + "\n", encoding="utf-8")
    set_teacher_seed(int(config["optimization"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal C1 requires the qualified CUDA device")
    train_loader, validation_loader, train_dataset, sampler_audit = build_loaders(
        config, data_root=data_root, num_workers=num_workers
    )
    model = ThermalR2Plus1D18Teacher(num_classes=40).to(device)
    accumulation = int(config["optimization"]["effective_batch_trials"])
    steps_per_epoch = (len(train_loader) + accumulation - 1) // accumulation
    optimizer, scheduler = build_teacher_optimizer_and_scheduler(model, config=config, steps_per_epoch=steps_per_epoch)
    checkpoint_path = output_root / "selected_checkpoint.pt"
    history_path = output_root / "history.json"
    run_manifest_path = output_root / "run_manifest.json"
    history: list[dict[str, Any]] = []
    best_rank, selected_epoch = None, None
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, 31):
        train_dataset.set_epoch(epoch - 1)
        epoch_started = time.perf_counter()
        train_metrics = run_train_epoch(model=model, loader=train_loader, config=config, device=device, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        validation_archive = collect_teacher_predictions(model=model, loader=validation_loader, device=device)
        validation_metrics = validation_archive["metrics"]
        rank = checkpoint_rank(validation_metrics, epoch=epoch)
        selected = best_rank is None or rank > best_rank
        if selected:
            best_rank, selected_epoch = rank, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "initialization_provenance": model.initialization_provenance,
                "training_only": True,
            }, checkpoint_path)
        row = build_epoch_history_row(
            epoch=epoch,
            selected=selected,
            seconds=time.perf_counter() - epoch_started,
            learning_rates=[float(group["lr"]) for group in optimizer.param_groups],
            online_train=compact(train_metrics),
            validation=compact(validation_metrics),
        )
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
    if selected_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError("30-epoch C1 run produced no selected checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    selected = collect_teacher_predictions(model=model, loader=validation_loader, device=device)
    archive_path = output_root / config["output"]["selected_prediction_archive"]
    np.savez_compressed(archive_path, **{key: value for key, value in selected.items() if key != "metrics"})
    metrics = selected["metrics"]
    teacher_gate = {
        "accuracy_at_least_0_60": metrics["accuracy"] >= 0.60,
        "macro_f1_at_least_0_45": metrics["macro_f1"] >= 0.45,
        "worst_user_at_least_0_50": metrics["worst_user_accuracy"] >= 0.50,
    }
    teacher_gate["passed"] = all(teacher_gate.values())
    manifest = {
        "schema_version": 1,
        "status": "completed_30_epoch_hard_stop",
        "experiment_id": config["experiment_id"],
        "repository_head_at_start": repository_head(),
        "authorization": config["authorization"],
        "selected_epoch": selected_epoch,
        "selected_metrics": metrics,
        "epochs_completed": len(history),
        "automatic_resume": False,
        "automatic_extension": False,
        "modality": "thermal_only",
        "forbidden_evidence_read": False,
        "teacher_is_training_only": True,
        "student_deployment_bytes_added": 0,
        "train_usable_trials": len(train_loader.dataset),
        "validation_usable_trials": len(validation_loader.dataset),
        "canonical_train_trials": len(train_dataset),
        "training_sampler": sampler_audit,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_archive_path": str(archive_path),
        "prediction_archive_sha256": sha256_file(archive_path),
        "total_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "teacher_gate": teacher_gate,
        "c2_status": "skipped_by_rule" if teacher_gate["passed"] else "requires_separate_approval",
        "teacher_logits_export_status": "eligible_for_separate_stage" if teacher_gate["passed"] else "blocked_by_teacher_quality",
        "history": history,
    }
    run_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_reports(manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single authorized Thermal C1 teacher job.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorize-training")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_teacher_config(args.config.resolve())
    if args.validate_only:
        print(json.dumps({"status": "validated_not_started", "experiment_id": config["experiment_id"], "training_authorized": config["training_authorized"]}))
        return
    require_teacher_training_authorization(config, token=args.authorize_training)
    output_root = args.output_root or PROJECT_ROOT / config["output"]["root"]
    manifest = run_formal_training(config, output_root=output_root, data_root=args.data_root, num_workers=args.num_workers)
    print(json.dumps({"status": manifest["status"], "selected_epoch": manifest["selected_epoch"], "teacher_gate_passed": manifest["teacher_gate"]["passed"]}))


if __name__ == "__main__":
    main()
