from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.thermal_native_dataset import ThermalNativeDataset
from src.train_thermal_generation2 import (
    build_generation2_student,
    build_optimizer_and_scheduler,
    checkpoint_rank,
    collect_generation2_predictions,
    load_generation2_config,
    require_training_authorization,
    run_generation2_epoch,
    set_deterministic_seed,
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
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def prepare_output_root(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("output root is non-empty; resume and overwrite are forbidden")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_loaders(
    config: dict,
    *,
    data_root: Path,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, ThermalNativeDataset, ThermalNativeDataset]:
    data = config["data"]

    def resolve(value: str) -> Path:
        return (PROJECT_ROOT / value).resolve()

    common = {
        "data_root": data_root.resolve(),
        "context_path": resolve(data["context_manifest"]),
        "normalization_path": resolve(data["normalization"]),
        "pose_cache_path": resolve(data["pose_cache"]),
        "seed": int(config["optimization"]["seed"]),
    }
    train_dataset = ThermalNativeDataset(
        **common, partition="train12", training=True
    )
    validation_dataset = ThermalNativeDataset(
        **common, partition="val_user6_user7", training=False
    )
    train_indices = [
        index for index, row in enumerate(train_dataset.records) if row.get("usable", False)
    ]
    validation_indices = [
        index
        for index, row in enumerate(validation_dataset.records)
        if row.get("usable", False)
    ]
    physical = int(config["optimization"]["physical_batch_trials"])
    generator = torch.Generator().manual_seed(int(config["optimization"]["seed"]))
    options = {
        "batch_size": physical,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        shuffle=True,
        generator=generator,
        **options,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices), shuffle=False, **options
    )
    return train_loader, validation_loader, train_dataset, validation_dataset


def compact_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"confusion_matrix", "per_class_recall"}
    }


def run_formal_training(
    config: dict,
    *,
    output_root: Path,
    data_root: Path,
    num_workers: int,
) -> dict:
    if config["route"] != "a_multistream" or config["objective"] != "direct":
        raise ValueError("this authorized A3 runner requires A-direct")
    output_root = prepare_output_root(output_root)
    seed = int(config["optimization"]["seed"])
    set_deterministic_seed(seed)
    train_loader, validation_loader, train_dataset, _ = build_loaders(
        config, data_root=data_root, num_workers=num_workers
    )
    model = build_generation2_student(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal A3 requires the A2-qualified CUDA device")
    model.to(device)
    accumulation = int(config["optimization"]["effective_batch_trials"]) // int(
        config["optimization"]["physical_batch_trials"]
    )
    steps_per_epoch = (len(train_loader) + accumulation - 1) // accumulation
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config=config, steps_per_epoch=steps_per_epoch
    )
    checkpoint_path = output_root / "selected_checkpoint.pt"
    history_path = output_root / "history.json"
    run_manifest_path = output_root / "run_manifest.json"
    history: list[dict] = []
    best_rank = None
    selected_epoch = None
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(config["optimization"]["maximum_epochs"]) + 1):
        train_dataset.set_epoch(epoch - 1)
        epoch_started = time.perf_counter()
        train_metrics = run_generation2_epoch(
            model=model,
            loader=train_loader,
            config=config,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        validation_metrics = run_generation2_epoch(
            model=model,
            loader=validation_loader,
            config=config,
            device=device,
        )
        rank = checkpoint_rank(validation_metrics, epoch=epoch)
        selected = best_rank is None or rank > best_rank
        if selected:
            best_rank = rank
            selected_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "initialization_provenance": model.initialization_provenance,
                    "teacher_logits_loaded": False,
                },
                checkpoint_path,
            )
        row = {
            "epoch": epoch,
            "selected": selected,
            "seconds": time.perf_counter() - epoch_started,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": compact_metrics(train_metrics),
            "validation": compact_metrics(validation_metrics),
        }
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
    if selected_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError("50-epoch run produced no selected checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    archive = collect_generation2_predictions(
        model=model, loader=validation_loader, config=config, device=device
    )
    archive_path = output_root / str(config["output"]["selected_prediction_archive"])
    np.savez_compressed(
        archive_path,
        **{key: value for key, value in archive.items() if key != "metrics"},
    )
    total_seconds = time.perf_counter() - started
    manifest = {
        "schema_version": 1,
        "status": "completed_50_epoch_hard_stop",
        "experiment_id": config["experiment_id"],
        "repository_head_at_start": repository_head(),
        "selected_epoch": selected_epoch,
        "selected_metrics": archive["metrics"],
        "epochs_completed": len(history),
        "automatic_resume": False,
        "automatic_extension": False,
        "teacher_logits_loaded": False,
        "train_usable_trials": len(train_loader.dataset),
        "validation_usable_trials": len(validation_loader.dataset),
        "canonical_train_trials": len(train_dataset),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_archive_path": str(archive_path),
        "prediction_archive_sha256": sha256_file(archive_path),
        "total_seconds": total_seconds,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "stream_norm_mean": (
            archive["stream_norms"].mean(axis=0).tolist()
            if "stream_norms" in archive else None
        ),
        "availability_rate": archive["availability"].mean(axis=0).tolist(),
        "history": history,
    }
    run_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strictly authorized Thermal generation-2 job.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorize-training")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_generation2_config(args.config.resolve())
    if args.validate_only:
        print(
            json.dumps(
                {
                    "experiment_id": config["experiment_id"],
                    "status": "validated_not_started",
                    "training_authorized": config["training_authorized"],
                },
                sort_keys=True,
            )
        )
        return
    require_training_authorization(config, token=args.authorize_training)
    output_root = args.output_root or PROJECT_ROOT / str(config["output"]["root"])
    manifest = run_formal_training(
        config,
        output_root=output_root,
        data_root=args.data_root,
        num_workers=args.num_workers,
    )
    print(json.dumps({"status": manifest["status"], "selected_epoch": manifest["selected_epoch"]}))


if __name__ == "__main__":
    main()
