from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import IMUDataset, RadarDataset, SkeletonDataset, VisualSequenceDataset, load_modality_frames
from src.data.common import compute_sequence_normalization
from src.engine import collect_predictions, run_epoch
from src.models import TemporalClassifier, VisualBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
VISUAL_MODALITIES = {"Depth_Color", "IR", "Thermal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one CUHK-X unimodal baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fold", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-id", type=str)
    return parser.parse_args()


def project_path(path: Path | str, base: Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def load_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    resolved_config = project_path(config_path)
    config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {resolved_config}")
    config["config_path"] = str(resolved_config)
    config["data_root"] = str((args.data_root or Path(config.get("data_root", DEFAULT_DATA_ROOT))).resolve())
    config["manifest"] = str(project_path(args.manifest or config.get("manifest", "metadata/manifest.csv")))
    config["fold"] = str(project_path(args.fold or config.get("fold", "metadata/splits/fold_0.json")))
    config["output_root"] = str(project_path(args.output_root or config.get("output_root", "outputs/task03")))
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_epochs is not None:
        config["epochs"] = args.max_epochs
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    config["smoke_test"] = bool(args.smoke_test)
    config["max_train_batches"] = args.max_train_batches
    config["max_val_batches"] = args.max_val_batches
    config["requested_run_id"] = args.run_id
    if int(config.get("num_workers", 0)) != 0:
        raise ValueError("This Windows project requires num_workers=0.")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive.")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_dir(config: dict[str, Any]) -> Path:
    modality_slug = str(config["modality"]).casefold()
    requested = config.get("requested_run_id")
    base_run_id = requested or f"{datetime.now():%Y%m%d_%H%M%S}_{modality_slug}_{config['model_name']}_fold0"
    modality_root = Path(config["output_root"]) / modality_slug
    modality_root.mkdir(parents=True, exist_ok=True)
    run_dir = modality_root / base_run_id
    suffix = 1
    while run_dir.exists():
        run_dir = modality_root / f"{base_run_id}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def build_datasets(config: dict[str, Any]) -> tuple[Dataset[dict[str, object]], Dataset[dict[str, object]]]:
    train_frame, val_frame = load_modality_frames(
        Path(config["manifest"]),
        Path(config["fold"]),
        Path(config["data_root"]),
        str(config["path_column"]),
    )
    modality = str(config["modality"])
    if modality in VISUAL_MODALITIES:
        return (
            VisualSequenceDataset(train_frame, modality, int(config["num_frames"]), int(config["image_size"])),
            VisualSequenceDataset(val_frame, modality, int(config["num_frames"]), int(config["image_size"])),
        )
    if modality == "IMU":
        return IMUDataset(train_frame, int(config["sequence_length"])), IMUDataset(val_frame, int(config["sequence_length"]))
    if modality == "Skeleton":
        return SkeletonDataset(train_frame, int(config["sequence_length"])), SkeletonDataset(val_frame, int(config["sequence_length"]))
    if modality == "Radar":
        return RadarDataset(train_frame, int(config["sequence_length"])), RadarDataset(val_frame, int(config["sequence_length"]))
    raise ValueError(f"Unsupported modality: {modality}")


def configure_normalization(
    config: dict[str, Any],
    train_dataset: Dataset[dict[str, object]],
    val_dataset: Dataset[dict[str, object]],
    run_dir: Path,
) -> None:
    if str(config["modality"]) in VISUAL_MODALITIES:
        return
    max_samples = 64 if config["smoke_test"] else None
    mean, std = compute_sequence_normalization(train_dataset, max_samples=max_samples)  # type: ignore[arg-type]
    train_dataset.set_normalization(mean, std)  # type: ignore[attr-defined]
    val_dataset.set_normalization(mean, std)  # type: ignore[attr-defined]
    (run_dir / "normalization_stats.json").write_text(
        json.dumps(
            {"source": "fold_0 train_users only", "samples_used": min(len(train_dataset), max_samples or len(train_dataset)), "mean": mean.tolist(), "std": std.tolist()},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def build_model(config: dict[str, Any], sample: dict[str, object]) -> nn.Module:
    embedding_dim = int(config.get("embedding_dim", 128))
    if str(config["modality"]) in VISUAL_MODALITIES:
        return VisualBaseline(embedding_dim=embedding_dim, dropout=float(config.get("dropout", 0.2)))
    input_tensor = sample["input"]
    if not isinstance(input_tensor, torch.Tensor) or input_tensor.ndim != 2:
        raise ValueError(f"Expected sequence tensor [T,F], got {type(input_tensor)}")
    channels = tuple(int(value) for value in config.get("tcn_channels", [64, 128]))
    return TemporalClassifier(
        input_features=int(input_tensor.shape[-1]),
        embedding_dim=embedding_dim,
        channels=channels,
        dropout=float(config.get("dropout", 0.2)),
    )


def loader_for(
    dataset: Dataset[dict[str, object]],
    config: dict[str, Any],
    training: bool,
) -> DataLoader[dict[str, object]]:
    actual: Dataset[dict[str, object]] = dataset
    if config["smoke_test"]:
        limit = min(len(dataset), 16 if training else 8)
        actual = Subset(dataset, range(limit))
    generator = torch.Generator().manual_seed(int(config["seed"]) + (0 if training else 1))
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        actual,
        batch_size=min(int(config["batch_size"]), len(actual)),
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_accuracy: float) -> None:
    torch.save({"epoch": epoch, "val_accuracy": val_accuracy, "model_state_dict": model.state_dict()}, path)


def save_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(np.asarray(matrix), interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("fold_0 validation confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def measure_inference_ms(model: nn.Module, sample: dict[str, object], device: torch.device, amp_enabled: bool) -> float:
    inputs = sample["input"].unsqueeze(0).to(device)  # type: ignore[union-attr]
    mask = sample["temporal_mask"].unsqueeze(0).to(device)  # type: ignore[union-attr]
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                model(inputs, temporal_mask=mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                model(inputs, temporal_mask=mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / 10.0


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; environment was not modified.")
    device = torch.device(requested_device)
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_dir = create_run_dir(config)
    config["run_dir"] = str(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    train_dataset, val_dataset = build_datasets(config)
    configure_normalization(config, train_dataset, val_dataset, run_dir)
    sample = train_dataset[0]
    if not isinstance(sample["input"], torch.Tensor):
        raise TypeError("Dataset input is not a Tensor.")
    if not 0 <= int(sample["label"]) < 40:
        raise ValueError("Dataset label is outside 0-39.")

    model = build_model(config, sample).to(device)
    train_loader = loader_for(train_dataset, config, training=True)
    val_loader = loader_for(val_dataset, config, training=False)
    first_batch = next(iter(train_loader))
    first_inputs = first_batch["input"].to(device)  # type: ignore[union-attr]
    first_mask = first_batch["temporal_mask"].to(device)  # type: ignore[union-attr]
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        first_output = model(first_inputs, temporal_mask=first_mask)
    expected_batch = int(first_inputs.shape[0])
    if tuple(first_output["logits"].shape) != (expected_batch, 40):
        raise ValueError(f"Unexpected logits shape: {tuple(first_output['logits'].shape)}")
    if tuple(first_output["embedding"].shape) != (expected_batch, int(config["embedding_dim"])):
        raise ValueError(f"Unexpected embedding shape: {tuple(first_output['embedding'].shape)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_accuracy = -1.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    patience = int(config.get("early_stopping_patience", epochs))
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=config.get("max_train_batches"),
            gradient_clip=float(config.get("gradient_clip", 1.0)),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled,
            max_batches=config.get("max_val_batches"),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": time.perf_counter() - epoch_start,
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False))
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(run_dir / "best_model.pt", model, epoch, best_accuracy)
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= patience:
            break

    save_checkpoint(run_dir / "last_model.pt", model, int(history[-1]["epoch"]), float(history[-1]["val_accuracy"]))
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_size_mb = (run_dir / "best_model.pt").stat().st_size / (1024 * 1024)
    if checkpoint_size_mb >= 95.0:
        raise RuntimeError(f"Checkpoint is {checkpoint_size_mb:.2f} MB, exceeding the 95 MB safety threshold.")

    predictions = collect_predictions(model, val_loader, criterion, device, amp_enabled)
    prediction_path = run_dir / "fold_0_val_predictions.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=predictions["sample_ids"],
        labels=predictions["labels"],
        logits=predictions["logits"],
        embeddings=predictions["embeddings"],
        class_order=np.arange(40, dtype=np.int64),
    )
    with np.load(prediction_path) as reloaded:
        count = len(reloaded["sample_ids"])
        if reloaded["labels"].shape != (count,) or reloaded["logits"].shape != (count, 40):
            raise ValueError("Saved validation prediction shapes are inconsistent.")
        if reloaded["embeddings"].shape != (count, int(config["embedding_dim"])):
            raise ValueError("Saved validation embedding shape is inconsistent.")
        if not np.array_equal(reloaded["class_order"], np.arange(40)):
            raise ValueError("Saved class_order is not 0-39.")

    metrics = predictions["metrics"]
    save_confusion_matrix(metrics["confusion_matrix"], run_dir / "confusion_matrix.png")  # type: ignore[arg-type]
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    result: dict[str, Any] = {
        "status": "passed",
        "smoke_test": bool(config["smoke_test"]),
        "modality": config["modality"],
        "model_name": config["model_name"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_epoch": best_epoch,
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_loss": metrics["loss"],
        "per_class_recall": metrics["per_class_recall"],
        "parameter_count": parameter_count,
        "checkpoint_size_mb": checkpoint_size_mb,
        "inference_ms_per_sample": measure_inference_ms(model, sample, device, amp_enabled),
        "gpu_memory_peak_mb": peak_memory_mb,
        "input_shape": list(first_inputs.shape),
        "logits_shape": list(first_output["logits"].shape),
        "embedding_shape": list(first_output["embedding"].shape),
        "loss": float(history[-1]["train_loss"]),
        "config_path": config["config_path"],
        "output_dir": str(run_dir),
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT_JSON={json.dumps(result, ensure_ascii=False)}")
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)
    run_experiment(config)


if __name__ == "__main__":
    main()
