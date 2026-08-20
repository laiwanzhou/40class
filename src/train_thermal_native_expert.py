from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.thermal_native_dataset import (
    SEALED_USER_IDS,
    ThermalNativeDataset,
    collate_thermal_trials,
)
from src.engine.metrics import classification_metrics
from src.models.expert_contract import ExpertOutput
from src.models.thermal_iformer_tsm import ThermalIFormerExpert


@dataclass(frozen=True)
class T1BRecipe:
    epochs: int = 30
    batch_size: int = 4
    gradient_accumulation: int = 4
    backbone_lr: float = 3e-5
    head_lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    seed: int = 20260715
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def development_membership(
    split: Mapping[str, Any], audit: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_users = set(map(str, split["train_user_ids"]))
    validation_users = set(map(str, split["validation_user_ids"]))
    if train_users & validation_users:
        raise ValueError("Train and validation users overlap")
    if (train_users | validation_users) & SEALED_USER_IDS:
        raise ValueError("Development membership includes sealed users")
    rows = audit["thermal_data_audit"]["canonical_trial_records"]
    train = [dict(row) for row in rows if row["user_id"] in train_users]
    validation = [dict(row) for row in rows if row["user_id"] in validation_users]
    if {row["user_id"] for row in train} != train_users:
        raise ValueError("Missing train12 users from canonical audit")
    if {row["user_id"] for row in validation} != validation_users:
        raise ValueError("Missing user6/user7 from canonical audit")
    if set(row["sample_id"] for row in train) & set(row["sample_id"] for row in validation):
        raise ValueError("Canonical trial appears in both partitions")
    return train, validation


def masked_trial_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    availability: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 40 or labels.shape != (logits.shape[0],):
        raise ValueError("Expected trial logits [B,40] and labels [B]")
    if availability.shape != labels.shape or availability.dtype != torch.bool:
        raise ValueError("availability must be boolean [B]")
    if not bool(availability.any()):
        raise ValueError("Batch contains no available Thermal trial")
    return nn.functional.cross_entropy(
        logits[availability],
        labels[availability],
        label_smoothing=label_smoothing,
    )


def thermal_metrics(
    labels: np.ndarray, predictions: np.ndarray, user_ids: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    user_ids = np.asarray(user_ids, dtype=np.str_)
    if labels.shape != predictions.shape or labels.shape != user_ids.shape or not len(labels):
        raise ValueError("Metrics require aligned, non-empty trial arrays")
    metrics = classification_metrics(labels, predictions, num_classes=40)
    per_user: dict[str, dict[str, Any]] = {}
    for user_id in sorted(set(user_ids.tolist())):
        selected = user_ids == user_id
        row = classification_metrics(labels[selected], predictions[selected], num_classes=40)
        per_user[user_id] = {
            "trial_count": int(selected.sum()),
            "accuracy": row["accuracy"],
            "macro_f1_fixed_0_39": row["macro_f1"],
            "present_class_count": int(len(set(labels[selected].tolist()))),
        }
    metrics["per_user"] = per_user
    metrics["worst_user_accuracy"] = min(row["accuracy"] for row in per_user.values())
    metrics["zero_recall_class_ids"] = [
        index for index, value in enumerate(metrics["per_class_recall"]) if value == 0.0
    ]
    return metrics


def checkpoint_rank(metrics: Mapping[str, Any], epoch: int) -> tuple[float, float, float, int]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        float(metrics["worst_user_accuracy"]),
        -int(epoch),
    )


def build_optimizer(model: ThermalIFormerExpert, recipe: T1BRecipe) -> torch.optim.Optimizer:
    backbone = list(model.backbone_parameters())
    head = list(model.head_parameters())
    if not backbone or not head or {id(item) for item in backbone} & {id(item) for item in head}:
        raise ValueError("Backbone/head parameter groups must be non-empty and disjoint")
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": recipe.backbone_lr, "name": "backbone"},
            {"params": head, "lr": recipe.head_lr, "name": "head"},
        ],
        weight_decay=recipe.weight_decay,
    )


def _schedule_multiplier(epoch: int, recipe: T1BRecipe) -> float:
    if epoch < recipe.warmup_epochs:
        return (epoch + 1) / recipe.warmup_epochs
    progress = (epoch - recipe.warmup_epochs) / max(1, recipe.epochs - recipe.warmup_epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, Any]
    sample_ids: tuple[str, ...]
    user_ids: tuple[str, ...]
    labels: np.ndarray
    logits: np.ndarray
    availability: np.ndarray
    quality: np.ndarray
    quality_mask: np.ndarray
    num_frames: np.ndarray
    routes: tuple[str, ...]
    preprocessing_latency_ms_mean: float
    model_latency_ms_per_trial: float


def run_epoch(
    model: ThermalIFormerExpert,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    recipe: T1BRecipe,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    quality_all: list[np.ndarray] = []
    quality_mask_all: list[np.ndarray] = []
    availability_all: list[np.ndarray] = []
    num_frames_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    user_ids: list[str] = []
    routes: list[str] = []
    losses: list[tuple[float, int]] = []
    preprocess_ms: list[float] = []
    model_seconds = 0.0
    trial_count = 0
    pending_batches = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            clips = batch["clips"].to(device)
            labels = batch["labels"].to(device)
            quality = batch["quality"].to(device)
            quality_mask = batch["quality_mask"].to(device)
            availability = batch["availability"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output: ExpertOutput = model(clips, quality, quality_mask, availability)
                loss = masked_trial_cross_entropy(
                    output.main_logits,
                    labels,
                    availability,
                    label_smoothing=recipe.label_smoothing,
                )
            if training:
                (loss / recipe.gradient_accumulation).backward()
                pending_batches += 1
                if pending_batches == recipe.gradient_accumulation:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending_batches = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_seconds += time.perf_counter() - started
            available_cpu = availability.detach().cpu().numpy().astype(bool)
            selected = np.flatnonzero(available_cpu)
            count = int(len(selected))
            losses.append((float(loss.detach()), count))
            labels_all.append(labels.detach().cpu().numpy()[selected])
            logits_all.append(output.main_logits.detach().float().cpu().numpy()[selected])
            quality_all.append(quality.detach().float().cpu().numpy()[selected])
            quality_mask_all.append(quality_mask.detach().cpu().numpy()[selected])
            availability_all.append(available_cpu[selected])
            num_frames_all.append(batch["num_frames"].numpy()[selected])
            sample_ids.extend(batch["sample_ids"][index] for index in selected)
            user_ids.extend(batch["user_ids"][index] for index in selected)
            routes.extend(batch["routes"][index] for index in selected)
            preprocess_ms.extend(batch["preprocessing_ms"].numpy().tolist())
            trial_count += count
    if training and pending_batches:
        correction = recipe.gradient_accumulation / pending_batches
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if not trial_count:
        raise ValueError("Epoch produced no available Thermal trials")
    labels_np = np.concatenate(labels_all)
    logits_np = np.concatenate(logits_all)
    metrics = thermal_metrics(labels_np, logits_np.argmax(axis=1), np.asarray(user_ids))
    weighted_loss = sum(loss * count for loss, count in losses) / sum(count for _, count in losses)
    metrics["loss"] = weighted_loss
    return EpochResult(
        loss=weighted_loss,
        metrics=metrics,
        sample_ids=tuple(sample_ids),
        user_ids=tuple(user_ids),
        labels=labels_np,
        logits=logits_np,
        availability=np.concatenate(availability_all),
        quality=np.concatenate(quality_all),
        quality_mask=np.concatenate(quality_mask_all),
        num_frames=np.concatenate(num_frames_all),
        routes=tuple(routes),
        preprocessing_latency_ms_mean=float(np.mean(preprocess_ms)),
        model_latency_ms_per_trial=model_seconds * 1000.0 / trial_count,
    )


def save_prediction_archive(path: Path, result: EpochResult, class_map_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.asarray(result.sample_ids),
        user_ids=np.asarray(result.user_ids),
        labels=result.labels,
        logits=result.logits,
        availability=result.availability,
        quality=result.quality,
        quality_mask=result.quality_mask,
        num_frames=result.num_frames,
        routes=np.asarray(result.routes),
        class_map_hash=np.asarray(class_map_hash),
    )


def train_development(
    model: ThermalIFormerExpert,
    train_dataset: ThermalNativeDataset,
    validation_dataset: ThermalNativeDataset,
    *,
    device: torch.device,
    output_dir: Path,
    class_map_hash: str,
    recipe: T1BRecipe = T1BRecipe(),
) -> dict[str, Any]:
    seed_everything(recipe.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(recipe.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=recipe.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=recipe.num_workers,
        collate_fn=collate_thermal_trials,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=recipe.batch_size,
        shuffle=False,
        num_workers=recipe.num_workers,
        collate_fn=collate_thermal_trials,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    optimizer = build_optimizer(model, recipe)
    history: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float, int] | None = None
    best_epoch = 0
    checkpoint_path = output_dir / "best_macro_f1.pt"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, recipe.epochs + 1):
        train_dataset.set_epoch(epoch)
        multiplier = _schedule_multiplier(epoch - 1, recipe)
        optimizer.param_groups[0]["lr"] = recipe.backbone_lr * multiplier
        optimizer.param_groups[1]["lr"] = recipe.head_lr * multiplier
        train_result = run_epoch(model, train_loader, device=device, recipe=recipe, optimizer=optimizer)
        validation_result = run_epoch(
            model, validation_loader, device=device, recipe=recipe, optimizer=None
        )
        row = {
            "epoch": epoch,
            "learning_rate_multiplier": multiplier,
            "train": train_result.metrics,
            "validation": validation_result.metrics,
            "latency": {
                "preprocessing_ms_per_trial": validation_result.preprocessing_latency_ms_mean,
                "model_ms_per_trial": validation_result.model_latency_ms_per_trial,
            },
        }
        history.append(row)
        rank = checkpoint_rank(validation_result.metrics, epoch)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "recipe": asdict(recipe),
                    "class_map_hash": class_map_hash,
                    "validation_metrics": validation_result.metrics,
                },
                checkpoint_path,
            )
            save_prediction_archive(
                output_dir / "best_validation_predictions.npz",
                validation_result,
                class_map_hash,
            )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch={epoch:02d} train_loss={train_result.loss:.5f} "
            f"val_macro_f1={validation_result.metrics['macro_f1']:.5f} "
            f"val_accuracy={validation_result.metrics['accuracy']:.5f}",
            flush=True,
        )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "recipe": asdict(recipe),
        "epochs_completed": recipe.epochs,
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "best_latency": history[best_epoch - 1]["latency"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "training_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
