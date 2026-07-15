from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics


def _move_batch(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = batch["input"].to(device, non_blocking=True)  # type: ignore[union-attr]
    labels = batch["label"].to(device, non_blocking=True)  # type: ignore[union-attr]
    temporal_mask = batch["temporal_mask"].to(device, non_blocking=True)  # type: ignore[union-attr]
    return inputs, labels, temporal_mask


def run_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    batches = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, labels, temporal_mask = _move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(inputs, temporal_mask=temporal_mask)
                loss = criterion(output["logits"], labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {batch_index}")
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach().cpu())
            predictions = output["logits"].argmax(dim=1)
            labels_all.append(labels.detach().cpu().numpy())
            predictions_all.append(predictions.detach().cpu().numpy())
            batches += 1
    if batches == 0:
        raise ValueError("DataLoader produced no batches.")
    labels_np = np.concatenate(labels_all)
    predictions_np = np.concatenate(predictions_all)
    metrics = classification_metrics(labels_np, predictions_np)
    return {
        "loss": total_loss / batches,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def collect_predictions(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    model.eval()
    sample_ids: list[str] = []
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    embeddings_all: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, temporal_mask = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(inputs, temporal_mask=temporal_mask)
                loss = criterion(output["logits"], labels)
            if not torch.isfinite(output["logits"]).all() or not torch.isfinite(output["embedding"]).all():
                raise FloatingPointError("Non-finite validation logits or embeddings.")
            sample_ids.extend(str(value) for value in batch["sample_id"])
            labels_all.append(labels.cpu().numpy())
            logits_all.append(output["logits"].float().cpu().numpy())
            embeddings_all.append(output["embedding"].float().cpu().numpy())
            losses.append(float(loss.cpu()))
    labels_np = np.concatenate(labels_all)
    logits_np = np.concatenate(logits_all)
    embeddings_np = np.concatenate(embeddings_all)
    predictions = logits_np.argmax(axis=1)
    metrics = classification_metrics(labels_np, predictions)
    metrics["loss"] = float(np.mean(losses))
    return {
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "labels": labels_np,
        "logits": logits_np,
        "embeddings": embeddings_np,
        "metrics": metrics,
    }
