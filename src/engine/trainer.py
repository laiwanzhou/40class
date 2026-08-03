from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics


def _move_input(value: object, device: torch.device) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict) and all(isinstance(item, torch.Tensor) for item in value.values()):
        return {str(key): item.to(device, non_blocking=True) for key, item in value.items()}
    raise TypeError(f"Unsupported model input type: {type(value)}")


def _move_batch(
    batch: dict[str, object], device: torch.device
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    inputs = _move_input(batch["input"], device)
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
    device_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    device_correct = torch.zeros((), device=device, dtype=torch.int64)
    device_finite = torch.ones((), device=device, dtype=torch.bool)
    labels_all: list[torch.Tensor] = []
    predictions_all: list[torch.Tensor] = []
    sample_count = 0
    batches = 0
    num_classes: int | None = None
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
            current_classes = int(output["logits"].shape[1])
            if num_classes is not None and current_classes != num_classes:
                raise ValueError("Model output class count changed within an epoch.")
            num_classes = current_classes
            device_finite.logical_and_(torch.isfinite(loss.detach()))
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            predictions = output["logits"].argmax(dim=1)
            batch_size = labels.shape[0]
            device_loss_sum += loss.detach().to(torch.float64) * batch_size
            device_correct += (predictions == labels).sum()
            sample_count += batch_size
            labels_all.append(labels.detach())
            predictions_all.append(predictions.detach())
            batches += 1
    if batches == 0:
        raise ValueError("DataLoader produced no batches.")
    if not bool(device_finite.item()):
        raise FloatingPointError("Non-finite loss encountered during epoch.")
    labels_np = torch.cat(labels_all).cpu().numpy()
    predictions_np = torch.cat(predictions_all).cpu().numpy()
    assert num_classes is not None
    metrics = classification_metrics(labels_np, predictions_np, num_classes)
    return {
        "loss": (device_loss_sum / sample_count).item(),
        "accuracy": (device_correct.to(torch.float64) / sample_count).item(),
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
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    embeddings_all: list[torch.Tensor] = []
    masks_all: list[torch.Tensor] = []
    roi_attention_all: list[torch.Tensor] = []
    modality_gate_all: list[torch.Tensor] = []
    device_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    device_finite = torch.ones((), device=device, dtype=torch.bool)
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels, temporal_mask = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(inputs, temporal_mask=temporal_mask)
                loss = criterion(output["logits"], labels)
            device_finite.logical_and_(torch.isfinite(output["logits"]).all())
            device_finite.logical_and_(torch.isfinite(output["embedding"]).all())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            batch_size = labels.shape[0]
            device_loss_sum += loss.detach().to(torch.float64) * batch_size
            sample_count += batch_size
            labels_all.append(labels.detach())
            logits_all.append(output["logits"].detach())
            embeddings_all.append(output["embedding"].detach())
            masks_all.append(temporal_mask.detach())
            if "roi_attention" in output:
                roi_attention_all.append(output["roi_attention"].detach())
            if "modality_gate" in output:
                modality_gate_all.append(output["modality_gate"].detach())
    if sample_count == 0:
        raise ValueError("DataLoader produced no prediction batches.")
    if not bool(device_finite.item()):
        raise FloatingPointError("Non-finite validation logits or embeddings.")
    labels_np = torch.cat(labels_all).cpu().numpy()
    logits_np = torch.cat(logits_all).float().cpu().numpy()
    embeddings_np = torch.cat(embeddings_all).float().cpu().numpy()
    predictions = logits_np.argmax(axis=1)
    metrics = classification_metrics(labels_np, predictions, logits_np.shape[1])
    metrics["loss"] = (device_loss_sum / sample_count).item()
    result: dict[str, object] = {
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "labels": labels_np,
        "logits": logits_np,
        "embeddings": embeddings_np,
        "temporal_mask": torch.cat(masks_all).cpu().numpy(),
        "metrics": metrics,
    }
    if roi_attention_all:
        result["roi_attention"] = torch.cat(roi_attention_all).float().cpu().numpy()
    if modality_gate_all:
        result["modality_gate"] = torch.cat(modality_gate_all).float().cpu().numpy()
    return result
