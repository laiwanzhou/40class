from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

from src.data.x3d_clip_dataset import X3DClipDataset, collate_x3d_clips
from src.engine.metrics import classification_metrics
from src.models.expert_contract import ExpertBatchResult, ExpertOutput
from src.models.mobilenet_tcn_visual_expert import build_mobilenet_tcn_visual_expert
from src.models.x3d_s_visual_expert import X3DSVisualExpert, build_x3d_s_feature_backbone


@dataclass(frozen=True)
class UserFold:
    fold: int
    train_user_ids: tuple[str, ...]
    validation_user_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrialPredictionResult:
    sample_ids: tuple[str, ...]
    user_ids: tuple[str, ...]
    labels: torch.Tensor
    class_map_hash: str
    output: ExpertOutput
    num_frames: torch.Tensor
    num_clips: torch.Tensor

    def validate(self) -> None:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Prediction result contains duplicate sample IDs")
        rows = len(self.sample_ids)
        if len(self.user_ids) != rows:
            raise ValueError("Prediction user rows do not match sample IDs")
        if self.labels.shape != (rows,):
            raise ValueError("Prediction labels must have shape [N]")
        if self.num_frames.shape != (rows,) or self.num_clips.shape != (rows,):
            raise ValueError("Prediction diagnostics must have shape [N]")
        ExpertBatchResult(
            sample_ids=self.sample_ids,
            class_map_hash=self.class_map_hash,
            output=self.output,
        ).validate()
        if self.output.main_logits.ndim != 2 or self.output.main_logits.shape[1] != 40:
            raise ValueError("Prediction logits must have shape [N,40]")
        if self.output.embedding.ndim != 2:
            raise ValueError("Prediction embeddings must have shape [N,D]")
        for name, value in (
            ("logits", self.output.main_logits),
            ("embeddings", self.output.embedding),
            ("quality", self.output.quality),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"Prediction {name} contains non-finite values")


@dataclass(frozen=True)
class EpochOutcome:
    metrics: dict[str, Any]
    predictions: TrialPredictionResult


class ClipBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        num_clips: Sequence[int],
        *,
        max_trials_per_batch: int,
        max_valid_clips_per_batch: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if not num_clips or any(count <= 0 for count in num_clips):
            raise ValueError("num_clips must contain positive counts")
        if max_trials_per_batch <= 0 or max_valid_clips_per_batch <= 0:
            raise ValueError("batch limits must be positive")
        if any(count > max_valid_clips_per_batch for count in num_clips):
            raise ValueError("A trial exceeds max_valid_clips_per_batch")
        self.num_clips = tuple(int(count) for count in num_clips)
        self.max_trials_per_batch = int(max_trials_per_batch)
        self.max_valid_clips_per_batch = int(max_valid_clips_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())

    def _batches(self) -> list[list[int]]:
        if self.shuffle:
            order = torch.randperm(
                len(self.num_clips), generator=torch.Generator().manual_seed(self.seed + self.epoch)
            ).tolist()
        else:
            order = list(range(len(self.num_clips)))
        batches: list[list[int]] = []
        current: list[int] = []
        current_clips = 0
        for index in order:
            count = self.num_clips[index]
            exceeds_trials = len(current) >= self.max_trials_per_batch
            exceeds_clips = current_clips + count > self.max_valid_clips_per_batch
            if current and (exceeds_trials or exceeds_clips):
                batches.append(current)
                current = []
                current_clips = 0
            current.append(index)
            current_clips += count
        if current:
            batches.append(current)
        return batches


def aggregate_clip_predictions(
    clip_view_logits: torch.Tensor,
    *,
    clip_mask: torch.Tensor,
) -> torch.Tensor:
    if clip_view_logits.ndim != 4:
        raise ValueError("clip_view_logits must have shape [B,K,V,C]")
    if clip_mask.shape != clip_view_logits.shape[:2] or clip_mask.dtype != torch.bool:
        raise ValueError("clip_mask must be boolean with shape [B,K]")
    if clip_view_logits.shape[2] != 1:
        raise ValueError("The first experiment requires exactly one validation view")
    counts = clip_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Each trial must have at least one valid clip")
    probabilities = torch.softmax(clip_view_logits.squeeze(2), dim=-1)
    masked = probabilities * clip_mask.unsqueeze(-1)
    return masked.sum(dim=1) / counts.to(probabilities.dtype).unsqueeze(1)


def run_model_epoch(
    model: torch.nn.Module,
    loader: Sequence[Mapping[str, object]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_accumulation: int,
    gradient_clip: float,
    amp_enabled: bool,
    label_smoothing: float = 0.0,
    max_batches: int | None = None,
) -> EpochOutcome:
    if gradient_accumulation <= 0 or gradient_clip <= 0:
        raise ValueError("gradient accumulation and clipping must be positive")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must lie in [0, 1)")
    available_batches = len(loader)
    total_batches = min(available_batches, max_batches or available_batches)
    if total_batches <= 0:
        raise ValueError("Epoch loader must contain at least one batch")

    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    sample_ids: list[str] = []
    user_ids: list[str] = []
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    embeddings_all: list[torch.Tensor] = []
    quality_all: list[torch.Tensor] = []
    quality_mask_all: list[torch.Tensor] = []
    availability_all: list[torch.Tensor] = []
    num_frames_all: list[torch.Tensor] = []
    num_clips_all: list[torch.Tensor] = []
    hashes: set[str] = set()
    loss_total = 0.0
    sample_count = 0
    processed_clips = 0
    model_seconds = 0.0
    latency_buckets: dict[str, list[float]] = {}
    backbone_received_finite_gradient = False
    head_received_finite_gradient = False
    gradient_scopes_with_finite_nonzero: dict[str, bool] = {}
    accumulated_trial_count = 0

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, batch in enumerate(loader):
            if batch_index >= total_batches:
                break
            clips_with_views = _tensor(batch, "clips").to(device)
            if clips_with_views.ndim != 7 or clips_with_views.shape[2] != 1:
                raise ValueError("Trainer requires clips with shape [B,K,V=1,C,T,H,W]")
            labels = _tensor(batch, "labels").to(device)
            clip_mask = _tensor(batch, "clip_mask").to(device)
            quality = _tensor(batch, "quality").to(device)
            quality_mask = _tensor(batch, "quality_mask").to(device)
            availability = _tensor(batch, "availability").to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            rows = int(labels.shape[0])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = model(
                    clips_with_views.squeeze(2),
                    clip_mask=clip_mask,
                    quality=quality,
                    quality_mask=quality_mask,
                    availability=availability,
                )
                loss_sum = _trial_nll_loss(
                    output.main_logits,
                    labels,
                    label_smoothing=label_smoothing,
                    reduction="sum",
                )
                loss = loss_sum / rows
            if training:
                loss_sum.backward()
                accumulated_trial_count += rows
                end_of_window = (batch_index + 1) % gradient_accumulation == 0
                end_of_epoch = batch_index + 1 == total_batches
                if end_of_window or end_of_epoch:
                    named_parameters = tuple(model.named_parameters())
                    for _, parameter in named_parameters:
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_trial_count)
                    backbone_received_finite_gradient |= _received_finite_gradient(
                        parameter
                        for name, parameter in named_parameters
                        if name.startswith("backbone.")
                    )
                    head_received_finite_gradient |= _received_finite_gradient(
                        parameter
                        for name, parameter in named_parameters
                        if not name.startswith("backbone.")
                    )
                    scoped_parameters: dict[str, list[torch.nn.Parameter]] = {}
                    for name, parameter in named_parameters:
                        scope = _gradient_scope(name)
                        if scope is not None:
                            scoped_parameters.setdefault(scope, []).append(parameter)
                    for scope, parameters in scoped_parameters.items():
                        gradient_scopes_with_finite_nonzero[scope] = (
                            gradient_scopes_with_finite_nonzero.get(scope, False)
                            or _received_finite_gradient(iter(parameters))
                        )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_trial_count = 0

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_seconds = time.perf_counter() - batch_started
            valid_clips = int(clip_mask.sum().item())
            processed_clips += valid_clips
            model_seconds += batch_seconds

            loss_total += float(loss_sum.detach())
            sample_count += rows
            sample_ids.extend(str(value) for value in batch["sample_ids"])
            user_ids.extend(str(value) for value in batch["user_ids"])
            hashes.add(str(batch["class_map_hash"]))
            labels_all.append(labels.detach().cpu())
            logits_all.append(output.main_logits.detach().float().cpu())
            embeddings_all.append(output.embedding.detach().float().cpu())
            quality_all.append(output.quality.detach().float().cpu())
            quality_mask_all.append(output.quality_mask.detach().cpu())
            availability_all.append(output.availability.detach().cpu())
            num_frames_all.append(_tensor(batch, "num_frames").detach().cpu())
            num_clips_all.append(_tensor(batch, "num_clips").detach().cpu())
            per_trial_ms = batch_seconds * 1000.0 / rows
            for num_frames in _tensor(batch, "num_frames").tolist():
                latency_buckets.setdefault(_length_bucket(int(num_frames)), []).append(per_trial_ms)

    if len(hashes) != 1:
        raise ValueError("Epoch batches contain different class-map hashes")
    labels_tensor = torch.cat(labels_all)
    logits_tensor = torch.cat(logits_all)
    prediction_result = TrialPredictionResult(
        sample_ids=tuple(sample_ids),
        user_ids=tuple(user_ids),
        labels=labels_tensor,
        class_map_hash=next(iter(hashes)),
        output=ExpertOutput(
            main_logits=logits_tensor,
            embedding=torch.cat(embeddings_all),
            quality=torch.cat(quality_all),
            quality_mask=torch.cat(quality_mask_all),
            availability=torch.cat(availability_all),
        ),
        num_frames=torch.cat(num_frames_all),
        num_clips=torch.cat(num_clips_all),
    )
    prediction_result.validate()
    metrics = _epoch_metrics(prediction_result, loss_total / sample_count)
    metrics.update(
        {
            "processed_clips": processed_clips,
            "processed_clips_per_second": processed_clips / max(model_seconds, 1e-12),
            "trial_latency_ms_by_length_bucket": {
                bucket: float(np.mean(values)) for bucket, values in sorted(latency_buckets.items())
            },
            "backbone_received_finite_gradient": backbone_received_finite_gradient,
            "head_received_finite_gradient": head_received_finite_gradient,
            "gradient_scopes_with_finite_nonzero": dict(
                sorted(gradient_scopes_with_finite_nonzero.items())
            ),
        }
    )
    return EpochOutcome(metrics=metrics, predictions=prediction_result)


def _trial_nll_loss(
    log_probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    reduction: str,
) -> torch.Tensor:
    if log_probabilities.ndim != 2 or labels.shape != (log_probabilities.shape[0],):
        raise ValueError("Trial logits and labels have incompatible shapes")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must lie in [0, 1)")
    if reduction not in {"sum", "mean", "none"}:
        raise ValueError("reduction must be sum, mean, or none")
    rows = torch.arange(labels.shape[0], device=labels.device)
    losses = -(
        (1.0 - label_smoothing) * log_probabilities[rows, labels]
        + label_smoothing * log_probabilities.mean(dim=1)
    )
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    return losses


def save_prediction_archive(
    path: Path,
    result: TrialPredictionResult,
    *,
    head_type: str = "projected",
) -> None:
    result.validate()
    if head_type not in {"projected", "direct"}:
        raise ValueError("head_type must be projected or direct")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.asarray(result.sample_ids, dtype=np.str_),
        user_ids=np.asarray(result.user_ids, dtype=np.str_),
        labels=result.labels.detach().cpu().numpy().astype(np.int64),
        logits=result.output.main_logits.detach().float().cpu().numpy(),
        embeddings=result.output.embedding.detach().float().cpu().numpy(),
        quality=result.output.quality.detach().float().cpu().numpy(),
        quality_mask=result.output.quality_mask.detach().cpu().numpy().astype(bool),
        availability=result.output.availability.detach().cpu().numpy().astype(bool),
        class_map_hash=np.asarray(result.class_map_hash, dtype=np.str_),
        num_frames=result.num_frames.detach().cpu().numpy().astype(np.int64),
        num_clips=result.num_clips.detach().cpu().numpy().astype(np.int64),
        head_type=np.asarray(head_type, dtype=np.str_),
        embedding_dim=np.asarray(result.output.embedding.shape[1], dtype=np.int64),
    )


def train_partition(
    *,
    model: X3DSVisualExpert,
    train_dataset: Dataset[Mapping[str, object]],
    validation_dataset: Dataset[Mapping[str, object]],
    config: Mapping[str, Any],
    run_directory: Path,
    device: torch.device,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict[str, Any]:
    validate_config(config)
    run_directory.mkdir(parents=True, exist_ok=True)
    loader_config = _mapping(config, "loader")
    optimizer_config = _mapping(config, "optimizer")
    training_config = _mapping(config, "training")
    amp_config = _mapping(config, "amp")
    seed = int(config["seed"])
    train_sampler = ClipBudgetBatchSampler(
        getattr(train_dataset, "num_clips"),
        max_trials_per_batch=int(loader_config["max_trials_per_batch"]),
        max_valid_clips_per_batch=int(loader_config["max_valid_clips_per_batch"]),
        shuffle=True,
        seed=seed,
    )
    validation_sampler = ClipBudgetBatchSampler(
        getattr(validation_dataset, "num_clips"),
        max_trials_per_batch=int(loader_config["max_trials_per_batch"]),
        max_valid_clips_per_batch=int(loader_config["max_valid_clips_per_batch"]),
        shuffle=False,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_x3d_clips,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collate_x3d_clips,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            float(optimizer_config["backbone_lr"]),
            float(optimizer_config["head_lr"]),
            float(optimizer_config["weight_decay"]),
            backbone_block_lrs=_backbone_block_lrs(optimizer_config),
        )
    )
    epochs = int(training_config["epochs"])
    scheduler_horizon_epochs = int(training_config.get("scheduler_horizon_epochs", epochs))
    warmup_epochs = int(training_config["warmup_epochs"])
    unfrozen_backbone_blocks = training_config.get("unfrozen_backbone_blocks")
    if unfrozen_backbone_blocks is not None:
        unfrozen_backbone_blocks = int(unfrozen_backbone_blocks)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda index: _warmup_cosine_multiplier(
            index, scheduler_horizon_epochs, warmup_epochs
        ),
    )
    amp_enabled = bool(amp_config["enabled"]) and device.type == "cuda"
    gradient_accumulation = int(optimizer_config["gradient_accumulation"])
    gradient_clip = float(optimizer_config["gradient_clip"])
    patience = int(training_config["patience"])
    early_stopping_enabled = bool(training_config.get("early_stopping_enabled", True))
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    class_names = list(getattr(validation_dataset, "class_names", [str(i) for i in range(40)]))

    history: list[dict[str, Any]] = []
    best: dict[str, tuple[float, float, int] | None] = {"accuracy": None, "macro_f1": None}
    best_records: dict[str, dict[str, Any]] = {}
    epochs_without_macro_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        getattr(train_dataset, "set_epoch")(epoch)
        train_sampler.set_epoch(epoch)
        backbone_enabled = epoch > warmup_epochs
        model.set_backbone_trainable(
            backbone_enabled,
            **(
                {"last_blocks": unfrozen_backbone_blocks}
                if backbone_enabled and unfrozen_backbone_blocks is not None
                else {}
            ),
        )
        train_outcome = run_model_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            gradient_accumulation=gradient_accumulation,
            gradient_clip=gradient_clip,
            amp_enabled=amp_enabled,
            label_smoothing=label_smoothing,
            max_batches=max_train_batches,
        )
        validation_outcome = run_model_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            gradient_accumulation=1,
            gradient_clip=gradient_clip,
            amp_enabled=amp_enabled,
            label_smoothing=0.0,
            max_batches=max_val_batches,
        )
        candidate = (
            float(validation_outcome.metrics["accuracy"]),
            float(validation_outcome.metrics["macro_f1"]),
            epoch,
        )
        row = _history_row(
            epoch,
            train_outcome.metrics,
            validation_outcome.metrics,
            optimizer,
            time.perf_counter() - epoch_started,
        )
        row["unfrozen_backbone_blocks"] = (
            unfrozen_backbone_blocks if backbone_enabled else 0
        )
        row["trainable_backbone_parameters"] = sum(
            parameter.numel()
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        )
        history.append(row)
        pd.DataFrame(history).to_csv(run_directory / "history.csv", index=False, encoding="utf-8-sig")

        macro_improved = False
        for objective in ("accuracy", "macro_f1"):
            if is_better_checkpoint(candidate, best[objective], objective=objective):
                best[objective] = candidate
                macro_improved = macro_improved or objective == "macro_f1"
                checkpoint_name = "best_accuracy" if objective == "accuracy" else "best_macro_f1"
                checkpoint_path = run_directory / f"{checkpoint_name}.pt"
                _save_checkpoint(checkpoint_path, model, epoch, validation_outcome.metrics, config)
                save_prediction_archive(
                    run_directory / f"val_predictions_{checkpoint_name}.npz",
                    validation_outcome.predictions,
                    head_type=_resolved_head_type(config),
                )
                _save_per_class_metrics(
                    run_directory / f"per_class_{checkpoint_name}.csv",
                    validation_outcome.metrics,
                    class_names,
                )
                best_records[checkpoint_name] = {
                    "epoch": epoch,
                    "accuracy": candidate[0],
                    "macro_f1": candidate[1],
                }
        epochs_without_macro_improvement = 0 if macro_improved else epochs_without_macro_improvement + 1
        scheduler.step()
        if early_stopping_enabled and epochs_without_macro_improvement >= patience:
            break

    checkpoint_bytes = {
        name: (run_directory / f"{name}.pt").stat().st_size
        for name in ("best_accuracy", "best_macro_f1")
    }
    prediction_archive_bytes = {
        name: (run_directory / f"val_predictions_{name}.npz").stat().st_size
        for name in ("best_accuracy", "best_macro_f1")
    }
    yolo_path_value = _mapping(config, "deployment_artifacts").get("yolo_checkpoint")
    yolo_bytes = Path(str(yolo_path_value)).stat().st_size if yolo_path_value else 0
    route_bytes = max(checkpoint_bytes.values()) + yolo_bytes
    size_limit = int(_mapping(config, "size_gate")["internal_limit_bytes"])
    resource_manifest = _resource_manifest(model, config, device)
    summary: dict[str, Any] = {
        "status": "passed",
        "seed": int(config["seed"]),
        "resolved_config_sha256": resolved_config_sha256(config),
        "epochs_completed": history[-1]["epoch"],
        "scheduler_horizon_epochs": scheduler_horizon_epochs,
        "early_stopping_enabled": early_stopping_enabled,
        "label_smoothing": label_smoothing,
        "unfrozen_backbone_blocks": unfrozen_backbone_blocks,
        "backbone_block_lrs": _serializable_backbone_block_lrs(optimizer_config),
        "trainable_backbone_parameters_last_epoch": sum(
            parameter.numel()
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ),
        "runtime_seconds": time.perf_counter() - started,
        "train_samples_evaluated_last_epoch": history[-1]["train_sample_count"],
        "val_samples_evaluated_last_epoch": history[-1]["val_sample_count"],
        "best_accuracy": best_records["best_accuracy"],
        "best_macro_f1": best_records["best_macro_f1"],
        "checkpoint_bytes": checkpoint_bytes,
        "prediction_archive_bytes": prediction_archive_bytes,
        "yolo_checkpoint_bytes": yolo_bytes,
        "ir_route_serialized_weight_subtotal": route_bytes,
        "internal_size_limit_bytes": size_limit,
        "ir_route_provisional_size_gate_passed": route_bytes < size_limit,
        "amp_enabled": amp_enabled,
        "amp_dtype": "bfloat16",
        "max_trials_per_batch": int(loader_config["max_trials_per_batch"]),
        "max_valid_clips_per_batch": int(loader_config["max_valid_clips_per_batch"]),
        "mean_train_clips_per_trial": float(np.mean(getattr(train_dataset, "num_clips"))),
        "max_train_clips_per_trial": int(max(getattr(train_dataset, "num_clips"))),
        "head_gradient_verified": any(
            bool(row["train_head_received_finite_gradient"]) for row in history
        ),
        "backbone_gradient_verified_after_unfreeze": any(
            int(row["epoch"]) > warmup_epochs
            and bool(row["train_backbone_received_finite_gradient"])
            for row in history
        ),
        "gradient_scopes_with_finite_nonzero": _merge_history_gradient_scopes(
            history
        ),
        "processed_clips_per_second": history[-1]["train_processed_clips_per_second"],
        "trial_latency_ms_by_length_bucket": validation_outcome.metrics[
            "trial_latency_ms_by_length_bucket"
        ],
        **resource_manifest,
    }
    (run_directory / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def finalize_train14(
    *,
    model: X3DSVisualExpert,
    train_dataset: Dataset[Mapping[str, object]],
    config: Mapping[str, Any],
    run_directory: Path,
    device: torch.device,
    max_train_batches: int | None,
) -> dict[str, Any]:
    validate_config(config)
    loader_config = _mapping(config, "loader")
    optimizer_config = _mapping(config, "optimizer")
    training_config = _mapping(config, "training")
    amp_config = _mapping(config, "amp")
    seed = int(config["seed"])
    sampler = ClipBudgetBatchSampler(
        getattr(train_dataset, "num_clips"),
        max_trials_per_batch=int(loader_config["max_trials_per_batch"]),
        max_valid_clips_per_batch=int(loader_config["max_valid_clips_per_batch"]),
        shuffle=True,
        seed=seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_x3d_clips,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            float(optimizer_config["backbone_lr"]),
            float(optimizer_config["head_lr"]),
            float(optimizer_config["weight_decay"]),
            backbone_block_lrs=_backbone_block_lrs(optimizer_config),
        )
    )
    epochs = int(training_config["epochs"])
    scheduler_horizon_epochs = int(training_config.get("scheduler_horizon_epochs", epochs))
    warmup_epochs = int(training_config["warmup_epochs"])
    unfrozen_backbone_blocks = training_config.get("unfrozen_backbone_blocks")
    if unfrozen_backbone_blocks is not None:
        unfrozen_backbone_blocks = int(unfrozen_backbone_blocks)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda index: _warmup_cosine_multiplier(
            index, scheduler_horizon_epochs, warmup_epochs
        ),
    )
    amp_enabled = bool(amp_config["enabled"]) and device.type == "cuda"
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_outcome: EpochOutcome | None = None
    for epoch in range(1, epochs + 1):
        getattr(train_dataset, "set_epoch")(epoch)
        sampler.set_epoch(epoch)
        backbone_enabled = epoch > warmup_epochs
        model.set_backbone_trainable(
            backbone_enabled,
            **(
                {"last_blocks": unfrozen_backbone_blocks}
                if backbone_enabled and unfrozen_backbone_blocks is not None
                else {}
            ),
        )
        last_outcome = run_model_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            gradient_accumulation=int(optimizer_config["gradient_accumulation"]),
            gradient_clip=float(optimizer_config["gradient_clip"]),
            amp_enabled=amp_enabled,
            label_smoothing=label_smoothing,
            max_batches=max_train_batches,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": last_outcome.metrics["loss"],
                "train_accuracy": last_outcome.metrics["accuracy"],
                "train_macro_f1": last_outcome.metrics["macro_f1"],
                "sample_count": last_outcome.metrics["sample_count"],
                "backbone_lr": _maximum_learning_rate(
                    optimizer, scope_prefix="backbone_"
                ),
                "head_lr": _maximum_learning_rate(
                    optimizer, scope_prefix="custom_head"
                ),
                "learning_rates_by_scope": json.dumps(
                    _learning_rates_by_scope(optimizer), sort_keys=True
                ),
                "active_learning_rates_by_scope": json.dumps(
                    _learning_rates_by_scope(optimizer, active_only=True), sort_keys=True
                ),
                "processed_clips_per_second": last_outcome.metrics[
                    "processed_clips_per_second"
                ],
                "unfrozen_backbone_blocks": (
                    unfrozen_backbone_blocks if backbone_enabled else 0
                ),
                "trainable_backbone_parameters": sum(
                    parameter.numel()
                    for parameter in model.backbone.parameters()
                    if parameter.requires_grad
                ),
            }
        )
        pd.DataFrame(history).to_csv(
            run_directory / "finalize_history.csv", index=False, encoding="utf-8-sig"
        )
        scheduler.step()
    assert last_outcome is not None
    _save_checkpoint(
        run_directory / "final_train14.pt",
        model,
        epochs,
        last_outcome.metrics,
        config,
    )
    summary = {
        "status": "passed",
        "role": "finalize_train14",
        "epochs_completed": epochs,
        "scheduler_horizon_epochs": scheduler_horizon_epochs,
        "label_smoothing": label_smoothing,
        "unfrozen_backbone_blocks": unfrozen_backbone_blocks,
        "backbone_block_lrs": _serializable_backbone_block_lrs(optimizer_config),
        "trainable_backbone_parameters": sum(
            parameter.numel()
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ),
        "seed": seed,
        "resolved_config_sha256": resolved_config_sha256(config),
        "runtime_seconds": time.perf_counter() - started,
        "train_sample_count": last_outcome.metrics["sample_count"],
        "final_checkpoint_bytes": (run_directory / "final_train14.pt").stat().st_size,
        "validation_used": False,
        **_resource_manifest(model, config, device),
    }
    yolo_bytes = int(summary["yolo_weight_bytes"])
    summary["ir_route_serialized_weight_subtotal"] = (
        int(summary["final_checkpoint_bytes"]) + yolo_bytes
    )
    summary["internal_size_limit_bytes"] = int(
        _mapping(config, "size_gate")["internal_limit_bytes"]
    )
    summary["ir_route_provisional_size_gate_passed"] = (
        summary["ir_route_serialized_weight_subtotal"]
        < summary["internal_size_limit_bytes"]
    )
    (run_directory / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def train_strict_oof_partition(
    *,
    model_factory: Callable[[], X3DSVisualExpert],
    inner_fit_dataset: Dataset[Mapping[str, object]],
    inner_validation_dataset: Dataset[Mapping[str, object]],
    outer_train_dataset: Dataset[Mapping[str, object]],
    outer_validation_dataset: Dataset[Mapping[str, object]],
    config: Mapping[str, Any],
    run_directory: Path,
    device: torch.device,
    fold_provenance: Mapping[str, Any],
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict[str, Any]:
    selection_directory = run_directory / "epoch_selection"
    selection_directory.mkdir(parents=False, exist_ok=False)
    _set_seed(int(config["seed"]))
    selection_summary = train_partition(
        model=model_factory(),
        train_dataset=inner_fit_dataset,
        validation_dataset=inner_validation_dataset,
        config=config,
        run_directory=selection_directory,
        device=device,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    selected_epoch = int(selection_summary["best_accuracy"]["epoch"])

    formal_config = _formal_refit_config(config, selected_epoch=selected_epoch)
    _set_seed(int(formal_config["seed"]))
    formal_model = model_factory()
    refit_summary = finalize_train14(
        model=formal_model,
        train_dataset=outer_train_dataset,
        config=formal_config,
        run_directory=run_directory,
        device=device,
        max_train_batches=max_train_batches,
    )
    temporary_checkpoint = run_directory / "final_train14.pt"
    formal_checkpoint = run_directory / "formal_outer_refit.pt"
    checkpoint = torch.load(temporary_checkpoint, map_location="cpu", weights_only=False)
    checkpoint["strict_oof_provenance"] = {
        **dict(fold_provenance),
        "actual_seed": int(formal_config["seed"]),
        "selected_epoch": selected_epoch,
        "scheduler_horizon_epochs": int(
            _mapping(formal_config, "training")["scheduler_horizon_epochs"]
        ),
        "resolved_config_sha256": resolved_config_sha256(formal_config),
        "outer_validation_labels_used_for_selection": False,
    }
    torch.save(checkpoint, formal_checkpoint)
    temporary_checkpoint.unlink()

    loader_config = _mapping(formal_config, "loader")
    validation_sampler = ClipBudgetBatchSampler(
        getattr(outer_validation_dataset, "num_clips"),
        max_trials_per_batch=int(loader_config["max_trials_per_batch"]),
        max_valid_clips_per_batch=int(loader_config["max_valid_clips_per_batch"]),
        shuffle=False,
        seed=int(formal_config["seed"]),
    )
    validation_loader = DataLoader(
        outer_validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collate_x3d_clips,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    formal_outcome = run_model_epoch(
        formal_model,
        validation_loader,
        device=device,
        optimizer=None,
        gradient_accumulation=1,
        gradient_clip=float(_mapping(formal_config, "optimizer")["gradient_clip"]),
        amp_enabled=bool(_mapping(formal_config, "amp")["enabled"]),
        max_batches=max_val_batches,
    )
    save_prediction_archive(
        run_directory / "formal_outer_predictions.npz",
        formal_outcome.predictions,
        head_type=_resolved_head_type(formal_config),
    )
    _save_per_class_metrics(
        run_directory / "formal_outer_per_class.csv",
        formal_outcome.metrics,
        list(getattr(outer_validation_dataset, "class_names", [str(i) for i in range(40)])),
    )
    summary = {
        "status": "passed",
        "role": "strict_checkpoint_selection_oof",
        "seed": int(formal_config["seed"]),
        "resolved_config_sha256": resolved_config_sha256(formal_config),
        "selection_labels_from_outer_validation": False,
        "selection_directory": "epoch_selection",
        "selected_epoch": selected_epoch,
        "scheduler_horizon_epochs": int(
            _mapping(formal_config, "training")["scheduler_horizon_epochs"]
        ),
        "formal_checkpoint": formal_checkpoint.name,
        "formal_checkpoint_bytes": formal_checkpoint.stat().st_size,
        "formal_checkpoint_sha256": _sha256_file(formal_checkpoint),
        "formal_outer_accuracy": float(formal_outcome.metrics["accuracy"]),
        "formal_outer_macro_f1": float(formal_outcome.metrics["macro_f1"]),
        "formal_outer_worst_user_accuracy": float(
            formal_outcome.metrics["worst_user_accuracy"]
        ),
        "formal_outer_sample_count": int(formal_outcome.metrics["sample_count"]),
        "fold_provenance": dict(fold_provenance),
        "epoch_selection_summary": selection_summary,
        "outer_refit_summary": refit_summary,
    }
    (run_directory / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def refit_strict_oof_partition(
    *,
    model_factory: Callable[[], X3DSVisualExpert],
    outer_train_dataset: Dataset[Mapping[str, object]],
    outer_validation_dataset: Dataset[Mapping[str, object]],
    config: Mapping[str, Any],
    run_directory: Path,
    device: torch.device,
    selected_epoch: int,
    fold_provenance: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict[str, Any]:
    """Run only the leakage-safe formal refit for a preselected epoch."""
    formal_config = _formal_refit_config(config, selected_epoch=selected_epoch)
    _set_seed(int(formal_config["seed"]))
    formal_model = model_factory()
    refit_summary = finalize_train14(
        model=formal_model,
        train_dataset=outer_train_dataset,
        config=formal_config,
        run_directory=run_directory,
        device=device,
        max_train_batches=max_train_batches,
    )
    temporary_checkpoint = run_directory / "final_train14.pt"
    formal_checkpoint = run_directory / "formal_outer_refit.pt"
    checkpoint = torch.load(temporary_checkpoint, map_location="cpu", weights_only=False)
    checkpoint["strict_oof_provenance"] = {
        **dict(fold_provenance),
        "actual_seed": int(formal_config["seed"]),
        "selected_epoch": int(selected_epoch),
        "scheduler_horizon_epochs": int(
            _mapping(formal_config, "training")["scheduler_horizon_epochs"]
        ),
        "resolved_config_sha256": resolved_config_sha256(formal_config),
        "outer_validation_labels_used_for_selection": False,
    }
    torch.save(checkpoint, formal_checkpoint)
    temporary_checkpoint.unlink()

    loader_config = _mapping(formal_config, "loader")
    validation_sampler = ClipBudgetBatchSampler(
        getattr(outer_validation_dataset, "num_clips"),
        max_trials_per_batch=int(loader_config["max_trials_per_batch"]),
        max_valid_clips_per_batch=int(loader_config["max_valid_clips_per_batch"]),
        shuffle=False,
        seed=int(formal_config["seed"]),
    )
    validation_loader = DataLoader(
        outer_validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collate_x3d_clips,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    formal_outcome = run_model_epoch(
        formal_model,
        validation_loader,
        device=device,
        optimizer=None,
        gradient_accumulation=1,
        gradient_clip=float(_mapping(formal_config, "optimizer")["gradient_clip"]),
        amp_enabled=bool(_mapping(formal_config, "amp")["enabled"]),
        max_batches=max_val_batches,
    )
    save_prediction_archive(
        run_directory / "formal_outer_predictions.npz",
        formal_outcome.predictions,
        head_type=_resolved_head_type(formal_config),
    )
    _save_per_class_metrics(
        run_directory / "formal_outer_per_class.csv",
        formal_outcome.metrics,
        list(getattr(outer_validation_dataset, "class_names", [str(i) for i in range(40)])),
    )
    summary = {
        "status": "passed",
        "role": "strict_fixed_epoch_oof_refit",
        "seed": int(formal_config["seed"]),
        "resolved_config_sha256": resolved_config_sha256(formal_config),
        "selection_labels_from_outer_validation": False,
        "selected_epoch": int(selected_epoch),
        "scheduler_horizon_epochs": int(
            _mapping(formal_config, "training")["scheduler_horizon_epochs"]
        ),
        "formal_checkpoint": formal_checkpoint.name,
        "formal_checkpoint_bytes": formal_checkpoint.stat().st_size,
        "formal_checkpoint_sha256": _sha256_file(formal_checkpoint),
        "formal_outer_accuracy": float(formal_outcome.metrics["accuracy"]),
        "formal_outer_macro_f1": float(formal_outcome.metrics["macro_f1"]),
        "formal_outer_worst_user_accuracy": float(
            formal_outcome.metrics["worst_user_accuracy"]
        ),
        "formal_outer_sample_count": int(formal_outcome.metrics["sample_count"]),
        "fold_provenance": dict(fold_provenance),
        "epoch_selection_summary": dict(selection_summary),
        "outer_refit_summary": refit_summary,
    }
    (run_directory / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _resolved_head_type(config: Mapping[str, Any]) -> str:
    head_type = str(config.get("head_type", "projected"))
    if head_type not in {"projected", "direct"}:
        raise ValueError("head_type must be projected or direct")
    return head_type


def _resolved_temporal_sampling_mode(config: Mapping[str, Any]) -> str:
    temporal = _mapping(config, "temporal")
    mode = str(temporal.get("sampling_mode", "adaptive_local_windows"))
    if mode not in {"adaptive_local_windows", "global_single_clip"}:
        raise ValueError(
            "temporal.sampling_mode must be adaptive_local_windows or "
            "global_single_clip"
        )
    return mode


def validate_config(config: Mapping[str, Any]) -> None:
    model_family = str(config.get("model_family", "x3d_s"))
    if model_family not in {"x3d_s", "mobilenet_v3_small_tcn"}:
        raise ValueError("model_family must be x3d_s or mobilenet_v3_small_tcn")
    head_type = _resolved_head_type(config)
    if head_type == "direct":
        if model_family != "x3d_s":
            raise ValueError("direct head is only supported for x3d_s")
        if int(config.get("embedding_dim", 0)) != 2048:
            raise ValueError("direct head embedding_dim must be 2048")
    if config.get("input_view") != "ir_context_path":
        raise ValueError("First-run input_view must be ir_context_path")
    if config.get("num_classes") != 40:
        raise ValueError("num_classes must be 40")
    temporal = _mapping(config, "temporal")
    _resolved_temporal_sampling_mode(config)
    expected_temporal = {
        "local_frames": 13,
        "target_window_frames": 32,
        "max_clips": 8,
        "train_views_per_window": 1,
        "val_views_per_window": 1,
        "aggregation": "mean_probability",
    }
    for key, expected in expected_temporal.items():
        if temporal.get(key) != expected:
            raise ValueError(f"temporal.{key} must be {expected!r}")
    loader = _mapping(config, "loader")
    if loader.get("max_trials_per_batch") != 2 or loader.get("max_valid_clips_per_batch") != 8:
        raise ValueError("loader limits must be two trials and eight valid clips")
    if loader.get("num_workers") != 0:
        raise ValueError("First-run loader.num_workers must be zero")
    batch_norm = _mapping(config, "backbone_bn")
    if batch_norm.get("update_running_stats") is not False:
        raise ValueError("First-run backbone BatchNorm running stats must remain frozen")
    if batch_norm.get("train_affine_after_unfreeze") is not True:
        raise ValueError("Backbone BatchNorm affine parameters must train after unfreeze")
    optimizer = _mapping(config, "optimizer")
    for key in ("backbone_lr", "head_lr", "gradient_clip"):
        if float(optimizer.get(key, 0.0)) <= 0:
            raise ValueError(f"optimizer.{key} must be positive")
    if float(optimizer.get("weight_decay", -1.0)) < 0:
        raise ValueError("optimizer.weight_decay must be non-negative")
    if int(optimizer.get("gradient_accumulation", 0)) != 4:
        raise ValueError("optimizer.gradient_accumulation must be 4")
    block_lrs = _backbone_block_lrs(optimizer)
    training = _mapping(config, "training")
    if int(training.get("epochs", 0)) <= 0 or int(training.get("warmup_epochs", -1)) < 0:
        raise ValueError("training must use positive epochs and non-negative warmup epochs")
    unfrozen_backbone_blocks = training.get("unfrozen_backbone_blocks")
    if unfrozen_backbone_blocks is not None and int(unfrozen_backbone_blocks) <= 0:
        raise ValueError("training.unfrozen_backbone_blocks must be positive")
    if block_lrs:
        if model_family != "x3d_s" or unfrozen_backbone_blocks is None:
            raise ValueError(
                "optimizer.backbone_block_lrs requires bounded X3D partial unfreezing"
            )
        expected_indices = set(
            range(6 - int(unfrozen_backbone_blocks), 6)
        )
        if set(block_lrs) != expected_indices:
            raise ValueError(
                "optimizer.backbone_block_lrs must exactly match the unfrozen X3D blocks"
            )
    label_smoothing = float(training.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("training.label_smoothing must lie in [0, 1)")
    scheduler_horizon_epochs = int(
        training.get("scheduler_horizon_epochs", training.get("epochs", 0))
    )
    if scheduler_horizon_epochs < int(training["epochs"]):
        raise ValueError("training.scheduler_horizon_epochs must cover every training epoch")
    if int(training.get("patience", 0)) <= 0:
        raise ValueError("training.patience must be positive")
    if int(_mapping(config, "size_gate").get("internal_limit_bytes", 0)) != 95_000_000:
        raise ValueError("size gate must use 95,000,000 bytes")


def validate_user_partition(
    train_user_ids: Sequence[str],
    validation_user_ids: Sequence[str],
    official_train_users: set[str],
    heldout_users: set[str],
) -> None:
    train = set(train_user_ids)
    validation = set(validation_user_ids)
    if not train or not validation:
        raise ValueError("Train and validation user partitions must be non-empty")
    if train & validation:
        raise ValueError("Train and validation users overlap")
    if (train | validation) & heldout_users:
        raise ValueError("Official held-out users are forbidden during training")
    unknown = (train | validation) - official_train_users
    if unknown:
        raise ValueError(f"Partition contains users outside official train-14: {sorted(unknown)}")


def validate_oof_assignment(
    assignment: Mapping[str, Any],
    *,
    allowed_users: set[str],
) -> tuple[UserFold, ...]:
    raw_folds = assignment.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != 3:
        raise ValueError("OOF assignment must contain exactly three folds")
    folds: list[UserFold] = []
    validation_occurrences: list[str] = []
    for expected_fold, raw in enumerate(raw_folds):
        if not isinstance(raw, Mapping) or int(raw.get("fold", -1)) != expected_fold:
            raise ValueError("OOF folds must be ordered and numbered 0, 1, 2")
        train = tuple(str(user) for user in raw.get("train_user_ids", ()))
        validation = tuple(str(user) for user in raw.get("validation_user_ids", ()))
        validate_user_partition(train, validation, allowed_users, set())
        if set(train) | set(validation) != allowed_users:
            raise ValueError("Every OOF fold must cover the complete allowed user set")
        validation_occurrences.extend(validation)
        folds.append(UserFold(expected_fold, train, validation))
    if sorted(validation_occurrences) != sorted(allowed_users):
        raise ValueError("Every allowed user must be validation exactly once")
    return tuple(folds)


def is_better_checkpoint(
    candidate: tuple[float, float, int],
    incumbent: tuple[float, float, int] | None,
    *,
    objective: str,
) -> bool:
    if objective not in {"accuracy", "macro_f1"}:
        raise ValueError("objective must be accuracy or macro_f1")
    if incumbent is None:
        return True
    candidate_accuracy, candidate_macro, candidate_epoch = candidate
    incumbent_accuracy, incumbent_macro, incumbent_epoch = incumbent
    if objective == "accuracy":
        candidate_key = (candidate_accuracy, candidate_macro, -candidate_epoch)
        incumbent_key = (incumbent_accuracy, incumbent_macro, -incumbent_epoch)
    else:
        candidate_key = (candidate_macro, candidate_accuracy, -candidate_epoch)
        incumbent_key = (incumbent_macro, incumbent_accuracy, -incumbent_epoch)
    return candidate_key > incumbent_key


def prepare_run_directory(output_root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    run_directory = output_root / run_id
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"Run directory already exists: {run_directory}") from error
    return run_directory


def apply_runtime_overrides(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int | None, int | None]:
    resolved = copy.deepcopy(dict(config))
    if getattr(args, "seed", None) is not None:
        resolved["seed"] = int(args.seed)
    if args.epochs is not None:
        resolved["training"]["epochs"] = int(args.epochs)
    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    if args.smoke_test:
        resolved["training"]["epochs"] = 3
        max_train_batches = max_train_batches or 1
        max_val_batches = max_val_batches or 1
    return resolved, max_train_batches, max_val_batches


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train adaptive multi-clip X3D-S IR expert")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--train-user-ids", nargs="+")
    parser.add_argument("--validation-user-ids", nargs="+")
    parser.add_argument("--oof-fold-assignment", type=Path)
    parser.add_argument("--oof-role", choices=("train14", "finalize_train14"))
    return parser


def resolved_config_sha256(config: Mapping[str, Any]) -> str:
    serialized = yaml.safe_dump(dict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def _tensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Batch field {key} must be a tensor")
    return value


def _received_finite_gradient(parameters: Iterator[torch.nn.Parameter]) -> bool:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return (
        bool(gradients)
        and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        and any(bool(torch.count_nonzero(gradient).item()) for gradient in gradients)
    )


def _gradient_scope(parameter_name: str) -> str | None:
    if parameter_name.startswith("classifier."):
        return "classifier"
    prefix = "backbone.blocks."
    if parameter_name.startswith(prefix):
        remainder = parameter_name[len(prefix):]
        block_index = remainder.split(".", 1)[0]
        if block_index.isdigit():
            return f"backbone_block_{int(block_index)}"
    return None


def _merge_history_gradient_scopes(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    merged: dict[str, bool] = {}
    for row in history:
        raw = row.get("train_gradient_scopes_with_finite_nonzero", "{}")
        scopes = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(scopes, Mapping):
            raise ValueError("Gradient scope history must be a mapping")
        for scope, value in scopes.items():
            merged[str(scope)] = merged.get(str(scope), False) or bool(value)
    return dict(sorted(merged.items()))


def _epoch_metrics(result: TrialPredictionResult, loss: float) -> dict[str, Any]:
    labels = result.labels.numpy()
    predictions = result.output.main_logits.argmax(dim=1).numpy()
    metrics = classification_metrics(labels, predictions, num_classes=40)
    user_accuracies = []
    users = np.asarray(result.user_ids)
    for user in np.unique(users):
        mask = users == user
        user_accuracies.append(float((predictions[mask] == labels[mask]).mean()))
    metrics.update(
        {
            "loss": float(loss),
            "sample_count": len(labels),
            "class_coverage": int(np.unique(labels).size),
            "predicted_class_count": int(np.unique(predictions).size),
            "zero_recall_class_count": int(np.count_nonzero(np.asarray(metrics["per_class_recall"]) == 0)),
            "worst_user_accuracy": min(user_accuracies),
        }
    )
    return metrics


def _warmup_cosine_multiplier(
    epoch_index: int, scheduler_horizon_epochs: int, warmup_epochs: int
) -> float:
    if epoch_index < warmup_epochs:
        return (epoch_index + 1) / warmup_epochs
    decay_intervals = max(1, scheduler_horizon_epochs - warmup_epochs - 1)
    progress = min(1.0, (epoch_index - warmup_epochs) / decay_intervals)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _formal_refit_config(
    config: Mapping[str, Any], *, selected_epoch: int
) -> dict[str, Any]:
    if selected_epoch <= 0:
        raise ValueError("selected_epoch must be positive")
    formal_config = copy.deepcopy(dict(config))
    selection_training = dict(_mapping(config, "training"))
    scheduler_horizon_epochs = int(
        selection_training.get("scheduler_horizon_epochs", selection_training["epochs"])
    )
    formal_config["training"] = {
        **selection_training,
        "epochs": int(selected_epoch),
        "scheduler_horizon_epochs": scheduler_horizon_epochs,
        "early_stopping_enabled": False,
        "patience": max(int(selected_epoch) + 1, 1),
    }
    validate_config(formal_config)
    return formal_config


def _history_row(
    epoch: int,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    epoch_seconds: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "train_weighted_f1": train_metrics["weighted_f1"],
        "train_sample_count": train_metrics["sample_count"],
        "train_class_coverage": train_metrics["class_coverage"],
        "train_zero_recall_class_count": train_metrics["zero_recall_class_count"],
        "train_worst_user_accuracy": train_metrics["worst_user_accuracy"],
        "train_backbone_received_finite_gradient": train_metrics[
            "backbone_received_finite_gradient"
        ],
        "train_head_received_finite_gradient": train_metrics["head_received_finite_gradient"],
        "train_gradient_scopes_with_finite_nonzero": json.dumps(
            train_metrics["gradient_scopes_with_finite_nonzero"], sort_keys=True
        ),
        "val_loss": validation_metrics["loss"],
        "val_accuracy": validation_metrics["accuracy"],
        "val_macro_f1": validation_metrics["macro_f1"],
        "val_weighted_f1": validation_metrics["weighted_f1"],
        "val_worst_user_accuracy": validation_metrics["worst_user_accuracy"],
        "val_class_coverage": validation_metrics["class_coverage"],
        "val_zero_recall_class_count": validation_metrics["zero_recall_class_count"],
        "val_sample_count": validation_metrics["sample_count"],
        "backbone_lr": _maximum_learning_rate(optimizer, scope_prefix="backbone_"),
        "head_lr": _maximum_learning_rate(optimizer, scope_prefix="custom_head"),
        "learning_rates_by_scope": json.dumps(
            _learning_rates_by_scope(optimizer), sort_keys=True
        ),
        "active_learning_rates_by_scope": json.dumps(
            _learning_rates_by_scope(optimizer, active_only=True), sort_keys=True
        ),
        "epoch_seconds": epoch_seconds,
        "train_processed_clips_per_second": train_metrics["processed_clips_per_second"],
        "val_processed_clips_per_second": validation_metrics["processed_clips_per_second"],
        "train_trial_latency_ms_by_length_bucket": json.dumps(
            train_metrics["trial_latency_ms_by_length_bucket"], sort_keys=True
        ),
        "val_trial_latency_ms_by_length_bucket": json.dumps(
            validation_metrics["trial_latency_ms_by_length_bucket"], sort_keys=True
        ),
    }


def _backbone_block_lrs(config: Mapping[str, Any]) -> dict[int, float] | None:
    raw = config.get("backbone_block_lrs")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("optimizer.backbone_block_lrs must be a non-empty mapping")
    resolved: dict[int, float] = {}
    for raw_index, raw_learning_rate in raw.items():
        try:
            index = int(raw_index)
            learning_rate = float(raw_learning_rate)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "optimizer.backbone_block_lrs requires integer indices and numeric rates"
            ) from error
        if index < 0 or learning_rate <= 0:
            raise ValueError(
                "optimizer.backbone_block_lrs indices must be non-negative and rates positive"
            )
        if index in resolved:
            raise ValueError("optimizer.backbone_block_lrs contains duplicate block indices")
        resolved[index] = learning_rate
    return resolved


def _serializable_backbone_block_lrs(config: Mapping[str, Any]) -> dict[str, float] | None:
    resolved = _backbone_block_lrs(config)
    if resolved is None:
        return None
    return {str(index): learning_rate for index, learning_rate in sorted(resolved.items())}


def _learning_rates_by_scope(
    optimizer: torch.optim.Optimizer, *, active_only: bool = False
) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        if active_only and not any(
            bool(parameter.requires_grad) for parameter in group["params"]
        ):
            continue
        scope = str(group.get("group_name", f"group_{index}"))
        learning_rate = float(group["lr"])
        previous = result.setdefault(scope, learning_rate)
        if abs(previous - learning_rate) > 1e-15:
            raise RuntimeError(f"Optimizer scope {scope} has inconsistent learning rates")
    return result


def _maximum_learning_rate(
    optimizer: torch.optim.Optimizer, *, scope_prefix: str
) -> float:
    scoped_learning_rates = _learning_rates_by_scope(optimizer, active_only=True)
    matching = [
        learning_rate
        for scope, learning_rate in scoped_learning_rates.items()
        if scope.startswith(scope_prefix)
    ]
    if not matching:
        matching = [
            learning_rate
            for scope, learning_rate in _learning_rates_by_scope(optimizer).items()
            if scope.startswith(scope_prefix)
        ]
    if not matching:
        raise RuntimeError(f"Optimizer has no learning-rate scope matching {scope_prefix}")
    return max(matching)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "model_state_dict": model.state_dict(),
            "num_classes": int(config["num_classes"]),
            "embedding_dim": int(config["embedding_dim"]),
            "head_type": _resolved_head_type(config),
            "backbone_bn": dict(_mapping(config, "backbone_bn")),
        },
        path,
    )


def _save_per_class_metrics(
    path: Path,
    metrics: Mapping[str, Any],
    class_names: Sequence[str],
) -> None:
    rows = []
    for class_id in range(40):
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                "precision": metrics["per_class_precision"][class_id],
                "recall": metrics["per_class_recall"][class_id],
                "f1": metrics["per_class_f1"][class_id],
                "support": metrics["per_class_support"][class_id],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def prepare_partition_manifest(
    frame: pd.DataFrame,
    *,
    train_user_ids: Sequence[str],
    validation_user_ids: Sequence[str],
) -> pd.DataFrame:
    train = set(train_user_ids)
    validation = set(validation_user_ids)
    selected = frame[frame["user_id"].astype(str).isin(train | validation)].copy()
    if selected.empty:
        raise ValueError("Selected user partition has no manifest rows")
    selected["split"] = np.where(selected["user_id"].astype(str).isin(train), "train", "val")
    observed = set(selected["user_id"].astype(str).unique())
    if observed != train | validation:
        raise ValueError(f"Manifest is missing partition users: {sorted((train | validation) - observed)}")
    return selected


def prepare_finalize_manifest(
    frame: pd.DataFrame,
    *,
    official_train_users: set[str],
) -> pd.DataFrame:
    selected = frame[frame["user_id"].astype(str).isin(official_train_users)].copy()
    observed = set(selected["user_id"].astype(str).unique())
    if observed != official_train_users:
        raise ValueError(
            "Finalization manifest is missing official train-14 users: "
            f"{sorted(official_train_users - observed)}"
        )
    selected["split"] = "train"
    return selected


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")
    config, max_train_batches, max_val_batches = apply_runtime_overrides(config, args)
    validate_config(config)
    _set_seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    split_path = Path(str(config["split_path"]))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    official_train_users = set(str(user) for user in split["train_users"])
    heldout_users = set(str(user) for user in split["val_users"])
    finalize_mode = args.oof_role == "finalize_train14"
    if finalize_mode:
        if args.epochs is None:
            raise ValueError("finalize_train14 requires an explicitly frozen --epochs value")
        if args.oof_fold_assignment is not None or args.validation_user_ids is not None:
            raise ValueError("finalize_train14 accepts no OOF assignment or validation users")
        if args.train_user_ids is not None and set(args.train_user_ids) != official_train_users:
            raise ValueError("finalize_train14 train users must equal the complete official train-14")

        output_root = Path(str(config["output_root"]))
        run_directory = prepare_run_directory(output_root, args.run_id)
        (run_directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
        train_manifest = prepare_finalize_manifest(
            manifest,
            official_train_users=official_train_users,
        )
        train_dataset = X3DClipDataset(
            train_manifest,
            split="train",
            training=True,
            temporal_sampling_mode=_resolved_temporal_sampling_mode(config),
            seed=int(config["seed"]),
        )
        model = _build_model(config)
        summary = finalize_train14(
            model=model,
            train_dataset=train_dataset,
            config=config,
            run_directory=run_directory,
            device=device,
            max_train_batches=max_train_batches,
        )
        summary["train_user_ids"] = sorted(official_train_users)
        (run_directory / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return

    partitions: list[UserFold]
    assignment: Mapping[str, Any] | None = None
    assignment_sha256: str | None = None
    if args.oof_fold_assignment is not None:
        if args.oof_role != "train14":
            raise ValueError("OOF assignment requires --oof-role train14")
        assignment = json.loads(args.oof_fold_assignment.read_text(encoding="utf-8"))
        assignment_sha256 = _sha256_file(args.oof_fold_assignment)
        partitions = list(validate_oof_assignment(assignment, allowed_users=official_train_users))
    elif args.train_user_ids is not None or args.validation_user_ids is not None:
        if args.train_user_ids is None or args.validation_user_ids is None:
            raise ValueError("Both explicit train and validation user lists are required")
        validate_user_partition(
            args.train_user_ids,
            args.validation_user_ids,
            official_train_users,
            heldout_users,
        )
        partitions = [UserFold(0, tuple(args.train_user_ids), tuple(args.validation_user_ids))]
    else:
        development = _mapping(config, "development_partition")
        development_train = tuple(str(user) for user in development.get("train_user_ids", ()))
        development_validation = tuple(
            str(user) for user in development.get("validation_user_ids", ())
        )
        validate_user_partition(
            development_train,
            development_validation,
            official_train_users,
            heldout_users,
        )
        partitions = [UserFold(0, development_train, development_validation)]

    output_root = Path(str(config["output_root"]))
    parent_run_directory = prepare_run_directory(output_root, args.run_id)
    (parent_run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    summaries = []
    multiple_partitions = len(partitions) > 1
    for partition in partitions:
        partition_directory = (
            parent_run_directory / f"fold_{partition.fold}" if multiple_partitions else parent_run_directory
        )
        partition_directory.mkdir(exist_ok=not multiple_partitions)
        partition_manifest = prepare_partition_manifest(
            manifest,
            train_user_ids=partition.train_user_ids,
            validation_user_ids=partition.validation_user_ids,
        )
        train_dataset = X3DClipDataset(
            partition_manifest,
            split="train",
            training=True,
            temporal_sampling_mode=_resolved_temporal_sampling_mode(config),
            seed=int(config["seed"]),
        )
        validation_dataset = X3DClipDataset(
            partition_manifest,
            split="val",
            training=False,
            temporal_sampling_mode=_resolved_temporal_sampling_mode(config),
            seed=int(config["seed"]),
        )
        if assignment is not None:
            raw_fold = assignment["folds"][partition.fold]
            epoch_selection = raw_fold.get("epoch_selection")
            if not isinstance(epoch_selection, Mapping):
                raise ValueError(f"OOF fold {partition.fold} has no frozen epoch_selection")
            inner_fit_users = tuple(str(user) for user in epoch_selection["fit_user_ids"])
            inner_validation_users = tuple(
                str(user) for user in epoch_selection["validation_user_ids"]
            )
            if (
                set(inner_fit_users) & set(inner_validation_users)
                or set(inner_fit_users) | set(inner_validation_users)
                != set(partition.train_user_ids)
            ):
                raise ValueError("Frozen epoch-selection users must partition outer-train")
            inner_manifest = prepare_partition_manifest(
                manifest,
                train_user_ids=inner_fit_users,
                validation_user_ids=inner_validation_users,
            )
            inner_fit_dataset = X3DClipDataset(
                inner_manifest,
                split="train",
                training=True,
                temporal_sampling_mode=_resolved_temporal_sampling_mode(config),
                seed=int(config["seed"]),
            )
            inner_validation_dataset = X3DClipDataset(
                inner_manifest,
                split="val",
                training=False,
                temporal_sampling_mode=_resolved_temporal_sampling_mode(config),
                seed=int(config["seed"]),
            )
            summary = train_strict_oof_partition(
                model_factory=lambda: _build_model(config),
                inner_fit_dataset=inner_fit_dataset,
                inner_validation_dataset=inner_validation_dataset,
                outer_train_dataset=train_dataset,
                outer_validation_dataset=validation_dataset,
                config=config,
                run_directory=partition_directory,
                device=device,
                fold_provenance={
                    "outer_fold": partition.fold,
                    "inner_fit_user_ids": list(inner_fit_users),
                    "inner_validation_user_ids": list(inner_validation_users),
                    "outer_train_user_ids": list(partition.train_user_ids),
                    "outer_validation_user_ids": list(partition.validation_user_ids),
                    "assignment_sha256": assignment_sha256,
                },
                max_train_batches=max_train_batches,
                max_val_batches=max_val_batches,
            )
        else:
            model = _build_model(config)
            summary = train_partition(
                model=model,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                config=config,
                run_directory=partition_directory,
                device=device,
                max_train_batches=max_train_batches,
                max_val_batches=max_val_batches,
            )
        summary["fold"] = partition.fold
        summary["train_user_ids"] = list(partition.train_user_ids)
        summary["validation_user_ids"] = list(partition.validation_user_ids)
        summaries.append(summary)
    (parent_run_directory / "partition_summaries.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


def _build_model(config: Mapping[str, Any]) -> torch.nn.Module:
    if str(config.get("model_family", "x3d_s")) == "mobilenet_v3_small_tcn":
        baseline = config.get("matched_baseline", {})
        if not isinstance(baseline, Mapping):
            raise ValueError("matched_baseline must be a mapping")
        return build_mobilenet_tcn_visual_expert(
            pretrained=bool(config["pretrained"]),
            num_classes=int(config["num_classes"]),
            tcn_channels=int(baseline.get("tcn_channels", 256)),
            embedding_dim=int(config["embedding_dim"]),
            dropout=float(config["dropout"]),
            update_backbone_bn_running_stats=bool(
                _mapping(config, "backbone_bn")["update_running_stats"]
            ),
        )
    return X3DSVisualExpert(
        backbone=build_x3d_s_feature_backbone(pretrained=bool(config["pretrained"])),
        num_classes=int(config["num_classes"]),
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
        head_type=_resolved_head_type(config),
        update_backbone_bn_running_stats=bool(
            _mapping(config, "backbone_bn")["update_running_stats"]
        ),
    )


def _length_bucket(num_frames: int) -> str:
    if num_frames <= 13:
        return "<=13"
    if num_frames <= 32:
        return "14-32"
    if num_frames <= 64:
        return "33-64"
    return ">64"


def _serialized_state(module: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getvalue()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_manifest(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    state_bytes = _serialized_state(model)
    if hasattr(model, "non_backbone_state_bytes"):
        custom_head_bytes = len(model.non_backbone_state_bytes())
    else:
        head_buffer = io.BytesIO()
        torch.save(
            {
                "embedding_head": model.embedding_head.state_dict(),
                "classifier": model.classifier.state_dict(),
            },
            head_buffer,
        )
        custom_head_bytes = len(head_buffer.getvalue())
    deployment_value = config.get("deployment_artifacts", {})
    deployment = deployment_value if isinstance(deployment_value, Mapping) else {}
    yolo_value = deployment.get("yolo_checkpoint")
    yolo_path = Path(str(yolo_value)) if yolo_value else None
    probe_value = deployment.get("phase0_probe")
    probe_path = Path(str(probe_value)) if probe_value else None
    probe: Mapping[str, Any] = {}
    if probe_path is not None and probe_path.is_file():
        loaded = json.loads(probe_path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            probe = loaded
    components = probe.get("components", {}) if isinstance(probe, Mapping) else {}
    x3d_probe = components.get("x3d_s", {}) if isinstance(components, Mapping) else {}
    is_x3d = not hasattr(model, "source_weight_sha256")
    return {
        "head_type": _resolved_head_type(config),
        "embedding_dim": int(
            getattr(model, "output_embedding_dim", config["embedding_dim"])
        ),
        "custom_head_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if not name.startswith("backbone.")
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "state_dict_bytes": len(state_bytes),
        "state_dict_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "custom_head_bytes": custom_head_bytes,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "pretrained_source": getattr(model, "pretrained_source", None)
        or (probe.get("source_url", "pytorchvideo x3d_s") if probe else "pytorchvideo x3d_s"),
        "pretraining_dataset": getattr(model, "pretraining_dataset", None)
        or (probe.get("pretraining", "kinetics_400") if probe else "kinetics_400"),
        "source_revision": getattr(model, "source_revision", None)
        or (probe.get("source_revision") if probe else None),
        "license": getattr(model, "license", None) or (probe.get("license") if probe else None),
        "backbone_source_weight_bytes": getattr(model, "source_weight_bytes", None),
        "backbone_source_weight_sha256": getattr(model, "source_weight_sha256", None),
        "x3d_source_weight_bytes": (
            x3d_probe.get("serialized_bytes") if is_x3d and isinstance(x3d_probe, Mapping) else None
        ),
        "x3d_source_weight_sha256": (
            x3d_probe.get("sha256") if is_x3d and isinstance(x3d_probe, Mapping) else None
        ),
        "yolo_weight_bytes": yolo_path.stat().st_size if yolo_path and yolo_path.is_file() else 0,
        "yolo_weight_sha256": _sha256_file(yolo_path) if yolo_path and yolo_path.is_file() else None,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
