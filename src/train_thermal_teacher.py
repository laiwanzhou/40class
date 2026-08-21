from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch import nn

from src.models.thermal_teachers import OFFICIAL_R2PLUS1D18, ThermalR2Plus1D18Teacher
from src.train_thermal_generation2 import CHECKPOINT_RANK, fixed_label_metrics


class TeacherTrainingAuthorizationError(PermissionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_teacher_config(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == 1, "schema_version must be 1")
    _require(config.get("route") == "c1_r2plus1d18", "only C1 is eligible")
    teacher = config.get("teacher", {})
    data = config.get("data", {})
    optimization = config.get("optimization", {})
    runtime = config.get("runtime", {})
    _require(teacher.get("architecture") == "torchvision_r2plus1d_18", "teacher changed")
    _require(teacher.get("num_classes") == 40, "teacher must emit 40 logits")
    _require(teacher.get("training_only") is True, "teacher must remain training-only")
    for key in ("repository_revision", "license", "checkpoint_url", "checkpoint_sha256", "checkpoint_bytes"):
        _require(teacher.get(key) == OFFICIAL_R2PLUS1D18[key], f"teacher provenance changed: {key}")
    _require(data.get("modality") == "thermal_only", "only Thermal inputs are allowed")
    _require(data.get("development_split") == "metadata/splits/train12_val2_user6_user7_development.json", "development split changed")
    _require(data.get("class_ids") == list(range(40)), "class map changed")
    _require(data.get("views") == ["full", "thermal_yolo_context"], "teacher views changed")
    _require(data.get("normalized_windows") == [[0.0, 0.5], [0.25, 0.75], [0.5, 1.0]], "windows changed")
    _require(data.get("frames_per_window") == 16, "frame count changed")
    _require(data.get("raster_crop_size") == 160, "crop size changed")
    for field in ("read_heldout4_labels", "read_competition_test", "read_quarantined_evidence", "import_ir_depth_inputs"):
        _require(data.get(field) is False, f"forbidden data flag must be false: {field}")
    _require(optimization.get("seed") == 20260715, "seed changed")
    _require(optimization.get("maximum_epochs") == 30, "maximum_epochs must be 30")
    _require(optimization.get("automatic_resume") is False, "resume is forbidden")
    _require(optimization.get("automatic_extension") is False, "extension is forbidden")
    _require(optimization.get("physical_batch_trials") == 1, "teacher physical batch must be 1")
    _require(optimization.get("effective_batch_trials") == 8, "effective batch must be 8")
    _require(optimization.get("checkpoint_rank") == CHECKPOINT_RANK, "checkpoint rank changed")
    _require(runtime.get("sequential_clip_execution") is True, "clip execution must be sequential")
    _require(runtime.get("average_available_clip_logits") is True, "clip logits must be averaged")
    _require(runtime.get("maximum_peak_allocated_mib_exclusive") == 7300, "VRAM gate changed")


def load_teacher_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("teacher config must be a mapping")
    validate_teacher_config(config)
    config["config_path"] = str(path.resolve())
    return config


def require_teacher_training_authorization(config: dict[str, Any], *, token: str | None) -> None:
    if config.get("training_authorized") is not True:
        raise TeacherTrainingAuthorizationError("C1 training is not authorized")
    if not token or token != config.get("authorization_token"):
        raise TeacherTrainingAuthorizationError("exact C1 authorization token is required")
    if config.get("authorization", {}).get("c2_authorized") is not False:
        raise TeacherTrainingAuthorizationError("C2 cannot be co-authorized with C1")


def set_teacher_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_teacher_optimizer_and_scheduler(
    model: ThermalR2Plus1D18Teacher,
    *,
    config: dict[str, Any],
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    validate_teacher_config(config)
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    optimization = config["optimization"]
    backbone, classifier = model.parameter_groups()
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(optimization["pretrained_backbone_learning_rate"])},
            {"params": classifier, "lr": float(optimization["classifier_learning_rate"])},
        ],
        weight_decay=float(optimization["weight_decay"]),
    )
    warmup_steps = int(optimization["warmup_epochs"]) * steps_per_epoch
    total_steps = int(optimization["maximum_epochs"]) * steps_per_epoch

    def scale(step: int) -> float:
        completed = step + 1
        if warmup_steps and completed <= warmup_steps:
            return completed / warmup_steps
        progress = (completed - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _bn_snapshot(model: nn.Module) -> list[tuple[nn.Module, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
    result = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            result.append((
                module,
                module.running_mean.clone() if module.running_mean is not None else None,
                module.running_var.clone() if module.running_var is not None else None,
                module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None,
            ))
    return result


def _restore_bn(snapshot: list[tuple[nn.Module, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]) -> None:
    with torch.no_grad():
        for module, mean, variance, tracked in snapshot:
            if mean is not None:
                module.running_mean.copy_(mean)
            if variance is not None:
                module.running_var.copy_(variance)
            if tracked is not None:
                module.num_batches_tracked.copy_(tracked)


def sequential_trial_backward(
    *,
    model: ThermalR2Plus1D18Teacher,
    batch: dict[str, Any],
    device: torch.device,
    label_smoothing: float,
    loss_scale: float,
    amp_enabled: bool,
) -> dict[str, Any]:
    """Exact mean-logit gradient while retaining activations for one clip at a time."""
    values = _batch_to_device(batch, device)
    if values["label"].shape != (1,) or not bool(values["loss_eligible"].item()):
        raise ValueError("sequential teacher backward requires one eligible trial")
    clips = list(model.iter_available_clips(
        values["full_rgb"], values["crop_rgb"],
        window_mask=values["window_mask"], availability=values["availability"],
    ))
    if not clips:
        raise ValueError("teacher trial has no available clips")
    snapshot = _bn_snapshot(model)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    autocast = lambda: torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=amp_enabled,
    )
    with torch.no_grad(), autocast():
        detached = [model.backbone(clip) for _, clip, _ in clips]
        mean_logits = torch.stack(detached).mean(dim=0).float()
    _restore_bn(snapshot)
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    proxy = mean_logits.detach().requires_grad_(True)
    loss = functional.cross_entropy(
        proxy, values["label"], label_smoothing=label_smoothing
    )
    gradient = torch.autograd.grad(loss * loss_scale, proxy)[0]
    share = gradient / len(clips)
    for _, clip, _ in clips:
        with autocast():
            clip_logits = model.backbone(clip)
        torch.autograd.backward(clip_logits, grad_tensors=share.to(clip_logits.dtype))
    if not bool(torch.isfinite(mean_logits).all()) or not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite teacher training values")
    return {
        "loss": float(loss.detach()),
        "logits": mean_logits.detach(),
        "clip_count": len(clips),
    }


def collect_teacher_predictions(
    *, model: ThermalR2Plus1D18Teacher, loader: Iterable[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    model.eval()
    logits, labels, users, sample_ids, clip_counts = [], [], [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = _batch_to_device(raw, device)
            output = model(
                batch["full_rgb"], batch["crop_rgb"],
                window_mask=batch["window_mask"], availability=batch["availability"],
            )
            eligible = batch["loss_eligible"].bool().cpu().numpy()
            logits.append(output["logits"].float().cpu().numpy()[eligible])
            labels.append(batch["label"].cpu().numpy()[eligible])
            clip_counts.append(output["clip_count"].cpu().numpy()[eligible])
            indices = np.flatnonzero(eligible)
            users.extend(str(raw["user_id"][index]) for index in indices)
            sample_ids.extend(str(raw["sample_id"][index]) for index in indices)
    archive = {
        "logits": np.concatenate(logits).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.int64),
        "users": np.asarray(users),
        "sample_ids": np.asarray(sample_ids),
        "clip_counts": np.concatenate(clip_counts).astype(np.int64),
    }
    archive["metrics"] = fixed_label_metrics(
        labels=archive["labels"], logits=archive["logits"], users=archive["users"]
    )
    return archive
