from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from p27_data import P27Dataset, compute_fold_imu_stats
from p27_model import (
    P27EventModel,
    compute_event_targets,
    fp16_size_mib,
    initialise_p27_from_fold,
    parameter_count,
    sha256,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "p27_a_fixed.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_a"
EVENT_COMPONENTS = ("skeleton", "visual", "imu", "shared_motion", "clip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed P27-A0/A1/A2 outer-fold protocol")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument(
        "--variants", nargs="*", choices=["a0", "a1", "a2"], default=["a0", "a1", "a2"]
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Engineering-only two-batch run; never contributes to formal OOF.",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help=(
            "Skip only variants that already have final checkpoint, held OOF, metrics, "
            "and history under the same frozen config."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def create_loader(
    dataset: P27Dataset,
    config: dict,
    train: bool,
) -> DataLoader:
    sampler = None
    shuffle = train
    if train and bool(config["balanced_sampling"]):
        counts = Counter(sample.class_id for sample in dataset.samples)
        weights = torch.tensor(
            [1.0 / counts[sample.class_id] for sample in dataset.samples],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(int(config["seed"])),
        )
        shuffle = False
    workers = int(config["num_workers"]) if train else 0
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        drop_last=train,
    )


def move_batch(
    batch: dict[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def apply_modality_dropout(
    batch: dict[str, torch.Tensor],
    probabilities: dict[str, float],
) -> None:
    batch_size = len(batch["label"])
    originals = {
        name: batch[f"{name}_present"].clone()
        for name in ("depth", "ir", "skeleton", "imu")
    }
    dropped: dict[str, torch.Tensor] = {}
    for name in ("depth", "ir", "skeleton", "imu"):
        present = batch[f"{name}_present"]
        random_drop = (
            torch.rand(batch_size, 1, device=present.device)
            < float(probabilities[name])
        )
        dropped[name] = random_drop & present.bool()
        present[dropped[name]] = 0.0
    total_present = sum(
        batch[f"{name}_present"] for name in ("depth", "ir", "skeleton", "imu")
    )
    empty = total_present.squeeze(1) == 0
    if bool(empty.any()):
        for index in torch.where(empty)[0].tolist():
            available = [
                name
                for name in ("skeleton", "depth", "ir", "imu")
                if bool(originals[name][index].item())
            ]
            if available:
                restored = available[0]
                batch[f"{restored}_present"][index] = originals[restored][index]
                dropped[restored][index] = False
    if bool(dropped["depth"].any()):
        batch["depth"][dropped["depth"].squeeze(1)] = 0.0
    if bool(dropped["ir"].any()):
        batch["ir"][dropped["ir"].squeeze(1)] = 0.0
    if bool(dropped["skeleton"].any()):
        batch["skeleton"][dropped["skeleton"].squeeze(1)] = 0.0
    if bool(dropped["imu"].any()):
        indices = dropped["imu"].squeeze(1)
        batch["imu"][indices] = 0.0
        batch["imu_time_mask"][indices] = 0.0
        batch["imu_device_mask"][indices] = 0.0


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(1)
    loss = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    expanded = mask.expand_as(loss)
    return (loss * expanded).sum() / expanded.sum().clamp_min(1.0)


def event_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = {
        "skeleton": masked_smooth_l1(
            outputs["skeleton_events"], targets["skeleton"], targets["skeleton_mask"]
        ),
        "visual": masked_smooth_l1(
            outputs["visual_events"], targets["visual"], targets["visual_mask"]
        ),
        "imu": masked_smooth_l1(
            outputs["imu_events"], targets["imu"], targets["imu_mask"]
        ),
        "shared_motion": masked_smooth_l1(
            outputs["shared_motion"],
            targets["shared_motion"],
            torch.maximum(targets["visual_mask"], targets["imu_mask"]),
        ),
        "clip": masked_smooth_l1(
            outputs["clip_events"], targets["clip"], targets["imu_mask"]
        ),
    }
    denominator = sum(float(weights[name]) for name in EVENT_COMPONENTS)
    total = sum(float(weights[name]) * components[name] for name in EVENT_COMPONENTS)
    return total / max(denominator, 1e-6), components


def set_stage_trainability(model: P27EventModel, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if stage == "a1_pretrain":
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(False)
    elif stage == "a1_classifier":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    elif stage not in {"a0", "a2"}:
        raise ValueError(stage)


def set_model_mode(model: P27EventModel, stage: str, training: bool) -> None:
    model.train(training)
    if training and stage == "a1_classifier":
        model.eval()
        model.classifier.train()


def make_optimizer(
    model: P27EventModel,
    stage_config: dict,
    weight_decay: float,
) -> torch.optim.Optimizer:
    base_lr = float(stage_config["learning_rate"])
    backbone_lr = float(stage_config.get("backbone_learning_rate", base_lr))
    backbone_ids = {
        id(parameter)
        for module in (model.visual, model.skeleton)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    backbone = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_ids
    ]
    other = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    if other:
        groups.append({"params": other, "lr": base_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def cosine_learning_rates(
    optimizer: torch.optim.Optimizer,
    initial_lrs: list[float],
    epoch: int,
    epochs: int,
) -> None:
    scale = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(epochs, 1)))
    for group, initial in zip(optimizer.param_groups, initial_lrs, strict=True):
        group["lr"] = initial * scale


def gradient_norm(
    loss: torch.Tensor,
    parameters: list[nn.Parameter],
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = sum(
        float(gradient.detach().float().square().sum())
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(total)


def estimate_a2_event_weight(
    model: P27EventModel,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    criterion: nn.Module,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    raw = next(iter(loader))
    batch = move_batch(raw, device)
    targets = compute_event_targets(batch)
    apply_modality_dropout(batch, config["modality_dropout"])
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        outputs = model(
            batch,
            temporal_mask_probability=float(config["a2"]["temporal_mask_probability"]),
        )
        class_loss = criterion(outputs["logits"], batch["label"])
        event_loss, _ = event_losses(
            outputs, targets, config["event_component_weights"]
        )
    shared_parameters = [
        parameter
        for module in (model.event_input, model.event_temporal)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    class_norm = gradient_norm(class_loss, shared_parameters, retain_graph=True)
    event_norm = gradient_norm(event_loss, shared_parameters, retain_graph=False)
    base = float(config["a2"]["event_base_weight"])
    target_ratio = float(config["a2"]["target_event_to_class_gradient_ratio"])
    raw_multiplier = target_ratio * class_norm / max(base * event_norm, 1e-12)
    multiplier = min(
        float(config["a2"]["event_weight_multiplier_max"]),
        max(float(config["a2"]["event_weight_multiplier_min"]), raw_multiplier),
    )
    return base * multiplier, {
        "classification_gradient_norm": class_norm,
        "event_gradient_norm": event_norm,
        "raw_event_weight_multiplier": raw_multiplier,
        "applied_event_weight_multiplier": multiplier,
        "event_weight": base * multiplier,
    }


def run_train_epoch(
    model: P27EventModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
    stage: str,
    event_weight: float,
    temporal_mask_probability: float,
    use_amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    set_model_mode(model, stage, training=True)
    totals: defaultdict[str, float] = defaultdict(float)
    labels: list[int] = []
    predictions: list[int] = []
    optimizer.zero_grad(set_to_none=True)
    batches = 0
    for batch_index, raw in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw, device)
        targets = compute_event_targets(batch)
        apply_modality_dropout(batch, config["modality_dropout"])
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                batch, temporal_mask_probability=temporal_mask_probability
            )
            class_loss = criterion(outputs["logits"], batch["label"])
            event_loss, components = event_losses(
                outputs, targets, config["event_component_weights"]
            )
            if stage == "a1_pretrain":
                loss = event_loss
            elif stage in {"a0", "a1_classifier"}:
                loss = class_loss
            else:
                loss = class_loss + event_weight * event_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["max_grad_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        batches += 1
        totals["loss"] += float(loss.detach())
        totals["classification_loss"] += float(class_loss.detach())
        totals["event_loss"] += float(event_loss.detach())
        for name, value in components.items():
            totals[f"event_{name}"] += float(value.detach())
        labels.extend(batch["label"].detach().cpu().tolist())
        predictions.extend(outputs["logits"].argmax(dim=1).detach().cpu().tolist())
    result = {key: value / max(batches, 1) for key, value in totals.items()}
    result.update(
        {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
            "macro_f1": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
            "batches": float(batches),
        }
    )
    return result


@torch.inference_mode()
def evaluate(
    model: P27EventModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
    use_amp: bool,
    max_batches: int | None = None,
    ablation: str | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []
    subjects: list[str] = []
    logits_all: list[np.ndarray] = []
    prediction_events: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    target_events: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    masks: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    total_loss = 0.0
    batches = 0
    started = time.perf_counter()
    for batch_index, raw in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw, device)
        model_ablation = ablation
        if ablation == "ir_zero":
            batch["ir"].zero_()
            batch["ir_present"].zero_()
        elif ablation == "imu_zero":
            batch["imu"].zero_()
            batch["imu_time_mask"].zero_()
            batch["imu_device_mask"].zero_()
            batch["imu_present"].zero_()
        elif ablation == "ir_shuffle":
            permutation = torch.arange(
                len(batch["label"]) - 1,
                -1,
                -1,
                device=device,
            )
            batch["ir"] = batch["ir"][permutation]
            batch["ir_present"] = batch["ir_present"][permutation]
            model_ablation = None
        elif ablation == "imu_shuffle":
            permutation = torch.arange(
                len(batch["label"]) - 1,
                -1,
                -1,
                device=device,
            )
            batch["imu"] = batch["imu"][permutation]
            batch["imu_time_mask"] = batch["imu_time_mask"][permutation]
            batch["imu_device_mask"] = batch["imu_device_mask"][permutation]
            batch["imu_present"] = batch["imu_present"][permutation]
            model_ablation = None
        targets = compute_event_targets(batch)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch, ablation=model_ablation)
            loss = criterion(outputs["logits"], batch["label"])
        total_loss += float(loss)
        batches += 1
        labels.extend(batch["label"].cpu().tolist())
        predictions.extend(outputs["logits"].argmax(dim=1).cpu().tolist())
        sample_ids.extend(raw["sample_id"])
        subjects.extend(raw["subject"])
        logits_all.append(outputs["logits"].float().cpu().numpy())
        for name, output_name in (
            ("skeleton", "skeleton_events"),
            ("visual", "visual_events"),
            ("imu", "imu_events"),
            ("shared_motion", "shared_motion"),
            ("clip", "clip_events"),
        ):
            prediction_events[name].append(outputs[output_name].float().cpu().numpy())
            target_events[name].append(targets[name].float().cpu().numpy())
        masks["skeleton"].append(targets["skeleton_mask"].cpu().numpy())
        masks["visual"].append(targets["visual_mask"].cpu().numpy())
        masks["imu"].append(targets["imu_mask"].cpu().numpy())
        masks["shared_motion"].append(
            torch.maximum(targets["visual_mask"], targets["imu_mask"]).cpu().numpy()
        )
        masks["clip"].append(targets["imu_mask"].cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    arrays = {
        "sample_ids": np.asarray(sample_ids),
        "subjects": np.asarray(subjects),
        "labels": np.asarray(labels, dtype=np.int64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "logits": np.concatenate(logits_all),
    }
    for name in EVENT_COMPONENTS:
        arrays[f"{name}_predictions"] = np.concatenate(prediction_events[name])
        arrays[f"{name}_targets"] = np.concatenate(target_events[name])
        arrays[f"{name}_masks"] = np.concatenate(masks[name])
    metrics = {
        "loss": total_loss / max(batches, 1),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "samples": len(labels),
        "elapsed_seconds": elapsed,
        "milliseconds_per_sample": 1000.0 * elapsed / max(len(labels), 1),
    }
    return metrics, arrays


def event_target_means(
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, float] = defaultdict(float)
    for batch_index, raw in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw, device)
        targets = compute_event_targets(batch)
        component_masks = {
            "skeleton": targets["skeleton_mask"],
            "visual": targets["visual_mask"],
            "imu": targets["imu_mask"],
            "shared_motion": torch.maximum(
                targets["visual_mask"], targets["imu_mask"]
            ),
            "clip": targets["imu_mask"],
        }
        for name in EVENT_COMPONENTS:
            value = targets[name]
            mask = component_masks[name]
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(1)
            expanded = mask.expand_as(value)
            reduced_axes = tuple(range(value.ndim - 1))
            part_sum = (value * expanded).sum(dim=reduced_axes).cpu().numpy()
            part_count = float(expanded[..., 0].sum())
            sums[name] = sums.get(name, np.zeros_like(part_sum)) + part_sum
            counts[name] += part_count
    return {
        name: sums[name] / max(counts[name], 1.0) for name in EVENT_COMPONENTS
    }


def add_event_metrics(
    metrics: dict[str, object],
    arrays: dict[str, np.ndarray],
    target_means: dict[str, np.ndarray],
) -> None:
    event_metrics: dict[str, object] = {}
    for name in EVENT_COMPONENTS:
        prediction = arrays[f"{name}_predictions"]
        target = arrays[f"{name}_targets"]
        mask = arrays[f"{name}_masks"]
        while mask.ndim < target.ndim:
            mask = np.expand_dims(mask, axis=1)
        expanded = np.broadcast_to(mask, target.shape)
        mean_shape = (1,) * (target.ndim - 1) + (-1,)
        baseline = np.broadcast_to(target_means[name].reshape(mean_shape), target.shape)
        mae = float(np.abs(prediction - target)[expanded > 0].mean())
        baseline_mae = float(np.abs(baseline - target)[expanded > 0].mean())
        event_metrics[name] = {
            "mae": mae,
            "train_mean_baseline_mae": baseline_mae,
            "relative_improvement": (baseline_mae - mae) / max(baseline_mae, 1e-12),
        }
    metrics["event_prediction"] = event_metrics


def save_checkpoint(
    path: Path,
    model: P27EventModel,
    config: dict,
    fold: int,
    variant: str,
    init_manifest: dict[str, object],
    metrics: dict[str, object],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "fold": fold,
            "variant": variant,
            "initialization": init_manifest,
            "metrics": metrics,
        },
        path,
    )


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_stage(
    model: P27EventModel,
    stage: str,
    stage_config: dict,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
    use_amp: bool,
    smoke: bool,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    set_stage_trainability(model, stage)
    optimizer = make_optimizer(model, stage_config, float(config["weight_decay"]))
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epochs = 1 if smoke else int(stage_config["epochs"])
    temporal_mask_probability = float(
        stage_config.get("temporal_mask_probability", 0.0)
    )
    history: list[dict[str, object]] = []
    last_gradient = {
        "classification_gradient_norm": 0.0,
        "event_gradient_norm": 0.0,
        "raw_event_weight_multiplier": 0.0,
        "applied_event_weight_multiplier": 0.0,
        "event_weight": float(stage_config.get("event_loss_weight", 0.0)),
    }
    for epoch in range(1, epochs + 1):
        cosine_learning_rates(optimizer, initial_lrs, epoch, epochs)
        if stage == "a2":
            event_weight, last_gradient = estimate_a2_event_weight(
                model, train_loader, device, config, criterion, use_amp
            )
        else:
            event_weight = float(stage_config.get("event_loss_weight", 0.0))
        started = time.perf_counter()
        train_metrics = run_train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            config,
            stage,
            event_weight,
            temporal_mask_probability,
            use_amp,
            max_batches=2 if smoke else None,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "stage": stage,
            "seconds": time.perf_counter() - started,
            "learning_rates": json.dumps(
                [float(group["lr"]) for group in optimizer.param_groups]
            ),
            "event_weight": event_weight,
            **last_gradient,
            **train_metrics,
        }
        history.append(row)
        print(
            f"{stage} epoch {epoch:02d}/{epochs:02d} "
            f"loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"event={train_metrics['event_loss']:.4f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    return history, last_gradient


def run_variant(
    variant: str,
    fold: int,
    config: dict,
    output: Path,
    train_loader: DataLoader,
    held_loader: DataLoader,
    target_means: dict[str, np.ndarray] | None,
    imu_mean: np.ndarray,
    imu_std: np.ndarray,
    device: torch.device,
    smoke: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray], P27EventModel]:
    model = P27EventModel(
        torch.from_numpy(imu_mean),
        torch.from_numpy(imu_std),
        event_dim=int(config["event_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    init_manifest = initialise_p27_from_fold(model, fold)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    use_amp = bool(config["use_amp"] and device.type == "cuda")
    history: list[dict[str, object]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if variant == "a0":
        stage_history, _ = train_stage(
            model,
            "a0",
            config["a0"],
            train_loader,
            criterion,
            device,
            config,
            use_amp,
            smoke,
        )
        history.extend(stage_history)
    elif variant == "a1":
        stage_history, _ = train_stage(
            model,
            "a1_pretrain",
            config["a1_pretrain"],
            train_loader,
            criterion,
            device,
            config,
            use_amp,
            smoke,
        )
        history.extend(stage_history)
        stage_history, _ = train_stage(
            model,
            "a1_classifier",
            config["a1_classifier"],
            train_loader,
            criterion,
            device,
            config,
            use_amp,
            smoke,
        )
        history.extend(stage_history)
    elif variant == "a2":
        a1_path = output.parent / "a1" / "final.pt"
        if not a1_path.is_file():
            raise FileNotFoundError(f"A2 requires the matching A1 checkpoint: {a1_path}")
        checkpoint = torch.load(a1_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        init_manifest = {
            **init_manifest,
            "a1_parent": {
                "path": str(a1_path),
                "sha256": sha256(a1_path),
            },
        }
        stage_history, _ = train_stage(
            model,
            "a2",
            config["a2"],
            train_loader,
            criterion,
            device,
            config,
            use_amp,
            smoke,
        )
        history.extend(stage_history)
    else:
        raise ValueError(variant)
    train_seconds = time.perf_counter() - started
    held_metrics, arrays = evaluate(
        model,
        held_loader,
        criterion,
        device,
        config,
        use_amp,
        max_batches=2 if smoke else None,
    )
    if target_means is not None:
        add_event_metrics(held_metrics, arrays, target_means)
    held_metrics.update(
        {
            "variant": variant,
            "fold": fold,
            "protocol": config["protocol"],
            "formal": not smoke,
            "train_seconds": train_seconds,
            "parameters": parameter_count(model),
            "fp16_size_mib": fp16_size_mib(model),
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated() / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
            "initialization": init_manifest,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    write_history(output / "history.csv", history)
    np.savez_compressed(output / "held_outputs.npz", **arrays)
    (output / "metrics.json").write_text(
        json.dumps(held_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_checkpoint(
        output / "final.pt",
        model,
        config,
        fold,
        variant,
        init_manifest,
        held_metrics,
    )
    fp16_state = {
        key: value.detach().half().cpu()
        if torch.is_floating_point(value)
        else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": fp16_state,
            "config": config,
            "fold": fold,
            "variant": variant,
        },
        output / "deployment_fp16.pt",
    )
    return held_metrics, arrays, model


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config["protocol"] != "p27-a-v1-fixed-before-training":
        raise ValueError("refusing an unregistered P27 protocol")
    output_root = args.output_dir.resolve()
    if args.smoke:
        output_root = output_root.parent / f"{output_root.name}_smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "config_used.json"
    if args.resume_completed and config_path.is_file():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config != config:
            raise ValueError("resume refused: config_used.json differs from requested config")
    else:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    seed_everything(int(config["seed"]))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = PROJECT_DIR / config["manifest"]
    all_summary: dict[str, object] = {
        "protocol": config["protocol"],
        "formal": not args.smoke,
        "device": str(device),
        "folds": {},
    }
    variants = list(args.variants)
    if "a2" in variants and "a1" not in variants:
        variants.insert(0, "a1")
    for fold in args.folds:
        fold_started = time.perf_counter()
        imu_mean, imu_std = compute_fold_imu_stats(manifest_path, fold)
        train_dataset = P27Dataset(
            manifest_path,
            fold,
            train=True,
            num_frames=int(config["num_frames"]),
            image_height=int(config["image_height"]),
            image_width=int(config["image_width"]),
            roi_padding=float(config["roi_padding"]),
        )
        held_dataset = P27Dataset(
            manifest_path,
            fold,
            train=False,
            num_frames=int(config["num_frames"]),
            image_height=int(config["image_height"]),
            image_width=int(config["image_width"]),
            roi_padding=float(config["roi_padding"]),
        )
        train_loader = create_loader(train_dataset, config, train=True)
        held_loader = create_loader(held_dataset, config, train=False)
        target_means = event_target_means(
            train_loader,
            device,
            max_batches=2 if args.smoke else None,
        )
        fold_summary: dict[str, object] = {
            "train_samples": len(train_dataset),
            "held_samples": len(held_dataset),
            "imu_mean": imu_mean.tolist(),
            "imu_std": imu_std.tolist(),
            "variants": {},
        }
        for variant in variants:
            variant_output = output_root / f"fold_{fold}" / variant
            required_outputs = (
                variant_output / "history.csv",
                variant_output / "held_outputs.npz",
                variant_output / "metrics.json",
                variant_output / "final.pt",
                variant_output / "deployment_fp16.pt",
            )
            if args.resume_completed and all(path.is_file() for path in required_outputs):
                metrics = json.loads(
                    (variant_output / "metrics.json").read_text(encoding="utf-8")
                )
                if (
                    metrics.get("protocol") != config["protocol"]
                    or int(metrics.get("fold", -1)) != fold
                    or metrics.get("variant") != variant
                    or not bool(metrics.get("formal", False))
                ):
                    raise ValueError(
                        f"resume refused: incompatible completed variant {variant_output}"
                    )
                fold_summary["variants"][variant] = metrics
                print(
                    f"resume: keeping completed fold={fold} variant={variant}",
                    flush=True,
                )
                continue
            metrics, _, model = run_variant(
                variant,
                fold,
                config,
                variant_output,
                train_loader,
                held_loader,
                target_means,
                imu_mean,
                imu_std,
                device,
                args.smoke,
            )
            fold_summary["variants"][variant] = metrics
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_summary["elapsed_seconds"] = time.perf_counter() - fold_started
        all_summary["folds"][str(fold)] = fold_summary
        (output_root / "summary_partial.json").write_text(
            json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_root / "summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
