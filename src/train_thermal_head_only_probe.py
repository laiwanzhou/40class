from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.thermal_native_dataset import (
    ThermalNativeDataset,
    collate_thermal_trials,
)
from src.diagnostics.activation_trace import state_dict_digest
from src.train_thermal_native_expert import (
    T1BRecipe,
    _schedule_multiplier,
    checkpoint_rank,
    run_epoch,
    save_prediction_archive,
    seed_everything,
)


@dataclass(frozen=True)
class T1B4HeadOnlyRecipe(T1BRecipe):
    epochs: int = 8
    hard_stop_epoch: int = 8
    automatic_extension: bool = False
    resume: bool = False


def _head_module(model: nn.Module) -> nn.Module:
    spatial = getattr(model, "spatial", None)
    backbone = getattr(spatial, "backbone", None)
    if backbone is not None and isinstance(getattr(backbone, "classifier", None), nn.Module):
        return backbone.classifier
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Module):
        return classifier
    raise TypeError("T1-B.4 requires the audited classifier module")


def _head_prefix(model: nn.Module) -> str:
    head = _head_module(model)
    matches = [name for name, module in model.named_modules() if module is head]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError("Could not identify one classifier state-dict prefix")
    return f"{matches[0]}."


def freeze_backbone_for_head_only(model: nn.Module) -> tuple[str, ...]:
    model.requires_grad_(False)
    _head_module(model).requires_grad_(True)
    model.eval()
    trainable = tuple(sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad))
    head_ids = {id(parameter) for parameter in _head_module(model).parameters()}
    if len(trainable) != 4:
        raise RuntimeError(f"Expected BN weight/bias and Linear weight/bias, got {trainable}")
    if any(parameter.requires_grad != (id(parameter) in head_ids) for parameter in model.parameters()):
        raise RuntimeError("Trainable parameter escaped the classifier boundary")
    return trainable


def configure_head_only_mode(model: nn.Module, training: bool) -> None:
    model.eval()
    if training:
        _head_module(model).train()
    head = _head_module(model)
    for module in model.modules():
        inside_head = module is head or any(module is child for child in head.modules())
        if module.training and not inside_head:
            raise RuntimeError("A frozen backbone module entered training mode")


def build_head_only_optimizer(
    model: nn.Module, recipe: T1B4HeadOnlyRecipe
) -> torch.optim.Optimizer:
    head = list(_head_module(model).parameters())
    if not head or any(not parameter.requires_grad for parameter in head):
        raise ValueError("Classifier must be the only unfrozen module before optimizer creation")
    return torch.optim.AdamW(
        [{"params": head, "lr": recipe.head_lr, "name": "head"}],
        weight_decay=recipe.weight_decay,
    )


def _partitioned_state(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prefix = _head_prefix(model)
    backbone: dict[str, torch.Tensor] = {}
    head: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        destination = head if name.startswith(prefix) else backbone
        destination[name] = value.detach().cpu().clone()
    return backbone, head


def _assert_state_equal(current: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    if current.keys() != expected.keys():
        raise RuntimeError("Frozen backbone state keys changed")
    changed = [name for name in current if not torch.equal(current[name], expected[name])]
    if changed:
        raise RuntimeError(f"Frozen backbone parameter or buffer changed: {changed[:5]}")


def train_head_only_probe(
    model: nn.Module,
    train_dataset: ThermalNativeDataset,
    validation_dataset: ThermalNativeDataset,
    *,
    device: torch.device,
    output_dir: Path,
    class_map_hash: str,
    recipe: T1B4HeadOnlyRecipe = T1B4HeadOnlyRecipe(),
) -> dict[str, Any]:
    if recipe.epochs != recipe.hard_stop_epoch or recipe.automatic_extension or recipe.resume:
        raise ValueError("T1-B.4 must execute one fresh, fixed-length run")
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
    trainable_names = freeze_backbone_for_head_only(model)
    backbone_before, head_before = _partitioned_state(model)
    backbone_digest_before = state_dict_digest(backbone_before)
    head_digest_before = state_dict_digest(head_before)
    optimizer = build_head_only_optimizer(model, recipe)
    history: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float, int] | None = None
    best_epoch = 0
    checkpoint_path = output_dir / "best_macro_f1.pt"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, recipe.hard_stop_epoch + 1):
        train_dataset.set_epoch(epoch)
        multiplier = _schedule_multiplier(epoch - 1, recipe)
        optimizer.param_groups[0]["lr"] = recipe.head_lr * multiplier
        train_result = run_epoch(
            model,
            train_loader,
            device=device,
            recipe=recipe,
            optimizer=optimizer,
            mode_setter=configure_head_only_mode,
        )
        current_backbone, _ = _partitioned_state(model)
        _assert_state_equal(current_backbone, backbone_before)
        validation_result = run_epoch(
            model,
            validation_loader,
            device=device,
            recipe=recipe,
            optimizer=None,
            mode_setter=configure_head_only_mode,
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
            "frozen_backbone_digest": state_dict_digest(current_backbone),
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
                    "probe_only": True,
                    "backbone_source": "official_imagenet_pretrained_iformer_t",
                    "epoch16_backbone_loaded": False,
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
            f"epoch={epoch:02d}/{recipe.hard_stop_epoch:02d} "
            f"train_loss={train_result.loss:.5f} "
            f"val_macro_f1={validation_result.metrics['macro_f1']:.5f} "
            f"val_accuracy={validation_result.metrics['accuracy']:.5f}",
            flush=True,
        )
    backbone_after, head_after = _partitioned_state(model)
    _assert_state_equal(backbone_after, backbone_before)
    backbone_digest_after = state_dict_digest(backbone_after)
    head_digest_after = state_dict_digest(head_after)
    if head_digest_after == head_digest_before:
        raise RuntimeError("Head-only probe completed without changing the classifier")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "recipe": asdict(recipe),
        "epochs_completed": len(history),
        "hard_stop_honored": len(history) == recipe.hard_stop_epoch,
        "automatic_extension_performed": False,
        "resume_performed": False,
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "best_latency": history[best_epoch - 1]["latency"],
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_backbone_state_digest_before": backbone_digest_before,
        "frozen_backbone_state_digest_after": backbone_digest_after,
        "frozen_backbone_exactly_unchanged": backbone_digest_before == backbone_digest_after,
        "head_state_digest_before": head_digest_before,
        "head_state_digest_after": head_digest_after,
        "head_state_changed": head_digest_before != head_digest_after,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "training_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
