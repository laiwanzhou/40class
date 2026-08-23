from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_ir_depth_videomaev2_teacher import load_probe_config
from src.data.ir_depth_videomaev2_dataset import IRDepthVideoMAEV2Dataset
from src.models.ir_depth_videomaev2_teacher import (
    IRDepthVideoMAEV2Teacher,
    build_official_videomaev2_vit_b,
    sequential_multiview_backward,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_depth_videomaev2_vit_b_p1.yaml"


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_training_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P1 config must be a mapping")
    training = config.get("training", {})
    if config.get("stage") != "P1":
        raise ValueError("P1 stage changed")
    if training.get("epochs") != 20 or training.get("gradient_accumulation") != 8:
        raise ValueError("P1 epoch or accumulation contract changed")
    if training.get("unfrozen_backbone_blocks") != 4:
        raise ValueError("P1 trainable backbone tail changed")
    if training.get("sampler") != "inverse_frequency_replacement":
        raise ValueError("P1 class balancing changed")
    if training.get("early_stopping_patience") != 6:
        raise ValueError("P1 early stopping changed")
    if config.get("policy", {}).get("automatic_logits_export") is not False:
        raise ValueError("P1 may not export logits automatically")
    load_probe_config(_project_path(str(config["p0_config"])))
    return config


def require_training_authorization(config: dict[str, Any], *, token: str | None) -> None:
    authorization = config.get("authorization", {})
    if authorization.get("p1_authorized") is not True or token != authorization.get("token"):
        raise PermissionError("P1 training authorization token is missing or invalid")


def build_inverse_frequency_sampler(
    class_ids: list[int], *, seed: int
) -> tuple[WeightedRandomSampler, dict[str, object]]:
    labels = np.asarray(class_ids, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    count_by_class = dict(zip(classes.tolist(), counts.tolist(), strict=True))
    weights = torch.tensor([1.0 / count_by_class[int(label)] for label in labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights, num_samples=len(labels), replacement=True, generator=generator
    )
    total = float(weights.sum())
    probability_mass = {
        str(class_id): float(weights[labels == class_id].sum() / total) for class_id in classes
    }
    return sampler, {
        "policy": "inverse_frequency_replacement",
        "class_counts": {str(key): int(value) for key, value in count_by_class.items()},
        "class_probability_mass": probability_mass,
        "epoch_samples": len(labels),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _metrics(
    *, labels: np.ndarray, logits: np.ndarray, users: np.ndarray
) -> dict[str, object]:
    predictions = logits.argmax(axis=1)
    probabilities = torch.from_numpy(logits).softmax(dim=1).numpy()
    user_accuracy = {
        str(user): float(accuracy_score(labels[users == user], predictions[users == user]))
        for user in sorted(set(users.tolist()))
    }
    recalls = []
    for class_id in range(40):
        selected = labels == class_id
        recalls.append(float(np.mean(predictions[selected] == class_id)) if selected.any() else 0.0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(40), average="macro", zero_division=0)),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(40))),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "predicted_class_count": int(np.unique(predictions).size),
        "zero_recall_classes": int(np.sum(np.asarray(recalls) == 0.0)),
        "per_class_recall": recalls,
    }


def _make_dataset(config: dict[str, Any], *, partition: str, training: bool) -> IRDepthVideoMAEV2Dataset:
    data = config["data"]
    p0 = load_probe_config(_project_path(str(config["p0_config"])))
    return IRDepthVideoMAEV2Dataset(
        manifest_path=_project_path(str(data["manifest"])),
        split_path=_project_path(str(data["split"])),
        data_root=Path(str(data["root"])),
        pose_cache_path=Path(str(data["pose_cache"])),
        partition=partition,
        training=training,
        frames=16,
        image_size=224,
        temporal_jitter=float(data["temporal_jitter"]) if training else 0.0,
        interaction_config=dict(p0["roi"]),
    )


def _configure_trainable_tail(model: IRDepthVideoMAEV2Teacher, blocks: int) -> int:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for block in list(model.backbone.blocks)[-blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for module in (model.backbone.fc_norm, model.backbone.head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.class_view_gate.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def evaluate(
    model: IRDepthVideoMAEV2Teacher,
    loader: DataLoader[dict[str, object]],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model.eval()
    logits, labels, users, sample_ids, gates = [], [], [], [], []
    for step, batch in enumerate(loader, start=1):
        clips = batch["clips"].to(device)
        availability = batch["availability"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            output = model(clips=clips, availability=availability)
        logits.append(output["logits"].float().cpu())
        gates.append(output["class_view_weights"].float().cpu())
        labels.append(batch["label"].cpu())
        users.extend(str(value) for value in batch["user_id"])
        sample_ids.extend(str(value) for value in batch["sample_id"])
        if max_batches is not None and step >= max_batches:
            break
    archive = {
        "logits": torch.cat(logits).numpy(),
        "labels": torch.cat(labels).numpy(),
        "users": np.asarray(users),
        "sample_ids": np.asarray(sample_ids),
        "class_view_weights": torch.cat(gates).numpy(),
    }
    return _metrics(labels=archive["labels"], logits=archive["logits"], users=archive["users"]), archive


def _save_checkpoint(
    path: Path,
    *,
    model: IRDepthVideoMAEV2Teacher,
    epoch: int,
    metrics: dict[str, object],
    config: dict[str, Any],
) -> None:
    torch.save(
        {"epoch": epoch, "metrics": metrics, "model_state_dict": model.state_dict(), "config": config},
        path,
    )


def run_training(
    config_path: Path,
    *,
    token: str,
    smoke_test: bool = False,
) -> dict[str, object]:
    config = load_training_config(config_path)
    require_training_authorization(config, token=token)
    p0_report = json.loads(_project_path(str(config["p0_report"])).read_text(encoding="utf-8"))
    if p0_report.get("status") != "passed" or p0_report.get("p1_status") != "eligible_to_start":
        raise RuntimeError("P1 is blocked because P0 did not pass")
    training = config["training"]
    seed = int(training["seed"])
    _set_seed(seed)
    device = torch.device("cuda")
    train_dataset = _make_dataset(config, partition="train", training=True)
    val_dataset = _make_dataset(config, partition="validation", training=False)
    sampler, sampler_audit = build_inverse_frequency_sampler(train_dataset.class_ids(), seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    p0_config = load_probe_config(_project_path(str(config["p0_config"])))
    backbone, provenance = build_official_videomaev2_vit_b(
        checkpoint_path=Path(str(p0_config["checkpoint"]["path"])), num_classes=40, with_cp=True
    )
    model = IRDepthVideoMAEV2Teacher(backbone=backbone, num_classes=40).to(device)
    trainable_parameters = _configure_trainable_tail(
        model, int(training["unfrozen_backbone_blocks"])
    )
    backbone_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("backbone.blocks")
    ]
    head_parameters = list(model.backbone.fc_norm.parameters()) + list(model.backbone.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(training["backbone_learning_rate"])},
            {"params": head_parameters, "lr": float(training["head_learning_rate"])},
            {"params": [model.class_view_gate], "lr": float(training["fusion_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    epochs = 1 if smoke_test else int(training["epochs"])
    output_root = _project_path(str(config["output_root"]))
    run_id = str(config["run_id"]) + ("_smoke" if smoke_test else "")
    run_dir = output_root / run_id
    if smoke_test:
        suffix = 2
        while run_dir.exists():
            run_id = str(config["run_id"]) + f"_smoke_{suffix:02d}"
            run_dir = output_root / run_id
            suffix += 1
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    history: list[dict[str, object]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    patience = 0
    accumulation_target = int(training["gradient_accumulation"])
    formal_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits, train_labels, train_users = [], [], []
        sampled_counts = np.zeros(40, dtype=np.int64)
        accumulated = 0
        optimizer_steps = 0
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for step, batch in enumerate(train_loader, start=1):
            clips = batch["clips"].to(device, non_blocking=True)
            availability = batch["availability"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                result = sequential_multiview_backward(
                    model=model,
                    clips=clips,
                    availability=availability,
                    labels=labels,
                    label_smoothing=float(training["label_smoothing"]),
                )
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in trainable):
                raise FloatingPointError("non-finite P1 gradient")
            accumulated += 1
            train_logits.append(result["logits"].float().cpu())
            train_labels.append(labels.cpu())
            train_users.extend(str(value) for value in batch["user_id"])
            sampled_counts[int(labels.item())] += 1
            is_last = step == len(train_loader) or (smoke_test and step >= 2)
            if accumulated == accumulation_target or is_last:
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0
            if smoke_test and step >= 2:
                break
        train_archive = {
            "logits": torch.cat(train_logits).numpy(),
            "labels": torch.cat(train_labels).numpy(),
            "users": np.asarray(train_users),
        }
        train_metrics = _metrics(**train_archive)
        val_metrics, val_archive = evaluate(
            model, val_loader, device, max_batches=8 if smoke_test else None
        )
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "train": train_metrics,
            "validation": val_metrics,
            "sampled_class_counts": sampled_counts.tolist(),
            "optimizer_steps": optimizer_steps,
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        }
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        rank = (
            float(val_metrics["macro_f1"]),
            float(val_metrics["accuracy"]),
            float(val_metrics["worst_user_accuracy"]),
            -float(val_metrics["nll"]),
            -float(epoch),
        )
        selected = best_key is None or rank > best_key
        if selected:
            best_key, best_epoch, patience = rank, epoch, 0
            _save_checkpoint(
                run_dir / "selected_checkpoint.pt",
                model=model,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
            )
            np.savez_compressed(run_dir / "selected_validation_predictions.npz", **val_archive)
        else:
            patience += 1
        print(json.dumps(row), flush=True)
        if (
            not smoke_test
            and epoch >= int(training["minimum_epochs"])
            and patience >= int(training["early_stopping_patience"])
        ):
            break
    selected_metrics = history[best_epoch - 1]["validation"]
    gate_config = config["teacher_gate"]
    gates = {
        "accuracy": float(selected_metrics["accuracy"]) >= float(gate_config["accuracy_at_least"]),
        "macro_f1": float(selected_metrics["macro_f1"]) >= float(gate_config["macro_f1_at_least"]),
        "worst_user_accuracy": float(selected_metrics["worst_user_accuracy"])
        >= float(gate_config["worst_user_accuracy_at_least"]),
    }
    gates["passed"] = all(gates.values())
    summary = {
        "status": "smoke_passed" if smoke_test else "completed",
        "run_id": run_id,
        "epochs_completed": len(history),
        "selected_epoch": best_epoch,
        "selected_metrics": selected_metrics,
        "teacher_gate": gates,
        "p0_report": str(_project_path(str(config["p0_report"]))),
        "checkpoint_provenance": provenance,
        "sampler": sampler_audit,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": trainable_parameters,
        "runtime_seconds": time.perf_counter() - formal_started,
        "logits_export_status": "blocked_until_teacher_gate_review",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single P1 IR/depth VideoMAE V2-B teacher")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--authorize-training", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    summary = run_training(
        args.config.resolve(), token=args.authorize_training, smoke_test=args.smoke_test
    )
    print(json.dumps({"status": summary["status"], "run_id": summary["run_id"]}))


if __name__ == "__main__":
    main()
