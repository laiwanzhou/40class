from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_ir_depth_videomaev2_p2a import _atomic_npz, _atomic_write_text, _metrics
from scripts.probe_ir_depth_videomaev2_teacher import load_probe_config
from src.data.ir_depth_videomaev2_dataset import IRDepthVideoMAEV2Dataset
from src.models.ir_anchored_midfusion_videomae import IRAnchoredMidFusionVideoMAE
from src.models.ir_depth_videomaev2_teacher import (
    build_official_videomaev2_vit_b,
    sha256_file,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_depth_videomaev2_aggressive_user6_user7.yaml"


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class ClassUserBalancedSampler(Sampler[int]):
    def __init__(self, *, labels: np.ndarray, users: np.ndarray, samples: int, seed: int) -> None:
        self.samples = int(samples)
        self.seed = int(seed)
        groups: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (label, user) in enumerate(zip(labels, users, strict=True)):
            groups[int(label)][str(user)].append(index)
        self.groups = {
            label: {user: tuple(indices) for user, indices in user_groups.items()}
            for label, user_groups in groups.items()
        }
        self.classes = tuple(sorted(self.groups))

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        for _ in range(self.samples):
            label = rng.choice(self.classes)
            user = rng.choice(tuple(sorted(self.groups[label])))
            yield rng.choice(self.groups[label][user])

    def __len__(self) -> int:
        return self.samples


def load_aggressive_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("aggressive config must be a mapping")
    split, model, policy = config.get("split", {}), config.get("model", {}), config.get("policy", {})
    if config.get("stage") != "P3-AGGRESSIVE":
        raise ValueError("aggressive stage changed")
    if split.get("validation_user_ids") != ["user6", "user7"]:
        raise ValueError("aggressive validation users changed")
    if split.get("train_class_count") != 40 or split.get("validation_class_count") != 40:
        raise ValueError("aggressive split must cover all 40 classes")
    if model.get("frozen_prefix_blocks") != 8 or model.get("load_previous_finetuned_checkpoint") is not False:
        raise ValueError("aggressive backbone contract changed")
    if policy.get("load_previous_teacher_checkpoint") is not False:
        raise ValueError("aggressive route cannot load prior teacher checkpoint")
    if any(
        policy.get(name) is not False
        for name in (
            "validation_users_enter_gradient",
            "validation_users_enter_normalization",
            "validation_users_enter_sampler",
        )
    ):
        raise ValueError("validation users leaked into aggressive training")
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_datasets(config: dict[str, Any]) -> tuple[IRDepthVideoMAEV2Dataset, IRDepthVideoMAEV2Dataset]:
    data = config["data"]
    p0_config = load_probe_config(_project_path(str(config["p0_input_config"])))
    common = {
        "manifest_path": _project_path(str(data["manifest"])),
        "split_path": _project_path(str(config["split"]["path"])),
        "data_root": Path(str(data["root"])),
        "pose_cache_path": Path(str(data["pose_cache"])),
        "pairing_audit_path": _project_path(str(data["pairing_audit"])),
        "frames": int(data["frames"]),
        "image_size": int(data["image_size"]),
        "interaction_config": dict(p0_config["roi"]),
    }
    return (
        IRDepthVideoMAEV2Dataset(
            **common,
            partition="train",
            training=True,
            temporal_jitter=float(data["temporal_jitter"]),
        ),
        IRDepthVideoMAEV2Dataset(
            **common,
            partition="validation",
            training=False,
            temporal_jitter=0.0,
        ),
    )


def _augment(
    clips: torch.Tensor,
    availability: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    augmentation = config["augmentation"]
    clips = clips.clone()
    available = availability.clone()
    if torch.rand(()) < float(augmentation["depth_dropout"]):
        available[:, 1] = False
    for view in range(4):
        probability = float(
            augmentation["global_person_view_dropout"] if view < 2 else augmentation["hand_view_dropout"]
        )
        if torch.rand(()) < probability:
            available[:, :, view] = False
    if not bool(available[:, 0].any()):
        available[:, 0, 2] = True
    mean = clips.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1, 1)
    std = clips.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1, 1)
    ir = (clips[:, 0] * std + mean).clamp(0.0, 1.0)
    brightness = torch.empty((), device=clips.device).uniform_(*augmentation["ir_brightness"])
    contrast = torch.empty((), device=clips.device).uniform_(*augmentation["ir_contrast"])
    gamma = torch.empty((), device=clips.device).uniform_(*augmentation["ir_gamma"])
    ir = (ir * brightness).clamp(0.0, 1.0)
    center = ir.mean(dim=(3, 4, 5), keepdim=True)
    ir = ((ir - center) * contrast + center).clamp(0.0, 1.0).pow(gamma)
    if torch.rand(()) < float(augmentation["ir_random_erasing_probability"]):
        height, width = ir.shape[-2:]
        erase_h, erase_w = max(1, height // 5), max(1, width // 5)
        top = int(torch.randint(max(1, height - erase_h + 1), (1,)).item())
        left = int(torch.randint(max(1, width - erase_w + 1), (1,)).item())
        ir[..., top : top + erase_h, left : left + erase_w] = 0.5
    clips[:, 0] = (ir - mean) / std
    return clips, available


def _model(config: dict[str, Any]) -> IRAnchoredMidFusionVideoMAE:
    checkpoint_config, model_config = config["checkpoint"], config["model"]
    backbone, _ = build_official_videomaev2_vit_b(
        checkpoint_path=Path(str(checkpoint_config["path"])),
        num_classes=int(model_config["num_classes"]),
        with_cp=True,
    )
    return IRAnchoredMidFusionVideoMAE(
        backbone=backbone,
        frozen_prefix_blocks=int(model_config["frozen_prefix_blocks"]),
        view_top_k=int(model_config["view_top_k"]),
        hand_prior_bias=float(model_config["hand_prior_bias"]),
        depth_hidden_dim=int(model_config["depth_hidden_dim"]),
        depth_gate_initial_bias=float(model_config["depth_gate_initial_bias"]),
    )


def _forward(model: IRAnchoredMidFusionVideoMAE, clips: torch.Tensor, availability: torch.Tensor) -> dict[str, torch.Tensor]:
    return model(ir=clips[:, 0], depth=clips[:, 1], availability=availability)


@torch.no_grad()
def _evaluate(
    model: IRAnchoredMidFusionVideoMAE,
    loader: DataLoader[dict[str, object]],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model.eval()
    logits, labels, users, sample_ids, gates, weights = [], [], [], [], [], []
    for step, batch in enumerate(loader, start=1):
        clips = batch["clips"].to(device, non_blocking=True)
        availability = batch["availability"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            output = _forward(model, clips, availability)
        logits.append(output["logits"].float().cpu())
        gates.append(output["depth_gates"].float().cpu())
        weights.append(output["view_weights"].float().cpu())
        labels.append(batch["label"].long().cpu())
        users.extend(str(value) for value in batch["user_id"])
        sample_ids.extend(str(value) for value in batch["sample_id"])
        if max_batches is not None and step >= max_batches:
            break
    logits_np = torch.cat(logits).numpy()
    labels_np = torch.cat(labels).numpy()
    users_np = np.asarray(users)
    metrics = _metrics(labels_np, logits_np, users_np)
    gate_np, weight_np = torch.cat(gates).numpy(), torch.cat(weights).numpy()
    metrics.update(
        {
            "mean_depth_gate": float(gate_np.mean()),
            "mean_view_entropy": float(
                (-(weight_np * np.log(np.clip(weight_np, 1e-12, 1.0))).sum(axis=2)).mean()
            ),
            "maximum_nonzero_views": int((weight_np > 0).sum(axis=2).max()),
        }
    )
    return metrics, {
        "sample_ids": np.asarray(sample_ids),
        "user_ids": users_np,
        "labels": labels_np,
        "logits": logits_np,
        "predictions": logits_np.argmax(axis=1),
        "depth_gates": gate_np,
        "view_weights": weight_np,
    }


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(config_path: Path, *, smoke_test: bool = False) -> dict[str, object]:
    config = load_aggressive_config(config_path)
    checkpoint_path = Path(str(config["checkpoint"]["path"]))
    if checkpoint_path.stat().st_size != int(config["checkpoint"]["bytes"]) or sha256_file(checkpoint_path) != config["checkpoint"]["sha256"]:
        raise RuntimeError("official checkpoint provenance mismatch")
    seed = int(config["training"]["seed"])
    _set_seed(seed)
    device = torch.device("cuda")
    train_dataset, val_dataset = _build_datasets(config)
    labels = np.asarray(train_dataset.class_ids(), dtype=np.int64)
    users = np.asarray([str(sample["user_id"]) for sample in train_dataset.samples])
    sampler = ClassUserBalancedSampler(labels=labels, users=users, samples=len(labels), seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model = _model(config).to(device)
    head_before = model.backbone.head.weight.detach().clone()
    finite_gradients = True
    new_ids = {
        id(parameter)
        for module in (model.depth_adapter, model.depth_gate)
        for parameter in module.parameters()
    } | {id(model.class_queries), id(model.class_view_bias)}
    new_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in new_ids]
    backbone_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in new_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(config["training"]["backbone_learning_rate"])},
            {"params": new_parameters, "lr": float(config["training"]["new_module_learning_rate"])},
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
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
    _atomic_write_text(run_dir / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=False))
    epochs = 1 if smoke_test else int(config["training"]["epochs"])
    accumulation_target = int(config["training"]["gradient_accumulation"])
    best_key: tuple[float, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, patience = 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits, train_labels, train_users = [], [], []
        accumulated = 0
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for step, batch in enumerate(train_loader, start=1):
            clips = batch["clips"].to(device, non_blocking=True)
            availability = batch["availability"].to(device, non_blocking=True)
            clips, availability = _augment(clips, availability, config)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                output = _forward(model, clips, availability)
                loss = nn.functional.cross_entropy(
                    output["logits"],
                    batch["label"].to(device),
                    label_smoothing=float(config["training"]["label_smoothing"]),
                )
            loss.backward()
            finite_gradients = finite_gradients and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            if not finite_gradients:
                raise FloatingPointError("non-finite aggressive-route gradient")
            accumulated += 1
            train_logits.append(output["logits"].detach().float().cpu())
            train_labels.append(batch["label"].long().cpu())
            train_users.extend(str(value) for value in batch["user_id"])
            is_last = step == len(train_loader) or (smoke_test and step >= 2)
            if accumulated == accumulation_target or is_last:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            if smoke_test and step >= 2:
                break
        train_metrics = _metrics(
            torch.cat(train_labels).numpy(),
            torch.cat(train_logits).numpy(),
            np.asarray(train_users),
        )
        val_metrics, val_archive = _evaluate(
            model, val_loader, device, max_batches=8 if smoke_test else None
        )
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "train": train_metrics,
            "validation": val_metrics,
            "generalization_gap": float(train_metrics["accuracy"] - val_metrics["accuracy"]),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        }
        history.append(row)
        _atomic_write_text(run_dir / "history.json", json.dumps(history, indent=2) + "\n")
        key = (
            float(val_metrics["accuracy"]),
            float(val_metrics["macro_f1"]),
            float(val_metrics["worst_user_accuracy"]),
            -float(val_metrics["nll"]),
            -float(epoch),
        )
        if best_key is None or key > best_key:
            best_key, best_epoch, patience = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            _atomic_torch_save(run_dir / "selected_checkpoint.pt", {"epoch": epoch, "model_state_dict": best_state, "metrics": val_metrics, "config": config})
            _atomic_npz(run_dir / "selected_validation_predictions.npz", **val_archive)
        else:
            patience += 1
        print(json.dumps({"epoch": epoch, "train_acc": train_metrics["accuracy"], "val_acc": val_metrics["accuracy"], "gap": row["generalization_gap"], "val_macro_f1": val_metrics["macro_f1"], "seconds": row["seconds"]}), flush=True)
        if not smoke_test and epoch >= int(config["training"]["minimum_epochs"]) and patience >= int(config["training"]["early_stopping_patience"]):
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    deterministic_train = IRDepthVideoMAEV2Dataset(
        manifest_path=_project_path(str(config["data"]["manifest"])),
        split_path=_project_path(str(config["split"]["path"])),
        data_root=Path(str(config["data"]["root"])),
        pose_cache_path=Path(str(config["data"]["pose_cache"])),
        pairing_audit_path=_project_path(str(config["data"]["pairing_audit"])),
        partition="train",
        training=False,
        frames=16,
        image_size=224,
        temporal_jitter=0.0,
    )
    train_eval_loader = DataLoader(deterministic_train, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    selected_train_metrics, _ = _evaluate(model, train_eval_loader, device, max_batches=8 if smoke_test else None)
    selected_val_metrics, _ = _evaluate(model, val_loader, device, max_batches=8 if smoke_test else None)
    gap = float(selected_train_metrics["accuracy"] - selected_val_metrics["accuracy"])
    gates = config["decision_gates"]
    decision = (
        "oof_worthy" if selected_val_metrics["accuracy"] >= gates["oof_worthy_accuracy"]
        else "promising" if selected_val_metrics["accuracy"] >= gates["promising_accuracy"]
        else "stop" if selected_val_metrics["accuracy"] < gates["stop_below_accuracy"]
        else "intermediate"
    )
    summary = {
        "status": "smoke_passed" if smoke_test else "completed",
        "run_id": run_id,
        "epochs_completed": len(history),
        "selected_epoch": best_epoch,
        "selected_train_metrics": selected_train_metrics,
        "selected_validation_metrics": selected_val_metrics,
        "generalization_gap": gap,
        "decision": decision,
        "target_accuracy_reached": selected_val_metrics["accuracy"] >= gates["target_accuracy"],
        "checkpoint_initialization": "official VideoMAE V2 ViT-B K710 distilled checkpoint",
        "previous_teacher_loaded": False,
        "validation_users_entered_training": False,
        "finite_gradients": finite_gradients,
        "optimizer_state_changed": not torch.equal(
            head_before, model.backbone.head.weight.detach()
        ),
        "p2b_started": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "output_dir": str(run_dir),
    }
    _atomic_write_text(run_dir / "run_summary.json", json.dumps(summary, indent=2) + "\n")
    if not smoke_test:
        report_json = _project_path(str(config["report_json"]))
        report_markdown = _project_path(str(config["report_markdown"]))
        _atomic_write_text(report_json, json.dumps(summary, indent=2) + "\n")
        _atomic_write_text(report_markdown, "\n".join([
            "# Aggressive IR + Depth VideoMAE development result", "",
            f"- Selected epoch: `{best_epoch}`",
            f"- Train Accuracy: `{selected_train_metrics['accuracy']:.6f}`",
            f"- Validation Accuracy: `{selected_val_metrics['accuracy']:.6f}`",
            f"- Generalization gap: `{gap:.6f}`",
            f"- Validation Macro-F1: `{selected_val_metrics['macro_f1']:.6f}`",
            f"- Worst-user Accuracy: `{selected_val_metrics['worst_user_accuracy']:.6f}`",
            f"- Decision: `{decision}`",
            "- Previous teacher loaded: `False`", "- P2-B started: `False`", ""
        ]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aggressive user6/user7 VideoMAE development route")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    summary = run(args.config.resolve(), smoke_test=args.smoke_test)
    print(json.dumps({"status": summary["status"], "train_acc": summary["selected_train_metrics"]["accuracy"], "val_acc": summary["selected_validation_metrics"]["accuracy"], "gap": summary["generalization_gap"], "decision": summary["decision"]}))


if __name__ == "__main__":
    main()
