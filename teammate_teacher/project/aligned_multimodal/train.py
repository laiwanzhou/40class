from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel, fp32_size_mb, parameter_count


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_DIR / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练同步多模态动作分类模型")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loader(dataset, config: dict, train: bool) -> DataLoader:
    sampler = None
    shuffle = train
    if train and config["balanced_sampling"]:
        counts = Counter(sample.class_id for sample in dataset.samples)
        weights = torch.tensor([1.0 / counts[sample.class_id] for sample in dataset.samples], dtype=torch.double)
        generator = torch.Generator().manual_seed(int(config["seed"]))
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        shuffle = False
    workers = int(config["num_workers"] if train else config.get("val_num_workers", 0))
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


def move_inputs(batch: dict, modalities: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    inputs = {
        modality: batch[modality].to(device, non_blocking=True)
        for modality in modalities
    }
    if "ir_motion" in batch:
        inputs["ir_motion"] = batch["ir_motion"].to(device, non_blocking=True)
    if "ir_local" in batch:
        inputs["ir_local"] = batch["ir_local"].to(device, non_blocking=True)
        inputs["ir_local_quality"] = batch["ir_local_quality"].to(
            device, non_blocking=True
        )
    return inputs


def set_encoder_trainable(model: AlignedMultimodalModel, trainable: bool) -> None:
    for module in (model.visual, model.skeleton):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)


def load_pretrained_encoder(model: AlignedMultimodalModel, module_name: str, checkpoint_path: str) -> None:
    module = getattr(model, module_name)
    if module is None:
        raise ValueError(f"当前模型没有 {module_name} 编码器")
    path = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    prefix = f"{module_name}."
    state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise KeyError(f"检查点中没有 {prefix} 参数：{path}")
    module.load_state_dict(state, strict=True)
    print(f"载入 {module_name} 独立预训练权重：{path}")


def load_pretrained_multimodal(model: AlignedMultimodalModel, checkpoint_path: str) -> None:
    path = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing_prefixes = (
        "depth_token_project.",
        "cross_attention.",
        "cross_attention_norm.",
        "cross_attention_scale",
        "visual.layer3_spatial.",
        "visual.layer3_residual_project.",
        "visual.layer3_residual_scale",
        "visual.ir_motion_stem.",
        "visual.ir_motion_norm.",
        "visual.ir_motion_residual_scale",
        "visual.ir_local_stem.",
        "visual.ir_local_norm.",
        "visual.ir_local_residual_scale",
    )
    unexpected_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"多模态预训练权重不兼容：missing={unexpected_missing}，"
            f"unexpected={incompatible.unexpected_keys}"
        )
    if (
        model.visual is not None
        and model.visual.ir_motion_stem is not None
        and "visual.ir_motion_stem.weight" in incompatible.missing_keys
    ):
        assert model.visual.ir_stem is not None
        with torch.no_grad():
            model.visual.ir_motion_stem.weight.copy_(model.visual.ir_stem.weight)
    if (
        model.visual is not None
        and model.visual.ir_local_stem is not None
        and "visual.ir_local_stem.weight" in incompatible.missing_keys
    ):
        assert model.visual.ir_stem is not None
        with torch.no_grad():
            model.visual.ir_local_stem.weight.copy_(model.visual.ir_stem.weight)
    print(
        f"载入多模态预训练权重：{path}；新初始化参数：{incompatible.missing_keys}"
    )


def load_residual_modality_initialization(
    model: AlignedMultimodalModel, checkpoint_path: str
) -> None:
    """Load all shape-compatible weights while leaving a new residual stem at zero."""
    path = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    allowed_missing = {
        "visual.depth_stem.weight",
        "visual.depth_residual_scale",
    }
    unexpected_missing = [
        key for key in incompatible.missing_keys if key not in allowed_missing
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "残差模态初始化不兼容："
            f"missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(
        f"载入零残差模态初始化：{path}；保留新参数："
        f"{sorted(incompatible.missing_keys)}"
    )


def scheduled_aux_weight(config: dict, epoch: int) -> float:
    start = float(config.get("auxiliary_loss_start", 0.0))
    end = float(config.get("auxiliary_loss_end", start))
    total_epochs = max(1, int(config["epochs"]) - 1)
    progress = min(1.0, max(0.0, (epoch - 1) / total_epochs))
    return start + (end - start) * progress


def add_modality_presence(
    inputs: dict[str, torch.Tensor],
    training: bool,
    depth_dropout: float,
    skeleton_dropout: float,
) -> None:
    reference = inputs["depth"]
    batch_size = len(reference)
    device = reference.device
    depth_present = torch.ones(batch_size, 1, device=device)
    skeleton_present = torch.ones(batch_size, 1, device=device)
    if training:
        depth_missing = torch.rand(batch_size, device=device) < depth_dropout
        skeleton_missing = torch.rand(batch_size, device=device) < skeleton_dropout
        both_missing = depth_missing & skeleton_missing
        if bool(both_missing.any()):
            keep_depth = torch.rand(batch_size, device=device) < 0.5
            depth_missing[both_missing & keep_depth] = False
            skeleton_missing[both_missing & ~keep_depth] = False
        depth_present[depth_missing] = 0.0
        skeleton_present[skeleton_missing] = 0.0
    inputs["depth_present"] = depth_present
    inputs["skeleton_present"] = skeleton_present


def run_epoch(
    model,
    loader,
    criterion,
    device,
    modalities,
    optimizer,
    scaler,
    use_amp,
    max_grad_norm,
    aux_weight: float = 0.0,
    depth_dropout: float = 0.0,
    skeleton_dropout: float = 0.0,
    gradient_accumulation_steps: int = 1,
):
    training = optimizer is not None
    model.train(training)
    if training:
        for module in (model.visual, model.skeleton):
            if module is not None and not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
    losses: list[float] = []
    main_losses: list[float] = []
    depth_losses: list[float] = []
    skeleton_losses: list[float] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []
    depth_predictions_all: list[int] = []
    skeleton_predictions_all: list[int] = []
    sample_ids_all: list[str] = []
    accumulation_steps = max(1, int(gradient_accumulation_steps)) if training else 1
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        labels = batch["label"].to(device, non_blocking=True)
        inputs = move_inputs(batch, modalities, device)
        if model.use_modality_masks:
            add_modality_presence(inputs, training, depth_dropout, skeleton_dropout)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs, return_aux=model.use_aux_heads)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                    main_loss = criterion(logits, labels)
                    depth_loss = criterion(outputs["depth_logits"], labels)
                    skeleton_loss = criterion(outputs["skeleton_logits"], labels)
                    loss = main_loss + aux_weight * (depth_loss + skeleton_loss)
                else:
                    logits = outputs
                    main_loss = criterion(logits, labels)
                    depth_loss = None
                    skeleton_loss = None
                    loss = main_loss
            if training:
                scaler.scale(loss / accumulation_steps).backward()
                should_step = (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == len(loader)
                if should_step:
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        main_losses.append(float(main_loss.detach().cpu()))
        if depth_loss is not None and skeleton_loss is not None:
            depth_losses.append(float(depth_loss.detach().cpu()))
            skeleton_losses.append(float(skeleton_loss.detach().cpu()))
            depth_predictions_all.extend(outputs["depth_logits"].argmax(1).detach().cpu().tolist())
            skeleton_predictions_all.extend(outputs["skeleton_logits"].argmax(1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(1).detach().cpu().tolist())
        sample_ids_all.extend(batch["sample_id"])
    result = {
        "loss": float(np.mean(losses)),
        "main_loss": float(np.mean(main_losses)),
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_all, predictions_all)),
        "macro_f1": float(f1_score(labels_all, predictions_all, average="macro", zero_division=0)),
        "labels": labels_all,
        "predictions": predictions_all,
        "sample_ids": sample_ids_all,
    }
    if depth_predictions_all and skeleton_predictions_all:
        result.update(
            {
                "depth_loss": float(np.mean(depth_losses)),
                "skeleton_loss": float(np.mean(skeleton_losses)),
                "depth_accuracy": float(accuracy_score(labels_all, depth_predictions_all)),
                "depth_balanced_accuracy": float(
                    balanced_accuracy_score(labels_all, depth_predictions_all)
                ),
                "skeleton_accuracy": float(accuracy_score(labels_all, skeleton_predictions_all)),
                "skeleton_balanced_accuracy": float(
                    balanced_accuracy_score(labels_all, skeleton_predictions_all)
                ),
            }
        )
    return result


def save_checkpoint(path: Path, model, config, epoch, metrics) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_accuracy": float(metrics["accuracy"]),
            "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
            "val_macro_f1": float(metrics["macro_f1"]),
        },
        path,
    )


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_predictions(path: Path, metrics: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prediction"])
        writer.writerows(zip(metrics["sample_ids"], metrics["labels"], metrics["predictions"]))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.resume is not None:
        config["resume_from"] = str(args.resume.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    seed_everything(int(config["seed"]))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["use_amp"] and device.type == "cuda")
    modalities = list(config["modalities"])

    common_dataset_args = {
        "manifest_path": args.manifest,
        "modalities": modalities,
        "num_frames": int(config["num_frames"]),
        "image_height": int(config["image_height"]),
        "image_width": int(config["image_width"]),
        "cache_dir": config.get("cache_dir"),
        "skeleton_strategy": config.get("skeleton_strategy", "first"),
        "depth_representation": config.get("depth_representation", "jet_rgb"),
        "visual_normalization": config.get("visual_normalization", "legacy"),
        "skeleton_representation": config.get("skeleton_representation", "frame_joint"),
        "skeleton_raw_cache_dir": config.get("skeleton_raw_cache_dir"),
        "ir_gain_jitter": float(config.get("ir_gain_jitter", 0.0)),
        "ir_offset_jitter": float(config.get("ir_offset_jitter", 0.0)),
        "ir_random_resized_crop_min_scale": float(
            config.get("ir_random_resized_crop_min_scale", 1.0)
        ),
        "temporal_sampling": str(config.get("temporal_sampling", "uniform")),
        "ir_motion_mode": str(config.get("ir_motion_mode", "none")),
        "ir_roi_csv": config.get("ir_roi_csv"),
        "ir_roi_context": float(config.get("ir_roi_context", 0.3)),
    }
    train_dataset = AlignedMultimodalDataset(split="train", augment=True, **common_dataset_args)
    val_dataset = AlignedMultimodalDataset(split="val", augment=False, **common_dataset_args)
    train_loader = create_loader(train_dataset, config, train=True)
    val_loader = create_loader(val_dataset, config, train=False)

    model = AlignedMultimodalModel(
        modalities,
        dropout=float(config["dropout"]),
        use_aux_heads=bool(config.get("use_aux_heads", False)),
        use_modality_masks=bool(config.get("use_modality_masks", False)),
        use_cross_attention=bool(config.get("use_cross_attention", False)),
        cross_attention_grid=tuple(config.get("cross_attention_grid", [2, 3])),
        cross_attention_heads=int(config.get("cross_attention_heads", 4)),
        depth_input_channels=int(config.get("depth_input_channels", 3)),
        use_layer3_spatial=bool(config.get("use_layer3_spatial", False)),
        imagenet_pretrained=bool(config.get("imagenet_pretrained", False)),
        ir_stem_initialization=str(config.get("ir_stem_initialization", "mean")),
        visual_stem_fusion=str(config.get("visual_stem_fusion", "concat")),
        skeleton_input_dim=int(config.get("skeleton_input_dim", 4)),
        use_ir_motion=bool(config.get("use_ir_motion", False)),
        use_ir_local=bool(config.get("use_ir_local", False)),
    ).to(device)
    start_epoch = 1
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        print(f"从 epoch {resume_checkpoint['epoch']} 继续：{args.resume.resolve()}")
    else:
        if config.get("pretrained_multimodal_checkpoint"):
            load_pretrained_multimodal(model, str(config["pretrained_multimodal_checkpoint"]))
        if config.get("pretrained_residual_modality_checkpoint"):
            load_residual_modality_initialization(
                model,
                str(config["pretrained_residual_modality_checkpoint"]),
            )
        if config.get("pretrained_depth_checkpoint"):
            load_pretrained_encoder(model, "visual", str(config["pretrained_depth_checkpoint"]))
        if config.get("pretrained_skeleton_checkpoint"):
            load_pretrained_encoder(model, "skeleton", str(config["pretrained_skeleton_checkpoint"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    base_learning_rate = float(config["learning_rate"])
    parameter_groups = []
    if (
        bool(config.get("use_ir_motion", False))
        or bool(config.get("use_ir_local", False))
    ) and config.get(
        "pretrained_multimodal_checkpoint"
    ):
        residual_prefix = (
            "visual.ir_motion_"
            if bool(config.get("use_ir_motion", False))
            else "visual.ir_local_"
        )
        residual_names = {
            name
            for name, _ in model.named_parameters()
            if name.startswith(residual_prefix)
        }
        residual_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name in residual_names
        ]
        base_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name not in residual_names
        ]
        parameter_groups.append(
            {
                "params": base_parameters,
                "lr": base_learning_rate
                * float(config.get("pretrained_base_learning_rate_scale", 0.1)),
                "name": "pretrained_base",
            }
        )
        parameter_groups.append(
            {
                "params": residual_parameters,
                "lr": base_learning_rate,
                "name": (
                    "new_ir_motion"
                    if bool(config.get("use_ir_motion", False))
                    else "new_ir_local"
                ),
            }
        )
    elif config.get("pretrained_residual_modality_checkpoint"):
        residual_names = {
            "visual.depth_stem.weight",
            "visual.depth_residual_scale",
        }
        residual_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name in residual_names
        ]
        base_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name not in residual_names
        ]
        available_residual_names = {
            name for name, _ in model.named_parameters()
        } & residual_names
        if available_residual_names != residual_names:
            raise RuntimeError(
                "残差模态配置缺少预期参数："
                f"{sorted(residual_names)}"
            )
        parameter_groups.append(
            {
                "params": base_parameters,
                "lr": base_learning_rate
                * float(config.get("pretrained_base_learning_rate_scale", 0.1)),
                "name": "pretrained_base",
            }
        )
        parameter_groups.append(
            {
                "params": residual_parameters,
                "lr": base_learning_rate,
                "name": "new_depth_residual",
            }
        )
    elif "backbone_learning_rate_scale" in config:
        if model.visual is None:
            raise ValueError("backbone_learning_rate_scale 需要视觉模态")
        backbone_modules = [
            model.visual.depth_stem,
            model.visual.ir_stem,
            model.visual.bn1,
            model.visual.layer1,
            model.visual.layer2,
            model.visual.layer3,
            model.visual.layer4,
        ]
        backbone_parameters = [
            parameter
            for module in backbone_modules
            if module is not None
            for parameter in module.parameters()
        ]
        backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
        new_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in backbone_parameter_ids
        ]
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": base_learning_rate * float(config["backbone_learning_rate_scale"]),
                "name": "imagenet_backbone",
            }
        )
        parameter_groups.append(
            {"params": new_parameters, "lr": base_learning_rate, "name": "new_layers"}
        )
    else:
        encoder_learning_rate_scale = float(config.get("encoder_learning_rate_scale", 1.0))
        encoder_parameters = list(model.visual.parameters()) if model.visual is not None else []
        if model.skeleton is not None:
            encoder_parameters.extend(model.skeleton.parameters())
        encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
        fusion_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in encoder_parameter_ids
        ]
        if encoder_parameters:
            parameter_groups.append(
                {
                    "params": encoder_parameters,
                    "lr": base_learning_rate * encoder_learning_rate_scale,
                    "name": "encoders",
                }
            )
        if fusion_parameters:
            parameter_groups.append(
                {"params": fusion_parameters, "lr": base_learning_rate, "name": "fusion"}
            )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(config["weight_decay"]))
    remaining_epochs = int(config["epochs"]) - start_epoch + 1
    if remaining_epochs <= 0:
        raise ValueError(f"resume epoch 已达到目标 epochs={config['epochs']}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_grad_norm = config.get("max_grad_norm")
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

    params = parameter_count(model)
    size_mb = fp32_size_mb(model)
    print(f"设备：{device}，模态：{'+'.join(modalities)}")
    print(f"训练/验证 trial：{len(train_dataset)}/{len(val_dataset)}")
    print(f"参数：{params:,}，FP32约 {size_mb:.2f} MB")
    print(f"输入：{config['num_frames']}帧，{config['image_width']}×{config['image_height']}，batch={config['batch_size']}")
    if bool(config.get("imagenet_pretrained", False)):
        print("视觉初始化：ImageNet-1K ResNet18（输入预处理由配置指定）")
    print(
        f"视觉归一化：{config.get('visual_normalization', 'legacy')}；"
        f"优化参数组：{[(group['name'], group['lr']) for group in optimizer.param_groups]}"
    )
    accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
    if accumulation_steps > 1:
        print(
            f"梯度累积：{accumulation_steps} 个 micro-batch，"
            f"有效 batch≈{int(config['batch_size']) * accumulation_steps}"
        )

    best_accuracy = -1.0
    best_f1 = -1.0
    best_accuracy_epoch = 0
    best_f1_epoch = 0
    best_accuracy_metrics = None
    best_f1_metrics = None
    history: list[dict] = []
    history_path = output / "history.csv"
    if args.resume is not None and history_path.is_file():
        with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))
        # 丢弃恢复点之后可能由被中断进程迟写入的不完整/重复 epoch。
        history = [row for row in history if int(row["epoch"]) < start_epoch]

        accuracy_path = output / "best_accuracy.pt"
        f1_path = output / "best_macro_f1.pt"
        if accuracy_path.is_file():
            checkpoint = torch.load(accuracy_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            best_accuracy_metrics = run_epoch(
                model, val_loader, criterion, device, modalities, None, scaler, use_amp, None
            )
            best_accuracy = float(best_accuracy_metrics["accuracy"])
            best_accuracy_epoch = int(checkpoint["epoch"])
        if f1_path.is_file():
            checkpoint = torch.load(f1_path, map_location="cpu", weights_only=False)
            best_f1_epoch = int(checkpoint["epoch"])
            if best_accuracy_metrics is not None and best_f1_epoch == best_accuracy_epoch:
                best_f1_metrics = best_accuracy_metrics
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
                best_f1_metrics = run_epoch(
                    model, val_loader, criterion, device, modalities, None, scaler, use_amp, None
                )
            best_f1 = float(best_f1_metrics["macro_f1"])
        model.load_state_dict(resume_checkpoint["model_state_dict"])
    started = time.time()

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        epoch_started = time.time()
        freeze_encoder_epochs = int(config.get("freeze_encoder_epochs", 0))
        set_encoder_trainable(model, epoch > freeze_encoder_epochs)
        aux_weight = scheduled_aux_weight(config, epoch) if model.use_aux_heads else 0.0
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            modalities,
            optimizer,
            scaler,
            use_amp,
            max_grad_norm,
            aux_weight=aux_weight,
            depth_dropout=float(config.get("depth_modality_dropout", 0.0)),
            skeleton_dropout=float(config.get("skeleton_modality_dropout", 0.0)),
            gradient_accumulation_steps=accumulation_steps,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, modalities, None, scaler, use_amp, None)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[-1]["lr"],
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "auxiliary_loss_weight": aux_weight,
            "train_loss": train_metrics["loss"],
            "train_main_loss": train_metrics["main_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_main_loss": val_metrics["main_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "seconds": round(time.time() - epoch_started, 2),
        }
        if model.use_aux_heads:
            row.update(
                {
                    "train_depth_accuracy": train_metrics["depth_accuracy"],
                    "train_skeleton_accuracy": train_metrics["skeleton_accuracy"],
                    "val_depth_accuracy": val_metrics["depth_accuracy"],
                    "val_depth_balanced_accuracy": val_metrics["depth_balanced_accuracy"],
                    "val_skeleton_accuracy": val_metrics["skeleton_accuracy"],
                    "val_skeleton_balanced_accuracy": val_metrics["skeleton_balanced_accuracy"],
                }
            )
        history.append(row)
        write_history(history_path, history)
        print(
            f"Epoch {epoch:02d} | train acc {row['train_accuracy']:.4f}, F1 {row['train_macro_f1']:.4f} | "
            f"val acc {row['val_accuracy']:.4f}, bal {row['val_balanced_accuracy']:.4f}, "
            f"F1 {row['val_macro_f1']:.4f}, loss {row['val_loss']:.4f} | "
            f"{row['seconds']:.1f}s"
        )
        if model.use_aux_heads:
            print(
                f"  aux weight {aux_weight:.3f} | val Depth {row['val_depth_accuracy']:.4f}, "
                f"Skeleton {row['val_skeleton_accuracy']:.4f}"
            )
        save_checkpoint(output / "last.pt", model, config, epoch, val_metrics)

        if float(val_metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            best_accuracy_epoch = epoch
            best_accuracy_metrics = val_metrics
            save_checkpoint(output / "best_accuracy.pt", model, config, epoch, val_metrics)
        if float(val_metrics["macro_f1"]) > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            best_f1_epoch = epoch
            best_f1_metrics = val_metrics
            save_checkpoint(output / "best_macro_f1.pt", model, config, epoch, val_metrics)

    assert best_accuracy_metrics is not None and best_f1_metrics is not None
    np.savetxt(
        output / "confusion_matrix_best_accuracy.csv",
        confusion_matrix(best_accuracy_metrics["labels"], best_accuracy_metrics["predictions"], labels=list(range(40))),
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        output / "confusion_matrix_best_macro_f1.csv",
        confusion_matrix(best_f1_metrics["labels"], best_f1_metrics["predictions"], labels=list(range(40))),
        delimiter=",",
        fmt="%d",
    )
    save_predictions(output / "val_predictions_best_accuracy.csv", best_accuracy_metrics)
    summary = {
        "device": str(device),
        "modalities": modalities,
        "train_trials": len(train_dataset),
        "val_trials": len(val_dataset),
        "parameters": params,
        "fp32_parameter_size_mb": size_mb,
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_val_accuracy": best_accuracy,
        "accuracy_checkpoint_balanced_accuracy": float(best_accuracy_metrics["balanced_accuracy"]),
        "accuracy_checkpoint_macro_f1": float(best_accuracy_metrics["macro_f1"]),
        "best_macro_f1_epoch": best_f1_epoch,
        "best_val_macro_f1": best_f1,
        "macro_f1_checkpoint_accuracy": float(best_f1_metrics["accuracy"]),
        "macro_f1_checkpoint_balanced_accuracy": float(best_f1_metrics["balanced_accuracy"]),
        "total_seconds": round(
            sum(float(row["seconds"]) for row in history), 2
        ),
    }
    if model.use_aux_heads:
        summary.update(
            {
                "accuracy_checkpoint_depth_accuracy": float(
                    best_accuracy_metrics["depth_accuracy"]
                ),
                "accuracy_checkpoint_depth_balanced_accuracy": float(
                    best_accuracy_metrics["depth_balanced_accuracy"]
                ),
                "accuracy_checkpoint_skeleton_accuracy": float(
                    best_accuracy_metrics["skeleton_accuracy"]
                ),
                "accuracy_checkpoint_skeleton_balanced_accuracy": float(
                    best_accuracy_metrics["skeleton_balanced_accuracy"]
                ),
            }
        )
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"完成：best accuracy={best_accuracy:.4f} (epoch {best_accuracy_epoch})；"
        f"best macro F1={best_f1:.4f} (epoch {best_f1_epoch})"
    )


if __name__ == "__main__":
    main()
