from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel, fp32_size_mb, parameter_count
from train import create_loader, run_epoch, seed_everything, set_encoder_trainable, write_history


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按预先固定的 epoch 在全部标注 subjects 上重训单模态专家；不使用伪验证集选 checkpoint"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def build_all_train_manifest(source: Path, destination: Path) -> tuple[int, list[str]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        row["split"] = "train"
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), sorted({row["user_id"] for row in rows})


def optimizer_for_model(
    model: AlignedMultimodalModel, config: dict
) -> torch.optim.Optimizer:
    base_learning_rate = float(config["learning_rate"])
    parameter_groups = []
    if "backbone_learning_rate_scale" in config:
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
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": base_learning_rate * float(config["backbone_learning_rate_scale"]),
                "name": "imagenet_backbone",
            }
        )
        parameter_groups.append(
            {
                "params": [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids],
                "lr": base_learning_rate,
                "name": "new_layers",
            }
        )
    else:
        encoder_parameters = list(model.visual.parameters()) if model.visual is not None else []
        if model.skeleton is not None:
            encoder_parameters.extend(model.skeleton.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        fusion_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
        if encoder_parameters:
            parameter_groups.append(
                {
                    "params": encoder_parameters,
                    "lr": base_learning_rate * float(config.get("encoder_learning_rate_scale", 1.0)),
                    "name": "encoders",
                }
            )
        if fusion_parameters:
            parameter_groups.append(
                {"params": fusion_parameters, "lr": base_learning_rate, "name": "fusion"}
            )
    return torch.optim.AdamW(parameter_groups, weight_decay=float(config["weight_decay"]))


def save_refit_checkpoint(
    path: Path,
    model: AlignedMultimodalModel,
    config: dict,
    epoch: int,
    train_metrics: dict,
    sample_count: int,
    subjects: list[str],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "final_refit": True,
            "refit_training_samples": sample_count,
            "refit_training_subjects": subjects,
            "refit_last_epoch_train_accuracy": float(train_metrics["accuracy"]),
            "refit_last_epoch_train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "refit_last_epoch_train_macro_f1": float(train_metrics["macro_f1"]),
            "selection_protocol": "fixed epoch chosen before final refit; no validation checkpoint selection",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    config["epochs"] = int(args.epochs)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
    config["final_refit"] = True
    config["final_refit_epoch_protocol"] = "median/best-epoch evidence frozen before all-subject refit"
    refit_manifest = output / "refit_manifest.csv"
    sample_count, subjects = build_all_train_manifest(args.manifest.resolve(), refit_manifest)
    config["final_refit_training_samples"] = sample_count
    config["final_refit_training_subjects"] = subjects
    (output / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    seed_everything(int(config["seed"]))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["use_amp"] and device.type == "cuda")
    modalities = list(config["modalities"])
    dataset = AlignedMultimodalDataset(
        manifest_path=refit_manifest,
        split="train",
        modalities=modalities,
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=True,
        cache_dir=config.get("cache_dir"),
        skeleton_strategy=config.get("skeleton_strategy", "first"),
        depth_representation=config.get("depth_representation", "jet_rgb"),
        visual_normalization=config.get("visual_normalization", "legacy"),
        skeleton_representation=config.get("skeleton_representation", "frame_joint"),
        skeleton_raw_cache_dir=config.get("skeleton_raw_cache_dir"),
    )
    loader = create_loader(dataset, config, train=True)
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
    ).to(device)
    optimizer = optimizer_for_model(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    max_grad_norm = config.get("max_grad_norm")
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
    accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))

    print(f"设备：{device}；最终全量重训：{'+'.join(modalities)}")
    print(f"标注 trial：{sample_count}；subjects：{len(subjects)}；固定 epochs：{config['epochs']}")
    print(f"参数：{parameter_count(model):,}；FP32约 {fp32_size_mb(model):.2f} MB", flush=True)
    history = []
    started = time.time()
    final_metrics = None
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.time()
        set_encoder_trainable(model, epoch > int(config.get("freeze_encoder_epochs", 0)))
        final_metrics = run_epoch(
            model,
            loader,
            criterion,
            device,
            modalities,
            optimizer,
            scaler,
            use_amp,
            max_grad_norm,
            aux_weight=0.0,
            depth_dropout=float(config.get("depth_modality_dropout", 0.0)),
            skeleton_dropout=float(config.get("skeleton_modality_dropout", 0.0)),
            gradient_accumulation_steps=accumulation_steps,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[-1]["lr"],
            "train_loss": final_metrics["loss"],
            "train_accuracy": final_metrics["accuracy"],
            "train_balanced_accuracy": final_metrics["balanced_accuracy"],
            "train_macro_f1": final_metrics["macro_f1"],
            "seconds": round(time.time() - epoch_started, 2),
        }
        history.append(row)
        write_history(output / "history.csv", history)
        save_refit_checkpoint(output / "last.pt", model, config, epoch, final_metrics, sample_count, subjects)
        print(
            f"Epoch {epoch:02d} | train acc {row['train_accuracy']:.4f}, "
            f"bal {row['train_balanced_accuracy']:.4f}, F1 {row['train_macro_f1']:.4f}, "
            f"loss {row['train_loss']:.4f} | {row['seconds']:.1f}s",
            flush=True,
        )
    assert final_metrics is not None
    save_refit_checkpoint(
        output / "final.pt", model, config, int(config["epochs"]), final_metrics, sample_count, subjects
    )
    summary = {
        "modalities": modalities,
        "training_samples": sample_count,
        "training_subjects": subjects,
        "fixed_epochs": int(config["epochs"]),
        "selection_protocol": "fixed epoch; no validation selection during final refit",
        "last_epoch_train_metrics": {
            key: float(final_metrics[key])
            for key in ["loss", "accuracy", "balanced_accuracy", "macro_f1"]
        },
        "seconds": round(time.time() - started, 2),
        "checkpoint": str((output / "final.pt").resolve()),
        "evaluation_note": "All labels were used for fitting, so this run has no unbiased local validation metric.",
    }
    (output / "refit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
