from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from thermal_model import fp32_size_mb, parameter_count
from thermal_oof_data import ThermalOOFDataset
from thermal_tsm_model import ThermalResNetTSM
from train_imagenet_oof import create_loader, run_epoch, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-epoch ImageNet Thermal refit on all usable training subjects."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    config["epochs"] = int(args.epochs)
    config["manifest"] = str(args.manifest.resolve())
    config["refit_protocol"] = (
        "All 18 training subjects; fixed epoch count is the median of the "
        "three subject-disjoint best-accuracy epochs (14, 14, 12 -> 14)."
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    seed_everything(int(config["seed"]))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["use_amp"] and device.type == "cuda")
    dataset = ThermalOOFDataset(
        manifest_path=args.manifest,
        split="train",
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=True,
        normalization=str(config.get("normalization", "legacy")),
    )
    loader = create_loader(dataset, config, True)
    model = ThermalResNetTSM(
        num_classes=40,
        dropout=float(config["dropout"]),
        imagenet_pretrained=bool(config["imagenet_pretrained"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_grad_norm = config.get("max_grad_norm")
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
    history: list[dict[str, float | int]] = []
    started = time.time()
    print(
        f"device={device}, train={len(dataset)}, epochs={config['epochs']}, "
        f"parameters={parameter_count(model):,}, fp32={fp32_size_mb(model):.2f} MiB",
        flush=True,
    )
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.time()
        train_metrics = run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            max_grad_norm,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    checkpoint_path = output_dir / "final_epoch.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": int(config["epochs"]),
            "train_accuracy": float(history[-1]["train_accuracy"]),
            "train_loss": float(history[-1]["train_loss"]),
        },
        checkpoint_path,
    )
    summary = {
        "protocol": config["refit_protocol"],
        "training_samples": len(dataset),
        "epochs": int(config["epochs"]),
        "device": str(device),
        "seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_mib": checkpoint_path.stat().st_size / 2**20,
        "final_train_accuracy": float(history[-1]["train_accuracy"]),
        "final_train_loss": float(history[-1]["train_loss"]),
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
