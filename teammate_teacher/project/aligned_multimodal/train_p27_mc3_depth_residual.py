from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from aligned_data import AlignedMultimodalDataset
from p27_ir_depth_mc3_model import P27IRDepthMC3ResidualModel
from probe_p27r3_incremental_information import metric_bundle
from train_p27_ir_s3d_skeleton import make_loader, metrics, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a zero-initialised shared-MC3 Depth residual."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_epoch(
    model: P27IRDepthMC3ResidualModel,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    accumulation_steps: int,
    ablation: str | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    model.base.eval()
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    base_logits: list[torch.Tensor] = []
    sample_ids: list[str] = []
    subjects: list[str] = []
    for step, batch in enumerate(loader):
        ir = batch["ir"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(
                    ir, depth, skeleton, ablation=ablation
                )
                loss = criterion(output["logits"], target)
            if training:
                scaler.scale(loss / accumulation_steps).backward()
                boundary = (
                    (step + 1) % accumulation_steps == 0
                    or step + 1 == len(loader)
                )
                if boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        5.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        batch_size = len(target)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        labels.append(target.detach().cpu())
        logits.append(output["logits"].detach().float().cpu())
        base_logits.append(output["base_logits"].detach().float().cpu())
        sample_ids.extend(batch["sample_id"])
        subjects.extend(batch["user_id"])
    label_array = torch.cat(labels).numpy()
    logit_array = torch.cat(logits).numpy()
    base_array = torch.cat(base_logits).numpy()
    return {
        "loss": total_loss / max(total_samples, 1),
        **metrics(label_array, logit_array.argmax(axis=1)),
        "labels": label_array,
        "logits": logit_array,
        "base_logits": base_array,
        "sample_ids": np.asarray(sample_ids),
        "subjects": np.asarray(subjects),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
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
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    dataset_args = {
        "manifest_path": args.manifest.resolve(),
        "modalities": ["ir", "depth", "skeleton"],
        "num_frames": int(config["num_frames"]),
        "image_height": int(config["image_height"]),
        "image_width": int(config["image_width"]),
        "cache_dir": config["cache_dir"],
        "depth_representation": "jet_rgb",
        "visual_normalization": "imagenet",
        "skeleton_representation": "clip_joint_bone_velocity",
        "skeleton_raw_cache_dir": config["skeleton_raw_cache_dir"],
        "temporal_sampling": str(config.get("temporal_sampling", "uniform")),
    }
    train_dataset = AlignedMultimodalDataset(
        split="train", augment=True, **dataset_args
    )
    held_dataset = AlignedMultimodalDataset(
        split="val", augment=False, **dataset_args
    )
    train_loader = make_loader(train_dataset, config, True)
    held_loader = make_loader(held_dataset, config, False)

    checkpoint_path = args.base_checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = P27IRDepthMC3ResidualModel(
        dropout=float(config["dropout"]), skeleton_input_dim=10
    ).to(device)
    model.base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.depth_project.load_state_dict(
        model.base.video_project.state_dict(), strict=True
    )
    model.freeze_base()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["residual_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["label_smoothing"])
    )

    first_batch = next(iter(held_loader))
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type, dtype=torch.float16, enabled=use_amp
    ):
        initial = model(
            first_batch["ir"].to(device),
            first_batch["depth"].to(device),
            first_batch["skeleton"].to(device),
        )
    initial_max_logit_diff = float(
        (initial["logits"] - initial["base_logits"]).abs().max()
    )
    if initial_max_logit_diff != 0.0:
        raise RuntimeError(
            f"Depth residual is not exactly zero: {initial_max_logit_diff}"
        )

    parameters = int(sum(value.numel() for value in model.parameters()))
    trainable_parameters = int(sum(value.numel() for value in trainable))
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_values = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            accumulation_steps=int(config["gradient_accumulation_steps"]),
        )
        held_values = run_epoch(
            model,
            held_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            accumulation_steps=1,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "residual_learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_values["loss"],
            "train_accuracy": train_values["accuracy"],
            "held_loss": held_values["loss"],
            "held_accuracy": held_values["accuracy"],
            "held_balanced_accuracy": held_values["balanced_accuracy"],
            "held_macro_f1": held_values["macro_f1"],
            "seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        if float(held_values["accuracy"]) > best_accuracy:
            best_accuracy = float(held_values["accuracy"])
            best_epoch = epoch
            torch.save(
                {
                    "model_type": "p27_ir_depth_shared_mc3_residual",
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "metrics": {
                        key: float(held_values[key])
                        for key in ("accuracy", "balanced_accuracy", "macro_f1")
                    },
                    "base_checkpoint_sha256": sha256(checkpoint_path),
                },
                output_dir / "best_accuracy.pt",
            )
            np.savez_compressed(
                output_dir / "best_accuracy_held_logits.npz",
                protocol=np.asarray("p27-ir-depth-shared-mc3-residual-inner-v1"),
                sample_ids=held_values["sample_ids"],
                subjects=held_values["subjects"],
                labels=held_values["labels"],
                logits=held_values["logits"].astype(np.float32),
                base_logits=held_values["base_logits"].astype(np.float32),
                outer_held_predictions_generated=np.asarray(False),
            )
        print(json.dumps(row), flush=True)

    best_path = output_dir / "best_accuracy.pt"
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    ablations: dict[str, Any] = {}
    for name in ("depth_zero", "depth_shuffle"):
        values = run_epoch(
            model,
            held_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            accumulation_steps=1,
            ablation=name,
        )
        ablations[name] = metric_bundle(
            values["labels"], values["logits"].argmax(axis=1)
        )
    with np.load(
        output_dir / "best_accuracy_held_logits.npz", allow_pickle=False
    ) as archive:
        best_metrics = metric_bundle(
            archive["labels"], archive["logits"].argmax(axis=1)
        )
        base_metrics = metric_bundle(
            archive["labels"], archive["base_logits"].argmax(axis=1)
        )
    summary = {
        "protocol": "p27-ir-depth-shared-mc3-residual-inner-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "initial_max_logit_diff": initial_max_logit_diff,
        "train_samples": len(train_dataset),
        "held_samples": len(held_dataset),
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "fp16_parameter_mib": parameters * 2 / 1024**2,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "best_subset_metrics": best_metrics,
        "corresponding_base_subset_metrics": base_metrics,
        "ablations": ablations,
        "total_seconds": float(time.perf_counter() - started),
        "checkpoint_sha256": sha256(best_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
