from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from aligned_data import AlignedMultimodalDataset
from p27_ir_mc3_model import P27IRMC3SkeletonModel
from p27_ir_r2plus1d_model import P27IRR2Plus1DSkeletonModel
from p27_ir_s3d_model import (
    P27IRS3DSkeletonModel,
    freeze_video_batch_norm_statistics,
)
from p27_ir_swin3d_model import P27IRSwin3DSkeletonModel
from probe_p27r3_incremental_information import metric_bundle, write_csv


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fold-pure Kinetics-S3D IR + Skeleton event model"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def make_loader(
    dataset: AlignedMultimodalDataset, config: dict[str, Any], train: bool
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
    workers = int(
        config["num_workers"] if train else config.get("val_num_workers", 0)
    )
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


def load_skeleton(
    model: P27IRS3DSkeletonModel, checkpoint_path: Path
) -> None:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    prefix = "skeleton."
    state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith(prefix)
    }
    model.skeleton.load_state_dict(state, strict=True)


def metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def run_epoch(
    model: P27IRS3DSkeletonModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    accumulation_steps: int,
    freeze_bn: bool,
    ir_input_key: str,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        freeze_video_batch_norm_statistics(model.video_features)
    total_loss = 0.0
    total_samples = 0
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    sample_ids: list[str] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        ir = batch[ir_input_key].to(device, non_blocking=True)
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(ir, skeleton)
                loss = criterion(output, target)
            if training:
                scaler.scale(loss / accumulation_steps).backward()
                boundary = (
                    (step + 1) % accumulation_steps == 0
                    or step + 1 == len(loader)
                )
                if boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(5.0),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        batch_size = len(target)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        labels.append(target.detach().cpu())
        logits.append(output.detach().float().cpu())
        sample_ids.extend(batch["sample_id"])
    label_array = torch.cat(labels).numpy()
    logit_array = torch.cat(logits).numpy()
    prediction_array = logit_array.argmax(axis=1)
    return {
        "loss": total_loss / max(total_samples, 1),
        **metrics(label_array, prediction_array),
        "labels": label_array,
        "logits": logit_array,
        "predictions": prediction_array,
        "sample_ids": np.asarray(sample_ids),
    }


def save_checkpoint(
    path: Path,
    model: P27IRS3DSkeletonModel,
    config: dict[str, Any],
    epoch: int,
    values: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_type": f"p27_ir_{config.get('video_backbone', 's3d')}_skeleton",
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": {
                key: float(values[key])
                for key in (
                    "loss",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                )
            },
        },
        path,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    seed_everything(int(config["seed"]))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    common = {
        "manifest_path": args.manifest.resolve(),
        "modalities": ["ir", "skeleton"],
        "num_frames": int(config["num_frames"]),
        "image_height": int(config["image_height"]),
        "image_width": int(config["image_width"]),
        "cache_dir": config["cache_dir"],
        "visual_normalization": "imagenet",
        "skeleton_representation": "clip_joint_bone_velocity",
        "skeleton_raw_cache_dir": config["skeleton_raw_cache_dir"],
        "ir_roi_csv": config.get("ir_roi_csv"),
        "ir_roi_context": float(config.get("ir_roi_context", 0.3)),
        "ir_random_resized_crop_min_scale": float(
            config.get("ir_random_resized_crop_min_scale", 1.0)
        ),
        "temporal_sampling": str(config.get("temporal_sampling", "uniform")),
    }
    train_dataset = AlignedMultimodalDataset(
        split="train", augment=True, **common
    )
    held_dataset = AlignedMultimodalDataset(
        split="val", augment=False, **common
    )
    train_loader = make_loader(train_dataset, config, True)
    held_loader = make_loader(held_dataset, config, False)
    video_backbone = str(config.get("video_backbone", "s3d"))
    model_class = {
        "s3d": P27IRS3DSkeletonModel,
        "mc3_18": P27IRMC3SkeletonModel,
        "r2plus1d_18": P27IRR2Plus1DSkeletonModel,
        "swin3d_t": P27IRSwin3DSkeletonModel,
    }.get(video_backbone)
    if model_class is None:
        raise ValueError(f"unknown video_backbone: {video_backbone}")
    model = model_class(
        dropout=float(config["dropout"]),
        skeleton_input_dim=10,
        kinetics_pretrained=bool(config["kinetics_pretrained"]),
    ).to(device)
    skeleton_checkpoint = (
        PROJECT_DIR / config["pretrained_skeleton_checkpoint"]
    ).resolve()
    load_skeleton(model, skeleton_checkpoint)
    video_parameters = list(model.video_features.parameters())
    video_ids = {id(parameter) for parameter in video_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in video_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": video_parameters,
                "lr": float(config["video_learning_rate"]),
                "name": "kinetics_video",
            },
            {
                "params": head_parameters,
                "lr": float(config["head_learning_rate"]),
                "name": "skeleton_fusion_head",
            },
        ],
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["label_smoothing"])
    )
    parameters = int(sum(value.numel() for value in model.parameters()))
    print(
        f"device={device} train={len(train_dataset)} held={len(held_dataset)} "
        f"params={parameters:,} fp16_mib={parameters * 2 / 1024**2:.3f}",
        flush=True,
    )
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_values: dict[str, Any] | None = None
    fixed_evaluation_epoch = int(config.get("fixed_evaluation_epoch", 0))
    fixed_values: dict[str, Any] | None = None
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
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
            freeze_bn=bool(config["freeze_video_batch_norm"]),
            ir_input_key=str(config.get("ir_input_key", "ir")),
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
            freeze_bn=False,
            ir_input_key=str(config.get("ir_input_key", "ir")),
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "video_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
            "train_loss": train_values["loss"],
            "train_accuracy": train_values["accuracy"],
            "held_loss": held_values["loss"],
            "held_accuracy": held_values["accuracy"],
            "held_balanced_accuracy": held_values["balanced_accuracy"],
            "held_macro_f1": held_values["macro_f1"],
            "seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(row)
        write_csv(output / "history.csv", history)
        save_checkpoint(output / "last.pt", model, config, epoch, held_values)
        if float(held_values["accuracy"]) > best_accuracy:
            best_accuracy = float(held_values["accuracy"])
            best_epoch = epoch
            best_values = held_values
            save_checkpoint(
                output / "best_accuracy.pt",
                model,
                config,
                epoch,
                held_values,
            )
            np.savez_compressed(
                output / "best_accuracy_held_logits.npz",
                protocol=np.asarray("p27-ir-s3d-skeleton-inner-v1"),
                sample_ids=held_values["sample_ids"],
                labels=held_values["labels"],
                logits=held_values["logits"].astype(np.float32),
                outer_held_predictions_generated=np.asarray(False),
            )
        if fixed_evaluation_epoch and epoch == fixed_evaluation_epoch:
            fixed_values = held_values
            save_checkpoint(
                output / f"fixed_epoch_{epoch:02d}.pt",
                model,
                config,
                epoch,
                held_values,
            )
            np.savez_compressed(
                output / f"fixed_epoch_{epoch:02d}_held_logits.npz",
                protocol=np.asarray("p27-ir-s3d-skeleton-inner-fixed-epoch-v1"),
                sample_ids=held_values["sample_ids"],
                labels=held_values["labels"],
                logits=held_values["logits"].astype(np.float32),
                outer_held_predictions_generated=np.asarray(False),
            )
        print(
            f"epoch={epoch:02d} train={train_values['accuracy']:.4f} "
            f"held={held_values['accuracy']:.4f} "
            f"bal={held_values['balanced_accuracy']:.4f} "
            f"f1={held_values['macro_f1']:.4f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    assert best_values is not None
    summary = {
        "protocol": "outer-fold-0 train subjects only; one fixed subject-disjoint inner fold",
        "video_backbone": video_backbone,
        "outer_held_predictions_generated": False,
        "device": str(device),
        "train_samples": len(train_dataset),
        "held_samples": len(held_dataset),
        "parameters": parameters,
        "fp16_parameter_mib": parameters * 2 / 1024**2,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "best_balanced_accuracy": float(
            best_values["balanced_accuracy"]
        ),
        "best_macro_f1": float(best_values["macro_f1"]),
        "total_seconds": float(time.perf_counter() - started),
        "checkpoint_sha256": sha256(output / "best_accuracy.pt"),
        "subset_metrics": metric_bundle(
            best_values["labels"], best_values["predictions"]
        ),
    }
    if fixed_evaluation_epoch:
        if fixed_values is None:
            raise RuntimeError(
                f"fixed_evaluation_epoch={fixed_evaluation_epoch} was not reached"
            )
        fixed_checkpoint = output / f"fixed_epoch_{fixed_evaluation_epoch:02d}.pt"
        summary["fixed_evaluation"] = {
            "epoch": fixed_evaluation_epoch,
            "checkpoint_sha256": sha256(fixed_checkpoint),
            "subset_metrics": metric_bundle(
                fixed_values["labels"], fixed_values["predictions"]
            ),
        }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
