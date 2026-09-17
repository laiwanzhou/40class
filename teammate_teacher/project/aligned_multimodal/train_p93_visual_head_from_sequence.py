from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from p86_cached_motion_data import (
    P86CachedSequenceMotionDataset,
    collate_p86_cached_motion,
)
from train_p86_cached_motion_proxy import load_visual
from train_p86_mobind_fusion_proxy import (
    to_device,
    visual_head_parameters,
)
from train_p86_visual_student_oof import class_weights, metric_dict, relation_loss


HERE = Path(__file__).resolve().parent
DEFAULT_MOTION = HERE / "runs/p86_motion_window_cache_t16_v1"
DEFAULT_PIXELS = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_TEACHER_FEATURES = (
    HERE / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_TEACHER_LOGITS = (
    HERE / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the P86 visual temporal/fusion head from a label-free "
            "Kinetics backbone sequence cache on explicit subject allowlists."
        )
    )
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-users", nargs="+", required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.08)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(
    args: argparse.Namespace,
    full: P86CachedSequenceMotionDataset,
    users: set[str],
    temporal_augment: bool,
) -> P86CachedSequenceMotionDataset:
    selected = np.asarray(
        [index for index, value in enumerate(full.users.astype(str)) if value in users],
        dtype=np.int64,
    )
    return P86CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        indices=selected,
        temporal_augment=temporal_augment,
    )


def make_loader(
    dataset: P86CachedSequenceMotionDataset,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_p86_cached_motion,
        drop_last=shuffle and len(dataset) >= args.batch_size,
    )


def forward_visual(model, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model.forward_from_backbone_sequence(
        batch["backbone_sequence"],
        batch["view_valid"],
        batch["view_quality"],
        batch["global_time_position"],
    )


def evaluate(model, data, device, max_batches: int):
    model.eval()
    rows = []
    logits_rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(data):
            if max_batches and batch_index >= max_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = forward_visual(model, batch)
            logits = output["logits"].float().cpu()
            probability = torch.softmax(logits, dim=1)
            labels = batch["label"].cpu()
            logits_rows.append(logits.numpy())
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "user_id": batch["user_id"][index],
                        "label": int(labels[index]),
                        "prediction": int(probability[index].argmax()),
                        "confidence": float(probability[index].max()),
                    }
                )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    metrics = metric_dict(labels, predictions, [row["user_id"] for row in rows])
    return metrics, rows, np.concatenate(logits_rows, axis=0)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
    train_users = set(map(str, args.train_users))
    holdout_users = set(map(str, args.holdout_users))
    if train_users & holdout_users:
        raise ValueError("visual train and holdout user allowlists overlap")
    seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full = P86CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
    )
    observed = set(full.users.astype(str).tolist())
    if not train_users <= observed or not holdout_users <= observed:
        raise ValueError("visual user allowlist contains an unknown subject")
    training = make_dataset(args, full, train_users, temporal_augment=True)
    holdout = make_dataset(args, full, holdout_users, temporal_augment=False)
    model, config = load_visual(args.initial_checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = visual_head_parameters(model)
    for parameter in trainable:
        parameter.requires_grad_(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    labels = np.asarray(
        [int(full.labels[index]) for index in training.indices], dtype=np.int64
    )
    weights = class_weights(labels, args.class_weight_power, device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1))
        )
        learning_rate = args.minimum_learning_rate + (
            args.learning_rate - args.minimum_learning_rate
        ) * cosine
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "relation": 0.0}
        samples = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(make_loader(training, args, True)):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                values = forward_visual(model, batch)
                ce = F.cross_entropy(
                    values["logits"],
                    batch["label"],
                    weight=weights,
                    label_smoothing=args.label_smoothing,
                )
                temperature = args.distillation_temperature
                kd = F.kl_div(
                    F.log_softmax(values["logits"] / temperature, dim=1),
                    F.softmax(batch["teacher_logits"] / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature**2
                relation = relation_loss(
                    values["clip_embeddings"],
                    batch["teacher_features"],
                    values["clip_mask"],
                )
                loss = (
                    ce
                    + args.distillation_weight * kd
                    + args.relation_weight * relation
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["label"])
            samples += count
            for key, value in (("loss", loss), ("ce", ce), ("kd", kd), ("relation", relation)):
                sums[key] += float(value.detach()) * count
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            **{f"train_{key}": value / max(samples, 1) for key, value in sums.items()},
            "train_samples": samples,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    holdout_metrics, predictions, logits = evaluate(
        model, make_loader(holdout, args, False), device, args.max_eval_batches
    )
    write_rows(output_dir / "training_history.csv", history)
    write_rows(output_dir / "holdout_predictions.csv", predictions)
    np.save(output_dir / "holdout_logits.npy", logits)
    training_subjects = sorted(train_users)
    holdout_subjects = sorted(holdout_users)
    excluded_subjects = sorted(observed - train_users - holdout_users)
    checkpoint = {
        "stage": "P93_subject_safe_cached_visual_head",
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": config,
        "fixed_epochs": args.epochs,
        "training_subjects": training_subjects,
        "holdout_subjects": holdout_subjects,
        "excluded_subjects": excluded_subjects,
        "holdout_metrics": holdout_metrics,
    }
    torch.save(checkpoint, output_dir / "visual_student.pt")
    summary = {
        "stage": "P93_subject_safe_cached_visual_head",
        "status": "smoke" if args.smoke else "formal",
        "protocol": (
            "Label-free Kinetics backbone sequences are fixed. Only the visual "
            "temporal/view/window/classifier head is trained on the explicit user "
            "allowlist for a fixed epoch budget; holdout is evaluated once."
        ),
        "counts": {"train": len(training), "holdout": len(holdout)},
        "training_subjects": training_subjects,
        "holdout_subjects": holdout_subjects,
        "excluded_subjects": excluded_subjects,
        "fixed_epochs": args.epochs,
        "holdout_metrics": holdout_metrics,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
