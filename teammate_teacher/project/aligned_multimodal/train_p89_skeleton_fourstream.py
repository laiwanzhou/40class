from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from p86_mobind_lite_data import P86MoBindMotionDataset, collate_p86_mobind
from p89_skeleton_fourstream_model import P89FourStreamSkeleton


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MOTION = PROJECT_DIR / "runs" / "p86_motion_window_cache_t16_v1"
DEFAULT_TEACHER = PROJECT_DIR / "runs" / "p85_videomae_large_multiclip_full40_head_v1" / "candidate_oof_logits.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P89 true four-stream subject-disjoint Skeleton model.")
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=0.08)
    parser.add_argument("--distillation-weight", type=float, default=0.25)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment_skeleton(features: torch.Tensor, joint_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.clone()
    joint_mask = joint_mask.clone()
    batch = len(features)
    scale = torch.empty(batch, 1, 1, 1, 1, device=features.device).uniform_(0.92, 1.08)
    features[..., :12] *= scale
    features[..., :12] += 0.005 * torch.randn_like(features[..., :12])

    mirror = torch.rand(batch, device=features.device) < 0.5
    if mirror.any():
        permutation = torch.tensor((0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13), device=features.device)
        mirrored = features[mirror][..., permutation, :]
        mirrored_mask = joint_mask[mirror][..., permutation]
        for start in (0, 3, 6, 9):
            mirrored[..., start] *= -1.0
        features[mirror] = mirrored
        joint_mask[mirror] = mirrored_mask

    dropped = torch.rand_like(joint_mask.float()) < 0.025
    joint_mask &= ~dropped
    features *= joint_mask.unsqueeze(-1).to(features.dtype)
    return features, joint_mask


def supervised_contrastive(embedding: torch.Tensor, labels: torch.Tensor, temperature: float = 0.12) -> torch.Tensor:
    normalized = F.normalize(embedding, dim=-1)
    logits = normalized @ normalized.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    logits = logits.masked_fill(identity, -1e4)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_sample = -(log_probability * positive.to(log_probability.dtype)).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return per_sample[valid].mean()


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.inference_mode()
def evaluate(model: P89FourStreamSkeleton, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits, labels, sample_ids, users, gates = [], [], [], [], []
    for raw in loader:
        batch = move(raw, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch["skeleton_features"], batch["skeleton_joint_mask"])
        logits.append(output["logits"].float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        gates.append(output["stream_gates"].float().cpu().numpy())
        sample_ids.extend(raw["sample_id"])
        users.extend(raw["user_id"])
    logits_array = np.concatenate(logits)
    label_array = np.concatenate(labels)
    prediction = logits_array.argmax(axis=1)
    user_array = np.asarray(users)
    return {
        "sample_ids": np.asarray(sample_ids), "users": user_array,
        "labels": label_array, "logits": logits_array,
        "gates": np.concatenate(gates), "predictions": prediction,
        "metrics": {
            "correct": int(np.sum(prediction == label_array)),
            "total": int(len(label_array)),
            "accuracy": float(np.mean(prediction == label_array)),
            "balanced_accuracy": float(balanced_accuracy_score(label_array, prediction)),
            "macro_f1": float(f1_score(label_array, prediction, average="macro", zero_division=0)),
            "per_user_accuracy": {user: float(np.mean(prediction[user_array == user] == label_array[user_array == user])) for user in sorted(set(users))},
        },
    }


def write_predictions(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "user_id", "label", "prediction"))
        writer.writerows(zip(result["sample_ids"], result["users"], result["labels"], result["predictions"], strict=True))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = P86MoBindMotionDataset(
        args.motion_cache.resolve(), temporal_augment=False, teacher_logits=args.teacher_logits.resolve(),
    )
    all_users = np.asarray([row["user_id"] for row in universe.rows])
    train_indices = np.flatnonzero(~np.isin(all_users, args.holdout_users))
    held_indices = np.flatnonzero(np.isin(all_users, args.holdout_users))
    train_set = P86MoBindMotionDataset(
        args.motion_cache.resolve(), indices=train_indices, temporal_augment=True, teacher_logits=args.teacher_logits.resolve(),
    )
    held_set = P86MoBindMotionDataset(
        args.motion_cache.resolve(), indices=held_indices, temporal_augment=False, teacher_logits=args.teacher_logits.resolve(),
    )
    counts = np.bincount(train_set.labels, minlength=40).astype(np.float64)
    sample_weights = np.power(np.maximum(counts[train_set.labels], 1.0), -0.65)
    sampler_generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(train_set), replacement=True, generator=sampler_generator)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        collate_fn=collate_p86_mobind, pin_memory=device.type == "cuda",
    )
    held_loader = DataLoader(
        held_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers,
        collate_fn=collate_p86_mobind, pin_memory=device.type == "cuda",
    )

    model = P89FourStreamSkeleton(args.width, 40, args.dropout).to(device)
    class_weight = torch.from_numpy(np.power(len(train_set) / np.maximum(counts, 1.0), 0.35).astype(np.float32)).to(device)
    class_weight /= class_weight.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        fraction = (epoch - 1) / max(args.epochs - 1, 1)
        learning_rate = args.minimum_learning_rate + 0.5 * (args.learning_rate - args.minimum_learning_rate) * (1.0 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        sums = {key: 0.0 for key in ("loss", "ce", "aux", "contrastive", "distillation")}
        samples = 0
        for raw in train_loader:
            batch = move(raw, device)
            skeleton, mask = augment_skeleton(batch["skeleton_features"], batch["skeleton_joint_mask"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(skeleton, mask)
                ce = F.cross_entropy(result["logits"], batch["label"], weight=class_weight, label_smoothing=0.05)
                aux = sum(F.cross_entropy(result["stream_logits"][:, index], batch["label"], weight=class_weight, label_smoothing=0.05) for index in range(4)) / 4.0
                contrastive = supervised_contrastive(result["embedding"], batch["label"])
                temperature = args.distillation_temperature
                distillation = F.kl_div(
                    F.log_softmax(result["logits"] / temperature, dim=1),
                    F.softmax(batch["teacher_logits"] / temperature, dim=1), reduction="batchmean",
                ) * (temperature * temperature)
                loss = ce + args.auxiliary_weight * aux + args.contrastive_weight * contrastive + args.distillation_weight * distillation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["label"])
            samples += count
            for key, value in (("loss", loss), ("ce", ce), ("aux", aux), ("contrastive", contrastive), ("distillation", distillation)):
                sums[key] += float(value.detach()) * count
        row = {"epoch": float(epoch), "learning_rate": learning_rate, **{key: value / samples for key, value in sums.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)

    evaluation = evaluate(model, held_loader, device)
    checkpoint = {
        "model": model.state_dict(), "args": vars(args),
        "stream_names": list(P89FourStreamSkeleton.STREAMS),
    }
    torch.save(checkpoint, output_dir / "fourstream_skeleton.pt")
    np.savez_compressed(
        output_dir / "subject_holdout_logits.npz",
        sample_ids=evaluation["sample_ids"], users=evaluation["users"], labels=evaluation["labels"],
        logits=evaluation["logits"], predictions=evaluation["predictions"], gates=evaluation["gates"],
    )
    write_predictions(output_dir / "subject_holdout_predictions.csv", evaluation)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    parameters = sum(value.numel() for value in model.parameters())
    summary = {
        "stage": "P89_true_fourstream_skeleton_subject_holdout_v1",
        "status": "formal_fixed_epoch",
        "holdout_users": sorted(args.holdout_users),
        "train_samples": len(train_set), "holdout_samples": len(held_set),
        "parameters": parameters, "fp32_mib": parameters * 4 / 2**20,
        "metrics": evaluation["metrics"],
        "mean_stream_gate": dict(zip(P89FourStreamSkeleton.STREAMS, evaluation["gates"].mean(axis=0).tolist(), strict=True)),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
