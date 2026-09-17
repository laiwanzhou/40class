"""HART/LIMU-inspired 128-step IMU teachers for the P90 ceiling study.

Two teachers are always comparable under the same architecture and folds:
random initialization, and fold-local masked-signal pretraining.  The SSL stage
uses only the outer fold's training users, including no validation-user inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from p90_teacher_common import (
    HERE,
    NUM_CLASSES,
    REPO_ROOT,
    classification_metrics,
    load_protocol,
    save_oof_artifact,
    seed_everything,
)


DEFAULT_CACHE = HERE / "cache" / "imu_128"
DEFAULT_RUN = REPO_ROOT / "runs" / "p90_imu_ssl_teacher_v1"


def load_aligned_imu(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    protocol = load_protocol()
    index = pd.read_csv(cache_dir / "index.csv")
    train = index.loc[index["split"].astype(str).eq("train")].copy()
    if train["sample_id"].duplicated().any():
        raise ValueError("duplicated IMU sample_id")
    lookup = train.set_index("sample_id")
    rows = lookup.loc[protocol.sample_ids]
    cache_indices = rows["cache_index"].to_numpy(dtype=np.int64)
    values = np.asarray(np.load(cache_dir / "imu_float32.npy", mmap_mode="r")[cache_indices])
    time_mask = np.asarray(
        np.load(cache_dir / "time_mask_uint8.npy", mmap_mode="r")[cache_indices]
    ).astype(bool)
    usable = rows["usable"].to_numpy(dtype=np.int64).astype(bool)
    if values.shape != (len(protocol.labels), 5, 128, 10):
        raise ValueError(f"expected aligned IMU [N,5,128,10], got {values.shape}")
    usable &= time_mask.any(axis=(1, 2))
    return values.astype(np.float32), time_mask, usable


def fit_normalizer(
    values: np.ndarray, time_mask: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = values[indices]
    mask = time_mask[indices]
    mean = np.zeros((5, 10), dtype=np.float64)
    std = np.ones((5, 10), dtype=np.float64)
    for device in range(5):
        valid = mask[:, device]
        for channel in range(10):
            channel_values = selected[:, device, :, channel][valid]
            if len(channel_values):
                mean[device, channel] = channel_values.mean(dtype=np.float64)
                std[device, channel] = channel_values.std(dtype=np.float64)
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_imu(
    values: np.ndarray, time_mask: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    output = (values - mean[None, :, None, :]) / std[None, :, None, :]
    output = np.clip(output, -8.0, 8.0).astype(np.float32)
    output[~time_mask] = 0.0
    return output


class IMUDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        time_mask: np.ndarray,
        labels: np.ndarray | None,
        augment: bool,
    ) -> None:
        self.values = values
        self.time_mask = time_mask
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        values = torch.from_numpy(self.values[index].copy())
        mask = torch.from_numpy(self.time_mask[index].copy())
        if self.augment:
            values[..., :6] += torch.randn_like(values[..., :6]) * 0.015
            values *= 0.96 + torch.rand((5, 1, 1)) * 0.08
            if torch.rand(()) < 0.35:
                device = int(torch.randint(0, 5, (1,)))
                other_devices = torch.arange(5) != device
                if mask[other_devices].any():
                    values[device] = 0.0
                    mask[device] = False
            shift = int(torch.randint(-4, 5, (1,)))
            values = torch.roll(values, shifts=shift, dims=1)
            mask = torch.roll(mask, shifts=shift, dims=1)
        if self.labels is None:
            return values, mask
        return values, mask, torch.tensor(self.labels[index], dtype=torch.long)


class IMUEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        depth: int = 4,
        heads: int = 6,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(10, dim, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, 32, 1, dim))
        self.device_embedding = nn.Parameter(torch.zeros(1, 1, 5, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        self.reconstruction_head = nn.Linear(dim, 10)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.device_embedding, std=0.02)

    def forward_tokens(
        self, values: torch.Tensor, time_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, devices, steps, channels = values.shape
        stem = self.stem(values.reshape(batch * devices, steps, channels).transpose(1, 2))
        reduced_steps = stem.shape[-1]
        tokens = stem.transpose(1, 2).reshape(batch, devices, reduced_steps, -1)
        tokens = tokens.permute(0, 2, 1, 3)
        reduced_mask = F.max_pool1d(
            time_mask.float().reshape(batch * devices, 1, steps), kernel_size=4, stride=4
        ).reshape(batch, devices, reduced_steps)
        reduced_mask = reduced_mask.permute(0, 2, 1).bool()
        tokens = tokens + self.time_embedding[:, :reduced_steps] + self.device_embedding
        flat_tokens = tokens.reshape(batch, reduced_steps * devices, -1)
        flat_mask = reduced_mask.reshape(batch, reduced_steps * devices)
        encoded = self.transformer(flat_tokens, src_key_padding_mask=~flat_mask)
        encoded = self.norm(encoded)
        encoded = encoded * flat_mask.unsqueeze(-1)
        return encoded.reshape(batch, reduced_steps, devices, -1), reduced_mask

    def pool(self, values: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        tokens, mask = self.forward_tokens(values, time_mask)
        weights = mask.unsqueeze(-1)
        mean = (tokens * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1)
        maximum = tokens.masked_fill(~weights, -1e4).amax(dim=(1, 2))
        return torch.cat((mean, maximum), dim=-1)

    def reconstruct(self, values: torch.Tensor, time_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.forward_tokens(values, time_mask)
        return self.reconstruction_head(tokens), mask


class IMUClassifier(nn.Module):
    def __init__(self, encoder: IMUEncoder, dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, NUM_CLASSES),
        )

    def forward(self, values: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.pool(values, time_mask))


def make_ssl_mask(time_mask: torch.Tensor, ratio: float = 0.30) -> torch.Tensor:
    batch, devices, steps = time_mask.shape
    starts = torch.rand((batch, devices, steps), device=time_mask.device) < (ratio / 8.0)
    mask = starts.float().reshape(batch * devices, 1, steps)
    mask = F.max_pool1d(mask, kernel_size=9, stride=1, padding=4).bool()
    mask = mask.reshape(batch, devices, steps)
    random_mask = torch.rand_like(time_mask.float()) < (ratio * 0.20)
    return (mask | random_mask) & time_mask


def pretrain_encoder(
    encoder: IMUEncoder,
    values: np.ndarray,
    time_mask: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> None:
    dataset = IMUDataset(values, time_mask, labels=None, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(epochs):
        encoder.train()
        total_loss = 0.0
        seen = 0
        for batch, mask in loader:
            batch = batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            ssl_mask = make_ssl_mask(mask)
            masked = batch.masked_fill(ssl_mask.unsqueeze(-1), 0.0)
            target = F.avg_pool1d(
                batch.reshape(-1, 128, 10).transpose(1, 2), kernel_size=4, stride=4
            ).transpose(1, 2).reshape(len(batch), 5, 32, 10).permute(0, 2, 1, 3)
            reduced_ssl_mask = F.max_pool1d(
                ssl_mask.float().reshape(-1, 1, 128), kernel_size=4, stride=4
            ).reshape(len(batch), 5, 32).permute(0, 2, 1).bool()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                prediction, valid = encoder.reconstruct(masked, mask)
                selected = reduced_ssl_mask & valid
                loss = F.smooth_l1_loss(prediction[selected], target[selected], beta=0.5)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        scheduler.step()
        print(f"    ssl epoch {epoch + 1:02d}/{epochs}: loss={total_loss / seen:.5f}", flush=True)


def train_classifier(
    model: IMUClassifier,
    values: np.ndarray,
    time_mask: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> None:
    dataset = IMUDataset(values, time_mask, labels=labels, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch, mask, batch_labels in loader:
            batch = batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(batch, mask)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(batch)
            correct += int((logits.argmax(dim=1) == batch_labels).sum())
            seen += len(batch)
        scheduler.step()
        print(
            f"    cls epoch {epoch + 1:02d}/{epochs}: loss={total_loss / seen:.4f}, train_acc={correct / seen:.4f}",
            flush=True,
        )


@torch.inference_mode()
def predict(
    model: IMUClassifier,
    values: np.ndarray,
    time_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = IMUDataset(values, time_mask, labels=None, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    chunks: list[np.ndarray] = []
    for batch, mask in loader:
        batch = batch.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(batch, mask)
        chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def run_teacher(args: argparse.Namespace, use_ssl: bool) -> None:
    protocol = load_protocol()
    values, time_mask, usable = load_aligned_imu(args.cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_name = "imu_hart128_maskedssl" if use_ssl else "imu_hart128_scratch"
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints" / teacher_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    oof_logits = np.zeros((len(protocol.labels), NUM_CLASSES), dtype=np.float32)
    completed = np.zeros(len(protocol.labels), dtype=bool)
    folds = [args.fold_only] if args.fold_only is not None else list(range(3))
    for fold in folds:
        seed_everything(args.seed + fold + (100 if use_ssl else 0))
        train_indices = protocol.train_indices(fold)
        val_indices = protocol.val_indices(fold)
        train_usable = train_indices[usable[train_indices]]
        val_usable = val_indices[usable[val_indices]]
        mean, std = fit_normalizer(values, time_mask, train_usable)
        normalized = normalize_imu(values, time_mask, mean, std)
        encoder = IMUEncoder(
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            dropout=args.transformer_dropout,
        ).to(device)
        print(
            f"{teacher_name}: fold={fold}, train_usable={len(train_usable)}, val_usable={len(val_usable)}",
            flush=True,
        )
        if use_ssl:
            pretrain_encoder(
                encoder,
                normalized[train_usable],
                time_mask[train_usable],
                args.ssl_epochs,
                args.batch_size,
                args.ssl_lr,
                args.weight_decay,
                device,
                args.seed + fold,
            )
        model = IMUClassifier(encoder, dim=args.dim, dropout=args.classifier_dropout).to(device)
        train_classifier(
            model,
            normalized[train_usable],
            time_mask[train_usable],
            protocol.labels[train_usable],
            args.cls_epochs,
            args.batch_size,
            args.cls_lr,
            args.weight_decay,
            device,
            args.seed + fold,
        )
        usable_logits = predict(
            model,
            normalized[val_usable],
            time_mask[val_usable],
            args.batch_size,
            device,
        )
        oof_logits[val_usable] = usable_logits
        completed[val_usable] = True
        missing = val_indices[~usable[val_indices]]
        counts = np.bincount(protocol.labels[train_usable], minlength=NUM_CLASSES).astype(np.float64)
        prior = np.log((counts + 1.0) / (counts.sum() + NUM_CLASSES)).astype(np.float32)
        oof_logits[missing] = prior
        completed[missing] = True
        all_metrics = classification_metrics(oof_logits[val_indices], protocol.labels[val_indices])
        usable_metrics = classification_metrics(usable_logits, protocol.labels[val_usable])
        print(
            f"  fold {fold}: all={all_metrics['accuracy']:.6f}, usable={usable_metrics['accuracy']:.6f}",
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "normalizer_mean": mean,
                "normalizer_std": std,
                "fold": fold,
                "use_ssl": use_ssl,
                "all_metrics": all_metrics,
                "usable_metrics": usable_metrics,
            },
            checkpoint_dir / f"fold_{fold}.pt",
        )
    if args.fold_only is not None:
        np.savez_compressed(
            output_dir / f"{teacher_name}_fold{args.fold_only}.npz",
            sample_ids=protocol.sample_ids[completed],
            labels=protocol.labels[completed],
            logits=oof_logits[completed],
            fold_id=protocol.fold_id[completed],
        )
        return
    payload = save_oof_artifact(
        output_dir,
        teacher_name,
        oof_logits,
        protocol,
        metadata={
            "cache": str(args.cache_dir.resolve()),
            "architecture": {
                "dim": args.dim,
                "depth": args.depth,
                "heads": args.heads,
                "stride": 4,
                "tokens": 160,
            },
            "ssl": {
                "enabled": use_ssl,
                "epochs": args.ssl_epochs if use_ssl else 0,
                "learning_rate": args.ssl_lr,
                "scope": "outer-fold training users only",
                "objective": "masked 4-step pooled signal reconstruction",
            },
            "classifier": {
                "epochs": args.cls_epochs,
                "learning_rate": args.cls_lr,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
            },
            "missing_imu": "outer-fold train prior with Laplace smoothing",
            "selection": "fixed epochs; outer validation evaluated once after training",
        },
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("scratch", "ssl", "both"), default="both")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--transformer-dropout", type=float, default=0.15)
    parser.add_argument("--classifier-dropout", type=float, default=0.30)
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--cls-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ssl-lr", type=float, default=3e-4)
    parser.add_argument("--cls-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=9017)
    parser.add_argument("--fold-only", type=int, choices=(0, 1, 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"scratch", "both"}:
        run_teacher(args, use_ssl=False)
    if args.stage in {"ssl", "both"}:
        run_teacher(args, use_ssl=True)


if __name__ == "__main__":
    main()
