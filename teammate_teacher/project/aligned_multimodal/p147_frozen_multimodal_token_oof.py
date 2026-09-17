"""Frozen P96 + Skeleton + IMU multimodal head with strict subject-fold OOF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from audit_p87_sequence_decoder import classification_metrics
from audit_p91_confusion_feature_capability import (
    imu_feature_builders,
    skeleton_feature_builders,
)
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import (
    TokenHead,
    class_weights,
    seed_everything,
    soft_cross_entropy,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VISUAL = REPO / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1/features.npy"
OUTPUT = HERE / "runs/p147_frozen_multimodal_token_single_seed_v1"


class MultimodalHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 192,
        dropout: float = 0.20,
        view_dropout: float = 0.15,
        modality_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.modality_dropout = float(modality_dropout)
        self.visual = TokenHead(
            hidden_dim=hidden_dim,
            heads=6,
            layers=2,
            dropout=dropout,
            view_dropout=view_dropout,
            num_tokens=24,
        )
        self.visual.head = nn.Identity()
        self.skeleton = nn.Sequential(
            nn.LayerNorm(696),
            nn.Linear(696, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.imu = nn.Sequential(
            nn.LayerNorm(100),
            nn.Linear(100, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.modality_position = nn.Parameter(torch.zeros(1, 4, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=6,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 40),
        )
        nn.init.normal_(self.fusion_token, std=0.02)
        nn.init.normal_(self.modality_position, std=0.02)

    def forward(
        self,
        visual: torch.Tensor,
        skeleton: torch.Tensor,
        imu: torch.Tensor,
    ) -> torch.Tensor:
        modalities = torch.stack(
            (
                self.visual.encode(visual),
                self.skeleton(skeleton),
                self.imu(imu),
            ),
            dim=1,
        )
        if self.training and self.modality_dropout > 0:
            dropped = torch.rand(
                (len(visual), 3), device=visual.device
            ) < self.modality_dropout
            all_dropped = dropped.all(dim=1)
            if all_dropped.any():
                dropped[all_dropped, 0] = False
            modalities = modalities.masked_fill(dropped[..., None], 0.0)
        fusion = self.fusion_token.expand(len(visual), -1, -1)
        values = torch.cat((fusion, modalities), dim=1) + self.modality_position
        return self.head(self.fusion(values)[:, 0])


def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    protocol = load_protocol()
    visual = np.load(VISUAL, mmap_mode="r")
    dummy = SimpleNamespace(
        sample_ids=protocol.sample_ids.astype(str),
        analysis_indices=np.arange(len(protocol.sample_ids), dtype=np.int64),
    )
    skeleton = skeleton_feature_builders(dummy)[
        "skeleton_hand_head_early_late"
    ]().values
    imu = imu_feature_builders(dummy)["imu_arms_full_trial_statistics"]().values
    if visual.shape != (len(protocol.labels), 24, 1024):
        raise RuntimeError(f"visual feature shape changed: {visual.shape}")
    if skeleton.shape != (len(protocol.labels), 696):
        raise RuntimeError(f"skeleton feature shape changed: {skeleton.shape}")
    if imu.shape != (len(protocol.labels), 100):
        raise RuntimeError(f"IMU feature shape changed: {imu.shape}")
    return visual, skeleton, imu


def train_fold(
    visual: np.ndarray,
    skeleton: np.ndarray,
    imu: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    seed_everything(seed)
    model = MultimodalHead(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        view_dropout=args.view_dropout,
        modality_dropout=args.modality_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * math.ceil(len(train_indices) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.learning_rate * 0.05
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_indices.astype(np.int64)),
            torch.from_numpy(labels[train_indices].astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    weights = torch.from_numpy(class_weights(labels[train_indices])).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        row_count = 0
        for indices, batch_labels in loader:
            numpy_indices = indices.numpy()
            visual_value = torch.from_numpy(
                np.asarray(visual[numpy_indices], dtype=np.float32)
            ).to(device)
            skeleton_value = torch.from_numpy(skeleton[numpy_indices]).to(device)
            imu_value = torch.from_numpy(imu[numpy_indices]).to(device)
            batch_labels = batch_labels.to(device)
            target = F.one_hot(batch_labels, num_classes=40).float()
            if args.mixup_alpha > 0 and len(indices) > 1:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                order = torch.randperm(len(indices), device=device)
                visual_value = lam * visual_value + (1.0 - lam) * visual_value[order]
                skeleton_value = (
                    lam * skeleton_value + (1.0 - lam) * skeleton_value[order]
                )
                imu_value = lam * imu_value + (1.0 - lam) * imu_value[order]
                target = lam * target + (1.0 - lam) * target[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.float16
            ):
                logits = model(visual_value, skeleton_value, imu_value)
                loss = soft_cross_entropy(logits, target, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += float(loss.detach()) * len(indices)
            row_count += len(indices)
        if epoch in (0, args.epochs - 1) or (epoch + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "epoch": epoch + 1,
                        "epochs": args.epochs,
                        "train_loss": epoch_loss / max(row_count, 1),
                    }
                ),
                flush=True,
            )
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(held_indices), args.batch_size * 2):
            indices = held_indices[start : start + args.batch_size * 2]
            visual_value = torch.from_numpy(
                np.asarray(visual[indices], dtype=np.float32)
            ).to(device)
            skeleton_value = torch.from_numpy(skeleton[indices]).to(device)
            imu_value = torch.from_numpy(imu[indices]).to(device)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.float16
            ):
                output.append(
                    model(visual_value, skeleton_value, imu_value)
                    .float()
                    .cpu()
                    .numpy()
                )
    return np.concatenate(output).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--view-dropout", type=float, default=0.15)
    parser.add_argument("--modality-dropout", type=float, default=0.10)
    parser.add_argument("--mixup-alpha", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=(14701,))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    visual, skeleton, imu = load_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = np.zeros((len(protocol.labels), 40), dtype=np.float64)
    folds = []
    for fold in range(3):
        train_indices = protocol.train_indices(fold)
        held_indices = protocol.val_indices(fold)
        members = [
            train_fold(
                visual,
                skeleton,
                imu,
                protocol.labels,
                train_indices,
                held_indices,
                int(seed + fold * 1000),
                args,
                device,
            )
            for seed in args.seeds
        ]
        fold_logits = np.mean(np.stack(members), axis=0)
        logits[held_indices] = fold_logits
        folds.append(
            {
                "fold": fold,
                "train_rows": int(len(train_indices)),
                "held_rows": int(len(held_indices)),
                "metrics": classification_metrics(
                    protocol.labels[held_indices], fold_logits.argmax(axis=1)
                ),
            }
        )
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    report = {
        "stage": "P147_frozen_multimodal_token_OOF",
        "status": "complete_strict_subject_fold_oof",
        "protocol": {
            "all_backbones_frozen": True,
            "epochs_fixed": args.epochs,
            "seeds_fixed": list(args.seeds),
            "held_fold_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "architecture": {
            "visual_tokens": "P96 dense24",
            "skeleton": "hand_head_early_late 696D",
            "imu": "arms_full_trial_statistics 100D",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "view_dropout": args.view_dropout,
            "modality_dropout": args.modality_dropout,
            "mixup_alpha": args.mixup_alpha,
        },
        "folds": folds,
        "metrics": classification_metrics(protocol.labels, probability.argmax(axis=1)),
    }
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        sample_ids=protocol.sample_ids,
        labels=protocol.labels,
        users=protocol.users,
        fold_id=protocol.fold_id,
        logits=logits.astype(np.float32),
        probability=probability.astype(np.float32),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
