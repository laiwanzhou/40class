"""Deep set router over the complete subject-safe candidate posterior bank."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p160_candidate_set_transformer_single_seed_v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def token_features(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, 1e-6, 1.0)
    ordered = np.sort(values, axis=2)[:, :, ::-1]
    entropy = -np.sum(values * np.log(values), axis=2) / np.log(40.0)
    scalar = np.stack(
        (
            ordered[:, :, 0],
            ordered[:, :, 0] - ordered[:, :, 1],
            entropy,
            ordered[:, :, :3].sum(axis=2),
            ordered[:, :, :5].sum(axis=2),
        ),
        axis=2,
    )
    return np.concatenate((values, np.log(values), scalar), axis=2).astype(np.float32)


class CandidateSetRouter(nn.Module):
    def __init__(
        self,
        candidate_count: int,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.25,
        teacher_dropout: float = 0.15,
        residual_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.teacher_dropout = float(teacher_dropout)
        self.residual_weight = float(residual_weight)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.teacher_embedding = nn.Parameter(
            torch.zeros(1, candidate_count, hidden_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.reliability = nn.Linear(hidden_dim, 1)
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 40),
        )
        nn.init.normal_(self.teacher_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        candidate_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.projection(features) + self.teacher_embedding
        dropped = torch.zeros(
            tokens.shape[:2], dtype=torch.bool, device=tokens.device
        )
        if self.training and self.teacher_dropout > 0:
            dropped = torch.rand(tokens.shape[:2], device=tokens.device) < self.teacher_dropout
            dropped[:, 0] = False  # P89 safe anchor always remains available.
            tokens = tokens.masked_fill(dropped[..., None], 0.0)
        cls = self.cls_token.expand(len(tokens), -1, -1)
        padding = torch.cat(
            (
                torch.zeros((len(tokens), 1), dtype=torch.bool, device=tokens.device),
                dropped,
            ),
            dim=1,
        )
        encoded = self.encoder(
            torch.cat((cls, tokens), dim=1),
            src_key_padding_mask=padding,
        )
        candidate_encoded = encoded[:, 1:]
        reliability = self.reliability(candidate_encoded).squeeze(-1)
        reliability = reliability.masked_fill(dropped, -1e4)
        weights = F.softmax(reliability, dim=1)
        mixture = torch.sum(weights[..., None] * candidate_probability, dim=1)
        logits = torch.log(mixture.clamp_min(1e-8)) + self.residual_weight * self.residual(
            encoded[:, 0]
        )
        return logits, weights


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    weights = np.sqrt(len(labels) / np.maximum(40.0 * counts, 1.0))
    weights /= weights.mean()
    return weights.astype(np.float32)


def train_outer(
    features: np.ndarray,
    probability: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    seed_everything(seed)
    model = CandidateSetRouter(
        candidate_count=probability.shape[1],
        input_dim=features.shape[2],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        teacher_dropout=args.teacher_dropout,
        residual_weight=args.residual_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices.astype(np.int64))),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    steps = args.epochs * math.ceil(len(train_indices) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1), eta_min=args.learning_rate * 0.05
    )
    weights = torch.from_numpy(class_weights(labels[train_indices])).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        rows = 0
        for (indices,) in loader:
            index = indices.numpy()
            feature_value = torch.from_numpy(features[index]).to(device)
            probability_value = torch.from_numpy(probability[index]).to(device)
            target = torch.from_numpy(labels[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.float16
            ):
                logits, _ = model(feature_value, probability_value)
                loss = F.cross_entropy(logits, target, weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(index)
            rows += len(index)
        if epoch in (0, args.epochs - 1) or (epoch + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "epoch": epoch + 1,
                        "epochs": args.epochs,
                        "train_loss": loss_sum / max(rows, 1),
                    }
                ),
                flush=True,
            )
    model.eval()
    logits_parts = []
    weight_parts = []
    with torch.inference_mode():
        for start in range(0, len(held_indices), args.batch_size * 2):
            index = held_indices[start : start + args.batch_size * 2]
            feature_value = torch.from_numpy(features[index]).to(device)
            probability_value = torch.from_numpy(probability[index]).to(device)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.float16
            ):
                logits, candidate_weights = model(feature_value, probability_value)
            logits_parts.append(logits.float().cpu().numpy())
            weight_parts.append(candidate_weights.float().cpu().numpy())
    return np.concatenate(logits_parts), np.concatenate(weight_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--teacher-dropout", type=float, default=0.15)
    parser.add_argument("--residual-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=(16001,))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=True,
        hand_object_bank=True,
        vjepa_dense_bank=True,
        nonvisual_bank=True,
        hierarchical_bank=True,
        epic_bank=True,
        expanded_bank=True,
    )
    split_names = list(data)
    candidate_names = ["p89_safe", *next(iter(data.values())).candidates.keys()]
    sample_ids = np.concatenate([data[name].split.sample_ids for name in split_names])
    labels = np.concatenate([data[name].split.labels for name in split_names])
    probability = np.concatenate(
        [
            np.stack(
                [
                    data[name].split.safe_probability,
                    *data[name].candidates.values(),
                ],
                axis=1,
            )
            for name in split_names
        ]
    ).astype(np.float32)
    features = token_features(probability)
    split_id = np.concatenate(
        [np.full(len(data[name].split.labels), index) for index, name in enumerate(split_names)]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = np.zeros((len(labels), 40), dtype=np.float64)
    candidate_weights = np.zeros((len(labels), len(candidate_names)), dtype=np.float32)
    cohorts = {}
    for held_index, held_name in enumerate(split_names):
        train_indices = np.flatnonzero(split_id != held_index)
        held_indices = np.flatnonzero(split_id == held_index)
        members = []
        member_weights = []
        for seed in args.seeds:
            member_logits, weights = train_outer(
                features,
                probability,
                labels,
                train_indices,
                held_indices,
                int(seed + held_index * 1000),
                args,
                device,
            )
            members.append(member_logits)
            member_weights.append(weights)
        logits[held_indices] = np.mean(np.stack(members), axis=0)
        candidate_weights[held_indices] = np.mean(np.stack(member_weights), axis=0)
        cohorts[held_name] = classification_metrics(
            labels[held_indices], logits[held_indices].argmax(axis=1)
        )
    output_probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    output_probability /= output_probability.sum(axis=1, keepdims=True)
    report = {
        "stage": "P160_candidate_set_transformer_OOF",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "candidate_names": candidate_names,
            "candidate_count": len(candidate_names),
            "epochs_fixed": args.epochs,
            "seeds_fixed": list(args.seeds),
            "held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "architecture": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "teacher_dropout": args.teacher_dropout,
            "residual_weight": args.residual_weight,
        },
        "cohorts": cohorts,
        "aggregate": classification_metrics(labels, output_probability.argmax(axis=1)),
        "target_0.91_correct": int(np.ceil(0.91 * len(labels))),
        "gap_to_0.91_correct": int(np.ceil(0.91 * len(labels)))
        - int(np.sum(output_probability.argmax(axis=1) == labels)),
    }
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        sample_ids=sample_ids,
        labels=labels,
        split_id=split_id,
        logits=logits.astype(np.float32),
        probability=output_probability.astype(np.float32),
        candidate_weights=candidate_weights,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
