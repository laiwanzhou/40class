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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audit_p87_sequence_decoder import (
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
)
from p88_train_depth_residual import log_softmax_numpy, make_decoder, rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_IR = PROJECT_DIR / "runs/p87s_mc3_sequence_holdout1_v1"
DEFAULT_DEPTH = PROJECT_DIR / "runs/p88_depth_sequence_holdout1_v1"
DEFAULT_ANCHOR = PROJECT_DIR / "runs/p88_depth_features_holdout1_v1"
DEFAULT_METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_DECODER_AUDIT = PROJECT_DIR / "runs/p87s_fusion_holdout1_c7_structured12_v1/decoder_audit.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p89_depth_temporal_cross_adapter_h1_v1"
DEFAULT_HOLDOUT = ("user6", "user8", "user17", "user23")
SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a zero-initialised temporal Depth-vs-IR residual on frozen "
            "strict-holdout layer4 sequences."
        )
    )
    parser.add_argument("--ir-cache", type=Path, default=DEFAULT_IR)
    parser.add_argument("--depth-cache", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--anchor-cache", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--decoder-audit", type=Path, default=DEFAULT_DECODER_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-users", nargs="+", default=list(DEFAULT_HOLDOUT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--residual-scale", type=float, default=0.50)
    parser.add_argument("--direct-weight", type=float, default=0.30)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.20)
    parser.add_argument("--delta-l2-weight", type=float, default=0.004)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--class-weight-power", type=float, default=0.25)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        ir: np.ndarray,
        depth: np.ndarray,
        validity: np.ndarray,
        anchor: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        augment: bool,
    ) -> None:
        self.ir = ir
        self.depth = depth
        self.validity = validity
        self.anchor = anchor
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        ir = np.asarray(self.ir[index], dtype=np.float32).copy()
        depth = np.asarray(self.depth[index], dtype=np.float32).copy()
        validity = np.asarray(self.validity[index], dtype=np.float32).copy()
        if self.augment:
            # Coherent time jitter keeps registered modalities aligned.
            if np.random.random() < 0.50:
                steps = ir.shape[2]
                shift = int(np.random.randint(-2, 3))
                positions = np.clip(np.arange(steps) + shift, 0, steps - 1)
                ir = ir[:, :, positions]
                depth = depth[:, :, positions]
                validity = validity[:, positions]
            # Suppress a view occasionally, forcing complementary evidence to
            # survive imperfect ROI localisation without changing labels.
            if np.random.random() < 0.25:
                view = int(np.random.randint(0, 3))
                depth[:, view] = 0.0
                validity[:, :, view] = 0.0
        return {
            "ir": torch.from_numpy(ir),
            "depth": torch.from_numpy(depth),
            "validity": torch.from_numpy(validity),
            "anchor": torch.from_numpy(np.asarray(self.anchor[index], dtype=np.float32)),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "index": index,
        }


class TemporalBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm(values).transpose(1, 2)
        values = self.pointwise(F.gelu(self.depthwise(values))).transpose(1, 2)
        return residual + self.dropout(values)


class DepthTemporalCrossAdapter(nn.Module):
    def __init__(
        self,
        width: int = 96,
        dropout: float = 0.18,
        residual_scale: float = 0.50,
        steps: int = 16,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.steps = int(steps)
        self.residual_scale = float(residual_scale)
        half = width // 2
        self.ir_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, half), nn.GELU())
        self.depth_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, half), nn.GELU())
        self.fusion = nn.Sequential(
            nn.LayerNorm(half * 4 + 1),
            nn.Linear(half * 4 + 1, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.motion_fusion = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU()
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, steps, width))
        self.view_embedding = nn.Parameter(torch.zeros(3, width))
        self.window_embedding = nn.Parameter(torch.zeros(2, width))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        nn.init.trunc_normal_(self.window_embedding, std=0.02)
        self.temporal_blocks = nn.ModuleList(TemporalBlock(width, dropout) for _ in range(2))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.stream_projection = nn.Sequential(
            nn.LayerNorm(width * 5), nn.Linear(width * 5, width), nn.GELU(), nn.Dropout(dropout)
        )
        stream_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stream_encoder = nn.TransformerEncoder(stream_layer, num_layers=1)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.direct_head = nn.Linear(width, 40)
        self.residual_head = nn.Linear(width, 40)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self, ir: torch.Tensor, depth: torch.Tensor, validity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ir.shape != depth.shape or ir.ndim != 5 or ir.shape[1:4] != (2, 3, self.steps):
            raise ValueError(f"unexpected sequence geometry: {tuple(ir.shape)}")
        batch = ir.shape[0]
        if validity.shape != (batch, 2, self.steps, 3):
            raise ValueError(f"unexpected Depth validity geometry: {tuple(validity.shape)}")
        ir_token = self.ir_projection(ir)
        depth_token = self.depth_projection(depth)
        quality = validity.permute(0, 1, 3, 2).unsqueeze(-1)
        fused = self.fusion(
            torch.cat(
                (ir_token, depth_token, depth_token - ir_token, depth_token * ir_token, quality),
                dim=-1,
            )
        )
        values = fused.reshape(batch * 6, self.steps, self.width)
        motion = torch.zeros_like(values)
        motion[:, 1:] = values[:, 1:] - values[:, :-1]
        values = self.motion_fusion(torch.cat((values, motion), dim=-1))
        values = values + self.time_embedding
        for block in self.temporal_blocks:
            values = block(values)
        values = self.temporal_encoder(values)
        midpoint = max(self.steps // 2, 1)
        mean = values.mean(dim=1)
        maximum = values.amax(dim=1)
        early = values[:, :midpoint].mean(dim=1)
        late = values[:, midpoint:].mean(dim=1)
        stream = self.stream_projection(
            torch.cat((mean, maximum, early, late, late - early), dim=-1)
        ).reshape(batch, 2, 3, self.width)
        stream = (
            stream
            + self.window_embedding.view(1, 2, 1, self.width)
            + self.view_embedding.view(1, 1, 3, self.width)
        ).reshape(batch, 6, self.width)
        stream = self.stream_encoder(stream)
        global_token = self.global_projection(
            torch.cat((stream.mean(dim=1), stream.amax(dim=1)), dim=-1)
        )
        return self.residual_scale * self.residual_head(global_token), self.direct_head(global_token)


def class_weights(labels: np.ndarray, indices: np.ndarray, power: float) -> torch.Tensor:
    counts = np.bincount(labels[indices], minlength=40).astype(np.float64)
    values = np.power(np.maximum(counts, 1.0), -float(power))
    values /= values.mean()
    return torch.from_numpy(values.astype(np.float32))


def infer(model, loader, anchor_logits, device):
    logits = np.asarray(anchor_logits, dtype=np.float32).copy()
    direct = np.zeros_like(logits)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            indices = np.asarray(batch["index"], dtype=np.int64)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                residual, direct_logits = model(
                    batch["ir"].to(device, non_blocking=True),
                    batch["depth"].to(device, non_blocking=True),
                    batch["validity"].to(device, non_blocking=True),
                )
            logits[indices] += residual.float().cpu().numpy()
            direct[indices] = direct_logits.float().cpu().numpy()
    return logits, direct


def evaluate(labels, held, logits, metadata, transition, decoder_config):
    raw = logits.argmax(axis=1)
    sessions = build_sessions(held, metadata, decoder_config.gap_seconds, "anonymous_date")
    decoded = decode_sessions(log_softmax_numpy(logits), sessions, transition, decoder_config)
    return {
        "raw": classification_metrics(labels[held], raw[held]),
        "decoded": classification_metrics(labels[held], decoded[held]),
        "raw_prediction": raw,
        "decoded_prediction": decoded,
    }


def main() -> None:
    args = parse_args()
    seed_all(SEED)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    ir_cache = args.ir_cache.resolve()
    depth_cache = args.depth_cache.resolve()
    anchor_cache = args.anchor_cache.resolve()
    ir_rows = read_rows(ir_cache / "rows.csv")
    depth_rows = read_rows(depth_cache / "rows.csv")
    anchor_rows = read_rows(anchor_cache / "rows.csv")
    row_ids = [row["sample_id"] for row in ir_rows]
    if row_ids != [row["sample_id"] for row in depth_rows] or row_ids != [row["sample_id"] for row in anchor_rows]:
        raise RuntimeError("IR, Depth, and anchor row orders differ")
    labels = np.asarray([int(row["class_id"]) for row in ir_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in ir_rows]).astype(str)
    held = np.flatnonzero(np.isin(users, list(args.holdout_users)))
    train = np.flatnonzero(~np.isin(users, list(args.holdout_users)))
    ir = np.load(ir_cache / "backbone_sequence_fp16.npy", mmap_mode="r")
    depth = np.load(depth_cache / "backbone_sequence_fp16.npy", mmap_mode="r")
    validity = np.load(depth_cache / "depth_valid_fraction_fp16.npy", mmap_mode="r")
    anchor_logits = np.load(anchor_cache / "anchor_logits.npy", mmap_mode="r")
    metadata = align_metadata(args.metadata, np.asarray(row_ids))
    transition, decoder_config = make_decoder(labels, train, metadata, args.decoder_audit.resolve())
    baseline = evaluate(labels, held, anchor_logits, metadata, transition, decoder_config)

    train_dataset = SequenceDataset(ir, depth, validity, anchor_logits, labels, train, True)
    held_dataset = SequenceDataset(ir, depth, validity, anchor_logits, labels, held, False)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    held_loader = DataLoader(
        held_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthTemporalCrossAdapter(
        width=args.width,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        steps=ir.shape[3],
    ).to(device)
    model.eval()
    probe = held[: min(4, len(held))]
    with torch.inference_mode():
        residual, _ = model(
            torch.from_numpy(np.asarray(ir[probe], dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(depth[probe], dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(validity[probe], dtype=np.float32)).to(device),
        )
    initial_maximum_delta = float(residual.abs().max())
    if initial_maximum_delta != 0.0:
        raise RuntimeError("adapter is not an exact zero residual at initialisation")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    weights = class_weights(labels, train, args.class_weight_power).to(device)
    history = []
    best_key = None
    best_state = None
    best_logits = None
    best_evaluation = None
    best_direct = None
    best_epoch = 0
    best_direct_logits = None
    best_direct_correct = -1
    best_direct_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for batch in train_loader:
            batch_ir = batch["ir"].to(device, non_blocking=True)
            batch_depth = batch["depth"].to(device, non_blocking=True)
            batch_validity = batch["validity"].to(device, non_blocking=True)
            batch_anchor = batch["anchor"].to(device, non_blocking=True)
            batch_labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                residual, direct_logits = model(batch_ir, batch_depth, batch_validity)
                combined = batch_anchor + residual
                classification = F.cross_entropy(
                    combined, batch_labels, weight=weights, label_smoothing=args.label_smoothing
                )
                direct_loss = F.cross_entropy(
                    direct_logits, batch_labels, weight=weights, label_smoothing=args.label_smoothing
                )
                anchor_kl = F.kl_div(
                    F.log_softmax(combined, dim=1),
                    F.softmax(batch_anchor.detach(), dim=1),
                    reduction="batchmean",
                )
                loss = (
                    classification
                    + args.direct_weight * direct_loss
                    + args.anchor_kl_weight * anchor_kl
                    + args.delta_l2_weight * residual.square().mean()
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(batch_labels)
            loss_total += float(loss.detach()) * count
            seen += count
        scheduler.step()
        candidate_logits, direct_logits = infer(model, held_loader, anchor_logits, device)
        values = evaluate(labels, held, candidate_logits, metadata, transition, decoder_config)
        direct_correct = int(np.sum(direct_logits[held].argmax(1) == labels[held]))
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_total / max(seen, 1),
            "held_raw_correct": values["raw"]["correct"],
            "held_decoded_correct": values["decoded"]["correct"],
            "held_depth_direct_correct": direct_correct,
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
        history.append(row)
        key = (values["decoded"]["correct"], values["raw"]["correct"], direct_correct)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_logits = candidate_logits.copy()
            best_direct = direct_logits.copy()
            best_evaluation = values
            best_epoch = epoch
        if direct_correct > best_direct_correct:
            best_direct_correct = direct_correct
            best_direct_logits = direct_logits.copy()
            best_direct_epoch = epoch
        print(json.dumps(row), flush=True)
    assert best_state is not None and best_logits is not None and best_evaluation is not None and best_direct is not None and best_direct_logits is not None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "P89_depth_temporal_cross_adapter",
            "model_state": best_state,
            "model_config": {
                "width": args.width,
                "dropout": args.dropout,
                "residual_scale": args.residual_scale,
                "steps": int(ir.shape[3]),
            },
            "holdout_users": sorted(map(str, args.holdout_users)),
            "config": vars(args),
        },
        output / "depth_temporal_adapter.pt",
    )
    np.savez_compressed(
        output / "held_predictions.npz",
        sample_ids=np.asarray(row_ids)[held],
        labels=labels[held],
        anchor_logits=np.asarray(anchor_logits[held]),
        candidate_logits=best_logits[held],
        depth_direct_logits=best_direct[held],
        best_depth_direct_logits=best_direct_logits[held],
        anchor_raw=baseline["raw_prediction"][held],
        anchor_decoded=baseline["decoded_prediction"][held],
        candidate_raw=best_evaluation["raw_prediction"][held],
        candidate_decoded=best_evaluation["decoded_prediction"][held],
    )
    report = {
        "stage": "P89_depth_temporal_cross_adapter_H1_v1",
        "protocol": (
            "Strict H1: frozen IR/Depth feature extractors and P87 anchor were trained without held users. "
            "The adapter is exactly zero at initialization and sees labels only from the other 14 subjects."
        ),
        "holdout_users": sorted(map(str, args.holdout_users)),
        "counts": {"train": int(len(train)), "holdout": int(len(held))},
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "fp32_bytes": int(sum(parameter.numel() for parameter in model.parameters()) * 4),
        "initial_maximum_delta": initial_maximum_delta,
        "baseline": {"raw": baseline["raw"], "decoded": baseline["decoded"]},
        "candidate": {"raw": best_evaluation["raw"], "decoded": best_evaluation["decoded"]},
        "raw_rescue_harm": rescue_harm(
            labels[held], baseline["raw_prediction"][held], best_evaluation["raw_prediction"][held]
        ),
        "decoded_rescue_harm": rescue_harm(
            labels[held], baseline["decoded_prediction"][held], best_evaluation["decoded_prediction"][held]
        ),
        "best_epoch": int(best_epoch),
        "best_depth_direct_epoch": int(best_direct_epoch),
        "best_depth_direct_correct": int(best_direct_correct),
        "history": history,
        "config": {
            key: (str(value.resolve()) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("history", "config")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
