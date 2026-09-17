from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from audit_p87_sequence_decoder import align_metadata
from p88_spatial_residual_model import P88Layer3SpatialResidual
from p88_train_depth_residual import (
    DEFAULT_DECODER_AUDIT,
    DEFAULT_METADATA,
    class_weights,
    evaluate,
    make_decoder,
    read_rows,
    rescue_harm,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DESCRIPTOR = PROJECT_DIR / "runs/p88_ir_layer3_descriptors_holdout1_v1"
DEFAULT_ANCHOR = PROJECT_DIR / "runs/p88_depth_features_holdout1_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_ir_layer3_residual_h1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a zero-initialised P88 layer3 spatial residual on frozen P87."
    )
    parser.add_argument("--descriptor-cache", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--anchor-cache", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--decoder-audit", type=Path, default=DEFAULT_DECODER_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.20)
    parser.add_argument("--auxiliary-weight", type=float, default=0.30)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.20)
    parser.add_argument("--delta-l2-weight", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    descriptor_dir = args.descriptor_cache.resolve()
    anchor_dir = args.anchor_cache.resolve()
    rows = read_rows(descriptor_dir / "rows.csv")
    anchor_rows = read_rows(anchor_dir / "rows.csv")
    if [row["sample_id"] for row in rows] != [row["sample_id"] for row in anchor_rows]:
        raise RuntimeError("P88 spatial and anchor row orders differ")
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    users = np.asarray([row["user_id"] for row in rows])
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    holdout_users = set(map(str, args.holdout_users))
    held = np.flatnonzero(np.isin(users, sorted(holdout_users)))
    train = np.flatnonzero(~np.isin(users, sorted(holdout_users)))
    descriptors = np.asarray(
        np.load(descriptor_dir / "layer3_descriptors_fp16.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    quality = np.asarray(
        np.load(descriptor_dir / "clip_quality_fp16.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    anchor_logits = np.asarray(
        np.load(anchor_dir / "anchor_logits.npy", mmap_mode="r"), dtype=np.float32
    )
    # Fold-pure standardisation for each temporal statistic/spatial contrast/channel.
    mean = descriptors[train].mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = descriptors[train].std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    descriptors = ((descriptors - mean[None, None, None]) / std[None, None, None]).astype(np.float32)
    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    transition, decoder_config = make_decoder(
        labels, train, metadata, args.decoder_audit.resolve()
    )
    baseline = evaluate(
        labels, held, anchor_logits, metadata, transition, decoder_config
    )
    dataset = TensorDataset(
        torch.from_numpy(descriptors[train]),
        torch.from_numpy(quality[train]),
        torch.from_numpy(anchor_logits[train]),
        torch.from_numpy(labels[train]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P88Layer3SpatialResidual(
        width=args.width,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)
    model.eval()
    with torch.inference_mode():
        initial_delta, _ = model(
            torch.from_numpy(descriptors[held[:4]]).to(device),
            torch.from_numpy(quality[held[:4]]).to(device),
        )
    if float(initial_delta.abs().max()) != 0.0:
        raise RuntimeError("P88 spatial residual is not an exact P87 anchor")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    weights = class_weights(labels, train, args.class_weight_power).to(device)
    held_descriptor = torch.from_numpy(descriptors[held]).to(device)
    held_quality = torch.from_numpy(quality[held]).to(device)
    history: list[dict[str, float | int]] = []
    best_key = None
    best_state = None
    best_values = None
    best_logits = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; samples = 0
        for batch_descriptor, batch_quality, batch_anchor, batch_label in loader:
            batch_descriptor = batch_descriptor.to(device, non_blocking=True)
            batch_quality = batch_quality.to(device, non_blocking=True)
            batch_anchor = batch_anchor.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            delta, auxiliary = model(batch_descriptor, batch_quality)
            combined = batch_anchor + delta
            main_loss = F.cross_entropy(
                combined, batch_label, weight=weights, label_smoothing=args.label_smoothing
            )
            auxiliary_loss = F.cross_entropy(
                auxiliary, batch_label, weight=weights, label_smoothing=args.label_smoothing
            )
            anchor_kl = F.kl_div(
                F.log_softmax(combined, dim=1),
                F.softmax(batch_anchor.detach(), dim=1),
                reduction="batchmean",
            )
            loss = (
                main_loss
                + args.auxiliary_weight * auxiliary_loss
                + args.anchor_kl_weight * anchor_kl
                + args.delta_l2_weight * delta.square().mean()
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_label); samples += len(batch_label)
        scheduler.step(); model.eval()
        with torch.inference_mode():
            held_delta, held_auxiliary = model(held_descriptor, held_quality)
        candidate_logits = anchor_logits.copy()
        candidate_logits[held] += held_delta.float().cpu().numpy()
        values = evaluate(
            labels, held, candidate_logits, metadata, transition, decoder_config
        )
        auxiliary_accuracy = float(
            np.mean(held_auxiliary.argmax(dim=1).cpu().numpy() == labels[held])
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(samples, 1),
            "held_auxiliary_accuracy": auxiliary_accuracy,
            "held_raw_accuracy": values["raw"]["accuracy"],
            "held_decoded_accuracy": values["decoded"]["accuracy"],
            "held_nll": values["nll"],
        }
        history.append(row); print(json.dumps(row), flush=True)
        key = (values["decoded"]["correct"], values["raw"]["correct"], -values["nll"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_values = values
            best_logits = candidate_logits.copy()
    assert best_state is not None and best_values is not None and best_logits is not None
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    torch.save(
        {
            "stage": "P88_zero_initialised_IR_layer3_spatial_residual",
            "model_state": best_state,
            "model_config": {"width": args.width, "dropout": args.dropout, "residual_scale": args.residual_scale},
            "descriptor_mean": mean,
            "descriptor_std": std,
            "holdout_users": sorted(holdout_users),
        },
        output / "spatial_residual.pt",
    )
    parameter_count = sum(value.numel() for value in best_state.values())
    base_raw = baseline["raw_prediction"][held]; base_decoded = baseline["decoded_prediction"][held]
    p88_raw = best_values["raw_prediction"][held]; p88_decoded = best_values["decoded_prediction"][held]
    best_epoch = max(history, key=lambda row: (row["held_decoded_accuracy"], row["held_raw_accuracy"], -row["held_nll"]))["epoch"]
    summary = {
        "stage": "P88_zero_initialised_IR_layer3_spatial_residual",
        "status": "complete",
        "protocol": "Frozen P87 layer3 descriptors; residual trained only on non-holdout labels. H1 selects the epoch and H2 requires a fresh cache/refit.",
        "holdout_users": sorted(holdout_users),
        "counts": {"train": len(train), "holdout": len(held)},
        "baseline": {"raw": baseline["raw"], "decoded": baseline["decoded"], "nll": baseline["nll"]},
        "p88": {"raw": best_values["raw"], "decoded": best_values["decoded"], "nll": best_values["nll"]},
        "raw_rescue_harm": rescue_harm(labels[held], base_raw, p88_raw),
        "decoded_rescue_harm": rescue_harm(labels[held], base_decoded, p88_decoded),
        "best_epoch": int(best_epoch),
        "residual_parameters": int(parameter_count),
        "residual_fp32_bytes": int(parameter_count * 4),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            "width": args.width, "dropout": args.dropout, "residual_scale": args.residual_scale,
            "auxiliary_weight": args.auxiliary_weight, "anchor_kl_weight": args.anchor_kl_weight,
            "learning_rate": args.learning_rate, "epochs": args.epochs,
            "descriptor_cache": str(descriptor_dir), "anchor_cache": str(anchor_dir),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
