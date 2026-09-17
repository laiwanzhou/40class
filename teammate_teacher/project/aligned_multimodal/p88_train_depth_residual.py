from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from audit_p87_sequence_decoder import (
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
    fit_transition_model,
)
from p88_depth_residual_model import P88DepthResidual


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs/p88_depth_features_holdout1_v1"
DEFAULT_METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_DECODER_AUDIT = PROJECT_DIR / "runs/p87s_fusion_holdout1_c7_structured12_v1/decoder_audit.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_depth_residual_holdout1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only a zero-initialised P88 Depth residual. Frozen P87 logits and "
            "embeddings are immutable inputs."
        )
    )
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--decoder-audit", type=Path, default=DEFAULT_DECODER_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--residual-scale", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.20)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.20)
    parser.add_argument("--delta-l2-weight", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260815)
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


def log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return values - np.logaddexp.reduce(values, axis=1, keepdims=True)


def nll(labels: np.ndarray, logits: np.ndarray) -> float:
    log_probability = log_softmax_numpy(logits)
    return float(-np.mean(log_probability[np.arange(len(labels)), labels]))


def rescue_harm(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, int]:
    baseline_correct = baseline == labels
    candidate_correct = candidate == labels
    return {
        "rescue": int(np.sum(~baseline_correct & candidate_correct)),
        "harm": int(np.sum(baseline_correct & ~candidate_correct)),
        "net": int(np.sum(candidate_correct) - np.sum(baseline_correct)),
        "changed": int(np.sum(baseline != candidate)),
    }


def class_weights(labels: np.ndarray, indices: np.ndarray, power: float) -> torch.Tensor:
    counts = np.bincount(labels[indices], minlength=40).astype(np.float64)
    weights = np.power(np.maximum(counts, 1.0), -float(power))
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def assemble_numpy_features(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.asarray(
        np.load(cache / "anchor_embedding_fp16.npy", mmap_mode="r"), dtype=np.float32
    )
    depth = np.asarray(
        np.load(cache / "depth_embedding_fp16.npy", mmap_mode="r"), dtype=np.float32
    )
    validity = np.asarray(
        np.load(cache / "depth_valid_statistics_fp16.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    features = np.concatenate(
        (depth, depth - anchor, depth * anchor, validity), axis=1
    ).astype(np.float32)
    return features, anchor


def make_decoder(
    labels: np.ndarray,
    train_indices: np.ndarray,
    metadata,
    decoder_audit_path: Path,
) -> tuple[Any, DecoderConfig]:
    audit = json.loads(decoder_audit_path.read_text(encoding="utf-8"))
    frozen = audit["selected_decoder_config"]
    config = DecoderConfig(
        gap_seconds=float(frozen["gap_seconds"]),
        transition_weight=float(frozen["transition_weight"]),
        trigram_backoff=float(frozen["trigram_backoff"]),
        beam_width=int(frozen["beam_width"]),
    )
    fit_sessions = build_sessions(
        train_indices, metadata, config.gap_seconds, grouping="known_user"
    )
    transition = fit_transition_model(
        labels, fit_sessions, num_classes=40, trigram_backoff=config.trigram_backoff
    )
    return transition, config


def evaluate(
    labels: np.ndarray,
    held_indices: np.ndarray,
    logits: np.ndarray,
    metadata,
    transition,
    decoder_config: DecoderConfig,
) -> dict[str, Any]:
    raw = logits.argmax(axis=1).astype(np.int64)
    sessions = build_sessions(
        held_indices,
        metadata,
        decoder_config.gap_seconds,
        grouping="anonymous_date",
    )
    decoded_all = decode_sessions(
        log_softmax_numpy(logits), sessions, transition, decoder_config
    )
    return {
        "raw": classification_metrics(labels[held_indices], raw[held_indices]),
        "decoded": classification_metrics(
            labels[held_indices], decoded_all[held_indices]
        ),
        "nll": nll(labels[held_indices], logits[held_indices]),
        "raw_prediction": raw,
        "decoded_prediction": decoded_all,
        "anonymous_sessions": len(sessions),
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    cache = args.feature_cache.resolve()
    rows = read_rows(cache / "rows.csv")
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    users = np.asarray([row["user_id"] for row in rows])
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    holdout_users = set(map(str, args.holdout_users))
    held_indices = np.flatnonzero(np.isin(users, sorted(holdout_users)))
    train_indices = np.flatnonzero(~np.isin(users, sorted(holdout_users)))
    if not len(held_indices) or set(users[held_indices]) != holdout_users:
        raise RuntimeError("requested P88 holdout users are missing")
    features, _ = assemble_numpy_features(cache)
    anchor_logits = np.asarray(
        np.load(cache / "anchor_logits.npy", mmap_mode="r"), dtype=np.float32
    )
    feature_mean = features[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = features[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-4)
    features = ((features - feature_mean) / feature_std).astype(np.float32)
    metadata = align_metadata(args.metadata, sample_ids)
    transition, decoder_config = make_decoder(
        labels, train_indices, metadata, args.decoder_audit.resolve()
    )
    baseline = evaluate(
        labels,
        held_indices,
        anchor_logits,
        metadata,
        transition,
        decoder_config,
    )
    dataset = TensorDataset(
        torch.from_numpy(features[train_indices]),
        torch.from_numpy(anchor_logits[train_indices]),
        torch.from_numpy(labels[train_indices]),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P88DepthResidual(
        input_width=features.shape[1],
        hidden_width=args.hidden_width,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)
    model.eval()
    with torch.inference_mode():
        initial_delta = model(torch.from_numpy(features[held_indices[:8]]).to(device))
    initial_maximum_delta = float(initial_delta.abs().max())
    if initial_maximum_delta != 0.0:
        raise RuntimeError("P88 residual does not preserve P87 at initialisation")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    weights = class_weights(labels, train_indices, args.class_weight_power).to(device)
    held_features = torch.from_numpy(features[held_indices]).to(device)
    history: list[dict[str, Any]] = []
    best_key: tuple[int, int, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_logits: np.ndarray | None = None
    best_evaluation: dict[str, Any] | None = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        samples = 0
        for batch_features, batch_anchor, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_anchor = batch_anchor.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            delta = model(batch_features)
            combined = batch_anchor + delta
            ce = F.cross_entropy(
                combined,
                batch_labels,
                weight=weights,
                label_smoothing=args.label_smoothing,
            )
            anchor_kl = F.kl_div(
                F.log_softmax(combined, dim=1),
                F.softmax(batch_anchor.detach(), dim=1),
                reduction="batchmean",
            )
            loss = (
                ce
                + args.anchor_kl_weight * anchor_kl
                + args.delta_l2_weight * delta.square().mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_size = len(batch_labels)
            loss_total += float(loss.detach()) * batch_size
            samples += batch_size
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            held_delta = model(held_features).float().cpu().numpy()
        candidate_logits = anchor_logits.copy()
        candidate_logits[held_indices] += held_delta
        values = evaluate(
            labels,
            held_indices,
            candidate_logits,
            metadata,
            transition,
            decoder_config,
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_total / max(samples, 1),
            "held_raw_accuracy": values["raw"]["accuracy"],
            "held_decoded_accuracy": values["decoded"]["accuracy"],
            "held_nll": values["nll"],
        }
        history.append(row)
        key = (
            int(values["decoded"]["correct"]),
            int(values["raw"]["correct"]),
            -float(values["nll"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_logits = candidate_logits.copy()
            best_evaluation = values
        print(json.dumps(row), flush=True)
    assert best_state is not None and best_logits is not None and best_evaluation is not None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.save(output / "held_logits.npy", best_logits[held_indices])
    np.savez_compressed(
        output / "held_predictions.npz",
        sample_ids=sample_ids[held_indices],
        labels=labels[held_indices],
        anchor_raw=baseline["raw_prediction"][held_indices],
        anchor_decoded=baseline["decoded_prediction"][held_indices],
        p88_raw=best_evaluation["raw_prediction"][held_indices],
        p88_decoded=best_evaluation["decoded_prediction"][held_indices],
    )
    torch.save(
        {
            "stage": "P88_zero_initialised_registered_depth_residual",
            "model_state": best_state,
            "model_config": {
                "input_width": features.shape[1],
                "hidden_width": args.hidden_width,
                "dropout": args.dropout,
                "residual_scale": args.residual_scale,
            },
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "holdout_users": sorted(holdout_users),
            "config": vars(args),
        },
        output / "depth_residual.pt",
    )
    baseline_raw = baseline["raw_prediction"][held_indices]
    baseline_decoded = baseline["decoded_prediction"][held_indices]
    p88_raw = best_evaluation["raw_prediction"][held_indices]
    p88_decoded = best_evaluation["decoded_prediction"][held_indices]
    best_epoch = max(
        history,
        key=lambda row: (
            row["held_decoded_accuracy"],
            row["held_raw_accuracy"],
            -row["held_nll"],
        ),
    )["epoch"]
    parameter_count = sum(value.numel() for value in best_state.values())
    summary = {
        "stage": "P88_zero_initialised_registered_depth_residual",
        "status": "complete",
        "protocol": (
            "P87 checkpoint and cached anchor outputs remain frozen. The residual is "
            "trained only on non-holdout labels. This development run selects its "
            "epoch on the declared holdout and must be independently retrained for H2."
        ),
        "holdout_users": sorted(holdout_users),
        "counts": {"train": len(train_indices), "holdout": len(held_indices)},
        "initial_maximum_delta": initial_maximum_delta,
        "baseline": {
            "raw": baseline["raw"],
            "decoded": baseline["decoded"],
            "nll": baseline["nll"],
        },
        "p88": {
            "raw": best_evaluation["raw"],
            "decoded": best_evaluation["decoded"],
            "nll": best_evaluation["nll"],
        },
        "raw_rescue_harm": rescue_harm(
            labels[held_indices], baseline_raw, p88_raw
        ),
        "decoded_rescue_harm": rescue_harm(
            labels[held_indices], baseline_decoded, p88_decoded
        ),
        "best_epoch": int(best_epoch),
        "residual_parameters": int(parameter_count),
        "residual_fp32_bytes": int(parameter_count * 4),
        "elapsed_seconds": time.perf_counter() - started,
        "decoder_config": {
            "gap_seconds": decoder_config.gap_seconds,
            "transition_weight": decoder_config.transition_weight,
            "trigram_backoff": decoder_config.trigram_backoff,
            "beam_width": decoder_config.beam_width,
        },
        "config": vars(args) | {
            "feature_cache": str(cache),
            "metadata": str(args.metadata.resolve()),
            "decoder_audit": str(args.decoder_audit.resolve()),
            "output_dir": str(output),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
