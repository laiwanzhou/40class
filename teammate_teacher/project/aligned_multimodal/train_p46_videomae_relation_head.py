from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from p46_protocol import HARD_CLASS_IDS
from train_p46_videomae_head import fit_temperature, metrics, p12_prediction, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs/p46_videomae_temporal_v2/complete_features.npz"
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_relation_head_v1"


@dataclass(frozen=True)
class Config:
    name: str
    width: int
    layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    class_weight: str


CONFIGS = (
    Config("compact_1l", 96, 1, 0.25, 1.0e-3, 0.05, "none"),
    Config("base_1l", 128, 1, 0.35, 7.0e-4, 0.05, "none"),
    Config("base_2l", 128, 2, 0.30, 7.0e-4, 0.05, "none"),
    Config("balanced_1l", 128, 1, 0.35, 7.0e-4, 0.05, "sqrt"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a compact unified attention head over VideoMAE's aligned "
            "3-view x 8-time tokens with held-user model selection."
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RelationHead(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(768)
        self.projection = nn.Linear(768, config.width)
        self.time_embedding = nn.Parameter(torch.zeros(8, config.width))
        self.view_embedding = nn.Parameter(torch.zeros(3, config.width))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.width))
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=4,
            dim_feedforward=2 * config.width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.layers, enable_nested_tensor=False
        )
        self.global_projection = nn.Linear(768, config.width)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * config.width),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.width, 21),
        )
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, temporal: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        batch = temporal.shape[0]
        tokens = self.projection(self.input_norm(temporal))
        tokens = tokens + self.time_embedding[None, None, :, :]
        tokens = tokens + self.view_embedding[None, :, None, :]
        # Time-major order puts the three synchronized views next to each other.
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch, 24, -1)
        cls = self.cls_token.expand(batch, -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1))[:, 0]
        global_feature = self.global_projection(
            self.input_norm(pooled).mean(dim=1)
        )
        return self.classifier(torch.cat((encoded, global_feature), dim=1))


def class_weights(labels: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=21).astype(np.float64)
    weights = np.zeros(21, dtype=np.float32)
    present = counts > 0
    weights[present] = np.sqrt(counts[present].mean() / counts[present]).astype(np.float32)
    weights[present] /= weights[present].mean()
    return torch.from_numpy(weights).to(device)


@torch.inference_mode()
def infer(
    model: RelationHead,
    temporal: torch.Tensor,
    pooled: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    dataset = TensorDataset(
        temporal[torch.from_numpy(indices)], pooled[torch.from_numpy(indices)]
    )
    for temporal_batch, pooled_batch in DataLoader(dataset, batch_size=batch_size):
        logits = model(temporal_batch.to(device), pooled_batch.to(device))
        result.append(logits.float().cpu().numpy())
    return np.concatenate(result)


def train_one(
    config: Config,
    temporal: torch.Tensor,
    pooled: torch.Tensor,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    held_indices: np.ndarray | None,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[RelationHead, int, dict[str, float]]:
    seed_everything(seed)
    model = RelationHead(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = max(1, math.ceil(len(fit_indices) / batch_size) * epochs)
    warmup_steps = max(1, int(0.1 * total_steps))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(labels[fit_indices], config.class_weight, device),
        label_smoothing=0.05,
    )
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        temporal[torch.from_numpy(fit_indices)],
        pooled[torch.from_numpy(fit_indices)],
        torch.from_numpy(labels[fit_indices]),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_accuracy = -1.0
    best_loss = float("inf")
    stale = 0
    last_train_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for temporal_batch, pooled_batch, label_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(temporal_batch.to(device), pooled_batch.to(device))
            loss = criterion(logits, label_batch.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        last_train_loss = float(np.mean(losses))
        if held_indices is None:
            continue
        held_logits = infer(model, temporal, pooled, held_indices, batch_size, device)
        held_labels = labels[held_indices]
        held_loss = float(
            nn.functional.cross_entropy(
                torch.from_numpy(held_logits), torch.from_numpy(held_labels)
            )
        )
        held_accuracy = float((held_logits.argmax(axis=1) == held_labels).mean())
        improved = held_accuracy > best_accuracy or (
            held_accuracy == best_accuracy and held_loss < best_loss
        )
        if improved:
            best_accuracy = held_accuracy
            best_loss = held_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if held_indices is not None:
        model.load_state_dict(best_state)
    return model, best_epoch if held_indices is not None else epochs, {
        "held_accuracy": best_accuracy,
        "held_loss": best_loss,
        "train_loss": last_train_loss,
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    with np.load(args.features.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        source_ids = np.asarray(data["source_ids"]).astype(str)
        users = np.asarray(data["users"]).astype(str)
        class_labels = np.asarray(data["labels"], dtype=np.int64)
        temporal_array = np.asarray(data["temporal_features"], dtype=np.float32)
        pooled_array = np.asarray(data["features"], dtype=np.float32)
    if temporal_array.shape != (1384, 3, 8, 768):
        raise RuntimeError(f"temporal feature contract changed: {temporal_array.shape}")
    temporal_array /= np.maximum(
        np.linalg.norm(temporal_array, axis=-1, keepdims=True), 1e-8
    )
    pooled_array /= np.maximum(
        np.linalg.norm(pooled_array, axis=-1, keepdims=True), 1e-8
    )
    temporal = torch.from_numpy(temporal_array)
    pooled = torch.from_numpy(pooled_array)
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    labels = np.asarray([class_to_index[value] for value in class_labels], dtype=np.int64)
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["detail_selected"] == "1"
        }
    split = np.asarray([manifest[value]["p46_split"] for value in sample_ids])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    folds = list(
        StratifiedGroupKFold(
            n_splits=args.cv_splits, shuffle=True, random_state=args.seed
        ).split(train_indices, labels[train_indices], groups=users[train_indices])
    )
    fold_vector = np.full(len(train_indices), -1, dtype=np.int64)
    cv_rows: list[dict[str, Any]] = []
    oof_by_name: dict[str, np.ndarray] = {}
    epochs_by_name: dict[str, list[int]] = {}
    for config_number, config in enumerate(CONFIGS, start=1):
        oof_logits = np.full((len(train_indices), 21), np.nan, dtype=np.float32)
        best_epochs: list[int] = []
        for fold, (fit_local, held_local) in enumerate(folds):
            fit_indices = train_indices[fit_local]
            held_indices = train_indices[held_local]
            fold_vector[held_local] = fold
            model, best_epoch, diagnostic = train_one(
                config,
                temporal,
                pooled,
                labels,
                fit_indices,
                held_indices,
                args.max_epochs,
                args.patience,
                args.batch_size,
                device,
                args.seed + 1000 * config_number + fold,
            )
            oof_logits[held_local] = infer(
                model, temporal, pooled, held_indices, args.batch_size, device
            )
            best_epochs.append(best_epoch)
            print(
                f"config={config.name} fold={fold} epoch={best_epoch} "
                f"held={100*diagnostic['held_accuracy']:.2f}%",
                flush=True,
            )
        result = metrics(labels[train_indices], oof_logits.argmax(axis=1))
        cv_rows.append(
            {
                **asdict(config),
                "median_best_epoch": int(np.median(best_epochs)),
                **result,
            }
        )
        oof_by_name[config.name] = oof_logits
        epochs_by_name[config.name] = best_epochs
        print(
            f"[{config_number}/{len(CONFIGS)}] {config.name} OOF "
            f"acc={100*float(result['accuracy']):.2f}% "
            f"macro={100*float(result['macro_f1']):.2f}% epochs={best_epochs}",
            flush=True,
        )
    selected_row = max(
        cv_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["macro_f1"]),
            -int(row["layers"]),
            -int(row["width"]),
        ),
    )
    selected = next(config for config in CONFIGS if config.name == selected_row["name"])
    write_csv(output / "group_cv_results.csv", cv_rows)
    selected_oof = oof_by_name[selected.name]
    temperature = fit_temperature(selected_oof, labels[train_indices])
    final_epochs = max(1, int(np.median(epochs_by_name[selected.name])))
    print(
        f"Selected on training users only: {selected.name}, final_epochs={final_epochs}, "
        f"temperature={temperature:.4f}",
        flush=True,
    )
    final_model, _, _ = train_one(
        selected,
        temporal,
        pooled,
        labels,
        train_indices,
        None,
        final_epochs,
        args.patience,
        args.batch_size,
        device,
        args.seed + 99999,
    )
    val_logits = infer(
        final_model, temporal, pooled, val_indices, args.batch_size, device
    ) / temperature
    selected_oof = selected_oof / temperature
    val_prediction = val_logits.argmax(axis=1)
    validation = metrics(labels[val_indices], val_prediction)
    p12 = p12_prediction(args.p12_oof, sample_ids[val_indices])
    oracle = np.logical_or(val_prediction == labels[val_indices], p12 == labels[val_indices])
    complete_indices = np.concatenate((train_indices, val_indices))
    complete_logits = np.concatenate((selected_oof, val_logits), axis=0)
    crossfit_path = output / "crossfit_logits.npz"
    temporary = crossfit_path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=sample_ids[complete_indices],
            source_ids=source_ids[complete_indices],
            labels=class_labels[complete_indices],
            users=users[complete_indices],
            folds=np.concatenate(
                (fold_vector, np.full(len(val_indices), args.cv_splits, dtype=np.int64))
            ),
            logits=complete_logits.astype(np.float32),
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[complete_logits.argmax(axis=1)],
            train_oof_mask=np.concatenate(
                (np.ones(len(train_indices), dtype=np.uint8), np.zeros(len(val_indices), dtype=np.uint8))
            ),
        )
    temporary.replace(crossfit_path)
    torch.save(
        {"config": asdict(selected), "state_dict": final_model.cpu().state_dict()},
        output / "final_head.pt",
    )
    summary = {
        "protocol": (
            "Unified attention over aligned 3-view x 8-time VideoMAE tokens; model "
            "selection and epoch count from training-user Group-CV only"
        ),
        "selected_cv": selected_row,
        "selected_fold_epochs": epochs_by_name[selected.name],
        "final_epochs": final_epochs,
        "temperature_from_train_oof": temperature,
        "validation": validation,
        "p12_validation": metrics(labels[val_indices], p12),
        "p12_relation_oracle_diagnostic": {
            "correct": int(oracle.sum()),
            "total": int(len(oracle)),
            "accuracy": float(oracle.mean()),
        },
        "crossfit_logits": str(crossfit_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
