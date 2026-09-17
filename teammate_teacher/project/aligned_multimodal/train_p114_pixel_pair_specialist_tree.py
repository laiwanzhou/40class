"""P114 pixel-level temporal pair-specialist tree over the frozen P89 anchor.

This model is not a replacement 40-class classifier.  One shared ImageNet
ResNet18 reads the scene and workspace channels of the existing source-safe
pixel cache; independent binary heads are trained only for source-derived P89
Top-2 confusion boundaries.  A held prediction is changed only on the exact
Top-2 edge and only when its binary head has at least 0.90 confidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from audit_p112_multimodal_pair_tree import OUTER_USERS, _align
from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_OUTPUT = HERE / "runs/p114_pixel_pair_specialist_tree_v1"
SEED = 20260824
FIXED_CALL_CONFIDENCE = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", default=",".join(OUTER_USERS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


class PairTrialDataset(Dataset):
    def __init__(
        self,
        images_path: Path,
        entries: list[tuple[int, int, int]],
        training: bool,
    ) -> None:
        self.images = np.load(images_path, mmap_mode="r")
        self.entries = entries
        self.training = training

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        task, row, target = self.entries[index]
        if self.training:
            frame_indices = np.sort(
                np.random.default_rng(SEED + index + random.randrange(1_000_000)).choice(
                    16, size=4, replace=False
                )
            )
            top = random.randint(0, 16)
            left = random.randint(0, 16)
        else:
            frame_indices = np.asarray([0, 5, 10, 15], dtype=np.int64)
            top = left = 8
        raw = np.array(self.images[row][:, frame_indices], copy=True)
        # Preserve separate scene and workspace evidence.  Each grayscale view
        # is repeated to RGB so ImageNet filters remain semantically valid.
        raw = raw[:, :, [0, 2], top : top + 144, left : left + 144]
        tensor = torch.from_numpy(raw).float().div_(255.0)
        if self.training:
            gain = 0.85 + 0.30 * torch.rand((1, 1, 1, 1, 1))
            bias = -0.08 + 0.16 * torch.rand((1, 1, 1, 1, 1))
            tensor = (tensor * gain + bias).clamp_(0.0, 1.0)
        tensor = tensor.unsqueeze(3).repeat(1, 1, 1, 3, 1, 1)
        mean = torch.tensor((0.485, 0.456, 0.406))[None, None, None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[None, None, None, :, None, None]
        tensor = (tensor - mean) / std
        return tensor, task, target, row


class PixelPairTree(nn.Module):
    def __init__(self, pair_count: int) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # Low-level ImageNet filters are fixed; layer3/layer4 adapt to local
        # depth/IR workspace appearance and action phase.
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad = name.startswith("6.") or name.startswith("7.")
        self.projector = nn.Sequential(
            nn.LayerNorm(512 * 8),
            nn.Linear(512 * 8, 384),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.heads = nn.ModuleList([nn.Linear(384, 2) for _ in range(pair_count)])

    def represent(self, images: torch.Tensor) -> torch.Tensor:
        batch, windows, frames, views, channels, height, width = images.shape
        flat = images.reshape(batch * windows * frames * views, channels, height, width)
        token = self.encoder(flat).flatten(1).reshape(batch, windows, frames, views, 512)
        mean = token.mean(dim=(1, 2))
        std = token.float().std(dim=(1, 2), unbiased=False)
        early = token[:, 0].mean(dim=1)
        late = token[:, 1].mean(dim=1)
        delta = late - early
        # Four statistics for each of two views: [B, 2, 4*512].
        value = torch.cat((mean, std, early, delta), dim=2).reshape(batch, -1)
        return self.projector(value)

    def forward(self, images: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        embedding = self.represent(images)
        logits = torch.empty((len(images), 2), device=images.device, dtype=embedding.dtype)
        for task in torch.unique(tasks).tolist():
            selected = tasks == int(task)
            logits[selected] = self.heads[int(task)](embedding[selected])
        return logits


def build_protocol() -> dict[str, np.ndarray]:
    splits = load_splits()
    fold_order = tuple(OUTER_USERS)
    return {
        "ids": np.concatenate([splits[name].sample_ids.astype(str) for name in fold_order]),
        "users": np.concatenate([splits[name].users.astype(str) for name in fold_order]),
        "labels": np.concatenate([splits[name].labels.astype(np.int64) for name in fold_order]),
        "safe": np.concatenate([splits[name].safe_prediction.astype(np.int64) for name in fold_order]),
        "probability": np.concatenate([splits[name].safe_probability.astype(np.float64) for name in fold_order]),
        "fold_names": np.concatenate(
            [np.repeat(name, len(splits[name].sample_ids)) for name in fold_order]
        ).astype(str),
    }


def pair_candidates(
    held: np.ndarray,
    labels: np.ndarray,
    safe: np.ndarray,
    top2_pair: np.ndarray,
) -> list[tuple[int, int]]:
    opportunities: Counter[tuple[int, int]] = Counter()
    for row in np.flatnonzero((~held) & (safe != labels)):
        pair = tuple(map(int, top2_pair[row]))
        if int(labels[row]) in pair and int(safe[row]) in pair:
            opportunities[pair] += 1
    return sorted(
        (pair for pair, count in opportunities.items() if count >= 3),
        key=lambda pair: (-opportunities[pair], pair),
    )


def train_fold(
    args: argparse.Namespace,
    outer: str,
    full_rows: list[dict[str, str]],
    protocol: dict[str, np.ndarray],
    cache_order: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    users = protocol["users"]
    labels = protocol["labels"]
    safe = protocol["safe"]
    probability = protocol["probability"]
    fold_names = protocol["fold_names"]
    held = fold_names == outer
    top2 = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    top2_pair = np.sort(top2, axis=1)
    pairs = pair_candidates(held, labels, safe, top2_pair)
    if args.smoke:
        pairs = pairs[:2]
    held_users = set(OUTER_USERS[outer])
    full_labels = np.asarray([int(row["class_id"]) for row in full_rows], dtype=np.int64)
    full_users = np.asarray([row["user_id"] for row in full_rows], dtype=str)
    entries: list[tuple[int, int, int]] = []
    for task, pair in enumerate(pairs):
        selected = np.flatnonzero((~np.isin(full_users, list(held_users))) & np.isin(full_labels, pair))
        for row in selected.tolist():
            entries.append((task, row, int(full_labels[row] == pair[1])))
    if args.smoke:
        entries = entries[: min(64, len(entries))]
    dataset = PairTrialDataset(args.cache / "images.npy", entries, training=True)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    model = PixelPairTree(len(pairs)).to(args.device)
    backbone_parameters = [value for value in model.encoder.parameters() if value.requires_grad]
    head_parameters = list(model.projector.parameters()) + list(model.heads.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": head_parameters, "lr": args.learning_rate},
        ],
        weight_decay=1e-4,
    )
    epochs = 1 if args.smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        correct = total = 0
        for images, tasks, targets, _ in loader:
            images = images.to(args.device, non_blocking=True)
            tasks = tasks.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if args.device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                logits = model(images, tasks)
                loss = F.cross_entropy(logits, targets, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(1) == targets).sum().detach().cpu())
            total += len(targets)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "training_pair_accuracy": correct / max(total, 1),
        }
        history.append(record)
        print(json.dumps({"p114": outer, **record}), flush=True)

    eval_entries: list[tuple[int, int, int]] = []
    route_meta: list[tuple[int, int]] = []
    for task, pair in enumerate(pairs):
        rows = np.flatnonzero(
            held
            & np.all(top2_pair == np.asarray(pair)[None], axis=1)
            & np.isin(safe, pair)
        )
        for protocol_row in rows.tolist():
            cache_row = int(cache_order[protocol_row])
            eval_entries.append((task, cache_row, int(labels[protocol_row] == pair[1])))
            route_meta.append((protocol_row, task))
    eval_dataset = PairTrialDataset(args.cache / "images.npy", eval_entries, training=False)
    eval_loader = DataLoader(eval_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False)
    model.eval()
    all_probability: list[np.ndarray] = []
    with torch.inference_mode():
        for images, tasks, _, _ in eval_loader:
            images = images.to(args.device, non_blocking=True)
            tasks = tasks.to(args.device, non_blocking=True)
            with torch.autocast(
                device_type="cuda" if args.device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                logits = model(images, tasks)
            all_probability.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    predicted_probability = (
        np.concatenate(all_probability, axis=0) if all_probability else np.zeros((0, 2), dtype=np.float32)
    )
    system = safe.copy()
    audit_rows: list[dict[str, Any]] = []
    for index, (row, task) in enumerate(route_meta):
        pair = pairs[task]
        local = int(predicted_probability[index].argmax())
        specialist = int(pair[local])
        confidence = float(predicted_probability[index, local])
        called = confidence >= FIXED_CALL_CONFIDENCE and specialist != int(safe[row])
        if called:
            system[row] = specialist
        audit_rows.append(
            {
                "sample_id": protocol["ids"][row],
                "subject": users[row],
                "outer_fold": outer,
                "pair": f"{pair[0]}<->{pair[1]}",
                "true_label": int(labels[row]),
                "p89_prediction": int(safe[row]),
                "specialist_prediction": specialist,
                "specialist_confidence": confidence,
                "called": int(called),
                "rescued": int(called and specialist == labels[row] and safe[row] != labels[row]),
                "harmed": int(called and safe[row] == labels[row] and specialist != labels[row]),
            }
        )
    base_correct = safe[held] == labels[held]
    final_correct = system[held] == labels[held]
    checkpoint = output / f"{outer}_best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "pairs": pairs,
            "outer_fold": outer,
            "call_confidence": FIXED_CALL_CONFIDENCE,
            "epochs": epochs,
            "seed": SEED,
        },
        checkpoint,
    )
    write_csv(output / f"{outer}_history.csv", history)
    write_csv(output / f"{outer}_route_audit.csv", audit_rows)
    return {
        "outer_fold": outer,
        "rows": int(held.sum()),
        "pairs": [f"{pair[0]}<->{pair[1]}" for pair in pairs],
        "training_entries": len(entries),
        "routed_rows": len(route_meta),
        "called": int(np.sum(system[held] != safe[held])),
        "p89_correct": int(base_correct.sum()),
        "system_correct": int(final_correct.sum()),
        "accuracy": float(final_correct.mean()),
        "rescue": int(np.sum((~base_correct) & final_correct)),
        "harm": int(np.sum(base_correct & (~final_correct))),
        "net": int(final_correct.sum() - base_correct.sum()),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    seed_everything()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.cache / "rows.csv")
    protocol = build_protocol()
    row_ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
    cache_order = _align(row_ids, protocol["ids"])
    requested = [item.strip() for item in args.folds.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(OUTER_USERS))
    if unknown:
        raise ValueError(f"unknown outer folds: {unknown}")
    results = [
        train_fold(args, outer, rows, protocol, cache_order, output)
        for outer in requested
    ]
    summary = {
        "stage": "P114_pixel_pair_specialist_tree",
        "status": "complete",
        "protocol": {
            "p89_frozen": True,
            "pair_discovery": "source P89 exact Top-2 error opportunities >=3",
            "training": "all source samples from each pair; no error-only training",
            "model": "shared ImageNet ResNet18 scene+workspace temporal encoder with binary pair heads",
            "routing": f"exact P89 Top-2 edge and fixed confidence >= {FIXED_CALL_CONFIDENCE}",
            "held_labels_used_for_selection": False,
        },
        "folds": results,
        "aggregate": {
            "rows": int(sum(row["rows"] for row in results)),
            "p89_correct": int(sum(row["p89_correct"] for row in results)),
            "system_correct": int(sum(row["system_correct"] for row in results)),
            "rescue": int(sum(row["rescue"] for row in results)),
            "harm": int(sum(row["harm"] for row in results)),
            "net": int(sum(row["net"] for row in results)),
        },
    }
    aggregate = summary["aggregate"]
    aggregate["accuracy"] = aggregate["system_correct"] / max(aggregate["rows"], 1)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
