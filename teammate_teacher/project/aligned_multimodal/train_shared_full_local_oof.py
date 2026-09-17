from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from aligned_model import VisualEncoder, fp32_size_mb, parameter_count
from local_roi_data import sample_positions


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FOLD_DIR = PROJECT_DIR / "data" / "subject_folds"
DEFAULT_FULL_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
DEFAULT_LOCAL_CACHE = PROJECT_DIR / "runs" / "p16_oracle_local_depth_cache"
DEFAULT_FULL_OOF = PROJECT_DIR / "runs" / "p5_depth_imagenet"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p16_shared_full_local_oracle_oof"
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an exploratory Full+Local dual-view model with one shared "
            "ResNet-18+TSM visual encoder."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--full-cache", type=Path, default=DEFAULT_FULL_CACHE)
    parser.add_argument("--local-cache", type=Path, default=DEFAULT_LOCAL_CACHE)
    parser.add_argument("--full-oof-root", type=Path, default=DEFAULT_FULL_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument(
        "--bn-mode",
        choices=("shared", "dsbn"),
        default="shared",
        help=(
            "shared reproduces the current baseline. dsbn keeps one set of "
            "convolution weights but gives Full and Local independent BN "
            "parameters and running statistics throughout the encoder."
        ),
    )
    parser.add_argument(
        "--roi-protocol",
        choices=(
            "exploratory_oracle_assisted_all286",
            "strict_fold_pure",
        ),
        default="exploratory_oracle_assisted_all286",
        help=(
            "Metadata label for the ROI-generation protocol. Use "
            "strict_fold_pure only with a cache whose held-fold ROI was "
            "generated without that fold's manual ROI supervision."
        ),
    )
    parser.add_argument("--fold", type=int, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class FullLocalCacheDataset(Dataset):
    def __init__(
        self,
        full_cache: Path,
        local_cache: Path,
        fold_csv: Path,
        held_fold: int,
        split: str,
        augment: bool,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        metadata = json.loads(
            (full_cache / "metadata.json").read_text(encoding="utf-8")
        )
        self.full_location = {
            str(sample_id): (int(offset), int(length))
            for sample_id, (offset, length) in zip(
                metadata["sample_ids"],
                metadata["offsets"],
                strict=True,
            )
        }
        self.full_path = full_cache / "depth_uint8.npy"
        nonfallback_ids = np.load(
            local_cache / "nonfallback_sample_ids.npy",
            allow_pickle=False,
        ).astype(str)
        fallback_ids = np.load(
            local_cache / "fallback_sample_ids.npy",
            allow_pickle=False,
        ).astype(str)
        self.local_location = {
            sample_id: ("nonfallback", index)
            for index, sample_id in enumerate(nonfallback_ids)
        }
        self.local_location.update(
            {
                sample_id: ("fallback", index)
                for index, sample_id in enumerate(fallback_ids)
            }
        )
        self.local_nonfallback_path = local_cache / "nonfallback_uint8.npy"
        self.local_fallback_path = (
            local_cache / f"fallback_fold_{held_fold}_uint8.npy"
        )
        self._full: np.ndarray | None = None
        self._local_nonfallback: np.ndarray | None = None
        self._local_fallback: np.ndarray | None = None
        self.samples = [
            row for row in read_csv(fold_csv) if row["split"] == split
        ]
        self.samples.sort(key=lambda row: row["sample_id"])
        for row in self.samples:
            sample_id = row["sample_id"]
            if sample_id not in self.full_location:
                raise ValueError(f"Full cache missing {sample_id}")
            if sample_id not in self.local_location:
                raise ValueError(f"Local cache missing {sample_id}")
        self.augment = augment

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._full is None:
            self._full = np.load(self.full_path, mmap_mode="r")
        if self._local_nonfallback is None:
            self._local_nonfallback = np.load(
                self.local_nonfallback_path,
                mmap_mode="r",
            )
        if self._local_fallback is None:
            self._local_fallback = np.load(
                self.local_fallback_path,
                mmap_mode="r",
            )
        return self._full, self._local_nonfallback, self._local_fallback

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.samples[index]
        sample_id = row["sample_id"]
        full_cache, local_nonfallback, local_fallback = self.arrays()
        offset, length = self.full_location[sample_id]
        positions = np.asarray(
            sample_positions(length, 12, augment=False),
            dtype=np.int64,
        )
        full = np.asarray(full_cache[offset + positions]).copy()
        source, local_index = self.local_location[sample_id]
        local = np.asarray(
            (
                local_nonfallback[local_index]
                if source == "nonfallback"
                else local_fallback[local_index]
            )
        ).copy()
        full_tensor = (
            torch.from_numpy(full).permute(0, 3, 1, 2).float().div_(255.0)
        )
        local_tensor = (
            torch.from_numpy(local).permute(0, 3, 1, 2).float().div_(255.0)
        )
        if self.augment and bool(torch.rand(1).item() < 0.5):
            full_tensor = torch.flip(full_tensor, dims=(3,))
            local_tensor = torch.flip(local_tensor, dims=(3,))
        full_tensor = (
            full_tensor - IMAGENET_MEAN[None, :, None, None]
        ) / IMAGENET_STD[None, :, None, None]
        local_tensor = (
            local_tensor - IMAGENET_MEAN[None, :, None, None]
        ) / IMAGENET_STD[None, :, None, None]
        return {
            "depth_full": full_tensor,
            "depth_local": local_tensor,
            "label": int(row["class_id"]),
            "sample_id": sample_id,
        }


class DomainSpecificBatchNorm2d(nn.Module):
    def __init__(self, source: nn.BatchNorm2d) -> None:
        super().__init__()
        self.full = copy.deepcopy(source)
        self.local = copy.deepcopy(source)
        self.domain = "full"

    def set_domain(self, domain: str) -> None:
        if domain not in {"full", "local"}:
            raise ValueError(f"Unknown BN domain: {domain}")
        self.domain = domain

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return getattr(self, self.domain)(inputs)


def replace_batch_norm_with_dsbn(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DomainSpecificBatchNorm2d(child))
            replaced += 1
        else:
            replaced += replace_batch_norm_with_dsbn(child)
    return replaced


def set_dsbn_domain(module: nn.Module, domain: str) -> None:
    for child in module.modules():
        if isinstance(child, DomainSpecificBatchNorm2d):
            child.set_domain(domain)


class SharedFullLocalModel(nn.Module):
    def __init__(self, dropout: float = 0.3, num_classes: int = 40) -> None:
        super().__init__()
        self.bn_mode = "shared"
        self.dsbn_layers = 0
        self.visual = VisualEncoder(
            ("depth",),
            dropout=dropout,
            imagenet_pretrained=True,
        )
        self.full_project = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.local_project = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.gate = nn.Linear(512, 256)
        self.fused_classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )
        self.full_classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.local_classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def enable_dsbn(self) -> int:
        if self.bn_mode != "shared":
            raise RuntimeError("DSBN has already been enabled")
        self.dsbn_layers = replace_batch_norm_with_dsbn(self.visual)
        if self.dsbn_layers <= 0:
            raise RuntimeError("No BatchNorm2d layers were replaced")
        self.bn_mode = "dsbn"
        return self.dsbn_layers

    def forward(
        self,
        full: torch.Tensor,
        local: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = len(full)
        if self.bn_mode == "shared":
            # Baseline path: Full and Local share both convolution and BN.
            combined = torch.cat([full, local], dim=0)
            encoded = self.visual({"depth": combined})
            if not isinstance(encoded, torch.Tensor):
                raise TypeError("VisualEncoder returned an unexpected tuple")
            full_feature, local_feature = encoded.split(batch_size, dim=0)
        else:
            # DSBN path: convolution weights remain shared, while every BN
            # layer has independent Full/Local affine parameters and stats.
            set_dsbn_domain(self.visual, "full")
            full_feature = self.visual({"depth": full})
            set_dsbn_domain(self.visual, "local")
            local_feature = self.visual({"depth": local})
            if not isinstance(full_feature, torch.Tensor) or not isinstance(
                local_feature,
                torch.Tensor,
            ):
                raise TypeError("VisualEncoder returned an unexpected tuple")
        full_projected = self.full_project(full_feature)
        local_projected = self.local_project(local_feature)
        gate = torch.sigmoid(
            self.gate(torch.cat([full_projected, local_projected], dim=1))
        )
        fused = gate * local_projected + (1.0 - gate) * full_projected
        return {
            "fused": self.fused_classifier(
                torch.cat(
                    [full_projected, local_projected, fused],
                    dim=1,
                )
            ),
            "full": self.full_classifier(full_projected),
            "local": self.local_classifier(local_projected),
            "gate_mean": gate.mean(),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def create_loader(
    dataset: FullLocalCacheDataset,
    batch_size: int,
    workers: int,
    train: bool,
    seed: int,
) -> DataLoader:
    sampler = None
    shuffle = train
    if train:
        counts = Counter(int(row["class_id"]) for row in dataset.samples)
        weights = torch.tensor(
            [1.0 / counts[int(row["class_id"])] for row in dataset.samples],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        drop_last=train,
    )


def metric_dict(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def load_full_initialization(
    model: SharedFullLocalModel,
    checkpoint_path: Path,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["model_state_dict"]
    visual_state = {
        key.removeprefix("visual."): value
        for key, value in state.items()
        if key.startswith("visual.")
    }
    model.visual.load_state_dict(visual_state, strict=True)


def run_epoch(
    model: SharedFullLocalModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    aux_weight: float,
) -> tuple[float, dict[str, np.ndarray], np.ndarray, list[str], float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    outputs: dict[str, list[np.ndarray]] = {
        "fused": [],
        "full": [],
        "local": [],
    }
    labels_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    gate_means: list[float] = []
    for batch in loader:
        full = batch["depth_full"].to(device, non_blocking=True)
        local = batch["depth_local"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                result = model(full, local)
                loss = (
                    criterion(result["fused"], labels)
                    + aux_weight * criterion(result["full"], labels)
                    + aux_weight * criterion(result["local"], labels)
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        for key in outputs:
            outputs[key].append(result[key].detach().float().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        sample_ids.extend(str(value) for value in batch["sample_id"])
        gate_means.append(float(result["gate_mean"].detach().cpu()))
    return (
        float(np.mean(losses)),
        {key: np.concatenate(value) for key, value in outputs.items()},
        np.concatenate(labels_all),
        sample_ids,
        float(np.mean(gate_means)),
    )


def train_fold(args: argparse.Namespace, held_fold: int) -> dict[str, object]:
    seed = 20260728 + held_fold
    seed_everything(seed)
    fold_csv = args.fold_dir.resolve() / f"fold_{held_fold}.csv"
    train_dataset = FullLocalCacheDataset(
        args.full_cache.resolve(),
        args.local_cache.resolve(),
        fold_csv,
        held_fold,
        "train",
        augment=True,
    )
    val_dataset = FullLocalCacheDataset(
        args.full_cache.resolve(),
        args.local_cache.resolve(),
        fold_csv,
        held_fold,
        "val",
        augment=False,
    )
    train_loader = create_loader(
        train_dataset,
        int(args.batch_size),
        int(args.workers),
        True,
        seed,
    )
    val_loader = create_loader(
        val_dataset,
        int(args.batch_size),
        max(0, int(args.workers) // 2),
        False,
        seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedFullLocalModel()
    initialization = (
        args.full_oof_root.resolve()
        / f"fold_{held_fold}"
        / "best_accuracy.pt"
    )
    load_full_initialization(model, initialization)
    if str(args.bn_mode) == "dsbn":
        model.enable_dsbn()
    model = model.to(device)
    backbone_parameters = list(model.visual.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(args.backbone_lr),
            },
            {
                "params": head_parameters,
                "lr": float(args.head_lr),
            },
        ],
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    fold_output = args.output_dir.resolve() / f"fold_{held_fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_epoch = 0
    no_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        train_loss, train_logits, train_labels, _, train_gate = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            float(args.aux_weight),
        )
        val_loss, val_logits, val_labels, val_ids, val_gate = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            scaler,
            float(args.aux_weight),
        )
        train_metrics = metric_dict(train_logits["fused"], train_labels)
        val_metrics = metric_dict(val_logits["fused"], val_labels)
        record = {
            "epoch": epoch,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_gate_mean": train_gate,
            "val_gate_mean": val_gate,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_full_accuracy": metric_dict(
                val_logits["full"], val_labels
            )["accuracy"],
            "val_local_accuracy": metric_dict(
                val_logits["local"], val_labels
            )["accuracy"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(
            f"fold={held_fold} epoch={epoch:02d} "
            f"train={train_metrics['accuracy']:.4f} "
            f"val={val_metrics['accuracy']:.4f} "
            f"full={record['val_full_accuracy']:.4f} "
            f"local={record['val_local_accuracy']:.4f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "held_fold": held_fold,
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "config": {
                        "architecture": "shared_full_local_resnet18_tsm",
                        "bn_mode": str(args.bn_mode),
                        "shared_batch_norm": str(args.bn_mode) == "shared",
                        "dsbn_layers": int(model.dsbn_layers),
                        "roi_protocol": str(args.roi_protocol),
                        "deployable_oof": (
                            str(args.roi_protocol) == "strict_fold_pure"
                        ),
                        "local_context": 0.15,
                        "wide_local": False,
                        "aux_weight": float(args.aux_weight),
                        "full_initialization": str(initialization),
                    },
                },
                fold_output / "best.pt",
            )
            np.savez_compressed(
                fold_output / "best_val_logits.npz",
                sample_ids=np.asarray(val_ids),
                labels=val_labels,
                held_fold=np.full(
                    len(val_ids),
                    held_fold,
                    dtype=np.int64,
                ),
                fused_logits=val_logits["fused"],
                full_aux_logits=val_logits["full"],
                local_aux_logits=val_logits["local"],
            )
        else:
            no_improvement += 1
        scheduler.step()
        if epoch >= 7 and no_improvement >= int(args.patience):
            print(
                f"fold={held_fold} early_stop epoch={epoch} "
                f"best={best_epoch}",
                flush=True,
            )
            break
    result = {
        "held_fold": held_fold,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "epochs_ran": len(history),
        "seconds": time.perf_counter() - started,
        "parameters": parameter_count(model),
        "fp32_size_mb": fp32_size_mb(model),
        "bn_mode": str(args.bn_mode),
        "dsbn_layers": int(model.dsbn_layers),
        "history": history,
    }
    (fold_output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    if args.fold is not None and args.fold not in (0, 1, 2):
        raise ValueError("--fold must be 0, 1, or 2")
    if not torch.cuda.is_available():
        raise RuntimeError("Shared Full+Local training requires CUDA")
    torch.backends.cudnn.benchmark = True
    folds = [args.fold] if args.fold is not None else [0, 1, 2]
    results = [train_fold(args, fold) for fold in folds]
    if args.fold is not None:
        return
    arrays = [
        np.load(
            args.output_dir.resolve()
            / f"fold_{fold}"
            / "best_val_logits.npz",
            allow_pickle=False,
        )
        for fold in range(3)
    ]
    combined = {
        key: np.concatenate([array[key] for array in arrays])
        for key in (
            "sample_ids",
            "labels",
            "held_fold",
            "fused_logits",
            "full_aux_logits",
            "local_aux_logits",
        )
    }
    order = np.argsort(combined["sample_ids"].astype(str))
    np.savez_compressed(
        args.output_dir.resolve() / "oof_logits.npz",
        **{key: value[order] for key, value in combined.items()},
    )
    pooled = metric_dict(
        combined["fused_logits"],
        combined["labels"],
    )
    summary = {
        "status": (
            "strict_fold_pure"
            if str(args.roi_protocol) == "strict_fold_pure"
            else "exploratory_oracle_assisted"
        ),
        "roi_protocol": str(args.roi_protocol),
        "deployable_oof": (
            str(args.roi_protocol) == "strict_fold_pure"
        ),
        "architecture": (
            (
                "One shared ResNet-18+TSM convolutional encoder; Full and "
                "standard Local use independent BN parameters and running "
                "statistics in every ResNet BN layer."
            )
            if str(args.bn_mode) == "dsbn"
            else (
                "One shared ResNet-18+TSM encoder; Full and standard Local "
                "are concatenated along the batch dimension and use shared BN."
            )
        ),
        "bn_mode": str(args.bn_mode),
        "wide_local": False,
        "folds": results,
        "pooled_oof": pooled,
    }
    (args.output_dir.resolve() / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
