from __future__ import annotations

import argparse
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

from aligned_model import AlignedMultimodalModel, fp32_size_mb, parameter_count
from local_roi_data import FoldPureLocalDepthDataset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_FOLD_DIR = PROJECT_DIR / "data" / "subject_folds"
DEFAULT_LOCATOR_DIR = PROJECT_DIR / "runs" / "p12_fold_pure_locator_predictions"
DEFAULT_CACHE_DIR = PROJECT_DIR / "runs" / "p12_local_depth_cache"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p12_local_depth_oof"
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a three-fold ImageNet ResNet-18+TSM Local Depth expert."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--locator-dir", type=Path, default=DEFAULT_LOCATOR_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class CachedLocalDepthDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        subject_fold_path: Path,
        held_fold: int,
        split: str,
        augment: bool,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.nonfallback_path = cache_dir / "nonfallback_uint8.npy"
        self.fallback_path = (
            cache_dir / f"fallback_fold_{held_fold}_uint8.npy"
        )
        self._nonfallback: np.ndarray | None = None
        self._fallback: np.ndarray | None = None
        nonfallback_ids = [
            str(value)
            for value in np.load(
                cache_dir / "nonfallback_sample_ids.npy",
                allow_pickle=False,
            )
        ]
        fallback_ids = [
            str(value)
            for value in np.load(
                cache_dir / "fallback_sample_ids.npy",
                allow_pickle=False,
            )
        ]
        self.location = {
            sample_id: ("nonfallback", index)
            for index, sample_id in enumerate(nonfallback_ids)
        }
        self.location.update(
            {
                sample_id: ("fallback", index)
                for index, sample_id in enumerate(fallback_ids)
            }
        )
        self.samples = [
            row
            for row in read_csv(subject_fold_path)
            if row["split"] == split
        ]
        self.samples.sort(key=lambda row: row["sample_id"])
        if any(row["sample_id"] not in self.location for row in self.samples):
            raise ValueError("Local cache does not cover subject-fold samples")
        self.held_fold = held_fold
        self.split = split
        self.augment = augment

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._nonfallback is None:
            self._nonfallback = np.load(
                self.nonfallback_path,
                mmap_mode="r",
            )
        if self._fallback is None:
            self._fallback = np.load(
                self.fallback_path,
                mmap_mode="r",
            )
        return self._nonfallback, self._fallback

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.samples[index]
        source, source_index = self.location[row["sample_id"]]
        nonfallback, fallback = self.arrays()
        array = (
            nonfallback[source_index]
            if source == "nonfallback"
            else fallback[source_index]
        )
        tensor = (
            torch.from_numpy(np.asarray(array).copy())
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        if self.augment and bool(torch.rand(1).item() < 0.5):
            tensor = torch.flip(tensor, dims=(3,))
        tensor = (tensor - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[
            None, :, None, None
        ]
        return {
            "depth_local": tensor,
            "label": int(row["class_id"]),
            "sample_id": row["sample_id"],
            "held_fold": self.held_fold,
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
    dataset: Dataset,
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


def run_epoch(
    model: AlignedMultimodalModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in loader:
        depth = batch["depth_local"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model({"depth": depth})
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().float().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        sample_ids.extend(str(value) for value in batch["sample_id"])
    return (
        float(np.mean(losses)),
        np.concatenate(logits_all),
        np.concatenate(labels_all),
        sample_ids,
    )


def metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, prediction),
        "balanced_accuracy": balanced_accuracy_score(labels, prediction),
        "macro_f1": f1_score(labels, prediction, average="macro"),
    }


def train_fold(args: argparse.Namespace, held_fold: int) -> dict[str, object]:
    seed = 20260727 + held_fold
    seed_everything(seed)
    fold_dir = args.fold_dir.resolve()
    locator_path = (
        args.locator_dir.resolve()
        / f"fold_{held_fold}_locator_predictions.csv"
    )
    cache_dir = args.cache_dir.resolve()
    if (cache_dir / "summary.json").exists():
        cache_summary = json.loads(
            (cache_dir / "summary.json").read_text(encoding="utf-8")
        )
        roi_protocol = str(cache_summary.get("status", "fold_pure"))
        train_dataset = CachedLocalDepthDataset(
            cache_dir,
            fold_dir / f"fold_{held_fold}.csv",
            held_fold,
            "train",
            augment=True,
        )
        val_dataset = CachedLocalDepthDataset(
            cache_dir,
            fold_dir / f"fold_{held_fold}.csv",
            held_fold,
            "val",
            augment=False,
        )
        input_source = "uint8_local_cache"
    else:
        roi_protocol = "fold_pure"
        train_dataset = FoldPureLocalDepthDataset(
            args.manifest.resolve(),
            fold_dir / f"fold_{held_fold}.csv",
            locator_path,
            held_fold,
            "train",
            num_frames=12,
            augment=True,
            context=0.15,
        )
        val_dataset = FoldPureLocalDepthDataset(
            args.manifest.resolve(),
            fold_dir / f"fold_{held_fold}.csv",
            locator_path,
            held_fold,
            "val",
            num_frames=12,
            augment=False,
            context=0.15,
        )
        input_source = "original_depth_on_the_fly"
    train_loader = create_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        True,
        seed,
    )
    val_loader = create_loader(
        val_dataset,
        args.batch_size,
        max(0, args.workers // 2),
        False,
        seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlignedMultimodalModel(
        ["depth"],
        num_classes=40,
        dropout=0.3,
        imagenet_pretrained=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    fold_output = args.output_dir.resolve() / f"fold_{held_fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_logits, train_labels, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
        )
        val_loss, val_logits, val_labels, val_ids = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            scaler,
            use_amp,
        )
        train_metrics = metrics(train_logits, train_labels)
        val_metrics = metrics(val_logits, val_labels)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(
            f"fold={held_fold} epoch={epoch:02d} "
            f"train={train_metrics['accuracy']:.4f} "
            f"val={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "held_fold": held_fold,
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "config": {
                        "modalities": ["depth"],
                        "local_depth": True,
                        "num_frames": 12,
                        "imagenet_pretrained": True,
                        "context": 0.15,
                        "locator_predictions": str(locator_path),
                        "input_source": input_source,
                        "roi_protocol": roi_protocol,
                    },
                },
                fold_output / "best.pt",
            )
            np.savez_compressed(
                fold_output / "best_val_logits.npz",
                sample_ids=np.asarray(val_ids),
                labels=val_labels,
                logits=val_logits,
                held_fold=np.full(len(val_ids), held_fold, dtype=np.int64),
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epoch >= 8 and epochs_without_improvement >= args.patience:
            print(
                f"fold={held_fold} early_stop epoch={epoch} best={best_epoch}",
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
        "input_source": input_source,
        "roi_protocol": roi_protocol,
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
        raise RuntimeError("Local Depth training requires CUDA for this experiment")
    torch.backends.cudnn.benchmark = True
    folds = [args.fold] if args.fold is not None else [0, 1, 2]
    results = [train_fold(args, fold) for fold in folds]
    if args.fold is None:
        arrays = [
            np.load(
                args.output_dir.resolve() / f"fold_{fold}" / "best_val_logits.npz",
                allow_pickle=False,
            )
            for fold in range(3)
        ]
        sample_ids = np.concatenate([array["sample_ids"] for array in arrays])
        labels = np.concatenate([array["labels"] for array in arrays])
        logits = np.concatenate([array["logits"] for array in arrays])
        held_folds = np.concatenate([array["held_fold"] for array in arrays])
        order = np.argsort(sample_ids)
        pooled = metrics(logits, labels)
        np.savez_compressed(
            args.output_dir.resolve() / "oof_logits.npz",
            sample_ids=sample_ids[order],
            labels=labels[order],
            logits=logits[order],
            held_fold=held_folds[order],
        )
        summary = {
            "protocol": (
                "Exploratory/oracle-assisted subject-disjoint classifier folds. "
                "The ROI locator used all available human ROI supervision, so "
                "this is not strict deployable OOF."
                if results[0]["roi_protocol"]
                == "exploratory_oracle_assisted"
                else (
                    "Three subject-disjoint folds; every fold uses a locator "
                    "trained only on the other two ROI annotation folds."
                )
            ),
            "deployable_oof": (
                results[0]["roi_protocol"] != "exploratory_oracle_assisted"
            ),
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
