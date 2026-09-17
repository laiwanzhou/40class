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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from p32_fused_data import BalancedLengthBucketBatchSampler, LengthBucketBatchSampler
from p44c_model import P44CPretrainModel, parameter_count
from p44c_spatial_fused_data import P44CSpatialFusedDataset, collate_p44c


PROJECT_DIR = Path(__file__).resolve().parent
CALIBRATION_SUBJECTS = {"user16", "user23"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P44-C 40-class fold0-train-only pretraining.")
    parser.add_argument(
        "--fold-csv", type=Path, default=PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv"
    )
    parser.add_argument(
        "--visual-run", type=Path, default=PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
    )
    parser.add_argument(
        "--motion-run", type=Path, default=PROJECT_DIR / "runs" / "p31_skeleton_imu_full"
    )
    parser.add_argument(
        "--spatial-run", type=Path, default=PROJECT_DIR / "runs" / "p44c_spatial_roi_fold0"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "runs" / "p44c_pretrain_fold0"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=22)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=44030)
    parser.add_argument("--skip-refit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    torch.save(value, temporary)
    temporary.replace(path)


def fold_train_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    for row in rows:
        row["source_id"] = f"{row['class_name']}/{row['user_id']}/{row['trial_id']}"
    return rows


def make_dataset(args: argparse.Namespace, ids: set[str]) -> P44CSpatialFusedDataset:
    return P44CSpatialFusedDataset(
        args.visual_run.resolve(), args.motion_run.resolve(), args.spatial_run.resolve(), ids
    )


def make_loader(
    dataset: P44CSpatialFusedDataset,
    batch_size: int,
    workers: int,
    seed: int,
    train: bool,
) -> tuple[DataLoader, Any]:
    if train:
        sampler = BalancedLengthBucketBatchSampler(
            dataset.frame_lengths,
            [int(row["class_id"]) for row in dataset.rows],
            batch_size=batch_size,
            seed=seed,
            bucket_multiplier=12,
        )
    else:
        sampler = LengthBucketBatchSampler(
            dataset.frame_lengths,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            bucket_multiplier=12,
        )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_p44c,
    )
    return loader, sampler


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def supervised_contrastive(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    users: list[str],
    temperature: float = 0.10,
) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    source = torch.nn.functional.normalize(embedding.float(), dim=1)
    similarity = source @ source.transpose(0, 1) / temperature
    identity = torch.eye(len(source), dtype=torch.bool, device=source.device)
    user_index = torch.tensor(
        [hash(value) % 1000003 for value in users], dtype=torch.long, device=source.device
    )
    positive = (
        labels[:, None].eq(labels[None, :])
        & user_index[:, None].ne(user_index[None, :])
        & ~identity
    )
    valid_anchor = positive.any(dim=1)
    if not valid_anchor.any():
        return embedding.sum() * 0.0
    logits = similarity.masked_fill(identity, -1e4)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_mean = (log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return -positive_mean[valid_anchor].mean()


def losses(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    users: list[str],
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    ce = torch.nn.functional.cross_entropy(
        output["logits"], labels, label_smoothing=label_smoothing
    )
    multimodal = torch.nn.functional.cross_entropy(
        output["multimodal_logits"], labels, label_smoothing=label_smoothing
    )
    spatial = torch.nn.functional.cross_entropy(
        output["spatial_logits"], labels, label_smoothing=label_smoothing
    )
    contrast = supervised_contrastive(output["embedding"], labels, users)
    return {
        "total": ce + 0.20 * multimodal + 0.35 * spatial + 0.08 * contrast,
        "main": ce,
        "multimodal": multimodal,
        "spatial": spatial,
        "contrast": contrast,
    }


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def run_epoch(
    model: P44CPretrainModel,
    loader: DataLoader,
    sampler: Any,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    label_smoothing: float,
    maximum_batches: int = 0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if training:
        sampler.set_epoch(epoch)
    totals = {key: 0.0 for key in ("total", "main", "multimodal", "spatial", "contrast")}
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if maximum_batches and batch_index >= maximum_batches:
            break
        batch = move_batch(batch, device)
        labels = batch["label"]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(batch)
            batch_losses = losses(output, labels, batch["user_id"], label_smoothing)
        if training:
            assert scaler is not None
            scaler.scale(batch_losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        weight = len(labels)
        for key in totals:
            totals[key] += float(batch_losses[key].detach()) * weight
        labels_all.append(labels.detach().cpu())
        logits_all.append(output["logits"].detach().float().cpu())
    label_array = torch.cat(labels_all).numpy()
    logit_array = torch.cat(logits_all).numpy()
    return {
        "losses": {key: value / len(label_array) for key, value in totals.items()},
        "metrics": metrics(label_array, logit_array),
        "seconds": time.perf_counter() - started,
        "labels": label_array,
        "logits": logit_array,
    }


def learning_rate(epoch: int, epochs: int, maximum: float, minimum: float) -> float:
    if epoch <= 2:
        return maximum * epoch / 2
    progress = (epoch - 2) / max(epochs - 2, 1)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        foreach=False,
    )


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_select(
    args: argparse.Namespace,
    fit: P44CSpatialFusedDataset,
    calibration: P44CSpatialFusedDataset,
    output: Path,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    fit_loader, fit_sampler = make_loader(
        fit, args.batch_size, args.workers, args.seed, True
    )
    cal_loader, cal_sampler = make_loader(
        calibration, args.eval_batch_size, args.workers, args.seed, False
    )
    model = P44CPretrainModel().to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    max_epochs = 2 if args.smoke else args.max_epochs
    for epoch in range(1, max_epochs + 1):
        lr = learning_rate(epoch, max_epochs, args.learning_rate, args.minimum_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train = run_epoch(
            model,
            fit_loader,
            fit_sampler,
            device,
            epoch,
            optimizer,
            scaler,
            args.label_smoothing,
            maximum_batches=2 if args.smoke else 0,
        )
        calibration_result = run_epoch(
            model,
            cal_loader,
            cal_sampler,
            device,
            epoch,
            None,
            None,
            0.0,
            maximum_batches=2 if args.smoke else 0,
        )
        score = calibration_result["metrics"]["macro_f1"]
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            atomic_checkpoint(
                output / "best_subject_calibration.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "calibration_metrics": calibration_result["metrics"],
                },
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train["losses"]["total"],
            "train_accuracy": train["metrics"]["accuracy"],
            "train_macro_f1": train["metrics"]["macro_f1"],
            "calibration_accuracy": calibration_result["metrics"]["accuracy"],
            "calibration_macro_f1": score,
            "train_seconds": train["seconds"],
            "calibration_seconds": calibration_result["seconds"],
            "best_epoch": best_epoch,
            "stale": stale,
        }
        history.append(row)
        write_history(output / "selection_history.csv", history)
        print(json.dumps({"stage": "pretrain_select", **row}, ensure_ascii=False), flush=True)
        if not args.smoke and epoch >= args.min_epochs and stale >= args.patience:
            break
    return best_epoch, history[-1]


def refit_all(
    args: argparse.Namespace,
    dataset: P44CSpatialFusedDataset,
    epochs: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    loader, sampler = make_loader(dataset, args.batch_size, args.workers, args.seed + 101, True)
    seed_everything(args.seed + 101)
    model = P44CPretrainModel().to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    epochs = 1 if args.smoke else max(1, epochs)
    for epoch in range(1, epochs + 1):
        lr = learning_rate(epoch, epochs, args.learning_rate, args.minimum_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        result = run_epoch(
            model,
            loader,
            sampler,
            device,
            epoch,
            optimizer,
            scaler,
            args.label_smoothing,
            maximum_batches=2 if args.smoke else 0,
        )
        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": result["losses"]["total"],
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "seconds": result["seconds"],
        }
        history.append(row)
        write_history(output / "refit_history.csv", history)
        print(json.dumps({"stage": "pretrain_refit", **row}, ensure_ascii=False), flush=True)
    checkpoint = {
        "protocol": "p44c-fold0-train-refit-v1",
        "model_state_dict": model.state_dict(),
        "epochs": epochs,
        "all_fold0_train_trials": len(dataset),
        "outer_held_or_fold0_val_predictions_generated": False,
    }
    atomic_checkpoint(output / "refit_all_fold0_train.pt", checkpoint)
    return history[-1]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = fold_train_rows(args.fold_csv.resolve())
    fit_rows = [row for row in rows if row["user_id"] not in CALIBRATION_SUBJECTS]
    calibration_rows = [row for row in rows if row["user_id"] in CALIBRATION_SUBJECTS]
    fit_ids = {row["source_id"] for row in fit_rows}
    calibration_ids = {row["source_id"] for row in calibration_rows}
    all_ids = {row["source_id"] for row in rows}
    if fit_ids & calibration_ids or len(all_ids) != 1348:
        raise RuntimeError("fold0 train/calibration partition contract failed")
    fit = make_dataset(args, fit_ids)
    calibration = make_dataset(args, calibration_ids)
    all_train = make_dataset(args, all_ids)
    device = torch.device(args.device)
    config = {
        "protocol": "p44c-shared-encoder-pretrain-fold0-v1",
        "fit_subjects": sorted({row["user_id"] for row in fit_rows}),
        "calibration_subjects": sorted(CALIBRATION_SUBJECTS),
        "fit_trials": len(fit),
        "calibration_trials": len(calibration),
        "all_train_trials": len(all_train),
        "calibration_class_note": "39/40 classes (class25 absent); fit contains all 40",
        "fold0_final_validation_used_for_training_or_epoch_selection": False,
        "spatial_layout": "all frames x D/IR x (LH,RH,workspace) x 3x3 x 128",
        "loss": "joint CE + 0.20 multimodal CE + 0.35 spatial CE + 0.08 cross-subject SupCon",
        "parameters": parameter_count(P44CPretrainModel()),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    atomic_json(output / "config.json", config)
    print(json.dumps({"stage": "start", **config}, ensure_ascii=False), flush=True)
    best_epoch, last_selection = train_select(args, fit, calibration, output, device)
    refit = None
    if not args.skip_refit:
        refit = refit_all(args, all_train, best_epoch, output, device)
    summary = {
        **config,
        "selected_epoch": best_epoch,
        "last_selection": last_selection,
        "refit_last": refit,
        "status": "pretraining_complete; no fold0 final validation was evaluated",
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
