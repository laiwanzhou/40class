from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from p44c_model import P44CPretrainModel
from p32_fused_data import BalancedCostBucketBatchSampler
from torch.utils.data import DataLoader
from p44c_spatial_fused_data import collate_p44c
from train_p44c_pretrain_fold0 import (
    atomic_checkpoint,
    fold_train_rows,
    learning_rate,
    make_dataset,
    make_loader,
    make_optimizer,
    run_epoch,
    seed_everything,
    write_history,
)


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable P44-C all-fold0-train refit.")
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
        "--pretrain-run", type=Path, default=PROJECT_DIR / "runs" / "p44c_pretrain_fold0"
    )
    parser.add_argument("--epochs", type=int, default=17)
    parser.add_argument("--max-new-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=44030)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed + 101)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = args.pretrain_run.resolve()
    rows = fold_train_rows(args.fold_csv.resolve())
    all_ids = {row["source_id"] for row in rows}
    source = make_dataset(args, all_ids)
    sampler = BalancedCostBucketBatchSampler(
        source.frame_lengths,
        source.imu_point_lengths,
        [int(row["class_id"]) for row in source.rows],
        maximum_batch_size=args.batch_size,
        seed=args.seed + 101,
    )
    source_loader = DataLoader(
        source,
        batch_sampler=sampler,
        collate_fn=collate_p44c,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    device = torch.device(args.device)
    model = P44CPretrainModel().to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    resume_path = output / "refit_resume.pt"
    history_path = output / "refit_history.csv"
    history: list[dict[str, object]] = []
    start_epoch = 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        if "torch_rng_state" in resume:
            torch.set_rng_state(resume["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in resume:
            torch.cuda.set_rng_state(resume["cuda_rng_state"], device)
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))
    end_epoch = min(args.epochs, start_epoch + args.max_new_epochs - 1)
    print(
        json.dumps(
            {
                "stage": "resumable_refit_start",
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "target_epochs": args.epochs,
                "trials": len(source),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for epoch in range(start_epoch, end_epoch + 1):
        lr = learning_rate(
            epoch, args.epochs, args.learning_rate, args.minimum_learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        result = run_epoch(
            model,
            source_loader,
            sampler,
            device,
            epoch,
            optimizer,
            scaler,
            args.label_smoothing,
        )
        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": result["losses"]["total"],
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "seconds": result["seconds"],
        }
        # Replace a stale partial row if this epoch was rerun after the original
        # non-resumable command lost its stdout pipe.
        history = [old for old in history if int(old["epoch"]) != epoch]
        history.append(row)
        history.sort(key=lambda old: int(old["epoch"]))
        write_history(history_path, history)
        resume = {
            "protocol": "p44c-resumable-refit-v1",
            "epoch": epoch,
            "target_epochs": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "all_fold0_train_trials": len(source),
            "outer_held_or_fold0_val_predictions_generated": False,
        }
        atomic_checkpoint(resume_path, resume)
        print(json.dumps({"stage": "resumable_refit_epoch", **row}, ensure_ascii=False), flush=True)
    complete = end_epoch >= args.epochs
    if complete:
        atomic_checkpoint(
            output / "refit_all_fold0_train.pt",
            {
                "protocol": "p44c-fold0-train-refit-v1",
                "model_state_dict": model.state_dict(),
                "epochs": args.epochs,
                "all_fold0_train_trials": len(source),
                "outer_held_or_fold0_val_predictions_generated": False,
            },
        )
    print(
        json.dumps(
            {
                "stage": "resumable_refit_stop",
                "last_epoch": end_epoch,
                "complete": complete,
                "resume_path": str(resume_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
