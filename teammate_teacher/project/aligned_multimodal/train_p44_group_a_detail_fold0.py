from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from p32_fused_data import (
    LengthBucketBatchSampler,
    P32FusedTrialDataset,
    collate_p32_trials,
)
from p32_part_fusion_temporal_model import (
    P32PartFusionTemporalModel,
    model_size_mib,
    parameter_count,
)


PROJECT_DIR = Path(__file__).resolve().parent
GROUP_CLASS_IDS = (6, 7, 8, 9, 10, 11, 14, 37)
LOCAL_REGION_INDICES = (1, 2, 3, 4, 5)  # L/R arms, L/R hands, hand workspace.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P44 fold0 pilot: food/tableware eight-class local Detail expert"
    )
    parser.add_argument(
        "--visual-run",
        type=Path,
        default=PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full",
    )
    parser.add_argument(
        "--motion-run",
        type=Path,
        default=PROJECT_DIR / "runs" / "p31_skeleton_imu_full",
    )
    parser.add_argument(
        "--fold-csv",
        type=Path,
        default=PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv",
    )
    parser.add_argument(
        "--base-logits",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p44_p12_base_inner_oof"
        / "fold0_pilot"
        / "fold0_base_logits.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p44_group_a_detail_fold0_pilot",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_id(row: dict[str, str]) -> str:
    return f"{row['class_name']}/{row['user_id']}/{row['trial_id']}"


def selected_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    selected = {"train": [], "val": []}
    for row in read_csv(path):
        split = row["split"]
        if split in selected and int(row["class_id"]) in GROUP_CLASS_IDS:
            item = dict(row)
            item["source_id"] = source_id(row)
            selected[split].append(item)
    return selected


def write_group_manifest(path: Path, rows: dict[str, list[dict[str, str]]]) -> None:
    combined = [dict(row, group_split=split) for split, values in rows.items() for row in values]
    fields = ["group_split"] + [key for key in combined[0] if key != "group_split"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)


def make_loaders(
    train_dataset: P32FusedTrialDataset,
    val_dataset: P32FusedTrialDataset,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, LengthBucketBatchSampler, DataLoader, LengthBucketBatchSampler]:
    train_sampler = LengthBucketBatchSampler(
        train_dataset.frame_lengths,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_sampler = LengthBucketBatchSampler(
        val_dataset.frame_lengths,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "collate_fn": collate_p32_trials,
    }
    return (
        DataLoader(train_dataset, batch_sampler=train_sampler, **common),
        train_sampler,
        DataLoader(val_dataset, batch_sampler=val_sampler, **common),
        val_sampler,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def mapped_labels(labels: torch.Tensor, lookup: torch.Tensor) -> torch.Tensor:
    mapped = lookup[labels]
    if bool((mapped < 0).any()):
        raise RuntimeError("Batch contains a label outside Group A")
    return mapped


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    pooled = (tokens * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)
    present = mask.any(dim=(1, 2))
    return pooled, present


class DetailHeads(nn.Module):
    def __init__(self, classes: int = len(GROUP_CLASS_IDS)) -> None:
        super().__init__()
        self.main = nn.Sequential(nn.LayerNorm(384), nn.Dropout(0.15), nn.Linear(384, classes))
        self.visual = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.10), nn.Linear(256, classes))
        self.skeleton = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.10), nn.Linear(256, classes))
        self.imu = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.10), nn.Linear(256, classes))

    def forward(self, output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual, visual_present = masked_mean(
            output["visual_part_tokens"], output["visual_part_mask"]
        )
        skeleton, skeleton_present = masked_mean(
            output["skeleton_part_tokens"], output["skeleton_part_mask"]
        )
        imu, imu_present = masked_mean(output["imu_part_tokens"], output["imu_part_mask"])
        return {
            "main": self.main(output["trial_embedding"]),
            "visual": self.visual(visual),
            "skeleton": self.skeleton(skeleton),
            "imu": self.imu(imu),
            "visual_present": visual_present,
            "skeleton_present": skeleton_present,
            "imu_present": imu_present,
        }


def auxiliary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    present: torch.Tensor,
    class_weights: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    if not bool(present.any()):
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(
        logits[present],
        labels[present],
        weight=class_weights,
        label_smoothing=smoothing,
    )


def learning_rate_for_epoch(
    epoch: int,
    max_epochs: int,
    warmup_epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_learning_rate * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_learning_rate + (base_learning_rate - minimum_learning_rate) * cosine


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    rows = []
    for class_id in GROUP_CLASS_IDS:
        selected = labels == class_id
        predicted = predictions == class_id
        correct = int((selected & predicted).sum())
        support = int(selected.sum())
        predicted_count = int(predicted.sum())
        recall = correct / support if support else 0.0
        precision = correct / predicted_count if predicted_count else 0.0
        class_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class_id": class_id,
                "support": support,
                "predicted": predicted_count,
                "correct": correct,
                "recall": recall,
                "precision": precision,
                "f1": class_f1,
            }
        )
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "per_class": rows,
    }


def perturb_local_roi(batch: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "none":
        return batch
    output = dict(batch)
    for key in ("features", "roi_quality", "roi_valid", "roi_source", "roi_clipped_ratio"):
        output[key] = batch[key].clone()
    local = LOCAL_REGION_INDICES
    if mode == "zero":
        output["features"][:, :, :, local] = 0
        output["roi_quality"][:, :, local] = 0
        output["roi_valid"][:, :, local] = False
        output["roi_source"][:, :, local] = 0
        output["roi_clipped_ratio"][:, :, local] = 0
        return output
    if mode != "shuffle":
        raise ValueError(f"Unknown ROI perturbation: {mode}")
    batch_size = int(batch["features"].shape[0])
    if batch_size < 2:
        return output
    shift = max(1, batch_size // 2)
    permutation = torch.roll(torch.arange(batch_size, device=batch["features"].device), shift)
    output["features"][:, :, :, local] = batch["features"][permutation][:, :, :, local]
    for key in ("roi_quality", "roi_valid", "roi_source", "roi_clipped_ratio"):
        output[key][:, :, local] = batch[key][permutation][:, :, local]
    return output


def train_epoch(
    model: P32PartFusionTemporalModel,
    heads: DetailHeads,
    loader: DataLoader,
    sampler: LengthBucketBatchSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    label_lookup: torch.Tensor,
    class_weights: torch.Tensor,
    smoothing: float,
) -> dict[str, float]:
    sampler.set_epoch(epoch)
    model.train()
    heads.train()
    started = time.perf_counter()
    total = 0
    correct = 0
    loss_sum = 0.0
    main_sum = 0.0
    visual_sum = 0.0
    skeleton_sum = 0.0
    imu_sum = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        labels = mapped_labels(batch["label"], label_lookup)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            detail = heads(model(batch))
            main_loss = nn.functional.cross_entropy(
                detail["main"],
                labels,
                weight=class_weights,
                label_smoothing=smoothing,
            )
            visual_loss = auxiliary_loss(
                detail["visual"], labels, detail["visual_present"], class_weights, smoothing
            )
            skeleton_loss = auxiliary_loss(
                detail["skeleton"], labels, detail["skeleton_present"], class_weights, smoothing
            )
            imu_loss = auxiliary_loss(
                detail["imu"], labels, detail["imu_present"], class_weights, smoothing
            )
            loss = main_loss + 0.20 * visual_loss + 0.15 * skeleton_loss + 0.15 * imu_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 5.0)
        scaler.step(optimizer)
        scaler.update()
        count = int(len(labels))
        total += count
        correct += int((detail["main"].argmax(1) == labels).sum().item())
        loss_sum += float(loss.detach()) * count
        main_sum += float(main_loss.detach()) * count
        visual_sum += float(visual_loss.detach()) * count
        skeleton_sum += float(skeleton_loss.detach()) * count
        imu_sum += float(imu_loss.detach()) * count
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "loss": loss_sum / total,
        "main_loss": main_sum / total,
        "visual_aux_loss": visual_sum / total,
        "skeleton_aux_loss": skeleton_sum / total,
        "imu_aux_loss": imu_sum / total,
        "accuracy": correct / total,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate(
    model: P32PartFusionTemporalModel,
    heads: DetailHeads,
    loader: DataLoader,
    sampler: LengthBucketBatchSampler,
    device: torch.device,
    label_lookup: torch.Tensor,
    perturbation: str = "none",
) -> dict[str, Any]:
    sampler.set_epoch(0)
    model.eval()
    heads.eval()
    sample_ids: list[str] = []
    users: list[str] = []
    labels_original: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    started = time.perf_counter()
    for batch in loader:
        sample_ids.extend(batch["sample_id"])
        users.extend(batch["user_id"])
        labels_original.append(batch["label"].numpy())
        batch = move_batch(batch, device)
        batch = perturb_local_roi(batch, perturbation)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = heads(model(batch))["main"]
        logits_all.append(logits.float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_original).astype(np.int64)
    group_indices = logits.argmax(1)
    predictions = np.asarray([GROUP_CLASS_IDS[index] for index in group_indices], dtype=np.int64)
    return {
        "sample_ids": sample_ids,
        "users": users,
        "labels": labels,
        "logits": logits,
        "predictions": predictions,
        "metrics": metric_dict(labels, predictions),
        "seconds": time.perf_counter() - started,
        "perturbation": perturbation,
    }


def checkpoint_payload(
    model: P32PartFusionTemporalModel,
    heads: DetailHeads,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "P44_GroupA_Detail_fold0_pilot_v1",
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "heads_state_dict": {key: value.detach().cpu() for key, value in heads.state_dict().items()},
    }


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[P32PartFusionTemporalModel, DetailHeads, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = P32PartFusionTemporalModel().to(device)
    heads = DetailHeads().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    heads.load_state_dict(checkpoint["heads_state_dict"])
    return model, heads, checkpoint


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(path: Path, evaluation: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        sample_ids=np.asarray(evaluation["sample_ids"]),
        users=np.asarray(evaluation["users"]),
        labels=evaluation["labels"],
        predictions=evaluation["predictions"],
        logits=evaluation["logits"].astype(np.float16),
        perturbation=np.asarray(evaluation["perturbation"]),
    )


def base_group_evaluation(
    base_path: Path,
    val_rows: list[dict[str, str]],
    detail: dict[str, Any],
) -> dict[str, Any]:
    with np.load(base_path, allow_pickle=False) as data:
        base_ids = data["sample_ids"].astype(str)
        base_labels = data["labels"].astype(np.int64)
        base_logits = data["base_logits"].astype(np.float32)
        outer_held = bool(data["outer_held_predictions_generated"].item())
    if outer_held:
        raise RuntimeError("Base artifact includes outer-held predictions")
    base_lookup = {sample_id: index for index, sample_id in enumerate(base_ids)}
    row_by_source = {row["source_id"]: row for row in val_rows}
    aligned_indices = []
    for detail_id in detail["sample_ids"]:
        row = row_by_source[detail_id]
        aligned_indices.append(base_lookup[row["sample_id"]])
    aligned_indices_array = np.asarray(aligned_indices, dtype=np.int64)
    aligned_labels = base_labels[aligned_indices_array]
    if not np.array_equal(aligned_labels, detail["labels"]):
        raise RuntimeError("Base/Detail labels do not align")
    aligned_logits = base_logits[aligned_indices_array]
    unrestricted = aligned_logits.argmax(1)
    group_logits = aligned_logits[:, np.asarray(GROUP_CLASS_IDS)]
    restricted = np.asarray(
        [GROUP_CLASS_IDS[index] for index in group_logits.argmax(1)], dtype=np.int64
    )
    detail_predictions = detail["predictions"]
    base_correct = restricted == aligned_labels
    detail_correct = detail_predictions == aligned_labels
    return {
        "unrestricted_predictions": unrestricted,
        "restricted_predictions": restricted,
        "metrics_unrestricted": metric_dict(aligned_labels, unrestricted),
        "metrics_restricted_oracle_group": metric_dict(aligned_labels, restricted),
        "rescue_base_wrong_detail_right": int((~base_correct & detail_correct).sum()),
        "new_error_base_right_detail_wrong": int((base_correct & ~detail_correct).sum()),
        "both_correct": int((base_correct & detail_correct).sum()),
        "both_wrong": int((~base_correct & ~detail_correct).sum()),
        "net_rescue": int((~base_correct & detail_correct).sum() - (base_correct & ~detail_correct).sum()),
    }


def write_comparison_rows(
    path: Path,
    detail: dict[str, Any],
    shuffled: dict[str, Any],
    zeroed: dict[str, Any],
    base: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "user_id",
                "label",
                "base_unrestricted",
                "base_restricted",
                "detail",
                "detail_local_shuffle",
                "detail_local_zero",
                "base_restricted_correct",
                "detail_correct",
                "outcome",
            ),
        )
        writer.writeheader()
        for index, sample_id in enumerate(detail["sample_ids"]):
            label = int(detail["labels"][index])
            base_prediction = int(base["restricted_predictions"][index])
            detail_prediction = int(detail["predictions"][index])
            base_correct = base_prediction == label
            detail_correct = detail_prediction == label
            if not base_correct and detail_correct:
                outcome = "rescue"
            elif base_correct and not detail_correct:
                outcome = "new_error"
            elif base_correct:
                outcome = "both_correct"
            else:
                outcome = "both_wrong"
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "user_id": detail["users"][index],
                    "label": label,
                    "base_unrestricted": int(base["unrestricted_predictions"][index]),
                    "base_restricted": base_prediction,
                    "detail": detail_prediction,
                    "detail_local_shuffle": int(shuffled["predictions"][index]),
                    "detail_local_zero": int(zeroed["predictions"][index]),
                    "base_restricted_correct": int(base_correct),
                    "detail_correct": int(detail_correct),
                    "outcome": outcome,
                }
            )


def main() -> None:
    args = parse_args()
    if args.min_epochs > args.max_epochs:
        raise ValueError("min-epochs cannot exceed max-epochs")
    seed_everything(int(args.seed))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(args.fold_csv.resolve())
    write_group_manifest(output / "group_manifest.csv", rows)
    train_ids = {row["source_id"] for row in rows["train"]}
    val_ids = {row["source_id"] for row in rows["val"]}
    if train_ids & val_ids:
        raise RuntimeError("Group train/val sample overlap")
    train_subjects = sorted({row["user_id"] for row in rows["train"]})
    val_subjects = sorted({row["user_id"] for row in rows["val"]})
    if set(train_subjects) & set(val_subjects):
        raise RuntimeError("Group train/val subject overlap")

    train_dataset = P32FusedTrialDataset(
        args.visual_run.resolve(), args.motion_run.resolve(), sample_ids=train_ids
    )
    val_dataset = P32FusedTrialDataset(
        args.visual_run.resolve(), args.motion_run.resolve(), sample_ids=val_ids
    )
    if len(train_dataset) != len(train_ids) or len(val_dataset) != len(val_ids):
        raise RuntimeError(
            f"Group/cache mismatch: train {len(train_dataset)}/{len(train_ids)}, "
            f"val {len(val_dataset)}/{len(val_ids)}"
        )
    if any(int(row["class_id"]) not in GROUP_CLASS_IDS for row in train_dataset.rows + val_dataset.rows):
        raise RuntimeError("Dataset contains a class outside Group A")

    train_loader, train_sampler, val_loader, val_sampler = make_loaders(
        train_dataset,
        val_dataset,
        int(args.batch_size),
        int(args.eval_batch_size),
        int(args.workers),
        int(args.seed),
    )
    device = torch.device(args.device)
    model = P32PartFusionTemporalModel().to(device)
    heads = DetailHeads().to(device)
    label_lookup = torch.full((40,), -1, dtype=torch.long, device=device)
    for group_index, class_id in enumerate(GROUP_CLASS_IDS):
        label_lookup[class_id] = group_index
    class_counts = np.asarray(
        [sum(int(row["class_id"]) == class_id for row in train_dataset.rows) for class_id in GROUP_CLASS_IDS],
        dtype=np.int64,
    )
    class_weight_values = len(train_dataset) / (len(GROUP_CLASS_IDS) * class_counts)
    class_weight_values = class_weight_values / class_weight_values.mean()
    class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)

    parameters = list(model.parameters()) + list(heads.parameters())
    decay = [parameter for parameter in parameters if parameter.requires_grad and parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters if parameter.requires_grad and parameter.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(args.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(args.learning_rate),
        foreach=False,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = {
        "protocol": "p44-group-a-detail-fold0-pilot-v1",
        "status": "exploratory_group_selected_on_same_fold0_validation",
        "group_class_ids": list(GROUP_CLASS_IDS),
        "fold_csv": str(args.fold_csv.resolve()),
        "visual_run": str(args.visual_run.resolve()),
        "motion_run": str(args.motion_run.resolve()),
        "base_logits": str(args.base_logits.resolve()),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "train_trials": len(train_dataset),
        "val_trials": len(val_dataset),
        "train_class_counts": class_counts.tolist(),
        "class_weights": class_weight_values.tolist(),
        "all_group_training_trials_used_once_per_epoch": True,
        "all_original_frames_preserved": True,
        "pretrained_p32_checkpoint_loaded": False,
        "loss": "group_ce + 0.20 visual_aux_ce + 0.15 skeleton_aux_ce + 0.15 imu_aux_ce",
        "local_roi_regions": ["left_arm", "right_arm", "left_hand", "right_hand", "hand_workspace"],
        "max_epochs": int(args.max_epochs),
        "min_epochs": int(args.min_epochs),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "minimum_learning_rate": float(args.minimum_learning_rate),
        "warmup_epochs": int(args.warmup_epochs),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "seed": int(args.seed),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": parameter_count(model) + parameter_count(heads),
        "fp32_mib": model_size_mib(model, 4) + model_size_mib(heads, 4),
    }
    atomic_json(output / "frozen_config.json", config)
    print(json.dumps({"stage": "start", **config}, ensure_ascii=False), flush=True)

    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_accuracy_epoch = 0
    best_macro_f1_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for epoch in range(1, int(args.max_epochs) + 1):
        learning_rate = learning_rate_for_epoch(
            epoch,
            int(args.max_epochs),
            int(args.warmup_epochs),
            float(args.learning_rate),
            float(args.minimum_learning_rate),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train = train_epoch(
            model,
            heads,
            train_loader,
            train_sampler,
            optimizer,
            scaler,
            device,
            epoch,
            label_lookup,
            class_weights,
            float(args.label_smoothing),
        )
        val = evaluate(model, heads, val_loader, val_sampler, device, label_lookup)
        accuracy = float(val["metrics"]["accuracy"])
        macro_f1 = float(val["metrics"]["macro_f1"])
        accuracy_improved = accuracy > best_accuracy + float(args.min_delta)
        macro_improved = macro_f1 > best_macro_f1 + float(args.min_delta)
        if accuracy_improved:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch
            atomic_checkpoint(
                output / "best_accuracy.pt",
                checkpoint_payload(model, heads, epoch, val["metrics"], config),
            )
        if macro_improved:
            best_macro_f1 = macro_f1
            best_macro_f1_epoch = epoch
            atomic_checkpoint(
                output / "best_macro_f1.pt",
                checkpoint_payload(model, heads, epoch, val["metrics"], config),
            )
        stale_epochs = 0 if (accuracy_improved or macro_improved) else stale_epochs + 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train["loss"],
                "train_main_loss": train["main_loss"],
                "train_visual_aux_loss": train["visual_aux_loss"],
                "train_skeleton_aux_loss": train["skeleton_aux_loss"],
                "train_imu_aux_loss": train["imu_aux_loss"],
                "train_accuracy": train["accuracy"],
                "train_seconds": train["seconds"],
                "val_accuracy": accuracy,
                "val_balanced_accuracy": val["metrics"]["balanced_accuracy"],
                "val_macro_f1": macro_f1,
                "val_seconds": val["seconds"],
                "best_accuracy_epoch": best_accuracy_epoch,
                "best_macro_f1_epoch": best_macro_f1_epoch,
                "stale_epochs": stale_epochs,
            }
        )
        write_history(output / "history.csv", history)
        atomic_json(output / "history.json", history)
        print(
            json.dumps(
                {
                    "stage": "epoch",
                    "epoch": epoch,
                    "train_accuracy": train["accuracy"],
                    "val_accuracy": accuracy,
                    "val_macro_f1": macro_f1,
                    "train_seconds": train["seconds"],
                    "val_seconds": val["seconds"],
                    "stale_epochs": stale_epochs,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if epoch >= int(args.min_epochs) and stale_epochs >= int(args.patience):
            break

    primary_model, primary_heads, primary_checkpoint = load_checkpoint(
        output / "best_macro_f1.pt", device
    )
    normal = evaluate(
        primary_model, primary_heads, val_loader, val_sampler, device, label_lookup, "none"
    )
    shuffled = evaluate(
        primary_model, primary_heads, val_loader, val_sampler, device, label_lookup, "shuffle"
    )
    zeroed = evaluate(
        primary_model, primary_heads, val_loader, val_sampler, device, label_lookup, "zero"
    )
    write_evaluation(output / "detail_normal_logits.npz", normal)
    write_evaluation(output / "detail_local_shuffle_logits.npz", shuffled)
    write_evaluation(output / "detail_local_zero_logits.npz", zeroed)
    base = base_group_evaluation(args.base_logits.resolve(), rows["val"], normal)
    write_comparison_rows(output / "per_sample_comparison.csv", normal, shuffled, zeroed, base)
    roi_audit = {
        "normal": normal["metrics"],
        "local_roi_shuffled_across_trials": shuffled["metrics"],
        "local_roi_zeroed": zeroed["metrics"],
        "normal_minus_shuffle_accuracy_pp": 100
        * (normal["metrics"]["accuracy"] - shuffled["metrics"]["accuracy"]),
        "normal_minus_shuffle_macro_f1_pp": 100
        * (normal["metrics"]["macro_f1"] - shuffled["metrics"]["macro_f1"]),
        "normal_minus_zero_accuracy_pp": 100
        * (normal["metrics"]["accuracy"] - zeroed["metrics"]["accuracy"]),
        "normal_minus_zero_macro_f1_pp": 100
        * (normal["metrics"]["macro_f1"] - zeroed["metrics"]["macro_f1"]),
    }
    summary = {
        "protocol": "p44-group-a-detail-fold0-pilot-v1",
        "status": "completed_exploratory_fold0_only_router_not_started",
        "warning": "Group A was selected from these same fold0 validation errors; fold1 is required for an unbiased confirmation.",
        "primary_checkpoint": {
            "selection": "best fold0 group macro-F1",
            "epoch": int(primary_checkpoint["epoch"]),
            "path": str((output / "best_macro_f1.pt").resolve()),
        },
        "counts": {
            "train_trials": len(train_dataset),
            "val_trials": len(val_dataset),
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
        },
        "base_comparison": {
            key: value
            for key, value in base.items()
            if key not in {"unrestricted_predictions", "restricted_predictions"}
        },
        "detail_metrics": normal["metrics"],
        "roi_information_audit": roi_audit,
        "training": {
            "epochs_completed": len(history),
            "elapsed_seconds": time.perf_counter() - run_started,
            "best_accuracy": best_accuracy,
            "best_accuracy_epoch": best_accuracy_epoch,
            "best_macro_f1": best_macro_f1,
            "best_macro_f1_epoch": best_macro_f1_epoch,
        },
        "stop_condition": "Detail feasibility evaluated; Router and global 40-class integration not started.",
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
