"""Full-layer LoRA adaptation of VideoMAE-L for P90 visual teachers.

Earlier P46 work trained only the last two blocks on a 21-class subset.  This
script adapts query/value projections in every one of the 24 encoder blocks on
all 40 classes and evaluates with the common subject-disjoint three-fold OOF
protocol.  IR uses early/late x scene/person/workspace clips; Depth uses the
existing scene/person/workspace preparation for a complementary screen.
"""

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
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from build_p46_videomae_cache import restore_legacy_attention_biases
from build_p46_videomae_modality_cache import prepare_trial as prepare_modality_trial
from build_p46_videomae_multiclip_cache import (
    WINDOW_BOUNDS,
    prepare_trial as prepare_ir_trial,
    window_indices,
)
from audit_yolo11_pose_skeleton import frame_map
from build_p30_shared_dir_roi_feature_cache import read_ir
from build_p46_videomae_cache import safe_relative, square_crop
from train_p46_videomae_head import l2_normalize, make_model
from p90_teacher_common import (
    HERE,
    NUM_CLASSES,
    REPO_ROOT,
    classification_metrics,
    load_protocol,
    save_oof_artifact,
    seed_everything,
)


MODEL_NAME = "MCG-NJU/videomae-large-finetuned-kinetics"
FULL_MANIFEST = HERE / "data" / "p46_single_split.csv"
P29_RUN = HERE / "runs" / "p29_dir_multiscale_roi_full"
DEFAULT_RUN = REPO_ROOT / "runs" / "p90_videomae_lora_teacher_v1"
P85_FEATURES = HERE / "runs" / "p85_videomae_large_multiclip_full40_v1" / "complete_features.npz"


def read_aligned_rows() -> list[dict[str, str]]:
    protocol = load_protocol()
    with FULL_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {row["sample_id"]: row for row in rows}
    if set(lookup) != set(protocol.sample_ids):
        raise ValueError("P46 full manifest does not match P90 master manifest")
    return [lookup[sample_id] for sample_id in protocol.sample_ids]


def prepare_clips(row: dict[str, str], modality: str) -> list[list[np.ndarray]]:
    if modality == "ir":
        clips, _ = prepare_ir_trial(row, P29_RUN)
    elif modality == "depth":
        clips, _ = prepare_modality_trial(row, P29_RUN, "depth")
    else:
        raise ValueError(modality)
    return clips


def prepare_ir_single_clip(row: dict[str, str], clip_index: int) -> list[np.ndarray]:
    if clip_index < 0 or clip_index >= 6:
        raise ValueError(clip_index)
    window = clip_index // 3
    view = clip_index % 3
    p29_path = P29_RUN / "trial_roi_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
    with np.load(p29_path, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"]).astype(str)
        region_names = tuple(np.asarray(data["region_names"]).astype(str))
        boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
        valid = np.asarray(data["roi_valid"], dtype=bool)
    person_index = region_names.index("full_body")
    workspace_index = region_names.index("hand_workspace")
    low, high = WINDOW_BOUNDS[window]
    chosen = window_indices(len(frame_ids), low, high)
    ir_paths = frame_map(Path(row["ir_dir"]), "ir")
    video: list[np.ndarray] = []
    for index in chosen:
        frame_id = frame_ids[index]
        image = read_ir(ir_paths[frame_id])
        if view == 0:
            video.append(image)
            continue
        person_box = (
            boxes[index, person_index]
            if valid[index, person_index]
            else np.full(4, np.nan)
        )
        if view == 1:
            video.append(square_crop(image, person_box, scale=1.15))
            continue
        workspace_box = (
            boxes[index, workspace_index]
            if valid[index, workspace_index]
            else person_box
        )
        video.append(square_crop(image, workspace_box, scale=1.40))
    return video


class TrainClipDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        labels: np.ndarray,
        modality: str,
        seed: int,
    ) -> None:
        self.rows = rows
        self.labels = labels
        self.modality = modality
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_count = 6 if self.modality == "ir" else 3
        clip_index = (index + self.epoch) % clip_count
        if self.modality == "ir":
            video = prepare_ir_single_clip(self.rows[index], clip_index)
        else:
            video = prepare_clips(self.rows[index], self.modality)[clip_index]
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        if rng.random() < 0.5:
            video = [np.ascontiguousarray(frame[:, ::-1]) for frame in video]
        return {
            "videos": [video],
            "clip_indices": [clip_index],
            "label": int(self.labels[index]),
            "sample_id": self.rows[index]["sample_id"],
        }


class EvalTrialDataset(Dataset):
    def __init__(
        self, rows: list[dict[str, str]], labels: np.ndarray, modality: str
    ) -> None:
        self.rows = rows
        self.labels = labels
        self.modality = modality

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "videos": (clips := prepare_clips(self.rows[index], self.modality)),
            "clip_indices": list(range(len(clips))),
            "label": int(self.labels[index]),
            "sample_id": self.rows[index]["sample_id"],
        }


class VideoCollator:
    def __init__(self, processor: VideoMAEImageProcessor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        views = len(items[0]["videos"])
        if any(len(item["videos"]) != views for item in items):
            raise ValueError("mixed clip count")
        videos = [video for item in items for video in item["videos"]]
        clip_indices = [index for item in items for index in item["clip_indices"]]
        pixel_values = self.processor(videos, return_tensors="pt").pixel_values
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
            "clip_indices": torch.tensor(clip_indices, dtype=torch.long),
            "sample_ids": [item["sample_id"] for item in items],
            "views": views,
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.lora_b(self.lora_a(self.dropout(values))) * self.scale


class MultiClipHead(nn.Module):
    def __init__(self, clips: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(clips, NUM_CLASSES, hidden_size))
        self.bias = nn.Parameter(torch.zeros(clips, NUM_CLASSES))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, features: torch.Tensor, clip_indices: torch.Tensor) -> torch.Tensor:
        weight = self.weight[clip_indices]
        bias = self.bias[clip_indices]
        return torch.einsum("nh,nch->nc", features, weight) + bias


def class_sample_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    reference = counts[counts > 0].mean()
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    present = counts > 0
    weights[present] = np.power(reference / counts[present], power)
    result = weights[labels]
    return result / result.mean()


def initialize_ir_head_from_fold_ridge(
    head: MultiClipHead,
    protocol_sample_ids: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
) -> dict[str, Any]:
    data = np.load(P85_FEATURES, allow_pickle=False)
    source_ids = data["sample_ids"].astype(str)
    lookup = {sample_id: row for row, sample_id in enumerate(source_ids)}
    order = np.asarray([lookup[sample_id] for sample_id in protocol_sample_ids], dtype=np.int64)
    features = l2_normalize(np.asarray(data["features"][order], dtype=np.float32))
    values = features.reshape(len(features), -1)
    ridge = make_model(3000.0)
    ridge.fit(
        values[train_indices],
        labels[train_indices],
        ridge__sample_weight=class_sample_weights(labels[train_indices], 0.75),
    )
    scaler = ridge.named_steps["scale"]
    estimator = ridge.named_steps["ridge"]
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.arange(NUM_CLASSES)):
        raise ValueError("fold Ridge does not contain all 40 classes")
    coefficient = np.asarray(estimator.coef_, dtype=np.float32)
    intercept = np.asarray(estimator.intercept_, dtype=np.float32)
    raw_weight = coefficient / np.asarray(scaler.scale_, dtype=np.float32)[None]
    raw_bias = intercept - raw_weight @ np.asarray(scaler.mean_, dtype=np.float32)
    clips = head.weight.shape[0]
    split_weight = raw_weight.reshape(NUM_CLASSES, clips, 1024).transpose(1, 0, 2)
    # Evaluation averages clip logits, so multiplying each chunk by clips
    # exactly recreates the concatenated Ridge decision function.
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(split_weight * clips))
        head.bias.copy_(torch.from_numpy(np.repeat(raw_bias[None], clips, axis=0)))
    train_accuracy = float(
        np.mean(ridge.predict(values[train_indices]) == labels[train_indices])
    )
    return {
        "source": str(P85_FEATURES.resolve()),
        "alpha": 3000.0,
        "class_weight_power": 0.75,
        "train_accuracy": train_accuracy,
        "contract": "per-fold 6144D Ridge split exactly into six averaged clip heads",
    }


def configure_lora(
    model: VideoMAEForVideoClassification,
    rank: int,
    alpha: float,
    dropout: float,
    train_layernorm: bool,
    clips: int,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced = []
    for layer_index, layer in enumerate(model.videomae.encoder.layer):
        attention = layer.attention.attention
        attention.query = LoRALinear(attention.query, rank, alpha, dropout)
        attention.value = LoRALinear(attention.value, rank, alpha, dropout)
        replaced.extend(
            [
                f"videomae.encoder.layer.{layer_index}.attention.attention.query",
                f"videomae.encoder.layer.{layer_index}.attention.attention.value",
            ]
        )
    model.classifier = MultiClipHead(clips, int(model.config.hidden_size))
    model.config.num_labels = NUM_CLASSES
    if train_layernorm:
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for parameter in module.parameters():
                    parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "targets": "query,value in all encoder blocks",
        "replaced_modules": replaced,
        "train_layernorm": train_layernorm,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def extract_features(
    model: VideoMAEForVideoClassification, pixel_values: torch.Tensor
) -> torch.Tensor:
    tokens = model.videomae(pixel_values).last_hidden_state
    features = tokens.mean(dim=1)
    if model.fc_norm is not None:
        features = model.fc_norm(features)
    return features


def classify_batch(
    model: VideoMAEForVideoClassification,
    pixel_values: torch.Tensor,
    clip_indices: torch.Tensor,
    views: int,
) -> torch.Tensor:
    features = F.normalize(extract_features(model, pixel_values), dim=-1)
    logits = model.classifier(features, clip_indices)
    if views > 1:
        logits = logits.reshape(-1, views, NUM_CLASSES).mean(dim=1)
    return logits


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@torch.inference_mode()
def evaluate(
    model: VideoMAEForVideoClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []
    for batch_number, batch in enumerate(loader):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = classify_batch(
                model, pixels, batch["clip_indices"].to(device), int(batch["views"])
            )
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(batch["labels"].numpy())
        all_ids.extend(batch["sample_ids"])
        if (batch_number + 1) % 20 == 0:
            print(f"    eval trials={(batch_number + 1) * loader.batch_size}", flush=True)
    return np.concatenate(all_logits), np.concatenate(all_labels), all_ids


def build_model_and_processor(args: argparse.Namespace, device: torch.device):
    snapshot = Path(
        snapshot_download(
            args.model,
            allow_patterns=("*.json", "*.safetensors", "*.txt"),
            local_files_only=True,
        )
    )
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(snapshot, local_files_only=True)
    bias_report = restore_legacy_attention_biases(model, snapshot)
    if int(model.config.hidden_size) != 1024 or len(model.videomae.encoder.layer) != 24:
        raise ValueError("P90 expects the official VideoMAE-L architecture")
    clips = 6 if args.modality == "ir" else 3
    lora_report = configure_lora(
        model, args.rank, args.alpha, args.lora_dropout, args.train_layernorm, clips
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to(device)
    return model, processor, snapshot, bias_report, lora_report


def train_fold(
    args: argparse.Namespace,
    fold: int,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    seed_everything(args.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, snapshot, bias_report, lora_report = build_model_and_processor(args, device)
    protocol = load_protocol()
    ridge_report = None
    if args.modality == "ir":
        ridge_report = initialize_ir_head_from_fold_ridge(
            model.classifier, protocol.sample_ids, labels, train_indices
        )
    selected_train = train_indices[: args.max_train_samples] if args.max_train_samples else train_indices
    selected_val = val_indices[: args.max_val_samples] if args.max_val_samples else val_indices
    train_dataset = TrainClipDataset(
        [rows[i] for i in selected_train], labels[selected_train], args.modality, args.seed + fold
    )
    val_dataset = EvalTrialDataset(
        [rows[i] for i in selected_val], labels[selected_val], args.modality
    )
    collator = VideoCollator(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + fold),
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_trial_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    lora_parameters = []
    norm_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith("classifier"):
            continue
        if "lora_" in name:
            lora_parameters.append(parameter)
        else:
            norm_parameters.append(parameter)
    groups = [
        {"params": lora_parameters, "lr": args.lora_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ]
    if norm_parameters:
        groups.append({"params": norm_parameters, "lr": args.norm_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = max(1, updates_per_epoch // 2)

    def schedule(step: int) -> float:
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    print(
        f"videomae_large_{args.modality}_lora: fold={fold}, train={len(train_dataset)}, val={len(val_dataset)}, "
        f"trainable={lora_report['trainable_parameters']:,}",
        flush=True,
    )
    started = time.time()
    update = 0
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch_number, batch in enumerate(train_loader):
            labels_batch = batch["labels"].to(device)
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = classify_batch(
                    model,
                    pixels,
                    batch["clip_indices"].to(device),
                    int(batch["views"]),
                )
                loss = criterion(logits, labels_batch) / args.gradient_accumulation
            scaler.scale(loss).backward()
            should_step = (
                (batch_number + 1) % args.gradient_accumulation == 0
                or batch_number + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update += 1
            total_loss += float(loss.detach()) * args.gradient_accumulation * len(labels_batch)
            correct += int((logits.argmax(dim=1) == labels_batch).sum())
            seen += len(labels_batch)
            if (batch_number + 1) % 100 == 0:
                print(
                    f"    epoch {epoch + 1}/{args.epochs} samples={seen}/{len(train_dataset)} "
                    f"loss={total_loss / seen:.4f} acc={correct / seen:.4f}",
                    flush=True,
                )
        print(
            f"  epoch {epoch + 1}/{args.epochs}: loss={total_loss / seen:.4f}, "
            f"train_acc={correct / seen:.4f}, elapsed_min={(time.time() - started) / 60:.1f}",
            flush=True,
        )
    logits, val_labels, sample_ids = evaluate(model, val_loader, device)
    if not np.array_equal(val_labels, labels[selected_val]):
        raise ValueError("visual validation label order changed")
    metrics = classification_metrics(logits, val_labels)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state": trainable_state(model),
            "fold": fold,
            "modality": args.modality,
            "metrics": metrics,
            "lora": lora_report,
            "snapshot": str(snapshot),
        },
        checkpoint_dir / f"videomae_large_{args.modality}_lora_r{args.rank}_fold{fold}.pt",
    )
    return logits, sample_ids, {
        "metrics": metrics,
        "snapshot": str(snapshot),
        "attention_bias": bias_report,
        "lora": lora_report,
        "ridge_initialization": ridge_report,
        "elapsed_seconds": time.time() - started,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }


def run(args: argparse.Namespace) -> None:
    protocol = load_protocol()
    rows = read_aligned_rows()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_name = f"videomae_large_{args.modality}_lora_r{args.rank}"
    oof_logits = np.zeros((len(protocol.labels), NUM_CLASSES), dtype=np.float32)
    completed = np.zeros(len(protocol.labels), dtype=bool)
    reports = []
    folds = [args.fold_only] if args.fold_only is not None else list(range(3))
    for fold in folds:
        val_indices = protocol.val_indices(fold)
        logits, sample_ids, report = train_fold(
            args,
            fold,
            rows,
            protocol.labels,
            protocol.train_indices(fold),
            val_indices,
            output_dir,
        )
        expected_indices = val_indices[: args.max_val_samples] if args.max_val_samples else val_indices
        if sample_ids != protocol.sample_ids[expected_indices].tolist():
            raise ValueError("visual validation sample order changed")
        oof_logits[expected_indices] = logits
        completed[expected_indices] = True
        report["fold"] = fold
        reports.append(report)
        print(f"fold {fold} accuracy={report['metrics']['accuracy']:.6f}", flush=True)
    if args.fold_only is not None or args.max_train_samples or args.max_val_samples:
        np.savez_compressed(
            output_dir / f"{teacher_name}_partial.npz",
            sample_ids=protocol.sample_ids[completed],
            labels=protocol.labels[completed],
            fold_id=protocol.fold_id[completed],
            logits=oof_logits[completed],
        )
        (output_dir / f"{teacher_name}_partial.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return
    payload = save_oof_artifact(
        output_dir,
        teacher_name,
        oof_logits,
        protocol,
        metadata={
            "model": args.model,
            "modality": args.modality,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "rank": args.rank,
            "alpha": args.alpha,
            "lora_lr": args.lora_lr,
            "norm_lr": args.norm_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "clip_policy": (
                "early/late x scene/person/workspace cyclic train; six-logit mean eval"
                if args.modality == "ir"
                else "scene/person/workspace cyclic train; three-logit mean eval"
            ),
            "selection": "fixed epochs; outer validation evaluated once after training",
            "fold_reports": reports,
        },
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--modality", choices=("ir", "depth"), default="ir")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.10)
    parser.add_argument("--train-layernorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-trial-batch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--norm-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=9029)
    parser.add_argument("--fold-only", type=int, choices=(0, 1, 2))
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    run(parse_args())
