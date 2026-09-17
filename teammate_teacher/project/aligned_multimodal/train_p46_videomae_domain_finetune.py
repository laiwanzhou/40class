from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from build_p46_videomae_cache import (
    DEFAULT_P29,
    MODEL_NAME,
    PROJECT_DIR,
    prepare_trial,
    read_rows,
    restore_legacy_attention_biases,
)
from p46_protocol import HARD_CLASS_IDS
from train_p46_videomae_head import metrics, write_csv


DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_RIDGE = PROJECT_DIR / "runs/p46_videomae_head_v1/final_head.joblib"
DEFAULT_TEACHER = PROJECT_DIR / "runs/p46_videomae_head_v1/crossfit_logits.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_domain_finetune_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Domain-adapt the final VideoMAE blocks on P46 training users. The "
            "classifier is initialized from the validated frozen-feature Ridge head."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--ridge-head", type=Path, default=DEFAULT_RIDGE)
    parser.add_argument("--teacher-oof", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--horizontal-flip", type=float, default=0.5)
    parser.add_argument("--train-all-views", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--protect-confidence", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrialDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        p29_run: Path,
        training: bool,
        seed: int,
        horizontal_flip: float,
        train_all_views: bool,
    ) -> None:
        self.rows = rows
        self.p29_run = p29_run
        self.training = training
        self.seed = seed
        self.horizontal_flip = horizontal_flip
        self.train_all_views = train_all_views
        self.epoch = 0
        self.class_to_index = {
            class_id: index for index, class_id in enumerate(HARD_CLASS_IDS)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        views, _ = prepare_trial(row, self.p29_run)
        label = self.class_to_index[int(row["class_id"])]
        if self.training:
            rng = random.Random(self.seed + 1000003 * self.epoch + index)
            if self.train_all_views:
                if rng.random() < self.horizontal_flip:
                    views = [
                        [np.ascontiguousarray(frame[:, ::-1]) for frame in video]
                        for video in views
                    ]
                return {"videos": views, "label": label, "sample_id": row["sample_id"]}
            # Deterministic cycling guarantees that every trial trains on every view
            # over three epochs, while the DataLoader still shuffles trial order.
            view_index = (index + self.epoch) % 3
            video = views[view_index]
            if rng.random() < self.horizontal_flip:
                video = [np.ascontiguousarray(frame[:, ::-1]) for frame in video]
            return {"videos": [video], "label": label, "sample_id": row["sample_id"]}
        return {"videos": views, "label": label, "sample_id": row["sample_id"]}


class VideoCollator:
    def __init__(self, processor: VideoMAEImageProcessor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        views_per_trial = len(items[0]["videos"])
        if any(len(item["videos"]) != views_per_trial for item in items):
            raise RuntimeError("mixed train/validation view counts in one batch")
        videos = [video for item in items for video in item["videos"]]
        return {
            "pixel_values": self.processor(videos, return_tensors="pt").pixel_values,
            "labels": torch.tensor([int(item["label"]) for item in items]),
            "sample_ids": [str(item["sample_id"]) for item in items],
            "views_per_trial": views_per_trial,
        }


def initialize_ridge_classifier(
    model: VideoMAEForVideoClassification, path: Path
) -> dict[str, Any]:
    pipeline = joblib.load(path.resolve())
    scaler = pipeline.named_steps["scale"]
    ridge = pipeline.named_steps["ridge"]
    classes = np.asarray(ridge.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.arange(21)):
        raise RuntimeError(f"Ridge class order changed: {classes}")
    coefficient = np.asarray(ridge.coef_, dtype=np.float32)
    intercept = np.asarray(ridge.intercept_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    raw_weight = coefficient / scale[None, :]
    raw_bias = intercept - raw_weight @ mean
    classifier = nn.Linear(768, 21)
    with torch.no_grad():
        classifier.weight.copy_(torch.from_numpy(raw_weight))
        classifier.bias.copy_(torch.from_numpy(raw_bias))
    model.classifier = classifier
    model.config.num_labels = 21
    return {
        "source": str(path.resolve()),
        "feature_contract": "L2-normalized mean of L2-normalized scene/person/workspace embeddings",
        "classes": classes.tolist(),
    }


def configure_trainable(
    model: VideoMAEForVideoClassification, unfreeze_layers: int
) -> dict[str, Any]:
    if unfreeze_layers < 1 or unfreeze_layers > len(model.videomae.encoder.layer):
        raise ValueError("--unfreeze-layers is outside the VideoMAE encoder")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected_layers = list(model.videomae.encoder.layer[-unfreeze_layers:])
    for layer in selected_layers:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    if model.fc_norm is not None:
        for parameter in model.fc_norm.parameters():
            parameter.requires_grad_(True)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        "unfrozen_encoder_layers": list(
            range(len(model.videomae.encoder.layer) - unfreeze_layers, len(model.videomae.encoder.layer))
        ),
        "trainable_parameters": int(sum(value.numel() for value in trainable.values())),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "trainable_names": list(trainable),
    }


def extract_features(
    model: VideoMAEForVideoClassification, pixel_values: torch.Tensor
) -> torch.Tensor:
    tokens = model.videomae(pixel_values).last_hidden_state
    feature = tokens.mean(dim=1)
    if model.fc_norm is not None:
        feature = model.fc_norm(feature)
    return F.normalize(feature, dim=-1)


def classify_trials(
    model: VideoMAEForVideoClassification,
    features: torch.Tensor,
    views_per_trial: int,
) -> torch.Tensor:
    if views_per_trial > 1:
        features = features.reshape(-1, views_per_trial, 768).mean(dim=1)
        features = F.normalize(features, dim=-1)
    return model.classifier(features)


def adapter_state(model: VideoMAEForVideoClassification) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_oof_teacher(path: Path, train_rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        labels = np.asarray(data["labels"], dtype=np.int64)
        logits = np.asarray(data["logits"], dtype=np.float32)
        train_mask = np.asarray(data["train_oof_mask"], dtype=bool)
    if logits.shape != (len(sample_ids), 21):
        raise RuntimeError(f"teacher logit schema changed: {logits.shape}")
    by_sample = {
        sample_id: logits[index]
        for index, sample_id in enumerate(sample_ids)
        if train_mask[index]
    }
    expected = {row["sample_id"] for row in train_rows}
    if not expected.issubset(by_sample):
        raise RuntimeError("OOF teacher does not cover all selected P46 training rows")
    label_by_sample = {
        sample_id: int(labels[index])
        for index, sample_id in enumerate(sample_ids)
        if train_mask[index]
    }
    for row in train_rows:
        if label_by_sample[row["sample_id"]] != int(row["class_id"]):
            raise RuntimeError(f"OOF teacher label mismatch: {row['sample_id']}")
    return by_sample


def load_adapter_state(
    model: VideoMAEForVideoClassification, state: dict[str, torch.Tensor]
) -> None:
    parameters = dict(model.named_parameters())
    if set(state) != {name for name, value in parameters.items() if value.requires_grad}:
        raise RuntimeError("adapter checkpoint trainable parameter contract changed")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


@torch.inference_mode()
def evaluate(
    model: VideoMAEForVideoClassification,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    losses: list[float] = []
    for batch in loader:
        labels = batch["labels"].to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            features = extract_features(model, batch["pixel_values"].to(device))
            logits = classify_trials(model, features, int(batch["views_per_trial"]))
            loss = F.cross_entropy(logits, labels)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_sample_ids.extend(batch["sample_ids"])
        losses.append(float(loss))
    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        all_sample_ids,
        float(np.mean(losses)),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not 0.0 <= args.horizontal_flip <= 1.0:
        raise ValueError("--horizontal-flip must be in [0, 1]")
    if args.distill_weight < 0.0:
        raise ValueError("--distill-weight must be non-negative")
    if not 0.0 <= args.protect_confidence <= 1.0:
        raise ValueError("--protect-confidence must be in [0, 1]")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest.resolve())
    train_rows = [row for row in rows if row["p46_split"] == "train"]
    val_rows = [row for row in rows if row["p46_split"] == "val"]
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_rows = val_rows[: args.max_val_samples]
    smoke_mode = args.max_train_samples > 0 or args.max_val_samples > 0
    if not smoke_mode and (len(train_rows), len(val_rows)) != (1094, 290):
        raise RuntimeError("Frozen P46 split counts changed")
    teacher_by_sample = load_oof_teacher(args.teacher_oof, train_rows)
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
    ridge_report = initialize_ridge_classifier(model, args.ridge_head)
    trainable_report = configure_trainable(model, args.unfreeze_layers)
    device = torch.device(args.device)
    model = model.to(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = TrialDataset(
        train_rows,
        args.p29_run.resolve(),
        training=True,
        seed=args.seed,
        horizontal_flip=args.horizontal_flip,
        train_all_views=args.train_all_views,
    )
    val_dataset = TrialDataset(
        val_rows,
        args.p29_run.resolve(),
        training=False,
        seed=args.seed,
        horizontal_flip=0.0,
        train_all_views=False,
    )
    collator = VideoCollator(processor)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
    )
    # Validation uses three views per trial, so cap its trial batch to keep the
    # actual video batch similar to training.
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 3),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    backbone_parameters: list[nn.Parameter] = []
    for layer in model.videomae.encoder.layer[-args.unfreeze_layers:]:
        backbone_parameters.extend(parameter for parameter in layer.parameters())
    norm_parameters = list(model.fc_norm.parameters()) if model.fc_norm is not None else []
    optimizer = torch.optim.AdamW(
        (
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": norm_parameters, "lr": 2.0 * args.backbone_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ),
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.max_epochs)
    warmup_steps = max(1, steps_per_epoch)

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    baseline_logits, val_labels, val_sample_ids, baseline_loss = evaluate(
        model, val_loader, device
    )
    baseline_prediction = baseline_logits.argmax(axis=1)
    baseline = metrics(val_labels, baseline_prediction)
    print(
        f"epoch=0 ridge-initialized val={100*float(baseline['accuracy']):.2f}% "
        f"loss={baseline_loss:.4f}",
        flush=True,
    )
    best_accuracy = float(baseline["accuracy"])
    best_loss = baseline_loss
    best_epoch = 0
    best_state = adapter_state(model)
    best_logits = baseline_logits
    stale = 0
    epoch_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        distill_losses: list[float] = []
        protected_total = 0
        train_correct = 0
        train_total = 0
        epoch_started = time.perf_counter()
        for batch_number, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                features = extract_features(model, batch["pixel_values"].to(device))
                logits = classify_trials(model, features, int(batch["views_per_trial"]))
                loss = criterion(logits, labels)
                teacher_logits = torch.from_numpy(
                    np.stack([teacher_by_sample[value] for value in batch["sample_ids"]])
                ).to(device)
                teacher_probability = F.softmax(teacher_logits, dim=1)
                teacher_confidence, teacher_prediction = teacher_probability.max(dim=1)
                protect = (teacher_prediction == labels) & (
                    teacher_confidence >= args.protect_confidence
                )
                if args.distill_weight > 0.0 and bool(protect.any()):
                    per_sample_distill = F.kl_div(
                        F.log_softmax(logits, dim=1),
                        teacher_probability,
                        reduction="none",
                    ).sum(dim=1)
                    distill_loss = per_sample_distill[protect].mean()
                    loss = loss + args.distill_weight * distill_loss
                    distill_losses.append(float(distill_loss.detach()))
                    protected_total += int(protect.sum())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(float(loss.detach()))
            train_correct += int((logits.argmax(dim=1) == labels).sum())
            train_total += len(labels)
            if batch_number % 50 == 0:
                print(
                    f"epoch={epoch} batch={batch_number}/{len(train_loader)} "
                    f"train={100*train_correct/train_total:.2f}% "
                    f"loss={np.mean(train_losses):.4f} protected={protected_total}",
                    flush=True,
                )
        val_logits, current_labels, current_sample_ids, val_loss = evaluate(
            model, val_loader, device
        )
        if not np.array_equal(current_labels, val_labels) or current_sample_ids != val_sample_ids:
            raise RuntimeError("validation order changed between epochs")
        val_prediction = val_logits.argmax(axis=1)
        validation = metrics(val_labels, val_prediction)
        val_accuracy = float(validation["accuracy"])
        improved = val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_loss < best_loss
        )
        if improved:
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            best_state = adapter_state(model)
            best_logits = val_logits
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": train_correct / train_total,
            "distill_loss": float(np.mean(distill_losses)) if distill_losses else 0.0,
            "protected_samples": protected_total,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in validation.items()},
            "improved": int(improved),
            "stale_epochs": stale,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[2]["lr"],
        }
        epoch_rows.append(row)
        write_csv(output / "epoch_metrics.csv", epoch_rows)
        torch.save(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "adapter_state": adapter_state(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            output / "latest_checkpoint.pt",
        )
        print(
            f"epoch={epoch} train={100*train_correct/train_total:.2f}% "
            f"val={100*val_accuracy:.2f}% ({int(validation['correct'])}/{len(val_labels)}) "
            f"best={100*best_accuracy:.2f}%@{best_epoch} stale={stale}/{args.patience} "
            f"seconds={row['elapsed_seconds']:.1f}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    load_adapter_state(model, best_state)
    final_logits, final_labels, final_sample_ids, final_loss = evaluate(model, val_loader, device)
    if final_sample_ids != val_sample_ids or not np.allclose(final_logits, best_logits, atol=2e-3):
        raise RuntimeError("reloaded best adapter does not reproduce best validation logits")
    final_prediction = final_logits.argmax(axis=1)
    final_metrics = metrics(final_labels, final_prediction)
    torch.save(
        {
            "model": args.model,
            "model_snapshot": str(snapshot),
            "unfreeze_layers": args.unfreeze_layers,
            "ridge_initialization": ridge_report,
            "adapter_state": best_state,
            "best_epoch": best_epoch,
            "validation": final_metrics,
        },
        output / "best_adapter.pt",
    )
    np.savez_compressed(
        output / "validation_logits.npz",
        sample_ids=np.asarray(final_sample_ids),
        labels=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[final_labels],
        logits=final_logits.astype(np.float32),
        predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[final_prediction],
    )
    prediction_rows: list[dict[str, Any]] = []
    row_by_sample = {row["sample_id"]: row for row in val_rows}
    for sample_id, label, prediction in zip(
        final_sample_ids, final_labels, final_prediction
    ):
        row = row_by_sample[sample_id]
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "source_id": row["source_id"],
                "user_id": row["user_id"],
                "true_class_id": int(HARD_CLASS_IDS[label]),
                "predicted_class_id": int(HARD_CLASS_IDS[prediction]),
                "correct": int(label == prediction),
            }
        )
    write_csv(output / "validation_predictions.csv", prediction_rows)
    summary = {
        "protocol": (
            "VideoMAE domain adaptation on 14 P46 training users; target four users "
            "used for epoch-level early stopping, matching the existing P46 protocol"
        ),
        "smoke_mode": smoke_mode,
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "model": args.model,
        "model_snapshot": str(snapshot),
        "attention_bias_compatibility": bias_report,
        "ridge_initialization": ridge_report,
        "train_all_views": args.train_all_views,
        "distillation": {
            "teacher_oof": str(args.teacher_oof.resolve()),
            "weight": args.distill_weight,
            "protect_confidence": args.protect_confidence,
            "policy": "only OOF-teacher-correct and sufficiently confident training samples",
        },
        "trainable": trainable_report,
        "baseline_validation": baseline,
        "best_epoch": best_epoch,
        "best_validation": final_metrics,
        "best_validation_loss": final_loss,
        "required_for_70_percent": math.ceil(0.70 * len(val_rows)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
