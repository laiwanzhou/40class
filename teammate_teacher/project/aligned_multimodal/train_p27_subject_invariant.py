from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Sampler

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel
from train import move_inputs, seed_everything, seed_worker


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the legal IR+Skeleton base with cross-subject "
            "same-class supervised contrastive regularization"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contrastive-weight", type=float, required=True)
    return parser.parse_args()


class CrossSubjectClassBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: AlignedMultimodalDataset,
        *,
        classes_per_batch: int,
        seed: int,
    ) -> None:
        self.classes_per_batch = int(classes_per_batch)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[int, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, sample in enumerate(dataset.samples):
            grouped[sample.class_id][sample.user_id].append(index)
        self.grouped = {
            class_id: dict(by_user)
            for class_id, by_user in grouped.items()
            if len(by_user) >= 2
        }
        if len(self.grouped) < self.classes_per_batch:
            raise RuntimeError(
                f"Only {len(self.grouped)} classes have two train subjects"
            )
        self.classes = np.asarray(sorted(self.grouped), dtype=np.int64)
        self.batches = max(
            1, len(dataset) // (2 * self.classes_per_batch)
        )

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches):
            selected_classes = rng.choice(
                self.classes, self.classes_per_batch, replace=False
            )
            batch: list[int] = []
            for class_id in selected_classes:
                by_user = self.grouped[int(class_id)]
                users = np.asarray(sorted(by_user), dtype=object)
                selected_users = rng.choice(users, 2, replace=False)
                for user in selected_users:
                    indices = by_user[str(user)]
                    batch.append(int(rng.choice(indices)))
            rng.shuffle(batch)
            yield batch


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 1152) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(embedding), dim=1)


def subject_contrastive_loss(
    projection: torch.Tensor,
    labels: torch.Tensor,
    users: list[str],
    temperature: float,
) -> torch.Tensor:
    similarity = projection @ projection.T / float(temperature)
    batch_size = len(labels)
    eye = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    same_class = labels[:, None].eq(labels[None, :])
    user_codes_lookup = {
        user: index for index, user in enumerate(sorted(set(users)))
    }
    user_codes = torch.tensor(
        [user_codes_lookup[user] for user in users],
        dtype=torch.long,
        device=labels.device,
    )
    different_subject = user_codes[:, None].ne(user_codes[None, :])
    positives = same_class & different_subject & ~eye
    valid = positives.sum(dim=1) > 0
    if not bool(valid.any()):
        return similarity.sum() * 0.0
    denominator_mask = ~eye
    masked_similarity = similarity.masked_fill(~denominator_mask, -torch.inf)
    log_probability = similarity - torch.logsumexp(
        masked_similarity, dim=1, keepdim=True
    )
    per_anchor = -(
        log_probability.masked_fill(~positives, 0.0).sum(dim=1)
        / positives.sum(dim=1).clamp_min(1)
    )
    return per_anchor[valid].mean()


def metric_values(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def make_model(config: dict[str, Any]) -> AlignedMultimodalModel:
    return AlignedMultimodalModel(
        list(config["modalities"]),
        dropout=float(config["dropout"]),
        use_aux_heads=False,
        use_modality_masks=bool(config.get("use_modality_masks", False)),
        use_cross_attention=bool(config.get("use_cross_attention", False)),
        cross_attention_grid=tuple(config.get("cross_attention_grid", [2, 3])),
        cross_attention_heads=int(config.get("cross_attention_heads", 4)),
        depth_input_channels=int(config.get("depth_input_channels", 3)),
        use_layer3_spatial=bool(config.get("use_layer3_spatial", False)),
        visual_stem_fusion=str(config.get("visual_stem_fusion", "concat")),
        skeleton_input_dim=int(config.get("skeleton_input_dim", 4)),
    )


def run_validation(
    model: AlignedMultimodalModel,
    loader: DataLoader,
    modalities: list[str],
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    sample_ids: list[str] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = move_inputs(batch, modalities, device)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(inputs)
            assert isinstance(output, torch.Tensor)
            labels.append(batch["label"])
            logits.append(output.float().cpu())
            sample_ids.extend(batch["sample_id"])
    label_array = torch.cat(labels).numpy()
    logit_array = torch.cat(logits).numpy()
    return {
        **metric_values(label_array, logit_array.argmax(axis=1)),
        "sample_ids": np.asarray(sample_ids),
        "labels": label_array,
        "logits": logit_array,
    }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    weight = float(args.contrastive_weight)
    if weight < 0:
        raise ValueError("--contrastive-weight must be non-negative")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    modalities = list(config["modalities"])
    dataset_args = {
        "manifest_path": args.manifest.resolve(),
        "modalities": modalities,
        "num_frames": int(config["num_frames"]),
        "image_height": int(config["image_height"]),
        "image_width": int(config["image_width"]),
        "cache_dir": config.get("cache_dir"),
        "skeleton_strategy": config.get("skeleton_strategy", "first"),
        "depth_representation": config.get("depth_representation", "jet_rgb"),
        "visual_normalization": config.get("visual_normalization", "legacy"),
        "skeleton_representation": config.get(
            "skeleton_representation", "frame_joint"
        ),
        "skeleton_raw_cache_dir": config.get("skeleton_raw_cache_dir"),
    }
    train_dataset = AlignedMultimodalDataset(
        split="train", augment=True, **dataset_args
    )
    val_dataset = AlignedMultimodalDataset(
        split="val", augment=False, **dataset_args
    )
    batch_sampler = CrossSubjectClassBatchSampler(
        train_dataset,
        classes_per_batch=int(config["classes_per_batch"]),
        seed=int(config["seed"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 0)) > 0,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("validation_batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("val_num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(
        args.base_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    model = make_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    projection = ProjectionHead().to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": float(config["base_learning_rate"]),
                "name": "base",
            },
            {
                "params": projection.parameters(),
                "lr": float(config["projection_learning_rate"]),
                "name": "projection",
            },
        ],
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["label_smoothing"])
    )

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        projection.train()
        train_ce = 0.0
        train_contrastive = 0.0
        train_total = 0.0
        train_samples = 0
        train_labels: list[torch.Tensor] = []
        train_predictions: list[torch.Tensor] = []
        epoch_started = time.perf_counter()
        for batch in train_loader:
            inputs = move_inputs(batch, modalities, device)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                model_output = model(inputs, return_embedding=True)
                assert isinstance(model_output, dict)
                logits = model_output["logits"]
                embedding = model_output["embedding"]
                ce = criterion(logits, target)
                projected = projection(embedding)
                contrastive = subject_contrastive_loss(
                    projected,
                    target,
                    list(batch["user_id"]),
                    float(config["temperature"]),
                )
                loss = ce + weight * contrastive
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *projection.parameters()],
                float(config.get("max_grad_norm", 5.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(target)
            train_samples += batch_size
            train_ce += float(ce.detach()) * batch_size
            train_contrastive += float(contrastive.detach()) * batch_size
            train_total += float(loss.detach()) * batch_size
            train_labels.append(target.detach().cpu())
            train_predictions.append(logits.detach().argmax(dim=1).cpu())
        scheduler.step()
        labels = torch.cat(train_labels).numpy()
        predictions = torch.cat(train_predictions).numpy()
        validation = run_validation(
            model, val_loader, modalities, device, use_amp
        )
        row = {
            "epoch": epoch,
            "base_learning_rate": optimizer.param_groups[0]["lr"],
            "projection_learning_rate": optimizer.param_groups[1]["lr"],
            "contrastive_weight": weight,
            "train_loss": train_total / train_samples,
            "train_ce": train_ce / train_samples,
            "train_contrastive": train_contrastive / train_samples,
            **{
                f"train_{key}": value
                for key, value in metric_values(labels, predictions).items()
            },
            **{
                f"val_{key}": validation[key]
                for key in ("accuracy", "balanced_accuracy", "macro_f1")
            },
            "seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(row)
        write_history(output / "history.csv", history)
        if float(validation["accuracy"]) > best_accuracy:
            best_accuracy = float(validation["accuracy"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": checkpoint["config"],
                    "epoch": epoch,
                    "metrics": {
                        key: float(validation[key])
                        for key in (
                            "accuracy",
                            "balanced_accuracy",
                            "macro_f1",
                        )
                    },
                    "training_protocol": "p27-cross-subject-supcon-inner-v1",
                    "contrastive_weight": weight,
                    "projection_not_required_for_inference": True,
                },
                output / "best_accuracy.pt",
            )
            np.savez_compressed(
                output / "best_accuracy_held_logits.npz",
                protocol=np.asarray("p27-cross-subject-supcon-inner-v1"),
                sample_ids=validation["sample_ids"],
                labels=validation["labels"],
                logits=validation["logits"].astype(np.float32),
                outer_held_predictions_generated=np.asarray(False),
            )
        print(json.dumps(row), flush=True)

    summary = {
        "protocol": "p27-cross-subject-supcon-inner-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "subject_used_as_prediction_feature": False,
        "subject_used_only_for_cross_subject_positive_sampling": True,
        "contrastive_weight": weight,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "total_seconds": float(time.perf_counter() - started),
        "train_samples_per_epoch": 2
        * int(config["classes_per_batch"])
        * len(batch_sampler),
        "held_samples": len(val_dataset),
        "projection_parameters": int(
            sum(parameter.numel() for parameter in projection.parameters())
        ),
        "projection_not_required_for_inference": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
