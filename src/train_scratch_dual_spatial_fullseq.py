from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.data import load_modality_frames
from src.data.dual_spatial_full_sequence_dataset import DualSpatialFullSequenceDataset
from src.models.dual_spatial_full_sequence_tcn import DualSpatialFullSequenceTCN
from src.train_depth_ir_pose_roi_40class import detailed_metrics, per_class_rows, plot_history
from src.train_unimodal import PROJECT_ROOT, save_confusion_matrix


RUN_ID = "scratch_dual_spatial_fullseq_14train_4val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_scratch_dual_spatial_fullseq.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_datasets(config: dict[str, Any]) -> tuple[DualSpatialFullSequenceDataset, DualSpatialFullSequenceDataset]:
    train_frame, val_frame = load_modality_frames(
        resolve_path(config["manifest"]), resolve_path(config["fold"]),
        resolve_path(config["data_root"]), str(config["path_column"]),
    )
    audit = pd.read_csv(resolve_path(config["pairing_audit"]), encoding="utf-8-sig")
    valid_ids = set(audit.loc[audit["complete_pairing"], "sample_id"].astype(str))
    train_frame = train_frame[train_frame["sample_id"].isin(valid_ids)].reset_index(drop=True)
    val_frame = val_frame[val_frame["sample_id"].isin(valid_ids)].reset_index(drop=True)
    actions = train_frame[["class_id", "action_name"]].drop_duplicates().sort_values("class_id")["action_name"].tolist()
    common = {
        "hard_actions": actions,
        "num_frames": int(config["stage_a_frames"]),
        "image_size": int(config["image_size"]),
        "pose_cache_path": resolve_path(config["pose_cache"]),
        "data_root": resolve_path(config["data_root"]),
        "interaction_config": dict(config["interaction_roi"]),
        "person_crop_config": dict(config["person_crop"]),
    }
    return (
        DualSpatialFullSequenceDataset(train_frame, training=True, **common),
        DualSpatialFullSequenceDataset(val_frame, training=False, **common),
    )


def loader_for_images(
    dataset: DualSpatialFullSequenceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    limit: int | None,
) -> DataLoader:
    actual = Subset(dataset, range(min(limit, len(dataset)))) if limit is not None else dataset
    return DataLoader(
        actual, batch_size=min(batch_size, len(actual)), shuffle=shuffle, num_workers=0,
        pin_memory=True, generator=torch.Generator().manual_seed(seed),
    )


def collect_result(
    losses: float,
    samples: int,
    labels: list[torch.Tensor],
    outputs: dict[str, list[torch.Tensor]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels_np = torch.cat(labels).numpy()
    arrays = {name: torch.cat(values).numpy() for name, values in outputs.items()}
    result = detailed_metrics(labels_np, arrays["logits"], losses / samples)
    return result, {"labels": labels_np, **arrays}


def image_epoch(
    model: DualSpatialFullSequenceTCN,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    gradient_clip: float,
    accumulation: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    losses = 0.0
    samples = 0
    labels_all: list[torch.Tensor] = []
    output_all: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("logits", "embedding", "short_attention", "long_attention", "spatial_gate")
    }
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, batch in enumerate(loader):
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(
                    batch["depth_input"].to(device, non_blocking=True),
                    batch["ir_input"].to(device, non_blocking=True),
                    batch["global_depth_input"].to(device, non_blocking=True),
                    batch["global_ir_input"].to(device, non_blocking=True),
                    batch["temporal_mask"].to(device, non_blocking=True),
                )
                loss = nn.functional.cross_entropy(output["logits"], labels)
            if training:
                assert scaler is not None and optimizer is not None
                scaler.scale(loss / accumulation).backward()
                should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad), gradient_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            count = len(labels)
            losses += float(loss.detach()) * count
            samples += count
            labels_all.append(labels.detach().cpu())
            for name in output_all:
                output_all[name].append(output[name].detach().float().cpu())
    return collect_result(losses, samples, labels_all, output_all)


def save_checkpoint(
    path: Path,
    model: DualSpatialFullSequenceTCN,
    stage: str,
    epoch: int,
    result: dict[str, Any],
) -> None:
    torch.save({
        "stage": stage,
        "epoch": epoch,
        "val_accuracy": float(result["accuracy"]),
        "val_macro_f1": float(result["macro_f1"]),
        "model_state_dict": model.state_dict(),
    }, path)


def train_stage_a(
    model: DualSpatialFullSequenceTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    smoke: bool,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["stage_a_learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["smoke_stage_a_epochs"] if smoke else config["stage_a_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(config["spatial_amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    accumulation = int(config["stage_a_gradient_accumulation"])
    history: list[dict[str, Any]] = []
    best = (-1.0, -1.0)
    no_improvement = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_result, _ = image_epoch(
            model, train_loader, device, amp_enabled, optimizer, scaler,
            float(config["gradient_clip"]), accumulation,
        )
        val_result, _ = image_epoch(model, val_loader, device, amp_enabled, None, None, 0.0, 1)
        row = make_history_row("spatial_warmup", epoch, train_result, val_result, optimizer, started)
        history.append(row)
        key = (float(val_result["macro_f1"]), float(val_result["accuracy"]))
        if key > best:
            best = key
            no_improvement = 0
            save_checkpoint(run_dir / "stage_a_best_macro_f1.pt", model, "stage_a", epoch, val_result)
        else:
            no_improvement += 1
        save_checkpoint(run_dir / "stage_a_last_complete.pt", model, "stage_a", epoch, val_result)
        pd.DataFrame(history).to_csv(run_dir / "stage_a_history.csv", index=False, encoding="utf-8-sig")
        scheduler.step()
        print(json.dumps(row), flush=True)
        if not smoke and no_improvement >= int(config["stage_a_patience"]):
            print(json.dumps({"stage_a_early_stop": epoch}), flush=True)
            break
    return history


def make_history_row(
    stage: str,
    epoch: int,
    train_result: dict[str, Any],
    val_result: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    started: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "epoch": epoch,
        "train_loss": train_result["loss"],
        "train_accuracy": train_result["accuracy"],
        "train_macro_f1": train_result["macro_f1"],
        "val_loss": val_result["loss"],
        "val_accuracy": val_result["accuracy"],
        "val_macro_f1": val_result["macro_f1"],
        "accuracy": val_result["accuracy"],
        "macro_f1": val_result["macro_f1"],
        "weighted_f1": val_result["weighted_f1"],
        "top3_accuracy": val_result["top3_accuracy"],
        "top5_accuracy": val_result["top5_accuracy"],
        "number_of_predicted_classes": val_result["number_of_predicted_classes"],
        "never_predicted_class_count": val_result["never_predicted_class_count"],
        "zero_f1_class_count": val_result["zero_f1_class_count"],
        "generalization_gap": train_result["accuracy"] - val_result["accuracy"],
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch_time_seconds": time.perf_counter() - started,
    }


def encode_spatial_item(
    model: DualSpatialFullSequenceTCN,
    item: dict[str, object],
    device: torch.device,
    chunk_frames: int,
    amp_enabled: bool,
) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in ("interaction_features", "global_features", "roi_attention", "modality_gate")
    }
    with torch.inference_mode():
        for start in range(0, len(item["depth_input"]), chunk_frames):
            stop = min(len(item["depth_input"]), start + chunk_frames)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model.encode_spatial(
                    item["depth_input"][start:stop].unsqueeze(0).to(device),
                    item["ir_input"][start:stop].unsqueeze(0).to(device),
                    item["global_depth_input"][start:stop].unsqueeze(0).to(device),
                    item["global_ir_input"][start:stop].unsqueeze(0).to(device),
                )
            for name in arrays:
                arrays[name].append(output[name].squeeze(0).float().cpu().numpy())
    return {name: np.concatenate(values) for name, values in arrays.items()}


def extract_cache(
    split: str,
    dataset: DualSpatialFullSequenceDataset,
    model: DualSpatialFullSequenceTCN,
    cache_root: Path,
    config: dict[str, Any],
    device: torch.device,
    limit: int | None,
) -> None:
    root = cache_root / split
    root.mkdir(parents=True, exist_ok=True)
    count = min(len(dataset), limit or len(dataset))
    shard_size = int(config["cache_shard_size"])
    amp_enabled = bool(config["spatial_amp"]) and device.type == "cuda"
    for first in range(0, count, shard_size):
        last = min(count, first + shard_size)
        path = root / f"shard_{first:05d}_{last:05d}.npz"
        sample_ids = np.asarray([str(dataset.samples[index]["sample_id"]) for index in range(first, last)])
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                if np.array_equal(existing["sample_ids"], sample_ids):
                    print(json.dumps({"cache_resume": split, "shard": path.name}), flush=True)
                    continue
            raise ValueError(f"Cache shard mismatch: {path}")
        pieces: dict[str, list[np.ndarray]] = {
            name: [] for name in ("interaction_features", "global_features", "roi_attention", "modality_gate")
        }
        masks, indices, labels, users, lengths = [], [], [], [], []
        started = time.perf_counter()
        for index in range(first, last):
            item = dataset[index]
            encoded = encode_spatial_item(
                model, item, device, int(config["feature_chunk_frames"]), amp_enabled,
            )
            for name in pieces:
                pieces[name].append(encoded[name].astype(np.float16))
            masks.append(np.asarray(item["temporal_mask"], dtype=bool))
            indices.append(np.asarray(item["frame_indices"], dtype=np.int64))
            labels.append(int(item["label"]))
            users.append(str(item["user_id"]))
            lengths.append(int(item["length"]))
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            sample_ids=sample_ids,
            user_ids=np.asarray(users),
            labels=np.asarray(labels, dtype=np.int64),
            original_lengths=np.asarray(lengths, dtype=np.int64),
            frame_indices=np.stack(indices),
            temporal_mask=np.stack(masks),
            **{name: np.stack(values) for name, values in pieces.items()},
        )
        temporary.replace(path)
        print(json.dumps({
            "cache_written": split, "shard": path.name, "samples": last - first,
            "seconds": time.perf_counter() - started,
        }), flush=True)


CACHE_KEYS = (
    "sample_ids", "user_ids", "labels", "original_lengths", "frame_indices", "temporal_mask",
    "interaction_features", "global_features", "roi_attention", "modality_gate",
)


def load_cache(split: str, root: Path, expected: int) -> dict[str, np.ndarray]:
    shards = sorted((root / split).glob("shard_*.npz"))
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in CACHE_KEYS}
    for path in shards:
        with np.load(path, allow_pickle=False) as shard:
            for name in CACHE_KEYS:
                pieces[name].append(np.asarray(shard[name]))
    result = {name: np.concatenate(values) for name, values in pieces.items()}
    if len(result["labels"]) != expected or len(np.unique(result["sample_ids"])) != expected:
        raise ValueError(f"Cache integrity failed for {split}: {len(result['labels'])}/{expected}")
    return result


def cache_loader(data: dict[str, np.ndarray], batch: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(data["interaction_features"].astype(np.float32)),
        torch.from_numpy(data["global_features"].astype(np.float32)),
        torch.from_numpy(data["temporal_mask"].astype(bool)),
        torch.from_numpy(data["labels"].astype(np.int64)),
    )
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def cached_epoch(
    model: DualSpatialFullSequenceTCN,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    gradient_clip: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    losses, samples = 0.0, 0
    labels_all: list[torch.Tensor] = []
    output_all: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("logits", "embedding", "short_attention", "long_attention", "spatial_gate")
    }
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for interaction, global_features, mask, labels in loader:
            interaction = interaction.to(device, non_blocking=True)
            global_features = global_features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model.forward_cached(interaction, global_features, mask)
                loss = nn.functional.cross_entropy(output["logits"], labels)
            if training:
                assert scaler is not None and optimizer is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad), gradient_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            count = len(labels)
            losses += float(loss.detach()) * count
            samples += count
            labels_all.append(labels.detach().cpu())
            for name in output_all:
                output_all[name].append(output[name].detach().float().cpu())
    return collect_result(losses, samples, labels_all, output_all)


def train_stage_b(
    model: DualSpatialFullSequenceTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["stage_b_learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["smoke_stage_b_epochs"] if smoke else config["stage_b_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(config["temporal_amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial_result, initial_output = cached_epoch(model, val_loader, device, amp_enabled, None, None, 0.0)
    save_checkpoint(run_dir / "best_accuracy.pt", model, "stage_b", 0, initial_result)
    save_checkpoint(run_dir / "best_macro_f1.pt", model, "stage_b", 0, initial_result)
    best_accuracy = (float(initial_result["accuracy"]), float(initial_result["macro_f1"]))
    best_macro = (float(initial_result["macro_f1"]), float(initial_result["accuracy"]))
    best_output = initial_output
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_result, _ = cached_epoch(
            model, train_loader, device, amp_enabled, optimizer, scaler, float(config["gradient_clip"]),
        )
        val_result, val_output = cached_epoch(model, val_loader, device, amp_enabled, None, None, 0.0)
        row = make_history_row("full_sequence", epoch, train_result, val_result, optimizer, started)
        history.append(row)
        accuracy_key = (float(val_result["accuracy"]), float(val_result["macro_f1"]))
        macro_key = (float(val_result["macro_f1"]), float(val_result["accuracy"]))
        if accuracy_key > best_accuracy:
            best_accuracy = accuracy_key
            save_checkpoint(run_dir / "best_accuracy.pt", model, "stage_b", epoch, val_result)
        if macro_key > best_macro:
            best_macro = macro_key
            best_output = val_output
            no_improvement = 0
            save_checkpoint(run_dir / "best_macro_f1.pt", model, "stage_b", epoch, val_result)
        else:
            no_improvement += 1
        save_checkpoint(run_dir / "last_complete.pt", model, "stage_b", epoch, val_result)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        scheduler.step()
        print(json.dumps(row), flush=True)
        if not smoke and no_improvement >= int(config["stage_b_patience"]):
            print(json.dumps({"stage_b_early_stop": epoch}), flush=True)
            break
    return history, best_output


def save_predictions(
    path: Path,
    output: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    result: dict[str, Any],
) -> None:
    logits = output["logits"]
    np.savez_compressed(
        path,
        sample_ids=cache["sample_ids"], user_ids=cache["user_ids"], labels=output["labels"],
        predictions=logits.argmax(axis=1), logits=logits,
        probabilities=np.asarray(result["probabilities"]),
        true_class_rank=np.asarray(result["true_class_rank"]),
        embeddings=output["embedding"], short_attention=output["short_attention"],
        long_attention=output["long_attention"], spatial_gate=output["spatial_gate"],
        temporal_mask=cache["temporal_mask"], frame_indices=cache["frame_indices"],
        original_lengths=cache["original_lengths"], roi_attention=cache["roi_attention"],
        modality_gate=cache["modality_gate"],
    )


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    smoke = bool(args.smoke_test)
    if smoke:
        config["stage_a_frames"] = int(config["smoke_stage_a_frames"])
        config["full_sequence_frames"] = int(config["smoke_full_sequence_frames"])
    set_seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    output_root = resolve_path(config["output_root"])
    run_id = f"{args.run_id}_smoke" if smoke and not args.run_id.endswith("_smoke") else args.run_id
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config.update({"smoke_test": smoke, "run_dir": str(run_dir), "test_read": False})
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_dataset, val_dataset = build_datasets(config)
    if (len(train_dataset), len(val_dataset)) != (2320, 590):
        raise ValueError(f"Unexpected split: {len(train_dataset)}/{len(val_dataset)}")
    if set(train_dataset.user_ids) & set(val_dataset.user_ids):
        raise ValueError("Train and validation users overlap.")
    smoke_train = int(config["smoke_train_samples"]) if smoke else None
    smoke_val = int(config["smoke_val_samples"]) if smoke else None
    train_loader = loader_for_images(
        train_dataset, int(config["stage_a_batch_size"]), True, int(config["seed"]), smoke_train,
    )
    val_loader = loader_for_images(
        val_dataset, int(config["stage_a_batch_size"]), False, int(config["seed"]) + 1, smoke_val,
    )
    model = DualSpatialFullSequenceTCN(
        num_classes=40, frame_feature_dim=int(config["frame_feature_dim"]),
        channels=int(config["tcn_channels"]), embedding_dim=int(config["embedding_dim"]),
        short_dilations=tuple(config["short_dilations"]),
        long_dilations=tuple(config["long_dilations"]), dropout=float(config["dropout"]),
        pretrained=True,
    ).to(device)
    audit = {
        "initialization": "ImageNet MobileNetV3-Small only; no B2 or prior TCN checkpoint",
        "b2_checkpoint_loaded": False,
        "prior_tcn_checkpoint_loaded": False,
        "pose_source": "IR frames processed by YOLO11n-pose cache",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "stage_a_trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "global_frame_views": 1,
        "person_interaction_views": 4,
        "shared_spatial_backbone": True,
        "test_read": False,
    }
    (run_dir / "model_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    stage_a_started = time.perf_counter()
    stage_a_history = train_stage_a(model, train_loader, val_loader, config, run_dir, device, smoke)
    stage_a_seconds = time.perf_counter() - stage_a_started
    stage_a_checkpoint = torch.load(run_dir / "stage_a_best_macro_f1.pt", map_location=device, weights_only=True)
    model.load_state_dict(stage_a_checkpoint["model_state_dict"], strict=True)
    model.freeze_spatial()
    audit["stage_b_trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    audit["stage_a_best_epoch"] = int(stage_a_checkpoint["epoch"])
    (run_dir / "model_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    train_dataset.num_frames = int(config["full_sequence_frames"])
    val_dataset.num_frames = int(config["full_sequence_frames"])
    cache_root = output_root / f"{run_id}_spatial_cache_t{config['full_sequence_frames']}"
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage_a_checkpoint_sha256": sha256(run_dir / "stage_a_best_macro_f1.pt"),
        "frames": int(config["full_sequence_frames"]), "views": 5,
        "sampling": "endpoint-preserving uniform complete-clip sampling with repeat-last padding",
        "test_read": False,
    }
    metadata_path = cache_root / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Existing cache metadata does not match the stage A checkpoint.")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    extract_cache("train", train_dataset, model, cache_root, config, device, smoke_train)
    extract_cache("validation", val_dataset, model, cache_root, config, device, smoke_val)
    train_count, val_count = smoke_train or len(train_dataset), smoke_val or len(val_dataset)
    train_cache = load_cache("train", cache_root, train_count)
    val_cache = load_cache("validation", cache_root, val_count)
    train_cached_loader = cache_loader(
        train_cache, int(config["stage_b_batch_size"]), True, int(config["seed"]) + 2,
    )
    val_cached_loader = cache_loader(
        val_cache, int(config["stage_b_batch_size"]), False, int(config["seed"]) + 3,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    stage_b_started = time.perf_counter()
    stage_b_history, _ = train_stage_b(
        model, train_cached_loader, val_cached_loader, config, run_dir, device, smoke,
    )
    stage_b_seconds = time.perf_counter() - stage_b_started

    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv").sort_values("class_id")
    class_names = class_map["action_name"].tolist()
    train_support = class_map["train_support"].to_numpy(dtype=np.int64)
    val_support = class_map["val_support"].to_numpy(dtype=np.int64)
    amp_enabled = bool(config["temporal_amp"]) and device.type == "cuda"
    checkpoints: dict[str, Any] = {}
    for name in ("best_accuracy", "best_macro_f1"):
        checkpoint = torch.load(run_dir / f"{name}.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        result, output = cached_epoch(model, val_cached_loader, device, amp_enabled, None, None, 0.0)
        rows = per_class_rows(
            int(checkpoint["epoch"]), output["labels"], output["logits"], result,
            class_names, train_support, val_support,
        )
        pd.DataFrame(rows).to_csv(run_dir / f"per_class_{name}.csv", index=False, encoding="utf-8-sig")
        save_confusion_matrix(
            np.asarray(result["confusion_matrix"]).tolist(), run_dir / f"confusion_matrix_{name}.png",
        )
        save_predictions(run_dir / f"val_predictions_{name}.npz", output, val_cache, result)
        checkpoints[name] = {
            "epoch": int(checkpoint["epoch"]), "accuracy": float(result["accuracy"]),
            "macro_f1": float(result["macro_f1"]), "weighted_f1": float(result["weighted_f1"]),
            "val_loss": float(result["loss"]), "top3_accuracy": float(result["top3_accuracy"]),
            "top5_accuracy": float(result["top5_accuracy"]),
            "predicted_classes": int(result["number_of_predicted_classes"]),
            "zero_f1_classes": int(result["zero_f1_class_count"]),
        }
    history = pd.DataFrame(stage_b_history)
    history.to_csv(run_dir / "epoch_summary.csv", index=False, encoding="utf-8-sig")
    plot_history(history, run_dir)
    summary = {
        "status": "passed", "test_read": False, "train_samples": train_count, "val_samples": val_count,
        "stage_a_epochs": len(stage_a_history), "stage_b_epochs": len(stage_b_history),
        "stage_a_seconds": stage_a_seconds, "stage_b_seconds": stage_b_seconds,
        "cache_root": str(cache_root), "model_audit": audit, "checkpoints": checkpoints,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run(parse_args())
