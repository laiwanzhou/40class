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
from src.data.ir_primary_full_sequence_dataset import IRPrimaryFullSequenceDataset
from src.models.ir_primary_depth_residual_tcn import IRPrimaryDepthResidualTCN
from src.train_depth_ir_pose_roi_40class import detailed_metrics, per_class_rows, plot_history
from src.train_unimodal import PROJECT_ROOT, save_confusion_matrix


RUN_ID = "ir_primary_depth_residual_fullseq_14train_4val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/ir_primary_depth_residual_fullseq.yaml",
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


def build_datasets(config: dict[str, Any]) -> tuple[IRPrimaryFullSequenceDataset, IRPrimaryFullSequenceDataset]:
    train_frame, val_frame = load_modality_frames(
        resolve_path(config["manifest"]), resolve_path(config["fold"]),
        resolve_path(config["data_root"]), str(config["path_column"]),
    )
    pairing = pd.read_csv(resolve_path(config["pairing_audit"]), encoding="utf-8-sig")
    valid_ids = set(pairing.loc[pairing["complete_pairing"], "sample_id"].astype(str))
    train_frame = train_frame[train_frame["sample_id"].isin(valid_ids)].reset_index(drop=True)
    val_frame = val_frame[val_frame["sample_id"].isin(valid_ids)].reset_index(drop=True)
    actions = train_frame[["class_id", "action_name"]].drop_duplicates().sort_values("class_id")["action_name"].tolist()
    common = {
        "hard_actions": actions,
        "num_frames": int(config["stage_a_frames"]),
        "image_size": int(config["image_size"]),
        "pose_cache_path": resolve_path(config["pose_cache"]),
        "data_root": resolve_path(config["data_root"]),
        "roi_config": dict(config["roi"]),
    }
    return (
        IRPrimaryFullSequenceDataset(train_frame, training=True, **common),
        IRPrimaryFullSequenceDataset(val_frame, training=False, **common),
    )


def image_loader(
    dataset: IRPrimaryFullSequenceDataset,
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


def target_lookup(class_ids: list[int], device: torch.device) -> torch.Tensor:
    lookup = torch.zeros(40, dtype=torch.bool, device=device)
    lookup[torch.as_tensor(class_ids, dtype=torch.long, device=device)] = True
    return lookup


def combined_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    lookup: torch.Tensor,
    route_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    route_labels = lookup[labels].to(torch.long)
    main = nn.functional.cross_entropy(output["logits"], labels)
    route = nn.functional.cross_entropy(output["route_logits"], route_labels)
    return main + route_weight * route, main, route, route_labels


def metrics_from_outputs(
    total_loss: float,
    main_loss: float,
    route_loss: float,
    samples: int,
    labels: list[torch.Tensor],
    outputs: dict[str, list[torch.Tensor]],
    route_labels: list[torch.Tensor],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels_np = torch.cat(labels).numpy()
    arrays = {name: torch.cat(values).numpy() for name, values in outputs.items()}
    route_np = torch.cat(route_labels).numpy()
    result = detailed_metrics(labels_np, arrays["logits"], total_loss / samples)
    result.update({
        "main_loss": main_loss / samples,
        "route_loss": route_loss / samples,
        "route_accuracy": float((arrays["route_logits"].argmax(axis=1) == route_np).mean()),
    })
    return result, {"labels": labels_np, "route_labels": route_np, **arrays}


OUTPUT_KEYS = (
    "logits", "route_logits", "embedding", "short_attention", "long_attention",
    "ir_roi_attention", "depth_relation_gate", "depth_gate",
)


def image_epoch(
    model: IRPrimaryDepthResidualTCN,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    lookup = target_lookup(list(config["small_object_class_ids"]), device)
    accumulation = int(config["stage_a_gradient_accumulation"])
    amp_enabled = bool(config["spatial_amp"]) and device.type == "cuda"
    totals = mains = routes = 0.0
    samples = 0
    labels_all: list[torch.Tensor] = []
    route_all: list[torch.Tensor] = []
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in OUTPUT_KEYS}
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, batch in enumerate(loader):
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(
                    batch["ir_input"].to(device, non_blocking=True),
                    batch["depth_input"].to(device, non_blocking=True),
                    batch["ir_valid_mask"].to(device, non_blocking=True),
                    batch["depth_valid_mask"].to(device, non_blocking=True),
                    batch["view_confidence"].to(device, non_blocking=True),
                    batch["temporal_mask"].to(device, non_blocking=True),
                )
                loss, main, route, route_labels = combined_loss(
                    output, labels, lookup, float(config["route_loss_weight"]),
                )
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(loss / accumulation).backward()
                should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            count = len(labels)
            totals += float(loss.detach()) * count
            mains += float(main.detach()) * count
            routes += float(route.detach()) * count
            samples += count
            labels_all.append(labels.detach().cpu())
            route_all.append(route_labels.detach().cpu())
            for name in OUTPUT_KEYS:
                outputs[name].append(output[name].detach().float().cpu())
    return metrics_from_outputs(totals, mains, routes, samples, labels_all, outputs, route_all)


def history_row(
    stage: str,
    epoch: int,
    train: dict[str, Any],
    val: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    started: float,
) -> dict[str, Any]:
    return {
        "stage": stage, "epoch": epoch,
        "train_loss": train["loss"], "train_main_loss": train["main_loss"],
        "train_route_loss": train["route_loss"], "train_route_accuracy": train["route_accuracy"],
        "train_accuracy": train["accuracy"], "train_macro_f1": train["macro_f1"],
        "val_loss": val["loss"], "val_main_loss": val["main_loss"],
        "val_route_loss": val["route_loss"], "val_route_accuracy": val["route_accuracy"],
        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
        "accuracy": val["accuracy"], "macro_f1": val["macro_f1"],
        "weighted_f1": val["weighted_f1"], "top3_accuracy": val["top3_accuracy"],
        "top5_accuracy": val["top5_accuracy"],
        "number_of_predicted_classes": val["number_of_predicted_classes"],
        "never_predicted_class_count": val["never_predicted_class_count"],
        "zero_f1_class_count": val["zero_f1_class_count"],
        "generalization_gap": train["accuracy"] - val["accuracy"],
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch_time_seconds": time.perf_counter() - started,
    }


def save_checkpoint(
    path: Path,
    model: IRPrimaryDepthResidualTCN,
    stage: str,
    epoch: int,
    result: dict[str, Any],
) -> None:
    torch.save({
        "stage": stage, "epoch": epoch,
        "val_accuracy": float(result["accuracy"]),
        "val_macro_f1": float(result["macro_f1"]),
        "model_state_dict": model.state_dict(),
    }, path)


def stop_requested(run_dir: Path, config: dict[str, Any]) -> bool:
    return (run_dir / str(config["stop_file_name"])).exists()


def train_stage_a(
    model: IRPrimaryDepthResidualTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    smoke: bool,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["stage_a_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["smoke_stage_a_epochs"] if smoke else config["stage_a_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history: list[dict[str, Any]] = []
    best = (-1.0, -1.0)
    no_improvement = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train, _ = image_epoch(model, train_loader, device, config, optimizer, scaler)
        val, _ = image_epoch(model, val_loader, device, config, None, None)
        row = history_row("spatial_warmup", epoch, train, val, optimizer, started)
        history.append(row)
        key = (float(val["macro_f1"]), float(val["accuracy"]))
        if key > best:
            best = key
            no_improvement = 0
            save_checkpoint(run_dir / "stage_a_best_macro_f1.pt", model, "stage_a", epoch, val)
        else:
            no_improvement += 1
        save_checkpoint(run_dir / "stage_a_last_complete.pt", model, "stage_a", epoch, val)
        pd.DataFrame(history).to_csv(run_dir / "stage_a_history.csv", index=False, encoding="utf-8-sig")
        scheduler.step()
        print(json.dumps(row), flush=True)
        if stop_requested(run_dir, config):
            print(json.dumps({"safe_stop_after_stage_a_epoch": epoch}), flush=True)
            break
        if not smoke and no_improvement >= int(config["stage_a_patience"]):
            print(json.dumps({"stage_a_early_stop": epoch}), flush=True)
            break
    return history


def encode_item(
    model: IRPrimaryDepthResidualTCN,
    item: dict[str, object],
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    keys = ("frame_features", "ir_roi_attention", "depth_relation_gate", "depth_gate")
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    chunk = int(config["feature_chunk_frames"])
    amp_enabled = bool(config["spatial_amp"]) and device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, len(item["ir_input"]), chunk):
            stop = min(len(item["ir_input"]), start + chunk)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model.encode_spatial(
                    item["ir_input"][start:stop].unsqueeze(0).to(device),
                    item["depth_input"][start:stop].unsqueeze(0).to(device),
                    item["ir_valid_mask"][start:stop].unsqueeze(0).to(device),
                    item["depth_valid_mask"][start:stop].unsqueeze(0).to(device),
                    item["view_confidence"][start:stop].unsqueeze(0).to(device),
                )
            for key in keys:
                values[key].append(output[key].squeeze(0).float().cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in values.items()}


CACHE_KEYS = (
    "sample_ids", "user_ids", "labels", "original_lengths", "frame_indices", "temporal_mask",
    "frame_features", "ir_roi_attention", "depth_relation_gate", "depth_gate",
)


def extract_cache(
    split: str,
    dataset: IRPrimaryFullSequenceDataset,
    model: IRPrimaryDepthResidualTCN,
    root: Path,
    config: dict[str, Any],
    device: torch.device,
    limit: int | None,
) -> None:
    split_root = root / split
    split_root.mkdir(parents=True, exist_ok=True)
    count = min(len(dataset), limit or len(dataset))
    shard_size = int(config["cache_shard_size"])
    for first in range(0, count, shard_size):
        last = min(count, first + shard_size)
        path = split_root / f"shard_{first:05d}_{last:05d}.npz"
        expected_ids = np.asarray([str(dataset.samples[index]["sample_id"]) for index in range(first, last)])
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                if np.array_equal(existing["sample_ids"], expected_ids):
                    continue
            raise ValueError(f"Cache shard mismatch: {path}")
        pieces = {key: [] for key in ("frame_features", "ir_roi_attention", "depth_relation_gate", "depth_gate")}
        masks, indices, labels, users, lengths = [], [], [], [], []
        for index in range(first, last):
            item = dataset[index]
            encoded = encode_item(model, item, device, config)
            for key in pieces:
                pieces[key].append(encoded[key].astype(np.float16))
            masks.append(np.asarray(item["temporal_mask"], dtype=bool))
            indices.append(np.asarray(item["frame_indices"], dtype=np.int64))
            labels.append(int(item["label"]))
            users.append(str(item["user_id"]))
            lengths.append(int(item["length"]))
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            sample_ids=expected_ids, user_ids=np.asarray(users), labels=np.asarray(labels, dtype=np.int64),
            original_lengths=np.asarray(lengths, dtype=np.int64), frame_indices=np.stack(indices),
            temporal_mask=np.stack(masks), **{key: np.stack(value) for key, value in pieces.items()},
        )
        temporary.replace(path)
        print(json.dumps({"cache_written": split, "shard": path.name, "samples": last - first}), flush=True)


def load_cache(split: str, root: Path, expected: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in CACHE_KEYS}
    for path in sorted((root / split).glob("shard_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            for key in CACHE_KEYS:
                pieces[key].append(np.asarray(shard[key]))
    if not pieces["labels"]:
        raise FileNotFoundError(f"No cache shards for {split}")
    result = {key: np.concatenate(value) for key, value in pieces.items()}
    if len(result["labels"]) != expected or len(np.unique(result["sample_ids"])) != expected:
        raise ValueError(f"Cache integrity failed for {split}")
    return result


def cached_loader(data: dict[str, np.ndarray], batch: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(data["frame_features"].astype(np.float32)),
        torch.from_numpy(data["temporal_mask"].astype(bool)),
        torch.from_numpy(data["labels"].astype(np.int64)),
    )
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def cached_epoch(
    model: IRPrimaryDepthResidualTCN,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    lookup = target_lookup(list(config["small_object_class_ids"]), device)
    amp_enabled = bool(config["temporal_amp"]) and device.type == "cuda"
    totals = mains = routes = 0.0
    samples = 0
    labels_all: list[torch.Tensor] = []
    route_all: list[torch.Tensor] = []
    outputs: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("logits", "route_logits", "embedding", "short_attention", "long_attention")
    }
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for features, temporal_mask, labels in loader:
            features = features.to(device, non_blocking=True)
            temporal_mask = temporal_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model.forward_cached(features, temporal_mask)
                loss, main, route, route_labels = combined_loss(
                    output, labels, lookup, float(config["route_loss_weight"]),
                )
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
            count = len(labels)
            totals += float(loss.detach()) * count
            mains += float(main.detach()) * count
            routes += float(route.detach()) * count
            samples += count
            labels_all.append(labels.detach().cpu())
            route_all.append(route_labels.detach().cpu())
            for key in outputs:
                outputs[key].append(output[key].detach().float().cpu())
    return metrics_from_outputs(totals, mains, routes, samples, labels_all, outputs, route_all)


def train_stage_b(
    model: IRPrimaryDepthResidualTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    smoke: bool,
) -> list[dict[str, Any]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config["stage_b_learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["smoke_stage_b_epochs"] if smoke else config["stage_b_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["temporal_amp"]) and device.type == "cuda")
    initial, _ = cached_epoch(model, val_loader, device, config, None, None)
    save_checkpoint(run_dir / "best_accuracy.pt", model, "stage_b", 0, initial)
    save_checkpoint(run_dir / "best_macro_f1.pt", model, "stage_b", 0, initial)
    best_accuracy = (float(initial["accuracy"]), float(initial["macro_f1"]))
    best_macro = (float(initial["macro_f1"]), float(initial["accuracy"]))
    history: list[dict[str, Any]] = []
    no_improvement = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train, _ = cached_epoch(model, train_loader, device, config, optimizer, scaler)
        val, _ = cached_epoch(model, val_loader, device, config, None, None)
        row = history_row("full_sequence", epoch, train, val, optimizer, started)
        history.append(row)
        accuracy_key = (float(val["accuracy"]), float(val["macro_f1"]))
        macro_key = (float(val["macro_f1"]), float(val["accuracy"]))
        if accuracy_key > best_accuracy:
            best_accuracy = accuracy_key
            save_checkpoint(run_dir / "best_accuracy.pt", model, "stage_b", epoch, val)
        if macro_key > best_macro:
            best_macro = macro_key
            no_improvement = 0
            save_checkpoint(run_dir / "best_macro_f1.pt", model, "stage_b", epoch, val)
        else:
            no_improvement += 1
        save_checkpoint(run_dir / "last_complete.pt", model, "stage_b", epoch, val)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        scheduler.step()
        print(json.dumps(row), flush=True)
        if stop_requested(run_dir, config):
            print(json.dumps({"safe_stop_after_stage_b_epoch": epoch}), flush=True)
            break
        if not smoke and no_improvement >= int(config["stage_b_patience"]):
            print(json.dumps({"stage_b_early_stop": epoch}), flush=True)
            break
    return history


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
        predictions=logits.argmax(axis=1), logits=logits, probabilities=np.asarray(result["probabilities"]),
        true_class_rank=np.asarray(result["true_class_rank"]), embeddings=output["embedding"],
        route_labels=output["route_labels"], route_logits=output["route_logits"],
        short_attention=output["short_attention"], long_attention=output["long_attention"],
        temporal_mask=cache["temporal_mask"], frame_indices=cache["frame_indices"],
        original_lengths=cache["original_lengths"], ir_roi_attention=cache["ir_roi_attention"],
        depth_relation_gate=cache["depth_relation_gate"], depth_gate=cache["depth_gate"],
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
        raise RuntimeError("CUDA requested but unavailable")
    output_root = resolve_path(config["output_root"])
    run_id = f"{args.run_id}_smoke" if smoke and not args.run_id.endswith("_smoke") else args.run_id
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    config.update({"smoke_test": smoke, "run_dir": str(run_dir), "test_read": False})
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_dataset, val_dataset = build_datasets(config)
    if (len(train_dataset), len(val_dataset)) != (2320, 590):
        raise ValueError(f"Unexpected split: {len(train_dataset)}/{len(val_dataset)}")
    if set(train_dataset.user_ids) & set(val_dataset.user_ids):
        raise ValueError("Train and validation users overlap")
    smoke_train = int(config["smoke_train_samples"]) if smoke else None
    smoke_val = int(config["smoke_val_samples"]) if smoke else None
    train_loader = image_loader(train_dataset, int(config["stage_a_batch_size"]), True, int(config["seed"]), smoke_train)
    val_loader = image_loader(val_dataset, int(config["stage_a_batch_size"]), False, int(config["seed"]) + 1, smoke_val)

    model = IRPrimaryDepthResidualTCN(
        num_classes=40, frame_feature_dim=int(config["frame_feature_dim"]),
        channels=int(config["tcn_channels"]), embedding_dim=int(config["embedding_dim"]),
        short_dilations=tuple(config["short_dilations"]), long_dilations=tuple(config["long_dilations"]),
        dropout=float(config["dropout"]), pretrained=bool(config["pretrained_ir"]),
        initial_depth_gate=float(config["initial_depth_gate"]),
    ).to(device)
    audit = {
        "ir_views": 4, "depth_views": 2, "global_frame_removed": True, "upper_body_removed": True,
        "fusion": "independent spatial encoders then IR-biased Depth residual before TCN",
        "initial_depth_gate": float(config["initial_depth_gate"]),
        "heads": ["40_class_main", "small_object_binary_route"],
        "conditional_small_object_head": False,
        "small_object_class_ids": list(config["small_object_class_ids"]),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test_read": False,
    }
    (run_dir / "model_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    stage_a = train_stage_a(model, train_loader, val_loader, config, run_dir, device, smoke)
    if stop_requested(run_dir, config):
        print(json.dumps({"status": "safely_stopped_after_stage_a", "epoch": stage_a[-1]["epoch"]}), flush=True)
        return

    checkpoint = torch.load(run_dir / "stage_a_best_macro_f1.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.freeze_spatial()
    train_dataset.num_frames = int(config["full_sequence_frames"])
    val_dataset.num_frames = int(config["full_sequence_frames"])
    cache_root = output_root / f"{run_id}_spatial_cache_t{config['full_sequence_frames']}"
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage_a_checkpoint_sha256": sha256(run_dir / "stage_a_best_macro_f1.pt"),
        "frames": int(config["full_sequence_frames"]), "ir_views": 4, "depth_views": 2,
        "sampling": "endpoint-preserving uniform complete-clip sampling with repeat-last padding",
        "test_read": False,
    }
    metadata_path = cache_root / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Existing cache metadata mismatch")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    extract_cache("train", train_dataset, model, cache_root, config, device, smoke_train)
    extract_cache("validation", val_dataset, model, cache_root, config, device, smoke_val)
    train_count, val_count = smoke_train or len(train_dataset), smoke_val or len(val_dataset)
    train_cache = load_cache("train", cache_root, train_count)
    val_cache = load_cache("validation", cache_root, val_count)
    train_cached = cached_loader(train_cache, int(config["stage_b_batch_size"]), True, int(config["seed"]) + 2)
    val_cached = cached_loader(val_cache, int(config["stage_b_batch_size"]), False, int(config["seed"]) + 3)
    stage_b = train_stage_b(model, train_cached, val_cached, config, run_dir, device, smoke)

    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv").sort_values("class_id")
    class_names = class_map["action_name"].tolist()
    train_support = class_map["train_support"].to_numpy(dtype=np.int64)
    val_support = class_map["val_support"].to_numpy(dtype=np.int64)
    checkpoints: dict[str, Any] = {}
    for name in ("best_accuracy", "best_macro_f1"):
        saved = torch.load(run_dir / f"{name}.pt", map_location=device, weights_only=True)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        result, output = cached_epoch(model, val_cached, device, config, None, None)
        rows = per_class_rows(
            int(saved["epoch"]), output["labels"], output["logits"], result,
            class_names, train_support, val_support,
        )
        pd.DataFrame(rows).to_csv(run_dir / f"per_class_{name}.csv", index=False, encoding="utf-8-sig")
        save_confusion_matrix(np.asarray(result["confusion_matrix"]).tolist(), run_dir / f"confusion_matrix_{name}.png")
        save_predictions(run_dir / f"val_predictions_{name}.npz", output, val_cache, result)
        checkpoints[name] = {
            "epoch": int(saved["epoch"]), "accuracy": float(result["accuracy"]),
            "macro_f1": float(result["macro_f1"]), "route_accuracy": float(result["route_accuracy"]),
            "val_loss": float(result["loss"]),
        }
    history = pd.DataFrame(stage_b)
    history.to_csv(run_dir / "epoch_summary.csv", index=False, encoding="utf-8-sig")
    plot_history(history, run_dir)
    summary = {
        "status": "passed", "test_read": False, "train_samples": train_count, "val_samples": val_count,
        "stage_a_epochs": len(stage_a), "stage_b_epochs": len(stage_b), "cache_root": str(cache_root),
        "checkpoints": checkpoints,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run(parse_args())
