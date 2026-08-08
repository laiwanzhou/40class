from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import types
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_modality_frames
from src.data.full_sequence_person_crop_dataset import FullSequencePersonCropPoseROIDataset
from src.data.pose_roi_dataset import PoseROIDataset
from src.models.depth_ir_pose_roi_expert import DepthIRPoseROIExpert
from src.models.full_sequence_multiscale_tcn import FullSequenceMultiScaleTCN
from src.train_depth_ir_pose_roi_40class import detailed_metrics, per_class_rows, plot_history
from src.train_unimodal import PROJECT_ROOT, save_confusion_matrix


RUN_ID = "full_sequence_multiscale_tcn_14train_4val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_person_crop_full_sequence_tcn.yaml",
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


def build_datasets(config: dict[str, Any]) -> tuple[FullSequencePersonCropPoseROIDataset, FullSequencePersonCropPoseROIDataset]:
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
        "num_frames": int(config["temporal_frames"]),
        "image_size": int(config["image_size"]),
        "pose_cache_path": resolve_path(config["pose_cache"]),
        "data_root": resolve_path(config["data_root"]),
        "interaction_config": dict(config["interaction_roi"]),
        "person_crop_config": dict(config["person_crop"]),
    }
    return (
        FullSequencePersonCropPoseROIDataset(train_frame, training=False, **common),
        FullSequencePersonCropPoseROIDataset(val_frame, training=False, **common),
    )


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_spatial_encoder(config: dict[str, Any], device: torch.device) -> tuple[DepthIRPoseROIExpert, dict[str, Any]]:
    checkpoint_path = resolve_path(config["b2_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("epoch", -1)) != 25:
        raise ValueError(f"Expected B2 Epoch 25, got {checkpoint.get('epoch')}.")
    model = DepthIRPoseROIExpert(
        num_classes=40,
        expected_views=4,
        embedding_dim=192,
        frame_feature_dim=int(config["frame_feature_dim"]),
        dropout=float(config["dropout"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    audit = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": 25,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "spatial_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_spatial_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "frame_feature_dim": int(config["frame_feature_dim"]),
    }
    return model, audit


def encode_item(
    model: DepthIRPoseROIExpert,
    item: dict[str, object],
    device: torch.device,
    chunk_frames: int,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = item["depth_input"]
    ir = item["ir_input"]
    if not isinstance(depth, torch.Tensor) or not isinstance(ir, torch.Tensor):
        raise TypeError("Full-sequence item is missing Depth/IR tensors.")
    features: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(depth), chunk_frames):
            stop = min(len(depth), start + chunk_frames)
            inputs = {
                "depth_input": depth[start:stop].unsqueeze(0).to(device),
                "ir_input": ir[start:stop].unsqueeze(0).to(device),
            }
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model.encode_frames(inputs)
            features.append(output["frame_features"].squeeze(0).float().cpu().numpy())
            attentions.append(output["roi_attention"].squeeze(0).float().cpu().numpy())
            gates.append(output["modality_gate"].squeeze(0).float().cpu().numpy())
    return np.concatenate(features), np.concatenate(attentions), np.concatenate(gates)


def verify_b2_reproduction(
    model: DepthIRPoseROIExpert,
    val_dataset: FullSequencePersonCropPoseROIDataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    archive_path = resolve_path(config["b2_val_predictions"])
    archive = np.load(archive_path, allow_pickle=False)
    sample_id = str(archive["sample_ids"][0])
    lookup = {str(sample["sample_id"]): index for index, sample in enumerate(val_dataset.samples)}
    if sample_id not in lookup:
        raise ValueError(f"B2 reproduction sample is absent from validation dataset: {sample_id}")
    original_frames = val_dataset.num_frames
    val_dataset.num_frames = 24
    val_dataset._window = types.MethodType(PoseROIDataset._window, val_dataset)
    try:
        item = val_dataset[lookup[sample_id]]
    finally:
        del val_dataset._window
        val_dataset.num_frames = original_frames
    inputs = {
        "depth_input": item["depth_input"].unsqueeze(0).to(device),
        "ir_input": item["ir_input"].unsqueeze(0).to(device),
    }
    mask = item["temporal_mask"].unsqueeze(0).to(device)
    amp_enabled = bool(config["spatial_amp"]) and device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.float16, enabled=amp_enabled,
    ):
        output = model(inputs, temporal_mask=mask)
    reproduced = output["logits"].squeeze(0).float().cpu().numpy()
    expected = np.asarray(archive["logits"][0], dtype=np.float32)
    maximum_difference = float(np.max(np.abs(reproduced - expected)))
    tolerance = 4e-3 if amp_enabled else 1e-4
    result = {
        "sample_id": sample_id,
        "archive": str(archive_path),
        "amp_enabled": amp_enabled,
        "absolute_tolerance": tolerance,
        "maximum_logit_difference": maximum_difference,
        "prediction_matches": int(reproduced.argmax()) == int(expected.argmax()),
        "passed": maximum_difference <= tolerance,
    }
    if not result["passed"]:
        raise RuntimeError(f"B2 Epoch 25 reproduction failed: {result}")
    return result


def extract_cache(
    split: str,
    dataset: FullSequencePersonCropPoseROIDataset,
    model: DepthIRPoseROIExpert,
    cache_root: Path,
    config: dict[str, Any],
    device: torch.device,
    limit: int | None,
) -> None:
    split_root = cache_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    count = min(len(dataset), limit or len(dataset))
    shard_size = int(config["cache_shard_size"])
    amp_enabled = bool(config["spatial_amp"]) and device.type == "cuda"
    for shard_start in range(0, count, shard_size):
        shard_stop = min(count, shard_start + shard_size)
        shard_path = split_root / f"shard_{shard_start:05d}_{shard_stop:05d}.npz"
        expected_ids = np.asarray(
            [str(dataset.samples[index]["sample_id"]) for index in range(shard_start, shard_stop)], dtype=str,
        )
        if shard_path.exists():
            existing = np.load(shard_path, allow_pickle=False)
            if np.array_equal(existing["sample_ids"], expected_ids):
                print(json.dumps({"cache_resume": split, "shard": shard_path.name}), flush=True)
                continue
            raise ValueError(f"Existing cache shard does not match dataset: {shard_path}")
        frame_features = []
        roi_attention = []
        modality_gate = []
        masks = []
        frame_indices = []
        labels = []
        users = []
        lengths = []
        started = time.perf_counter()
        for index in range(shard_start, shard_stop):
            item = dataset[index]
            features, attention, gate = encode_item(
                model, item, device, int(config["feature_chunk_frames"]), amp_enabled,
            )
            frame_features.append(features.astype(np.float16))
            roi_attention.append(attention.astype(np.float16))
            modality_gate.append(gate.astype(np.float16))
            masks.append(np.asarray(item["temporal_mask"], dtype=bool))
            frame_indices.append(np.asarray(item["frame_indices"], dtype=np.int64))
            labels.append(int(item["label"]))
            users.append(str(item["user_id"]))
            lengths.append(int(item["length"]))
        temporary = shard_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            sample_ids=expected_ids,
            user_ids=np.asarray(users, dtype=str),
            labels=np.asarray(labels, dtype=np.int64),
            original_lengths=np.asarray(lengths, dtype=np.int64),
            frame_indices=np.stack(frame_indices),
            temporal_mask=np.stack(masks),
            frame_features=np.stack(frame_features),
            roi_attention=np.stack(roi_attention),
            modality_gate=np.stack(modality_gate),
        )
        temporary.replace(shard_path)
        print(json.dumps({
            "cache_written": split,
            "shard": shard_path.name,
            "samples": shard_stop - shard_start,
            "seconds": time.perf_counter() - started,
        }), flush=True)


def load_cache(split: str, cache_root: Path, expected_count: int) -> dict[str, np.ndarray]:
    shards = sorted((cache_root / split).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No cache shards found for {split}.")
    keys = (
        "sample_ids", "user_ids", "labels", "original_lengths", "frame_indices",
        "temporal_mask", "frame_features", "roi_attention", "modality_gate",
    )
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in shards:
        shard = np.load(path, allow_pickle=False)
        for key in keys:
            pieces[key].append(np.asarray(shard[key]))
    result = {key: np.concatenate(values) for key, values in pieces.items()}
    if len(result["labels"]) != expected_count or len(np.unique(result["sample_ids"])) != expected_count:
        raise ValueError(f"Cache integrity failed for {split}: {len(result['labels'])}/{expected_count}.")
    return result


def cache_loader(data: dict[str, np.ndarray], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(data["frame_features"].astype(np.float32)),
        torch.from_numpy(data["temporal_mask"].astype(bool)),
        torch.from_numpy(data["labels"].astype(np.int64)),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed), pin_memory=True,
    )


def temporal_epoch(
    model: FullSequenceMultiScaleTCN,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    gradient_clip: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    losses = 0.0
    samples = 0
    labels_all = []
    logits_all = []
    embeddings_all = []
    short_attention_all = []
    long_attention_all = []
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for features, mask, labels in loader:
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(features, mask)
                loss = nn.functional.cross_entropy(output["logits"], labels)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            count = len(labels)
            losses += float(loss.detach()) * count
            samples += count
            labels_all.append(labels.detach().cpu())
            logits_all.append(output["logits"].detach().float().cpu())
            embeddings_all.append(output["embedding"].detach().float().cpu())
            short_attention_all.append(output["short_attention"].detach().float().cpu())
            long_attention_all.append(output["long_attention"].detach().float().cpu())
    labels_np = torch.cat(labels_all).numpy()
    logits_np = torch.cat(logits_all).numpy()
    result = detailed_metrics(labels_np, logits_np, losses / samples)
    return result, {
        "labels": labels_np,
        "logits": logits_np,
        "embeddings": torch.cat(embeddings_all).numpy(),
        "short_attention": torch.cat(short_attention_all).numpy(),
        "long_attention": torch.cat(long_attention_all).numpy(),
    }


def save_checkpoint(path: Path, model: FullSequenceMultiScaleTCN, epoch: int, result: dict[str, Any]) -> None:
    torch.save({
        "epoch": epoch,
        "val_accuracy": result["accuracy"],
        "val_macro_f1": result["macro_f1"],
        "model_state_dict": model.state_dict(),
    }, path)


def save_predictions(
    path: Path,
    output: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    target_result: dict[str, Any],
) -> None:
    logits = output["logits"]
    np.savez_compressed(
        path,
        sample_ids=cache["sample_ids"],
        user_ids=cache["user_ids"],
        labels=output["labels"],
        predictions=logits.argmax(axis=1),
        logits=logits,
        probabilities=np.asarray(target_result["probabilities"]),
        true_class_rank=np.asarray(target_result["true_class_rank"]),
        embeddings=output["embeddings"],
        short_attention=output["short_attention"],
        long_attention=output["long_attention"],
        temporal_mask=cache["temporal_mask"],
        frame_indices=cache["frame_indices"],
        original_lengths=cache["original_lengths"],
    )


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    config["smoke_test"] = bool(args.smoke_test)
    if args.smoke_test:
        config["temporal_frames"] = int(config["smoke_temporal_frames"])
    set_seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    output_root = resolve_path(config["output_root"])
    run_id = f"{args.run_id}_smoke" if args.smoke_test and not args.run_id.endswith("_smoke") else args.run_id
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config["run_dir"] = str(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_dataset, val_dataset = build_datasets(config)
    if len(train_dataset) != 2320 or len(val_dataset) != 590:
        raise ValueError(f"Unexpected split sizes: {len(train_dataset)}/{len(val_dataset)}.")
    if getattr(train_dataset, "original_class_ids") != list(range(40)):
        raise ValueError("Training class map is not 0-39.")
    if set(train_dataset.user_ids) & set(val_dataset.user_ids):
        raise ValueError("Train and validation users overlap.")
    spatial_model, spatial_audit = load_spatial_encoder(config, device)
    spatial_audit["b2_reproduction"] = verify_b2_reproduction(
        spatial_model, val_dataset, config, device,
    )
    (run_dir / "spatial_encoder_audit.json").write_text(json.dumps(spatial_audit, indent=2) + "\n", encoding="utf-8")

    cache_name = f"b2_spatial_features_v2_t{config['temporal_frames']}" + ("_smoke" if args.smoke_test else "")
    cache_root = output_root / cache_name
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_meta = {
        **spatial_audit,
        "temporal_frames": int(config["temporal_frames"]),
        "sampling": "all frames with repeat-last padding; endpoint-preserving uniform sampling above cap",
        "image_size": int(config["image_size"]),
        "expected_views": 4,
        "test_read": False,
    }
    metadata_path = cache_root / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != cache_meta:
            raise ValueError("Existing cache metadata differs from current configuration.")
    else:
        metadata_path.write_text(json.dumps(cache_meta, indent=2) + "\n", encoding="utf-8")
    smoke_train = int(config["smoke_train_samples"]) if args.smoke_test else None
    smoke_val = int(config["smoke_val_samples"]) if args.smoke_test else None
    extract_cache("train", train_dataset, spatial_model, cache_root, config, device, smoke_train)
    extract_cache("validation", val_dataset, spatial_model, cache_root, config, device, smoke_val)
    train_count = smoke_train or len(train_dataset)
    val_count = smoke_val or len(val_dataset)
    train_cache = load_cache("train", cache_root, train_count)
    val_cache = load_cache("validation", cache_root, val_count)
    del spatial_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = FullSequenceMultiScaleTCN(
        frame_feature_dim=int(config["frame_feature_dim"]),
        channels=int(config["tcn_channels"]),
        embedding_dim=int(config["embedding_dim"]),
        num_classes=40,
        short_dilations=tuple(config["short_dilations"]),
        long_dilations=tuple(config["long_dilations"]),
        dropout=float(config["dropout"]),
    ).to(device)
    temporal_audit = {
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "short_receptive_field": model.short_branch.receptive_field,
        "long_receptive_field": model.long_branch.receptive_field,
        "temporal_frames": int(config["temporal_frames"]),
    }
    (run_dir / "temporal_model_audit.json").write_text(json.dumps(temporal_audit, indent=2) + "\n", encoding="utf-8")
    train_loader = cache_loader(train_cache, int(config["batch_size"]), True, int(config["seed"]))
    val_loader = cache_loader(val_cache, int(config["batch_size"]), False, int(config["seed"]) + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["smoke_epochs"] if args.smoke_test else config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(config["temporal_amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv").sort_values("class_id")
    class_names = class_map["action_name"].tolist()
    train_support = class_map["train_support"].to_numpy(dtype=np.int64)
    val_support = class_map["val_support"].to_numpy(dtype=np.int64)
    history = []
    diagnostics = []
    best_accuracy = (-1.0, -1.0)
    best_macro = (-1.0, -1.0)
    no_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_result, _ = temporal_epoch(
            model, train_loader, device, amp_enabled, optimizer, scaler, float(config["gradient_clip"]),
        )
        val_result, val_output = temporal_epoch(model, val_loader, device, amp_enabled, None, None, 0.0)
        row = {
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
            "epoch_time_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        diagnostics.extend(per_class_rows(
            epoch, val_output["labels"], val_output["logits"],
            val_result, class_names, train_support, val_support,
        ))
        accuracy_key = (float(val_result["accuracy"]), float(val_result["macro_f1"]))
        macro_key = (float(val_result["macro_f1"]), float(val_result["accuracy"]))
        if accuracy_key > best_accuracy:
            best_accuracy = accuracy_key
            save_checkpoint(run_dir / "best_accuracy.pt", model, epoch, val_result)
        if macro_key > best_macro:
            best_macro = macro_key
            no_improvement = 0
            save_checkpoint(run_dir / "best_macro_f1.pt", model, epoch, val_result)
        else:
            no_improvement += 1
        scheduler.step()
        save_checkpoint(run_dir / "last_complete.pt", model, epoch, val_result)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(diagnostics).to_csv(
            run_dir / "epoch_per_class_diagnostics.csv", index=False, encoding="utf-8-sig",
        )
        print(json.dumps(row), flush=True)
        if not args.smoke_test and no_improvement >= int(config["early_stopping_patience"]):
            print(json.dumps({"early_stop_epoch": epoch, "patience": int(config["early_stopping_patience"])}), flush=True)
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(run_dir / "epoch_summary.csv", index=False, encoding="utf-8-sig")
    plot_history(history_frame, run_dir)
    checkpoint_results = {}
    for name in ("best_accuracy", "best_macro_f1"):
        checkpoint = torch.load(run_dir / f"{name}.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        result, output = temporal_epoch(model, val_loader, device, amp_enabled, None, None, 0.0)
        rows = per_class_rows(
            int(checkpoint["epoch"]), output["labels"], output["logits"], result,
            class_names, train_support, val_support,
        )
        pd.DataFrame(rows).to_csv(run_dir / f"per_class_{name}.csv", index=False, encoding="utf-8-sig")
        matrix = np.asarray(result["confusion_matrix"])
        save_confusion_matrix(matrix.tolist(), run_dir / f"confusion_matrix_{name}.png")
        save_predictions(run_dir / f"val_predictions_{name}.npz", output, val_cache, result)
        checkpoint_results[name] = {
            "epoch": int(checkpoint["epoch"]),
            "accuracy": float(result["accuracy"]),
            "macro_f1": float(result["macro_f1"]),
            "weighted_f1": float(result["weighted_f1"]),
            "val_loss": float(result["loss"]),
            "top3_accuracy": float(result["top3_accuracy"]),
            "top5_accuracy": float(result["top5_accuracy"]),
        }
    summary = {
        "status": "passed",
        "test_read": False,
        "train_samples": train_count,
        "val_samples": val_count,
        "epochs_completed": int(history[-1]["epoch"]),
        "runtime_seconds_excluding_cache": time.perf_counter() - started,
        "cache_root": str(cache_root),
        "spatial_encoder": spatial_audit,
        "temporal_model": temporal_audit,
        "checkpoints": checkpoint_results,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run(parse_args())
