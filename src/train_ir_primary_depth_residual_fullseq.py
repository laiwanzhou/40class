from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Subset

from src.data.ir_primary_full_sequence_dataset import (
    FrameBudgetBatchSampler,
    IRPrimaryFullSequenceDataset,
    QUALITY_NAMES,
    collate_full_sequences,
)
from src.models.expert_contract import ExpertBatchResult, ExpertOutput
from src.models.ir_primary_depth_residual_tcn import IRPrimaryDepthResidualTCN
from src.train_depth_ir_pose_roi_40class import detailed_metrics, per_class_rows, plot_history
from src.train_unimodal import PROJECT_ROOT, save_confusion_matrix


RUN_ID = "ir_primary_ordinal_depth_variable_fullseq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
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


def build_datasets(config: dict[str, Any]) -> tuple[IRPrimaryFullSequenceDataset, IRPrimaryFullSequenceDataset]:
    manifest = resolve_path(config["input_manifest"])
    representation = str(config["depth_representation"])
    return (
        IRPrimaryFullSequenceDataset(manifest, split="train", depth_representation=representation),
        IRPrimaryFullSequenceDataset(manifest, split="val", depth_representation=representation),
    )


def build_loader(
    dataset: IRPrimaryFullSequenceDataset,
    config: dict[str, Any],
    *,
    training: bool,
    smoke_limit: int | None,
) -> tuple[DataLoader, FrameBudgetBatchSampler]:
    actual: IRPrimaryFullSequenceDataset | Subset = dataset
    lengths = dataset.lengths
    if smoke_limit is not None:
        count = min(smoke_limit, len(dataset))
        actual = Subset(dataset, range(count))
        lengths = lengths[:count]
    prefix = "train" if training else "val"
    sampler = FrameBudgetBatchSampler(
        lengths,
        max_frames=int(config[f"{prefix}_max_padded_frames"]),
        max_samples=int(config[f"{prefix}_max_samples_per_batch"]),
        shuffle=training,
        seed=int(config["seed"]) + (0 if training else 1),
        bucket_size=int(config["length_bucket_size"]),
    )
    workers = int(config["num_workers"])
    loader = DataLoader(
        actual,
        batch_sampler=sampler,
        collate_fn=collate_full_sequences,
        num_workers=workers,
        pin_memory=bool(config["pin_memory"]),
        persistent_workers=workers > 0,
    )
    return loader, sampler


def make_model(config: dict[str, Any]) -> IRPrimaryDepthResidualTCN:
    return IRPrimaryDepthResidualTCN(
        num_classes=40,
        frame_feature_dim=int(config["frame_feature_dim"]),
        channels=int(config["tcn_channels"]),
        embedding_dim=int(config["embedding_dim"]),
        short_dilations=tuple(config["short_dilations"]),
        long_dilations=tuple(config["long_dilations"]),
        dropout=float(config["dropout"]),
        pretrained=bool(config["pretrained_ir"]),
        initial_depth_gate=float(config["initial_depth_gate"]),
        activation_checkpointing=bool(config["activation_checkpointing"]),
        spatial_view_chunk_size=int(config["spatial_view_chunk_size"]),
    )


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "ir", "depth", "depth_pixel_valid", "view_valid", "view_reliability",
        "temporal_mask", "quality", "quality_mask", "labels",
    )
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in names
        if isinstance(batch[name], torch.Tensor)
    }


def _small_lookup(class_ids: list[int], device: torch.device) -> torch.Tensor:
    lookup = torch.zeros(40, dtype=torch.bool, device=device)
    lookup[torch.tensor(class_ids, dtype=torch.long, device=device)] = True
    return lookup


def _losses(
    main_logits: torch.Tensor,
    small_logits: torch.Tensor,
    labels: torch.Tensor,
    lookup: torch.Tensor,
    small_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    small_labels = lookup[labels].long()
    main = nn.functional.cross_entropy(main_logits, labels)
    small = nn.functional.cross_entropy(small_logits, small_labels)
    return main + small_weight * small, main, small, small_labels


def run_epoch(
    model: IRPrimaryDepthResidualTCN,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> tuple[dict[str, Any], ExpertBatchResult, np.ndarray, tuple[str, ...]]:
    training = optimizer is not None
    model.train(training)
    lookup = _small_lookup(list(config["small_object_class_ids"]), device)
    accumulation = int(config["gradient_accumulation"])
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(config["amp_dtype"]) == "bfloat16" else torch.float16
    totals = mains = smalls = 0.0
    samples = 0
    sample_ids: list[str] = []
    user_ids: list[str] = []
    labels_all: list[torch.Tensor] = []
    small_labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    qualities: list[torch.Tensor] = []
    quality_masks: list[torch.Tensor] = []
    availability: list[torch.Tensor] = []
    small_logits_all: list[torch.Tensor] = []
    if training:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, batch in enumerate(loader):
            tensors = _to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(
                    tensors["ir"], tensors["depth"], tensors["depth_pixel_valid"],
                    tensors["view_valid"], tensors["view_reliability"], tensors["temporal_mask"],
                    tensors["quality"], tensors["quality_mask"],
                )
                loss, main, small, small_labels = _losses(
                    output.expert.main_logits,
                    output.small_gate_logits,
                    tensors["labels"],
                    lookup,
                    float(config["small_gate_loss_weight"]),
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
            count = len(tensors["labels"])
            totals += float(loss.detach()) * count
            mains += float(main.detach()) * count
            smalls += float(small.detach()) * count
            samples += count
            sample_ids.extend(batch["sample_ids"])
            user_ids.extend(batch["user_ids"])
            labels_all.append(tensors["labels"].detach().cpu())
            small_labels_all.append(small_labels.detach().cpu())
            logits_all.append(output.expert.main_logits.detach().float().cpu())
            embeddings.append(output.expert.embedding.detach().float().cpu())
            qualities.append(output.expert.quality.detach().float().cpu())
            quality_masks.append(output.expert.quality_mask.detach().cpu())
            availability.append(output.expert.availability.detach().cpu())
            small_logits_all.append(output.small_gate_logits.detach().float().cpu())

    labels_np = torch.cat(labels_all).numpy()
    logits = torch.cat(logits_all)
    result = detailed_metrics(labels_np, logits.numpy(), totals / samples)
    small_labels_tensor = torch.cat(small_labels_all)
    small_logits = torch.cat(small_logits_all)
    result.update({
        "main_loss": mains / samples,
        "small_gate_loss": smalls / samples,
        "small_gate_accuracy": float((small_logits.argmax(dim=1) == small_labels_tensor).float().mean()),
    })
    expert = ExpertOutput(
        main_logits=logits,
        embedding=torch.cat(embeddings),
        quality=torch.cat(qualities),
        quality_mask=torch.cat(quality_masks),
        availability=torch.cat(availability),
    )
    batch_result = ExpertBatchResult(
        sample_ids=tuple(sample_ids),
        class_map_hash=str(config["class_map_hash"]),
        output=expert,
        small_gate_logits=small_logits,
    )
    batch_result.validate()
    return result, batch_result, labels_np, tuple(user_ids)


def _checkpoint(
    path: Path,
    model: IRPrimaryDepthResidualTCN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save({
        "epoch": epoch,
        "val_accuracy": float(metrics["accuracy"]),
        "val_macro_f1": float(metrics["macro_f1"]),
        "class_map_hash": config["class_map_hash"],
        "quality_names": list(QUALITY_NAMES),
        "depth_representation": config["depth_representation"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def _history_row(
    epoch: int,
    train: dict[str, Any],
    val: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    seconds: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_loss": train["loss"],
        "train_accuracy": train["accuracy"],
        "train_macro_f1": train["macro_f1"],
        "train_small_gate_accuracy": train["small_gate_accuracy"],
        "val_loss": val["loss"],
        "val_accuracy": val["accuracy"],
        "val_macro_f1": val["macro_f1"],
        "val_weighted_f1": val["weighted_f1"],
        "val_top3_accuracy": val["top3_accuracy"],
        "val_top5_accuracy": val["top5_accuracy"],
        "val_small_gate_accuracy": val["small_gate_accuracy"],
        # Compatibility aliases used by the shared validation-history plots.
        "accuracy": val["accuracy"],
        "macro_f1": val["macro_f1"],
        "weighted_f1": val["weighted_f1"],
        "zero_f1_class_count": val["zero_f1_class_count"],
        "never_predicted_class_count": val["never_predicted_class_count"],
        "number_of_predicted_classes": val["number_of_predicted_classes"],
        "generalization_gap": train["accuracy"] - val["accuracy"],
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch_time_seconds": seconds,
    }


def _save_predictions(
    path: Path,
    result: ExpertBatchResult,
    labels: np.ndarray,
    user_ids: tuple[str, ...],
) -> None:
    logits = result.output.main_logits.numpy()
    np.savez_compressed(
        path,
        sample_ids=np.asarray(result.sample_ids),
        user_ids=np.asarray(user_ids),
        labels=labels,
        predictions=logits.argmax(axis=1),
        logits=logits,
        embeddings=result.output.embedding.numpy(),
        quality=result.output.quality.numpy(),
        quality_mask=result.output.quality_mask.numpy(),
        availability=result.output.availability.numpy(),
        small_gate_logits=result.small_gate_logits.numpy() if result.small_gate_logits is not None else None,
        class_map_hash=np.asarray(result.class_map_hash),
    )


def stop_requested(run_dir: Path, config: dict[str, Any]) -> bool:
    return (run_dir / str(config["stop_file_name"])).exists()


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_dataset, val_dataset = build_datasets(config)
    if (len(train_dataset), len(val_dataset)) != (2320, 590):
        raise ValueError(f"Unexpected split: {len(train_dataset)}/{len(val_dataset)}")
    if set(train_dataset.user_ids) & set(val_dataset.user_ids):
        raise ValueError("Train and validation users overlap")
    if train_dataset.class_map_hash != val_dataset.class_map_hash:
        raise ValueError("Train and validation class maps differ")
    config["class_map_hash"] = train_dataset.class_map_hash
    config["quality_names"] = list(QUALITY_NAMES)
    config["competition_test_read"] = False
    config["skeleton_connected"] = False
    config["imu_connected"] = False

    smoke = bool(args.smoke_test)
    smoke_train = int(config["smoke_train_samples"]) if smoke else None
    smoke_val = int(config["smoke_val_samples"]) if smoke else None
    train_loader, train_sampler = build_loader(
        train_dataset, config, training=True, smoke_limit=smoke_train,
    )
    val_loader, _ = build_loader(val_dataset, config, training=False, smoke_limit=smoke_val)
    run_id = f"{args.run_id}_smoke" if smoke else args.run_id
    run_dir = resolve_path(config["output_root"]) / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["smoke_epochs"] if smoke else config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]) and device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_accuracy = (-1.0, -1.0)
    best_macro = (-1.0, -1.0)
    no_improvement = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        train, _, _, _ = run_epoch(model, train_loader, device, config, optimizer=optimizer, scaler=scaler)
        val, _, _, _ = run_epoch(model, val_loader, device, config, optimizer=None, scaler=None)
        row = _history_row(epoch, train, val, optimizer, time.perf_counter() - started)
        history.append(row)
        accuracy_key = (float(val["accuracy"]), float(val["macro_f1"]))
        macro_key = (float(val["macro_f1"]), float(val["accuracy"]))
        if accuracy_key > best_accuracy:
            best_accuracy = accuracy_key
            _checkpoint(run_dir / "best_accuracy.pt", model, optimizer, epoch, val, config)
        if macro_key > best_macro:
            best_macro = macro_key
            no_improvement = 0
            _checkpoint(run_dir / "best_macro_f1.pt", model, optimizer, epoch, val, config)
        else:
            no_improvement += 1
        _checkpoint(run_dir / "last_complete.pt", model, optimizer, epoch, val, config)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        scheduler.step()
        print(json.dumps(row), flush=True)
        if stop_requested(run_dir, config):
            print(json.dumps({"safe_stop_after_epoch": epoch}), flush=True)
            break
        if not smoke and no_improvement >= int(config["patience"]):
            print(json.dumps({"early_stop_epoch": epoch}), flush=True)
            break

    class_names = train_dataset.class_names
    train_support = np.bincount(
        [int(sample.iloc[0].class_id) for sample in train_dataset.samples], minlength=40,
    )
    val_support = np.bincount(
        [int(sample.iloc[0].class_id) for sample in val_dataset.samples], minlength=40,
    )
    checkpoint_summary: dict[str, Any] = {}
    for name in ("best_accuracy", "best_macro_f1"):
        saved = torch.load(run_dir / f"{name}.pt", map_location=device, weights_only=True)
        if saved["class_map_hash"] != config["class_map_hash"]:
            raise ValueError("Checkpoint class map differs")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        metrics, output, labels, users = run_epoch(
            model, val_loader, device, config, optimizer=None, scaler=None,
        )
        rows = per_class_rows(
            int(saved["epoch"]), labels, output.output.main_logits.numpy(), metrics,
            class_names, train_support, val_support,
        )
        pd.DataFrame(rows).to_csv(run_dir / f"per_class_{name}.csv", index=False, encoding="utf-8-sig")
        save_confusion_matrix(
            np.asarray(metrics["confusion_matrix"]).tolist(), run_dir / f"confusion_matrix_{name}.png",
        )
        _save_predictions(run_dir / f"val_predictions_{name}.npz", output, labels, users)
        checkpoint_summary[name] = {
            "epoch": int(saved["epoch"]),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "val_loss": float(metrics["loss"]),
        }
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(run_dir / "epoch_summary.csv", index=False, encoding="utf-8-sig")
    plot_history(history_frame, run_dir)
    summary = {
        "status": "complete",
        "depth_representation": config["depth_representation"],
        "train_samples": smoke_train or len(train_dataset),
        "val_samples": smoke_val or len(val_dataset),
        "epochs_completed": len(history),
        "class_map_hash": config["class_map_hash"],
        "quality_names": list(QUALITY_NAMES),
        "competition_test_read": False,
        "skeleton_connected": False,
        "imu_connected": False,
        "checkpoints": checkpoint_summary,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run(parse_args())
