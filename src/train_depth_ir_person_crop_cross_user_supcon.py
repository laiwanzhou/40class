from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from src.data.cross_user_batch_sampler import CrossUserActionBatchSampler
from src.engine import collect_predictions
from src.models.cross_user_supcon import CrossUserSupConModel, cross_user_supcon_loss
from src.train_depth_ir_pose_roi_40class import (
    detailed_metrics,
    evaluate_checkpoint,
    generic_args,
    per_class_rows,
    plot_history,
    save_checkpoint,
)
from src.train_unimodal import PROJECT_ROOT, build_datasets, load_config, loader_for, seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_person_crop_cross_user_supcon.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-id", default="depth_ir_person_crop_cross_user_supcon_14train_4val")
    return parser.parse_args()


def train_loader_for(dataset: Any, config: dict[str, Any]) -> tuple[DataLoader[dict[str, object]], CrossUserActionBatchSampler]:
    sampler = CrossUserActionBatchSampler.from_dataset(
        dataset, batch_size=int(config["batch_size"]), seed=int(config["seed"]),
    )
    workers = int(config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2, "multiprocessing_context": "spawn"})
    return DataLoader(**kwargs), sampler


def move_batch(batch: dict[str, object], device: torch.device) -> tuple[dict[str, torch.Tensor], ...]:
    inputs = {
        "depth_input": batch["depth_input"].to(device, non_blocking=True),
        "ir_input": batch["ir_input"].to(device, non_blocking=True),
    }
    return (
        inputs,
        batch["temporal_mask"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
        batch["user_index"].to(device, non_blocking=True),
    )


def train_epoch(
    model: CrossUserSupConModel,
    loader: DataLoader[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce_loss": 0.0, "supcon_loss": 0.0}
    labels_all: list[torch.Tensor] = []
    predictions_all: list[torch.Tensor] = []
    samples = 0
    eligible_anchors = 0
    contrastive_mode = str(config.get("contrastive_mode", "in_batch"))
    limit = config.get("max_train_batches")
    for step, batch in enumerate(loader, start=1):
        inputs, temporal_mask, labels, users = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda" and bool(config["amp"])):
            output = model(inputs, temporal_mask=temporal_mask)
            ce_loss = nn.functional.cross_entropy(output["logits"], labels)
            if contrastive_mode == "prototype_bank":
                if model.prototype_bank is None:
                    raise RuntimeError("Prototype-bank mode requires a prototype bank")
                supcon_loss, batch_eligible = model.prototype_bank.loss(
                    output["projection"], labels, users,
                    temperature=float(config["supcon_temperature"]),
                    same_user_negative_weight=float(config["same_user_negative_weight"]),
                )
            elif contrastive_mode == "in_batch":
                supcon_loss = cross_user_supcon_loss(
                    output["projection"], labels, users,
                    temperature=float(config["supcon_temperature"]),
                    same_user_negative_weight=float(config["same_user_negative_weight"]),
                )
                batch_eligible = len(labels)
            else:
                raise ValueError(f"Unknown contrastive mode: {contrastive_mode}")
            loss = ce_loss + float(config["supcon_lambda"]) * supcon_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        scaler.step(optimizer)
        scaler.update()
        if contrastive_mode == "prototype_bank":
            assert model.prototype_bank is not None
            model.prototype_bank.update(output["projection"], labels, users)
        count = len(labels)
        totals["loss"] += float(loss.detach()) * count
        totals["ce_loss"] += float(ce_loss.detach()) * count
        totals["supcon_loss"] += float(supcon_loss.detach()) * count
        labels_all.append(labels.detach().cpu())
        predictions_all.append(output["logits"].detach().argmax(1).cpu())
        samples += count
        eligible_anchors += batch_eligible
        if limit and step >= int(limit):
            break
    labels_np = torch.cat(labels_all).numpy()
    predictions_np = torch.cat(predictions_all).numpy()
    _, _, f1, _ = precision_recall_fscore_support(
        labels_np, predictions_np, labels=np.arange(40), zero_division=0,
    )
    return {
        **{name: value / samples for name, value in totals.items()},
        "accuracy": float(np.mean(labels_np == predictions_np)),
        "macro_f1": float(f1.mean()),
        "supcon_eligible_anchor_rate": eligible_anchors / samples,
        "prototype_count": float(model.prototype_bank.prototype_count) if model.prototype_bank is not None else 0.0,
    }


def save_prediction_archive(path: Path, output: dict[str, object], user_ids: list[str]) -> None:
    labels = np.asarray(output["labels"]).astype(np.int64)
    logits = np.asarray(output["logits"])
    probabilities = torch.from_numpy(logits).softmax(1).numpy()
    np.savez_compressed(
        path,
        sample_ids=np.asarray(output["sample_ids"]),
        user_ids=np.asarray(user_ids),
        labels=labels,
        predictions=logits.argmax(1),
        logits=logits,
        probabilities=probabilities,
        embeddings=np.asarray(output["embeddings"]),
        roi_attention=np.asarray(output["roi_attention"]),
        modality_gate=np.asarray(output["modality_gate"]),
        temporal_mask=np.asarray(output["temporal_mask"]),
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config, generic_args(args))
    set_seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    run_id = f"{args.run_id}_smoke" if args.smoke_test and not args.run_id.endswith("_smoke") else args.run_id
    run_dir = Path(config["output_root"]) / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config["run_dir"] = str(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_dataset, val_dataset = build_datasets(config)
    if len(train_dataset) != 2320 or len(val_dataset) != 590:
        raise ValueError(f"Unexpected split sizes: {len(train_dataset)}/{len(val_dataset)}")
    if set(getattr(train_dataset, "user_ids")) & set(getattr(val_dataset, "user_ids")):
        raise ValueError("Train and validation users overlap")
    contrastive_mode = str(config.get("contrastive_mode", "in_batch"))
    if contrastive_mode == "prototype_bank":
        train_loader = loader_for(train_dataset, config, training=True)
        batch_sampler = None
        samples = getattr(train_dataset, "samples")
        class_user_pairs = {
            (int(sample["label"]), str(sample["user_id"])) for sample in samples
        }
        users_per_class = {
            label: len({user for current_label, user in class_user_pairs if current_label == label})
            for label in range(int(config["num_classes"]))
        }
        sampler_audit = {
            "sampler": "b2_random_shuffle",
            "samples_per_epoch": len(train_dataset),
            "unique_sample_rate": 1.0,
            "class_exposure": "original_dataset_distribution",
            "target_prototype_count": len(class_user_pairs),
            "train_users_per_class_min": min(users_per_class.values()),
            "train_users_per_class_max": max(users_per_class.values()),
        }
    elif contrastive_mode == "in_batch":
        train_loader, batch_sampler = train_loader_for(train_dataset, config)
        sampler_audit = batch_sampler.audit(epochs=3)
        if sampler_audit["cross_user_positive_anchor_rate"] != 1.0 or sampler_audit["same_user_negative_anchor_rate"] != 1.0:
            raise RuntimeError(f"Sampler audit failed: {sampler_audit}")
    else:
        raise ValueError(f"Unknown contrastive mode: {contrastive_mode}")
    val_loader = loader_for(val_dataset, config, training=False)
    (run_dir / "batch_sampler_audit.json").write_text(json.dumps(sampler_audit, indent=2) + "\n", encoding="utf-8")

    model = CrossUserSupConModel(
        num_classes=int(config["num_classes"]),
        embedding_dim=int(config["embedding_dim"]),
        frame_feature_dim=int(config["frame_feature_dim"]),
        projection_dim=int(config["projection_dim"]),
        dropout=float(config["dropout"]),
        pretrained=bool(config["pretrained"]),
        prototype_num_users=(len(getattr(train_dataset, "user_ids")) if contrastive_mode == "prototype_bank" else None),
        prototype_momentum=float(config.get("prototype_momentum", 0.9)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    class_names = list(getattr(train_dataset, "class_names"))
    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv").sort_values("class_id")
    train_support = class_map["train_support"].to_numpy(dtype=np.int64)
    val_support = class_map["val_support"].to_numpy(dtype=np.int64)
    history: list[dict[str, Any]] = []
    class_history: list[dict[str, Any]] = []
    best_accuracy = (-1.0, -1.0)
    best_macro = (-1.0, -1.0)
    best_success = (-1.0, -1.0)
    started = time.perf_counter()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch - 1)
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, config)
        validation = collect_predictions(model, val_loader, criterion, device, amp_enabled)
        labels = np.asarray(validation["labels"])
        logits = np.asarray(validation["logits"])
        metrics = detailed_metrics(labels, logits, float(validation["metrics"]["loss"]))
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_supcon_loss": train_metrics["supcon_loss"],
            "train_supcon_eligible_anchor_rate": train_metrics["supcon_eligible_anchor_rate"],
            "prototype_count": int(train_metrics["prototype_count"]),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "top3_accuracy": metrics["top3_accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "zero_f1_class_count": metrics["zero_f1_class_count"],
            "zero_recall_class_count": metrics["zero_recall_class_count"],
            "never_predicted_class_count": metrics["never_predicted_class_count"],
            "number_of_predicted_classes": metrics["number_of_predicted_classes"],
            "generalization_gap": train_metrics["accuracy"] - metrics["accuracy"],
            "epoch_time_seconds": time.perf_counter() - epoch_started,
            "gpu_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
            "gpu_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        class_history.extend(per_class_rows(epoch, labels, logits, metrics, class_names, train_support, val_support))
        accuracy_key = (float(metrics["accuracy"]), float(metrics["macro_f1"]))
        macro_key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
        if accuracy_key > best_accuracy:
            best_accuracy = accuracy_key
            save_checkpoint(run_dir / "best_accuracy.pt", model, epoch, metrics)
        if macro_key > best_macro:
            best_macro = macro_key
            save_checkpoint(run_dir / "best_macro_f1.pt", model, epoch, metrics)
        if float(metrics["accuracy"]) + 1e-12 >= float(config["baseline_accuracy"]):
            success_key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
            if success_key > best_success:
                best_success = success_key
                save_checkpoint(run_dir / "best_success.pt", model, epoch, metrics)
        scheduler.step()
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(class_history).to_csv(run_dir / "epoch_per_class_diagnostics.csv", index=False, encoding="utf-8-sig")
        save_checkpoint(run_dir / "last_complete.pt", model, epoch, metrics)
        print(json.dumps(row), flush=True)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(run_dir / "epoch_summary.csv", index=False, encoding="utf-8-sig")
    plot_history(history_frame, run_dir)
    checkpoint_files = {"best_accuracy": "best_accuracy.pt", "best_macro_f1": "best_macro_f1.pt"}
    primary_name = "best_success" if (run_dir / "best_success.pt").exists() else "best_accuracy"
    if primary_name == "best_success":
        checkpoint_files["best_success"] = "best_success.pt"
    evaluations = {
        suffix: evaluate_checkpoint(
            suffix, run_dir / filename, model, val_loader, criterion, device, amp_enabled, run_dir,
            class_names, np.arange(40, dtype=np.int64), train_support, val_support,
        )
        for suffix, filename in checkpoint_files.items()
    }
    primary_checkpoint = torch.load(run_dir / checkpoint_files[primary_name], map_location=device, weights_only=True)
    model.load_state_dict(primary_checkpoint["model_state_dict"])
    setattr(train_dataset, "training", False)
    train_eval_loader = loader_for(train_dataset, {**config, "num_workers": int(config["num_workers"])}, training=False)
    train_output = collect_predictions(model, train_eval_loader, criterion, device, amp_enabled)
    val_output = collect_predictions(model, val_loader, criterion, device, amp_enabled)
    train_user_lookup = {str(sample["sample_id"]): str(sample["user_id"]) for sample in getattr(train_dataset, "samples")}
    val_user_lookup = {str(sample["sample_id"]): str(sample["user_id"]) for sample in getattr(val_dataset, "samples")}
    save_prediction_archive(
        run_dir / "train_predictions_primary.npz", train_output,
        [train_user_lookup[str(sample_id)] for sample_id in train_output["sample_ids"]],
    )
    save_prediction_archive(
        run_dir / "val_predictions_primary.npz", val_output,
        [val_user_lookup[str(sample_id)] for sample_id in val_output["sample_ids"]],
    )
    result = {
        "status": "passed",
        "test_read": False,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_users": list(getattr(train_dataset, "user_ids")),
        "val_users": list(getattr(val_dataset, "user_ids")),
        "epochs_completed": epochs,
        "primary_checkpoint": checkpoint_files[primary_name],
        "primary_epoch": int(primary_checkpoint["epoch"]),
        "success_checkpoint_found": (run_dir / "best_success.pt").exists(),
        "batch_sampler_audit": sampler_audit,
        "evaluations": evaluations,
        "total_training_time_seconds": time.perf_counter() - started,
        "peak_allocated_mb": float(history_frame["gpu_peak_allocated_mb"].max()),
        "peak_reserved_mb": float(history_frame["gpu_peak_reserved_mb"].max()),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT_JSON={json.dumps(result)}", flush=True)


if __name__ == "__main__":
    run(parse_args())
