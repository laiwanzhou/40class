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
from torch import nn

from src.data import load_modality_frames
from src.data.person_crop_pose_roi_dataset import PersonCropPoseROIDataset
from src.engine import collect_predictions, run_epoch
from src.models.target16_conditional_expert import Target16ConditionalExpert
from src.train_depth_ir_pose_roi_40class import detailed_metrics, generic_args, per_class_rows, plot_history
from src.train_unimodal import PROJECT_ROOT, build_datasets, load_config, loader_for, set_seed


RUN_ID = "target16_conditional_e2_14train_4val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_person_crop_target16_e2.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args()


def _save_checkpoint(
    path: Path,
    model: Target16ConditionalExpert,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "target_class_ids": list(model.target_class_ids),
            "target_actions": list(config["target_actions"]),
            "b2_checkpoint": str(config["b2_checkpoint"]),
            "b2_epoch": 25,
            "frozen_shared_blocks": model.frozen_shared_blocks,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _full_validation_dataset(config: dict[str, Any]) -> PersonCropPoseROIDataset:
    _, frame = load_modality_frames(
        Path(config["manifest"]), Path(config["fold"]), Path(config["data_root"]), str(config["path_column"]),
    )
    if config.get("pairing_audit"):
        audit = pd.read_csv(Path(config["pairing_audit"]), encoding="utf-8-sig")
        valid = set(audit.loc[audit["complete_pairing"], "sample_id"].astype(str))
        frame = frame[frame["sample_id"].isin(valid)].reset_index(drop=True)
    actions = frame[["class_id", "action_name"]].drop_duplicates().sort_values("class_id")["action_name"].tolist()
    return PersonCropPoseROIDataset(
        frame=frame,
        hard_actions=actions,
        num_frames=int(config["num_frames"]),
        image_size=int(config["image_size"]),
        training=False,
        pose_cache_path=Path(config["pose_cache"]),
        data_root=Path(config["data_root"]),
        interaction_config=dict(config["interaction_roi"]),
        person_crop_config=dict(config["person_crop"]),
    )


def _collect_full_predictions(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    sample_ids: list[str] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {str(key): value.to(device, non_blocking=True) for key, value in batch["input"].items()}
            mask = batch["temporal_mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(inputs, temporal_mask=mask)
            sample_ids.extend(str(value) for value in batch["sample_id"])
            labels.append(batch["label"].numpy())
            logits.append(output["logits"].float().cpu().numpy())
            embeddings.append(output["embedding"].float().cpu().numpy())
    return {
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "labels": np.concatenate(labels),
        "logits": np.concatenate(logits),
        "embeddings": np.concatenate(embeddings),
    }


def _save_archive(
    path: Path,
    output: dict[str, object],
    user_lookup: dict[str, str],
    original_class_ids: np.ndarray,
) -> None:
    logits = np.asarray(output["logits"], dtype=np.float32)
    sample_ids = np.asarray(output["sample_ids"], dtype=str)
    np.savez_compressed(
        path,
        sample_ids=sample_ids,
        user_ids=np.asarray([user_lookup[value] for value in sample_ids], dtype=str),
        labels=np.asarray(output["labels"], dtype=np.int64),
        predictions=logits.argmax(axis=1),
        logits=logits,
        probabilities=torch.from_numpy(logits).softmax(1).numpy(),
        embeddings=np.asarray(output["embeddings"], dtype=np.float32),
        original_class_ids=original_class_ids,
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config, generic_args(args))
    set_seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    run_id = f"{args.run_id}_smoke" if args.smoke_test and not args.run_id.endswith("_smoke") else args.run_id
    run_dir = Path(config["output_root"]) / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config["run_dir"] = str(run_dir)
    config["b2_checkpoint"] = str((PROJECT_ROOT / str(config["b2_checkpoint"])).resolve())
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_dataset, val_dataset = build_datasets(config)
    target_ids = np.asarray(getattr(train_dataset, "original_class_ids"), dtype=np.int64)
    class_names = list(getattr(train_dataset, "class_names"))
    if len(train_dataset) != 880 or len(val_dataset) != 222 or len(target_ids) != 16:
        raise ValueError(f"Unexpected Target16 dataset: {len(train_dataset)}/{len(val_dataset)}/{len(target_ids)} classes.")
    if getattr(val_dataset, "original_class_ids") != target_ids.tolist():
        raise ValueError("Train and validation target class maps differ.")
    if set(getattr(train_dataset, "user_ids")) & set(getattr(val_dataset, "user_ids")):
        raise ValueError("Train and validation users overlap.")

    b2 = torch.load(config["b2_checkpoint"], map_location="cpu", weights_only=True)
    if int(b2.get("epoch", -1)) != 25:
        raise ValueError(f"Expected B2 Epoch 25, got {b2.get('epoch')}.")
    model = Target16ConditionalExpert(
        target_class_ids=target_ids,
        embedding_dim=int(config["embedding_dim"]),
        frame_feature_dim=int(config["frame_feature_dim"]),
        dropout=float(config["dropout"]),
        pretrained=False,
        frozen_shared_blocks=int(config["frozen_shared_blocks"]),
    )
    model.initialize_from_b2(b2)
    parameter_audit = model.freeze_b2_early_layers()
    (run_dir / "parameter_audit.json").write_text(json.dumps(parameter_audit, indent=2) + "\n", encoding="utf-8")
    model.to(device)

    train_loader = loader_for(train_dataset, config, training=True)
    val_loader = loader_for(val_dataset, config, training=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv").sort_values("class_id")
    target_rows = class_map.set_index("class_id").loc[target_ids]
    train_support = target_rows["train_support"].to_numpy(dtype=np.int64)
    val_support = target_rows["val_support"].to_numpy(dtype=np.int64)
    history: list[dict[str, Any]] = []
    class_history: list[dict[str, Any]] = []
    best_accuracy = (-1.0, -1.0)
    best_macro = (-1.0, -1.0)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(
            model, train_loader, criterion, device, amp_enabled,
            optimizer=optimizer, scaler=scaler, max_batches=config.get("max_train_batches"),
            gradient_clip=float(config["gradient_clip"]),
        )
        validation = collect_predictions(model, val_loader, criterion, device, amp_enabled)
        labels = np.asarray(validation["labels"])
        logits = np.asarray(validation["logits"])
        metrics = detailed_metrics(labels, logits, float(validation["metrics"]["loss"]))
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
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
            "predicted_class_count": metrics["number_of_predicted_classes"],
            "number_of_predicted_classes": metrics["number_of_predicted_classes"],
            "never_predicted_class_count": metrics["never_predicted_class_count"],
            "zero_f1_class_count": metrics["zero_f1_class_count"],
            "generalization_gap": train_metrics["accuracy"] - metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": time.perf_counter() - epoch_started,
            "gpu_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        }
        history.append(row)
        class_history.extend(per_class_rows(epoch, labels, logits, metrics, class_names, train_support, val_support))
        accuracy_key = (float(metrics["accuracy"]), float(metrics["macro_f1"]))
        macro_key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
        if accuracy_key > best_accuracy:
            best_accuracy = accuracy_key
            _save_checkpoint(run_dir / "best_accuracy.pt", model, epoch, metrics, config)
        if macro_key > best_macro:
            best_macro = macro_key
            _save_checkpoint(run_dir / "best_macro_f1.pt", model, epoch, metrics, config)
        scheduler.step()
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(class_history).to_csv(run_dir / "epoch_per_class_diagnostics.csv", index=False, encoding="utf-8-sig")
        _save_checkpoint(run_dir / "last_complete.pt", model, epoch, metrics, config)
        print(json.dumps(row), flush=True)
        if (run_dir / "STOP_AFTER_CURRENT_EPOCH").exists():
            print(json.dumps({"graceful_stop_after_epoch": epoch}), flush=True)
            break

    completed_epochs = int(history[-1]["epoch"])
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(run_dir / "epoch_summary.csv", index=False, encoding="utf-8-sig")
    plot_history(history_frame, run_dir)

    checkpoint = torch.load(run_dir / "best_macro_f1.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    target_output = collect_predictions(model, val_loader, criterion, device, amp_enabled)
    target_lookup = {str(sample["sample_id"]): str(sample["user_id"]) for sample in getattr(val_dataset, "samples")}
    _save_archive(run_dir / "val_predictions_target16_best_macro_f1.npz", target_output, target_lookup, target_ids)

    full_val_dataset = _full_validation_dataset(config)
    if len(full_val_dataset) != 590 or getattr(full_val_dataset, "original_class_ids") != list(range(40)):
        raise ValueError("Full validation dataset integrity check failed.")
    full_loader = loader_for(full_val_dataset, {**config, "smoke_test": False}, training=False)
    full_output = _collect_full_predictions(model, full_loader, device, amp_enabled)
    full_lookup = {str(sample["sample_id"]): str(sample["user_id"]) for sample in getattr(full_val_dataset, "samples")}
    _save_archive(run_dir / "val_predictions_all40_best_macro_f1.npz", full_output, full_lookup, target_ids)
    summary = {
        "status": "passed",
        "test_read": False,
        "train_samples": len(train_dataset),
        "target_val_samples": len(val_dataset),
        "full_val_samples": len(full_val_dataset),
        "epochs_completed": completed_epochs,
        "best_macro_epoch": int(checkpoint["epoch"]),
        "best_macro_f1": float(checkpoint["val_macro_f1"]),
        "best_macro_accuracy": float(checkpoint["val_accuracy"]),
        "runtime_seconds": time.perf_counter() - started,
        "target_class_ids": target_ids.tolist(),
        "parameter_audit": parameter_audit,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run(parse_args())
