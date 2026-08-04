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
from torch.utils.data import DataLoader, Subset

from src.data.common import load_modality_frames
from src.data.object_interaction_roi_dataset import ObjectInteractionROIDataset
from src.data.pose_roi_dataset import PoseROIDataset
from src.models.depth_ir_pose_roi_expert import DepthIRPoseROIExpert
from src.models.object_interaction_tcn_expert import ObjectInteractionTCNExpert
from src.train_unimodal import PROJECT_ROOT, seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_object_interaction_tcn_expert.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-id")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in (
        "manifest", "fold", "pairing_audit", "class_map", "pose_cache", "base_checkpoint",
        "base_predictions", "base_logits_cache", "output_root",
    ):
        path = Path(config[key])
        config[key] = str(path if path.is_absolute() else PROJECT_ROOT / path)
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.run_id:
        config["run_id"] = args.run_id
    config["smoke_test"] = bool(args.smoke_test)
    config["probe"] = bool(args.probe)
    config["max_train_batches"] = args.max_train_batches
    config["max_val_batches"] = args.max_val_batches
    return config


def filtered_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = load_modality_frames(
        Path(config["manifest"]), Path(config["fold"]), Path(config["data_root"]), "depth_color_path"
    )
    audit = pd.read_csv(config["pairing_audit"], encoding="utf-8-sig")
    valid_ids = set(audit.loc[audit["complete_pairing"].astype(bool), "sample_id"].astype(str))
    train = train[train["sample_id"].isin(valid_ids)].reset_index(drop=True)
    val = val[val["sample_id"].isin(valid_ids)].reset_index(drop=True)
    return train, val


def make_loader(
    dataset: Any, config: dict[str, Any], training: bool, smoke_limit: int | None = None,
) -> DataLoader[dict[str, object]]:
    if smoke_limit:
        indices = np.linspace(0, len(dataset) - 1, min(len(dataset), smoke_limit)).round().astype(int).tolist()
        actual = Subset(dataset, indices)
    else:
        actual = dataset
    workers = int(config["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]) + (0 if training else 1))
    kwargs: dict[str, Any] = {
        "dataset": actual, "batch_size": min(int(config["batch_size"]), len(actual)), "shuffle": training,
        "num_workers": workers, "pin_memory": True, "generator": generator, "worker_init_fn": seed_worker,
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    return DataLoader(**kwargs)


def base_loader(dataset: PoseROIDataset, config: dict[str, Any]) -> DataLoader[dict[str, object]]:
    workers = int(config["num_workers"])
    kwargs: dict[str, Any] = {
        "dataset": dataset, "batch_size": 4, "shuffle": False, "num_workers": workers,
        "pin_memory": True, "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(int(config["seed"]) + 11),
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    return DataLoader(**kwargs)


def build_base_logits_cache(
    config: dict[str, Any], train_frame: pd.DataFrame, val_frame: pd.DataFrame, device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = Path(config["base_logits_cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        selected = cached["split"].astype(str) == "validation"
        saved = np.load(config["base_predictions"], allow_pickle=False)
        saved_predictions = saved["predictions"] if "predictions" in saved else saved["predicted"]
        reproduced = (
            np.array_equal(cached["sample_ids"][selected].astype(str), saved["sample_ids"].astype(str))
            and np.array_equal(cached["labels"][selected], saved["labels"])
            and np.array_equal(cached["logits"][selected].argmax(axis=1), saved_predictions)
        )
        metrics = metric_bundle(cached["labels"][selected], cached["logits"][selected], np.arange(16))
        verification = {
            "strict_prediction_reproduction": bool(reproduced),
            "checkpoint_epoch": int(cached["base_checkpoint_epoch"]),
            "validation_samples": int(selected.sum()), "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
            "zero_f1_count": metrics["zero_f1_count"], "source": "verified deterministic cache",
        }
        if not reproduced:
            raise RuntimeError(f"Cached Epoch 8 predictions did not reproduce: {verification}")
        return (
            {str(sample_id): cached["logits"][index] for index, sample_id in enumerate(cached["sample_ids"])},
            verification,
        )
    actions = train_frame[["class_id", "action_name"]].drop_duplicates().sort_values("class_id")["action_name"].tolist()
    common = dict(
        hard_actions=actions, num_frames=int(config["base_num_frames"]), image_size=int(config["image_size"]),
        use_pose_roi=True, pose_cache_path=Path(config["pose_cache"]), use_ir_input=True,
        data_root=Path(config["data_root"]), training=False,
    )
    datasets = (PoseROIDataset(train_frame, **common), PoseROIDataset(val_frame, **common))
    checkpoint = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=True)
    model = DepthIRPoseROIExpert(num_classes=40, expected_views=4, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    sample_ids: list[str] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    splits: list[str] = []
    with torch.inference_mode():
        for split, dataset in zip(("train", "validation"), datasets, strict=True):
            for batch in base_loader(dataset, config):
                inputs = {
                    "depth_input": batch["depth_input"].to(device, non_blocking=True),
                    "ir_input": batch["ir_input"].to(device, non_blocking=True),
                }
                mask = batch["temporal_mask"].to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=device.type == "cuda" and bool(config["amp"])):
                    output = model(inputs, temporal_mask=mask)
                count = len(batch["sample_id"])
                sample_ids.extend(str(value) for value in batch["sample_id"])
                labels.append(batch["label"].numpy())
                logits.append(output["logits"].float().cpu().numpy())
                splits.extend([split] * count)
    all_labels = np.concatenate(labels)
    all_logits = np.concatenate(logits)
    np.savez_compressed(
        cache_path, sample_ids=np.asarray(sample_ids), labels=all_labels, logits=all_logits,
        split=np.asarray(splits), base_checkpoint_epoch=int(checkpoint["epoch"]),
    )
    val_selected = np.asarray(splits) == "validation"
    saved = np.load(config["base_predictions"], allow_pickle=False)
    current_ids = np.asarray(sample_ids)[val_selected]
    current_predictions = all_logits[val_selected].argmax(axis=1)
    reproduced = (
        np.array_equal(current_ids.astype(str), saved["sample_ids"].astype(str))
        and np.array_equal(all_labels[val_selected], saved["labels"])
        and np.array_equal(current_predictions, saved["predictions"] if "predictions" in saved else saved["predicted"])
    )
    metrics = metric_bundle(all_labels[val_selected], all_logits[val_selected], np.arange(16))
    verification = {
        "strict_prediction_reproduction": bool(reproduced), "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_samples": int(val_selected.sum()), "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
        "zero_f1_count": metrics["zero_f1_count"],
    }
    if not reproduced:
        raise RuntimeError(f"Epoch 8 predictions did not reproduce: {verification}")
    return {sample_id: all_logits[index] for index, sample_id in enumerate(sample_ids)}, verification


def load_base_logits(config: dict[str, Any]) -> dict[str, np.ndarray]:
    data = np.load(config["base_logits_cache"], allow_pickle=False)
    return {str(sample_id): data["logits"][index] for index, sample_id in enumerate(data["sample_ids"])}


def roi_config(config: dict[str, Any]) -> dict[str, float]:
    return {
        "keypoint_threshold": float(config["pose_keypoint_confidence_threshold"]),
        "projection_alpha": float(config["projection_alpha"]), "long_axis_scale": float(config["long_axis_scale"]),
        "short_axis_scale": float(config["short_axis_scale"]),
        "wrist_fallback_person_width": float(config["wrist_fallback_person_width"]),
        "minimum_pixel_side": float(config["minimum_pixel_side"]),
        "merge_iou_threshold": float(config["merge_iou_threshold"]),
        "merge_projected_distance": float(config["merge_projected_distance"]),
        "merge_wrist_distance": float(config["merge_wrist_distance"]),
        "merge_elbow_distance": float(config["merge_elbow_distance"]),
        "merge_wide_wrist_distance": float(config["merge_wide_wrist_distance"]),
        "two_hand_horizontal_padding": float(config["two_hand_horizontal_padding"]),
        "two_hand_top_padding": float(config["two_hand_top_padding"]),
        "two_hand_bottom_padding": float(config["two_hand_bottom_padding"]),
        "hand_head_distance": float(config["hand_head_distance"]),
    }


def target_ids(config: dict[str, Any], class_map: pd.DataFrame) -> list[int]:
    lookup = dict(zip(class_map["action_name"], class_map["class_id"], strict=True))
    missing = set(config["target_actions"]) - set(lookup)
    if missing:
        raise ValueError(f"Target actions missing from class map: {sorted(missing)}")
    return [int(lookup[name]) for name in config["target_actions"]]


def target_weights(train_frame: pd.DataFrame, ids: list[int]) -> np.ndarray:
    counts = train_frame["class_id"].value_counts().reindex(ids).to_numpy(dtype=np.float64)
    weights = np.sqrt(np.median(counts) / counts)
    weights = weights / weights.mean()
    return np.clip(weights, 0.5, 3.0).astype(np.float32)


def metric_bundle(labels: np.ndarray, logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(40), zero_division=0,
    )
    weighted = float(np.average(f1, weights=support))
    ranks = 1 + (logits > logits[np.arange(len(labels)), labels, None]).sum(axis=1)
    probabilities = torch.from_numpy(logits).softmax(dim=1).numpy()
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, 16)[:-1], np.linspace(0.0, 1.0, 16)[1:]):
        selected = (confidences > lower) & (confidences <= upper)
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidences[selected].mean()))
    return {
        "accuracy": float((predictions == labels).mean()), "macro_f1": float(f1.mean()),
        "weighted_f1": weighted, "target16_macro_f1": float(f1[targets].mean()),
        "zero_f1_count": int((f1 == 0).sum()), "target16_zero_f1_count": int((f1[targets] == 0).sum()),
        "per_class_precision": precision, "per_class_recall": recall, "per_class_f1": f1,
        "support": support, "predictions": predictions, "true_rank": ranks,
        "expected_calibration_error": ece,
    }


def group_f1(metrics: dict[str, Any], names: list[str], lookup: dict[str, int]) -> float:
    return float(np.asarray(metrics["per_class_f1"])[[lookup[name] for name in names]].mean())


def compute_loss(
    output: dict[str, torch.Tensor], labels: torch.Tensor, target_index: torch.Tensor,
    local_map: torch.Tensor, weights: torch.Tensor, config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    main = nn.functional.cross_entropy(output["logits"], labels)
    local_labels = local_map[labels]
    selected = local_labels >= 0
    target = (
        nn.functional.cross_entropy(output["delta_logits_target"][selected], local_labels[selected], weight=weights)
        if selected.any() else main.new_zeros(())
    )
    non_target = ~selected
    false_activation = (
        torch.relu(output["delta_logits_target"][non_target]).square().mean()
        if non_target.any() else main.new_zeros(())
    )
    residual = output["delta_logits_target"].square().mean()
    total = (
        main + float(config["loss_target_coefficient"]) * target
        + float(config["loss_non_target_coefficient"]) * false_activation
        + float(config["loss_residual_coefficient"]) * residual
    )
    return total, {"main": main, "target": target, "non_target": false_activation, "residual": residual}


def to_device(batch: dict[str, object], device: torch.device) -> tuple[dict[str, torch.Tensor], ...]:
    inputs = {
        "depth_input": batch["depth_input"].to(device, non_blocking=True),
        "ir_input": batch["ir_input"].to(device, non_blocking=True),
    }
    return (
        inputs, batch["view_valid_mask"].to(device, non_blocking=True),
        batch["temporal_mask"].to(device, non_blocking=True),
        batch["base_logits"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def train_epoch(
    model: ObjectInteractionTCNExpert, loader: DataLoader[dict[str, object]], optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, device: torch.device, config: dict[str, Any], local_map: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    model.train()
    model.enforce_frozen_encoder_eval()
    optimizer.zero_grad(set_to_none=True)
    totals = {name: 0.0 for name in ("loss", "main", "target", "non_target", "residual")}
    samples = 0
    accumulation = int(config["gradient_accumulation_steps"])
    limit = config.get("max_train_batches")
    for step, batch in enumerate(loader, start=1):
        inputs, views, temporal, base, labels = to_device(batch, device)
        with torch.autocast(device.type, enabled=device.type == "cuda" and bool(config["amp"])):
            output = model(inputs, views, temporal, base)
            loss, pieces = compute_loss(output, labels, model.target_index, local_map, weights, config)
        scaler.scale(loss / accumulation).backward()
        if step % accumulation == 0 or step == len(loader) or (limit and step == int(limit)):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        count = len(labels)
        totals["loss"] += float(loss.detach()) * count
        for name, value in pieces.items():
            totals[name] += float(value.detach()) * count
        samples += count
        if limit and step >= int(limit):
            break
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: ObjectInteractionTCNExpert, loader: DataLoader[dict[str, object]], device: torch.device,
    config: dict[str, Any], targets: np.ndarray,
) -> dict[str, Any]:
    validation_started = time.perf_counter()
    model.eval()
    collected: dict[str, list[Any]] = {name: [] for name in (
        "sample_ids", "labels", "logits", "base_logits", "embeddings", "delta", "view_weights",
        "view_entropy", "temporal_attention", "modality_gate", "activation_norms", "temporal_mask",
        "view_valid", "sampled_indices", "two_hand", "hand_head", "patterns",
    )}
    loss_total = 0.0
    count_total = 0
    limit = config.get("max_val_batches")
    for step, batch in enumerate(loader, start=1):
        inputs, views, temporal, base, labels = to_device(batch, device)
        with torch.autocast(device.type, enabled=device.type == "cuda" and bool(config["amp"])):
            output = model(inputs, views, temporal, base)
            loss = nn.functional.cross_entropy(output["logits"], labels)
        count = len(labels)
        loss_total += float(loss) * count
        count_total += count
        collected["sample_ids"].extend(str(value) for value in batch["sample_id"])
        collected["patterns"].extend(str(value) for value in batch["valid_keypoint_pattern"])
        for name, value in (
            ("labels", labels), ("logits", output["logits"]), ("base_logits", output["base_logits"]),
            ("embeddings", output["embedding"]), ("delta", output["delta_logits_target"]),
            ("view_weights", output["view_weights"]), ("view_entropy", output["view_gate_entropy"]),
            ("temporal_attention", output["temporal_attention"]), ("modality_gate", output["modality_gate"]),
            ("activation_norms", output["tcn_activation_norms"]), ("temporal_mask", temporal),
            ("view_valid", views),
            ("sampled_indices", batch["sampled_indices"]), ("two_hand", batch["two_hand_roi_valid"]),
            ("hand_head", batch["hand_head_roi_valid"]),
        ):
            collected[name].append(value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
        if limit and step >= int(limit):
            break
    arrays = {
        name: np.concatenate(values) for name, values in collected.items()
        if name not in {"sample_ids", "patterns"}
    }
    metrics = metric_bundle(arrays["labels"].astype(np.int64), arrays["logits"], targets)
    base_predictions = arrays["base_logits"].argmax(axis=1)
    final_predictions = arrays["logits"].argmax(axis=1)
    correct_base = base_predictions == arrays["labels"]
    correct_final = final_predictions == arrays["labels"]
    metrics["rescued_count"] = int((~correct_base & correct_final).sum())
    metrics["harmed_count"] = int((correct_base & ~correct_final).sum())
    metrics["net_rescue"] = metrics["rescued_count"] - metrics["harmed_count"]
    return {
        **arrays, "sample_ids": np.asarray(collected["sample_ids"]), "patterns": np.asarray(collected["patterns"]),
        "metrics": metrics, "val_loss": loss_total / max(count_total, 1),
        "validation_seconds": time.perf_counter() - validation_started,
    }


def serial_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }


def save_checkpoint(
    path: Path, model: ObjectInteractionTCNExpert, epoch: int, stage: str, evaluation: dict[str, Any],
    config: dict[str, Any], weights: np.ndarray,
) -> None:
    torch.save(
        {
            "epoch": epoch, "stage": stage, "model_state_dict": model.state_dict(),
            "metrics": serial_metrics(evaluation["metrics"]), "val_loss": evaluation["val_loss"],
            "validation_seconds": evaluation.get("validation_seconds"),
            "target_actions": config["target_actions"], "target_class_ids": list(model.target_class_ids),
            "target_class_weights": weights.tolist(), "config": config,
        }, path,
    )


def probe_gradients(
    model: ObjectInteractionTCNExpert, loader: DataLoader[dict[str, object]], device: torch.device,
    config: dict[str, Any], local_map: torch.Tensor, weights: torch.Tensor,
) -> dict[str, Any]:
    model.set_stage("warmup")
    model.train()
    model.enforce_frozen_encoder_eval()
    optimizer = torch.optim.AdamW(model.parameter_groups(0.0, float(config["warmup_learning_rate"])), weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(config["amp"]),
        init_scale=float(config.get("amp_initial_scale", 1024.0)),
    )
    batch = None
    for candidate in loader:
        candidate_labels = candidate["label"].to(device)
        if (local_map[candidate_labels] >= 0).any():
            batch = candidate
            break
    if batch is None:
        raise RuntimeError("Gradient probe could not find a Target16 training sample")
    inputs, views, temporal, base, labels = to_device(batch, device)
    architecture_probe = model.architecture_probe(inputs) if hasattr(model, "architecture_probe") else None
    model.set_stage("warmup")
    model.train()
    model.enforce_frozen_encoder_eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device.type, enabled=device.type == "cuda" and bool(config["amp"])):
        initial = model(inputs, views, temporal, base)
        initial_difference = float((initial["logits"] - base).abs().max())
        loss, _ = compute_loss(initial, labels, model.target_index, local_map, weights, config)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if hasattr(model, "layer4"):
        model.set_stage("finetune")
        model.train()
        model.enforce_frozen_encoder_eval()
    with torch.autocast(device.type, enabled=device.type == "cuda" and bool(config["amp"])):
        second = model(inputs, views, temporal, base)
        second_loss, _ = compute_loss(second, labels, model.target_index, local_map, weights, config)
    scaler.scale(second_loss).backward()
    modules = {
        "view_gate": model.view_gate, "tcn": model.tcn_blocks, "attention_pool": model.attention_pool,
        "residual_head": model.residual_head,
    }
    if hasattr(model, "layer4"):
        modules["resnet_layer4"] = model.layer4
        modules["modality_gate"] = model.modality_gate
    gradient_norms = {
        name: float(sum((parameter.grad.float().norm() for parameter in module.parameters() if parameter.grad is not None), torch.tensor(0.0, device=device)))
        for name, module in modules.items()
    }
    invalid_weight = second["view_weights"].masked_select(~views).abs().max().item() if (~views).any() else 0.0
    attention_padding = second["temporal_attention"].masked_select(~temporal).abs().max().item() if (~temporal).any() else 0.0
    result = {
        "initial_max_abs_final_minus_base": initial_difference,
        "gradient_norms_after_head_update": gradient_norms,
        "invalid_view_max_weight": float(invalid_weight),
        "padding_attention_max_weight": float(attention_padding),
        "attention_sum_min": float(second["temporal_attention"].sum(dim=1).min()),
        "attention_sum_max": float(second["temporal_attention"].sum(dim=1).max()),
        "tcn_receptive_field": model.receptive_field,
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
    }
    if architecture_probe is not None:
        result["architecture_probe"] = architecture_probe
    if initial_difference >= 1e-6 or invalid_weight != 0 or attention_padding != 0 or any(value <= 0 for value in gradient_norms.values()):
        raise RuntimeError(f"Probe failed: {result}")
    return result


def run(
    args: argparse.Namespace,
    model_factory: Any | None = None,
) -> None:
    config = resolve_config(args)
    set_seed(int(config["seed"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    train_frame, val_frame = filtered_frames(config)
    class_map = pd.read_csv(config["class_map"], encoding="utf-8-sig").sort_values("class_id")
    if class_map["class_id"].tolist() != list(range(40)):
        raise ValueError("Class map is not 0-39")
    targets = target_ids(config, class_map)
    weights_np = target_weights(train_frame, targets)
    base_logits, reproduction = build_base_logits_cache(config, train_frame, val_frame, device)
    train_dataset = ObjectInteractionROIDataset(
        train_frame, Path(config["data_root"]), Path(config["pose_cache"]), int(config["expert_num_frames"]),
        int(config["image_size"]), True, roi_config(config), base_logits,
    )
    val_dataset = ObjectInteractionROIDataset(
        val_frame, Path(config["data_root"]), Path(config["pose_cache"]), int(config["expert_num_frames"]),
        int(config["image_size"]), False, roi_config(config), base_logits,
    )
    run_id = str(config["run_id"])
    if args.smoke_test:
        run_id += "_smoke"
    if args.probe:
        run_id += "_probe"
    run_dir = Path(config["output_root"]) / run_id
    if run_dir.exists():
        raise FileExistsError(f"Output exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "base_reproduction.json").write_text(json.dumps(reproduction, indent=2) + "\n", encoding="utf-8")
    train_dataset.roi_audit_rows().to_csv(run_dir / "roi_audit_train_samples.csv", index=False, encoding="utf-8-sig")
    val_dataset.roi_audit_rows().to_csv(run_dir / "roi_audit_val_samples.csv", index=False, encoding="utf-8-sig")
    val_dataset.temporal_diagnostic_rows().to_csv(run_dir / "temporal_diagnostics_base.csv", index=False, encoding="utf-8-sig")
    def build_model() -> ObjectInteractionTCNExpert:
        if model_factory is not None:
            return model_factory(config, targets)
        return ObjectInteractionTCNExpert(
            targets, frame_feature_dim=int(config["frame_feature_dim"]), tcn_channels=int(config["tcn_channels"]),
            embedding_dim=int(config["expert_embedding_dim"]), kernel_size=int(config["tcn_kernel_size"]),
            dilations=tuple(int(value) for value in config["tcn_dilations"]), dropout=float(config["dropout"]),
        )

    model = build_model()
    checkpoint = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=True)
    model.initialize_encoder(checkpoint["model_state_dict"])
    model.to(device)
    local_map = torch.full((40,), -1, dtype=torch.long, device=device)
    local_map[torch.tensor(targets, device=device)] = torch.arange(len(targets), device=device)
    weights = torch.from_numpy(weights_np).to(device)
    smoke_train = 8 if args.smoke_test or args.probe else None
    smoke_val = 8 if args.smoke_test or args.probe else None
    train_loader = make_loader(train_dataset, config, True, smoke_train)
    val_loader = make_loader(val_dataset, config, False, smoke_val)
    probe = probe_gradients(model, train_loader, device, config, local_map, weights)
    (run_dir / "probe.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    if args.probe:
        print(f"RESULT_JSON={json.dumps({'status': 'passed', 'probe': probe, 'reproduction': reproduction})}")
        return
    # Restore exact zero-residual initialization after the destructive gradient probe.
    model = build_model()
    model.initialize_encoder(checkpoint["model_state_dict"])
    model.to(device)
    lookup = dict(zip(class_map["action_name"], class_map["class_id"], strict=True))
    history: list[dict[str, Any]] = []
    best_target = (-1.0, -1.0, -1.0)
    best_overall = (-1.0, -1.0)
    best_accuracy = (-1.0, -1.0)
    no_improvement = 0
    total_started = time.perf_counter()
    global_epoch = 0
    stages = (("warmup", 1 if args.smoke_test else int(config["warmup_epochs"])),
              ("finetune", 0 if args.smoke_test else int(config["finetune_max_epochs"])))
    for stage, stage_epochs in stages:
        if stage_epochs == 0:
            continue
        model.set_stage(stage)
        groups = model.parameter_groups(
            float(config["expert_last_backbone_learning_rate"]),
            float(config["warmup_learning_rate"] if stage == "warmup" else config["expert_new_modules_learning_rate"]),
        )
        optimizer = torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and bool(config["amp"]),
            init_scale=float(config.get("amp_initial_scale", 1024.0)),
        )
        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            epoch_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, config, local_map, weights)
            validation = evaluate(model, val_loader, device, config, np.asarray(targets))
            metrics = validation["metrics"]
            row = {
                "epoch": global_epoch, "stage": stage, "stage_epoch": stage_epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                "val_loss": validation["val_loss"], "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"], "target16_macro_f1": metrics["target16_macro_f1"],
                "target16_zero_f1_count": metrics["target16_zero_f1_count"], "zero_f1_count": metrics["zero_f1_count"],
                "hand_head7_macro_f1": group_f1(metrics, config["hand_head_actions"], lookup),
                "table7_macro_f1": group_f1(metrics, config["table_actions"], lookup),
                "screen2_macro_f1": group_f1(metrics, config["screen_actions"], lookup),
                "control10_macro_f1": group_f1(metrics, config["control_actions"], lookup),
                "other14_macro_f1": float(np.asarray(metrics["per_class_f1"])[[i for i in range(40) if i not in targets and i not in [lookup[n] for n in config["control_actions"]]]].mean()),
                "expected_calibration_error": metrics["expected_calibration_error"],
                "validation_seconds": validation["validation_seconds"],
                "epoch_seconds": time.perf_counter() - epoch_started,
                "gpu_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
                "gpu_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
            save_checkpoint(run_dir / "last_complete.pt", model, global_epoch, stage, validation, config, weights_np)
            target_key = (metrics["target16_macro_f1"], metrics["macro_f1"], metrics["accuracy"])
            if target_key > best_target:
                best_target = target_key
                no_improvement = 0
                save_checkpoint(run_dir / "best_target16_macro_f1.pt", model, global_epoch, stage, validation, config, weights_np)
            else:
                no_improvement += 1
            overall_key = (metrics["macro_f1"], metrics["accuracy"])
            if overall_key > best_overall:
                best_overall = overall_key
                save_checkpoint(run_dir / "best_overall_macro_f1.pt", model, global_epoch, stage, validation, config, weights_np)
            accuracy_key = (metrics["accuracy"], metrics["macro_f1"])
            if accuracy_key > best_accuracy:
                best_accuracy = accuracy_key
                save_checkpoint(run_dir / "best_accuracy.pt", model, global_epoch, stage, validation, config, weights_np)
            print(json.dumps(row), flush=True)
            if (
                stage == "finetune" and stage_epoch >= int(config["finetune_min_epochs"])
                and no_improvement >= int(config["early_stopping_patience"])
            ):
                break
    best = torch.load(run_dir / "best_target16_macro_f1.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    final = evaluate(model, val_loader, device, config, np.asarray(targets))
    np.savez_compressed(
        run_dir / "val_predictions_best_target16.npz",
        sample_ids=final["sample_ids"], labels=final["labels"].astype(np.int64), logits=final["logits"],
        base_logits=final["base_logits"], predictions=final["metrics"]["predictions"],
        embeddings=final["embeddings"], delta_logits_target=final["delta"], view_weights=final["view_weights"],
        view_gate_entropy=final["view_entropy"], temporal_attention=final["temporal_attention"],
        modality_gate=final["modality_gate"], tcn_activation_norms=final["activation_norms"],
        temporal_mask=final["temporal_mask"].astype(bool), view_valid_mask=final["view_valid"].astype(bool),
        sampled_indices=final["sampled_indices"].astype(np.int64),
        two_hand_roi_valid=final["two_hand"].astype(bool), hand_head_roi_valid=final["hand_head"].astype(bool),
        valid_keypoint_pattern=final["patterns"], target_class_ids=np.asarray(targets),
    )
    total_seconds = time.perf_counter() - total_started
    summary = {
        "status": "passed", "smoke_test": bool(args.smoke_test), "train_samples": len(train_dataset),
        "val_samples": len(val_dataset), "target_train_samples": int(train_frame["class_id"].isin(targets).sum()),
        "target_val_samples": int(val_frame["class_id"].isin(targets).sum()), "epochs_completed": global_epoch,
        "best_target16_epoch": int(best["epoch"]), "base_reproduction": reproduction,
        "final_metrics": serial_metrics(final["metrics"]), "target_class_ids": targets,
        "target_class_weights": weights_np.tolist(), "probe": probe, "training_seconds": total_seconds,
        "expert_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "base_parameter_bytes": sum(value.numel() * value.element_size() for value in checkpoint["model_state_dict"].values()),
        "total_inference_parameter_bytes": (
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
            + sum(value.numel() * value.element_size() for value in checkpoint["model_state_dict"].values())
        ),
        "checkpoint_bytes": (run_dir / "best_target16_macro_f1.pt").stat().st_size,
        "peak_allocated_mb": max(row["gpu_peak_allocated_mb"] for row in history),
        "peak_reserved_mb": max(row["gpu_peak_reserved_mb"] for row in history),
    }
    if hasattr(model, "experiment_metadata"):
        summary["expert_metadata"] = model.experiment_metadata()
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT_JSON={json.dumps(summary)}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
