from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel, temporal_shift
from imu_data import read_index
from p27_strong_event_model import P27StrongEventModule, parameter_count
from probe_p27r3_incremental_information import metric_bundle


PROJECT_DIR = Path(__file__).resolve().parent
PASSED_EVENT_INDICES = np.asarray([2, 5, 6, 7, 8, 9, 11, 12, 13], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a strong fold-pure temporal event module on frozen "
            "IR/Skeleton sequences plus raw device-structured IMU."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--variant", choices=("ce", "event"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--imu-cache", type=Path, default=PROJECT_DIR / "cache" / "imu_32"
    )
    parser.add_argument(
        "--event-cache",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p27_r2_event_audit"
        / "event_cache_v2.npz",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--event-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=27083)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_base(checkpoint: dict, device: torch.device) -> AlignedMultimodalModel:
    config = checkpoint["config"]
    model = AlignedMultimodalModel(
        list(config["modalities"]),
        dropout=float(config["dropout"]),
        use_aux_heads=bool(config.get("use_aux_heads", False)),
        use_modality_masks=bool(config.get("use_modality_masks", False)),
        use_cross_attention=bool(config.get("use_cross_attention", False)),
        depth_input_channels=int(config.get("depth_input_channels", 3)),
        use_layer3_spatial=bool(config.get("use_layer3_spatial", False)),
        imagenet_pretrained=False,
        ir_stem_initialization=str(config.get("ir_stem_initialization", "mean")),
        visual_stem_fusion=str(config.get("visual_stem_fusion", "concat")),
        skeleton_input_dim=int(config.get("skeleton_input_dim", 4)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def base_sequences(
    model: AlignedMultimodalModel,
    ir: torch.Tensor,
    skeleton_input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert model.visual is not None and model.skeleton is not None
    visual = model.visual
    batch_size, time_steps = ir.shape[:2]
    x = visual.ir_stem(ir.reshape(batch_size * time_steps, 1, *ir.shape[-2:]))
    x = visual.maxpool(visual.relu(visual.bn1(x)))
    x = visual.layer1(temporal_shift(x, batch_size, time_steps))
    x = visual.layer2(temporal_shift(x, batch_size, time_steps))
    x = visual.layer3(temporal_shift(x, batch_size, time_steps))
    x = visual.layer4(temporal_shift(x, batch_size, time_steps))
    visual_sequence = x.mean(dim=(-2, -1)).reshape(batch_size, time_steps, 512)
    visual_sequence = visual.temporal(
        visual_sequence.transpose(1, 2)
    ).transpose(1, 2)
    skeleton_sequence = model.skeleton.encode_sequence(skeleton_input)
    visual_pooled = visual.pool(visual_sequence)
    skeleton_pooled = model.skeleton.pool(skeleton_sequence)
    visual_projected = model.visual_project(visual_pooled)
    skeleton_projected = model.skeleton_project(skeleton_pooled)
    gate = torch.sigmoid(
        model.gate(torch.cat([visual_projected, skeleton_projected], dim=1))
    )
    fused = gate * visual_projected + (1.0 - gate) * skeleton_projected
    logits = model.classifier(
        torch.cat([visual_projected, skeleton_projected, fused], dim=1)
    )
    return visual_sequence, skeleton_sequence, logits


def fit_imu_stats(
    imu: np.ndarray,
    time_mask: np.ndarray,
    train_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = imu[:train_count]
    mask = time_mask[:train_count].astype(bool)
    valid = values[mask]
    if not len(valid):
        return np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32)
    return valid.mean(axis=0).astype(np.float32), np.maximum(
        valid.std(axis=0), 1e-4
    ).astype(np.float32)


def extract_or_load(
    args: argparse.Namespace,
    checkpoint: dict,
    base: AlignedMultimodalModel,
    device: torch.device,
) -> dict[str, np.ndarray]:
    output = args.output_dir.resolve()
    cache_path = output.parent / f"fold_{args.fold}_frozen_sequences.npz"
    if cache_path.is_file():
        archive = np.load(cache_path, allow_pickle=False)
        return {key: archive[key] for key in archive.files}
    config = checkpoint["config"]
    common = {
        "manifest_path": args.manifest.resolve(),
        "modalities": ["ir", "skeleton"],
        "num_frames": int(config["num_frames"]),
        "image_height": int(config["image_height"]),
        "image_width": int(config["image_width"]),
        "augment": False,
        "cache_dir": config.get("cache_dir"),
        "skeleton_strategy": config.get("skeleton_strategy", "first"),
        "visual_normalization": config.get("visual_normalization", "legacy"),
        "skeleton_representation": config.get(
            "skeleton_representation", "frame_joint"
        ),
        "skeleton_raw_cache_dir": config.get("skeleton_raw_cache_dir"),
    }
    imu_rows = {
        row.sample_id: row
        for row in read_index(args.imu_cache.resolve() / "index.csv")
        if row.split == "train" and row.usable
    }
    imu_values = np.load(args.imu_cache.resolve() / "imu_float32.npy", mmap_mode="r")
    imu_time = np.load(
        args.imu_cache.resolve() / "time_mask_uint8.npy", mmap_mode="r"
    )
    imu_device = np.load(
        args.imu_cache.resolve() / "device_mask_uint8.npy", mmap_mode="r"
    )
    events = np.load(args.event_cache.resolve(), allow_pickle=False)
    event_location = {
        str(sample_id): index
        for index, sample_id in enumerate(events["sample_ids"])
    }
    outputs: dict[str, list[np.ndarray] | list[str]] = {
        key: []
        for key in (
            "sample_ids",
            "subjects",
            "labels",
            "visual_sequence",
            "skeleton_sequence",
            "base_logits",
            "imu",
            "imu_time_mask",
            "imu_device_mask",
            "event_targets",
            "event_quality",
        )
    }
    split_indicator: list[int] = []
    for split_value, split in enumerate(("train", "val")):
        dataset = AlignedMultimodalDataset(split=split, **common)
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for batch in loader:
            ir = batch["ir"].to(device, non_blocking=True)
            skeleton = batch["skeleton"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                visual_sequence, skeleton_sequence, base_logits = base_sequences(
                    base, ir, skeleton
                )
            ids = [str(value) for value in batch["sample_id"]]
            outputs["sample_ids"].extend(ids)
            sample_lookup = {
                sample.sample_id: sample for sample in dataset.samples
            }
            outputs["subjects"].extend(sample_lookup[value].user_id for value in ids)
            outputs["labels"].append(batch["label"].numpy())
            outputs["visual_sequence"].append(
                visual_sequence.float().cpu().numpy()
            )
            outputs["skeleton_sequence"].append(
                skeleton_sequence.float().cpu().numpy()
            )
            outputs["base_logits"].append(base_logits.float().cpu().numpy())
            raw_imu = np.zeros((len(ids), 5, 32, 10), dtype=np.float32)
            raw_time = np.zeros((len(ids), 5, 32), dtype=np.float32)
            raw_device = np.zeros((len(ids), 5), dtype=np.float32)
            target = np.zeros((len(ids), len(PASSED_EVENT_INDICES)), dtype=np.float32)
            quality = np.zeros_like(target)
            for index, sample_id in enumerate(ids):
                imu_row = imu_rows.get(sample_id)
                if imu_row is not None:
                    raw_imu[index] = imu_values[imu_row.cache_index]
                    raw_time[index] = imu_time[imu_row.cache_index]
                    raw_device[index] = imu_device[imu_row.cache_index]
                event_index = event_location[sample_id]
                target[index] = events["event_targets"][
                    event_index, PASSED_EVENT_INDICES
                ]
                quality[index] = events["event_quality"][
                    event_index, PASSED_EVENT_INDICES
                ]
            outputs["imu"].append(raw_imu)
            outputs["imu_time_mask"].append(raw_time)
            outputs["imu_device_mask"].append(raw_device)
            outputs["event_targets"].append(target)
            outputs["event_quality"].append(quality)
            split_indicator.extend([split_value] * len(ids))
    result = {
        "sample_ids": np.asarray(outputs["sample_ids"]),
        "subjects": np.asarray(outputs["subjects"]),
        "split": np.asarray(split_indicator, dtype=np.int64),
    }
    for key in (
        "labels",
        "visual_sequence",
        "skeleton_sequence",
        "base_logits",
        "imu",
        "imu_time_mask",
        "imu_device_mask",
        "event_targets",
        "event_quality",
    ):
        result[key] = np.concatenate(outputs[key])  # type: ignore[arg-type]
    check_item = AlignedMultimodalDataset(split="val", **common)[0]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        direct = base(
            {
                "ir": check_item["ir"].unsqueeze(0).to(device),
                "skeleton": check_item["skeleton"].unsqueeze(0).to(device),
            }
        )
    first_held = int(np.flatnonzero(result["split"] == 1)[0])
    reconstruction_error = float(
        np.max(
            np.abs(
                direct.float().cpu().numpy()[0]
                - result["base_logits"][first_held]
            )
        )
    )
    result["base_reconstruction_max_abs_error"] = np.asarray(
        reconstruction_error, dtype=np.float32
    )
    result["outer_held_predictions_generated"] = np.asarray(False)
    np.savez_compressed(cache_path, **result)
    return result


def batch_to_device(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in batch)


@torch.no_grad()
def infer(
    model: P27StrongEventModule,
    loader: DataLoader,
    device: torch.device,
    ablation: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    events: list[np.ndarray] = []
    for batch in loader:
        values = batch_to_device(batch, device)
        output = model(
            values[0], values[1], values[2], values[3], values[4],
            ablation=ablation,
        )
        logits.append(output["logits"].float().cpu().numpy())
        events.append(output["event_predictions"].float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(events)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    base = build_base(checkpoint, device)
    data = extract_or_load(args, checkpoint, base, device)
    if bool(data["outer_held_predictions_generated"]):
        raise RuntimeError("outer-held predictions are forbidden")
    train = data["split"] == 0
    held = data["split"] == 1
    train_count = int(train.sum())
    order = np.concatenate([np.flatnonzero(train), np.flatnonzero(held)])
    for key in (
        "visual_sequence",
        "skeleton_sequence",
        "imu",
        "imu_time_mask",
        "imu_device_mask",
        "event_targets",
        "event_quality",
    ):
        data[key] = data[key][order]
    labels = data["labels"][order].astype(np.int64)
    imu_mean, imu_std = fit_imu_stats(
        data["imu"], data["imu_time_mask"], train_count
    )
    tensors = TensorDataset(
        torch.from_numpy(data["visual_sequence"]).float(),
        torch.from_numpy(data["skeleton_sequence"]).float(),
        torch.from_numpy(data["imu"]).float(),
        torch.from_numpy(data["imu_time_mask"]).float(),
        torch.from_numpy(data["imu_device_mask"]).float(),
        torch.from_numpy(data["event_targets"]).float(),
        torch.from_numpy(data["event_quality"]).float(),
        torch.from_numpy(labels).long(),
    )
    counts = np.bincount(labels[:train_count], minlength=40).clip(min=1)
    weights = 1.0 / counts[labels[:train_count]]
    train_loader = DataLoader(
        torch.utils.data.Subset(tensors, range(train_count)),
        batch_size=int(args.batch_size),
        sampler=WeightedRandomSampler(
            torch.from_numpy(weights).double(),
            num_samples=train_count,
            replacement=True,
            generator=torch.Generator().manual_seed(int(args.seed)),
        ),
    )
    held_loader = DataLoader(
        torch.utils.data.Subset(tensors, range(train_count, len(tensors))),
        batch_size=int(args.batch_size),
        shuffle=False,
    )
    model = P27StrongEventModule(
        torch.from_numpy(imu_mean),
        torch.from_numpy(imu_std),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs)
    )
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = ce_sum = event_sum = 0.0
        seen = 0
        for batch in train_loader:
            values = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(values[0], values[1], values[2], values[3], values[4])
            ce = F.cross_entropy(
                output_batch["logits"], values[7], label_smoothing=0.05
            )
            quality = values[6]
            event_loss = (
                (output_batch["event_predictions"] - values[5]).square()
                * quality
            ).sum() / quality.sum().clamp_min(1.0)
            loss = ce
            if args.variant == "event":
                loss = loss + float(args.event_weight) * event_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(values[7])
            seen += count
            loss_sum += float(loss.detach()) * count
            ce_sum += float(ce.detach()) * count
            event_sum += float(event_loss.detach()) * count
        scheduler.step()
        held_logits, held_events = infer(model, held_loader, device)
        held_labels = labels[train_count:]
        accuracy = float(np.mean(held_logits.argmax(1) == held_labels))
        target = data["event_targets"][train_count:]
        quality = data["event_quality"][train_count:]
        event_mae = float(
            (np.abs(held_events - target) * quality).sum()
            / max(float(quality.sum()), 1.0)
        )
        row = {
            "epoch": float(epoch),
            "train_loss": loss_sum / seen,
            "train_ce": ce_sum / seen,
            "train_event": event_sum / seen,
            "held_accuracy_monitor": accuracy,
            "held_event_mae": event_mae,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    held_logits, held_events = infer(model, held_loader, device)
    ablations = {
        name: infer(model, held_loader, device, ablation=name)[0]
        for name in ("imu_zero", "time_reverse", "time_permute", "event_zero", "event_shuffle")
    }
    held_labels = labels[train_count:]
    held_ids = data["sample_ids"][order][train_count:]
    held_subjects = data["subjects"][order][train_count:]
    np.savez_compressed(
        output / "held_logits.npz",
        protocol=np.asarray(f"p27-strong-event-{args.variant}-v1"),
        sample_ids=held_ids,
        labels=held_labels,
        subjects=held_subjects,
        logits=held_logits.astype(np.float32),
        base_logits=data["base_logits"][order][train_count:].astype(np.float32),
        event_predictions=held_events.astype(np.float32),
        event_targets=data["event_targets"][train_count:].astype(np.float32),
        event_quality=data["event_quality"][train_count:].astype(np.float32),
        **{
            f"{name}_logits": values.astype(np.float32)
            for name, values in ablations.items()
        },
        outer_held_predictions_generated=np.asarray(False),
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "config": vars(args),
            "imu_mean": imu_mean,
            "imu_std": imu_std,
            "best_epoch": int(np.argmax([row["held_accuracy_monitor"] for row in history])) + 1,
        },
        output / "best.pt",
    )
    metrics = {
        "event_model": metric_bundle(held_labels, held_logits.argmax(1)),
        "base_reconstructed": metric_bundle(
            held_labels,
            data["base_logits"][order][train_count:].argmax(1),
        ),
    }
    summary = {
        "protocol": f"p27-strong-event-{args.variant}-v1",
        "outer_fold": 0,
        "inner_fold": int(args.fold),
        "outer_held_predictions_generated": False,
        "variant": args.variant,
        "parameters": parameter_count(model),
        "fp16_parameter_mib": parameter_count(model) * 2 / 1024**2,
        "base_reconstruction_max_abs_error": float(
            data["base_reconstruction_max_abs_error"]
        ),
        "best_accuracy": best_accuracy,
        "metrics": metrics,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
