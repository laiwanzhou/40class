"""MotionBERT teacher screening and fine-tuning for the P90 ceiling study.

This deliberately differs from all earlier skeleton experiments: it transfers
the official MotionBERT pretrained/action weights on projected H36M-17 poses and
evaluates only with the common three subject-disjoint OOF folds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from p90_teacher_common import (
    HERE,
    NUM_CLASSES,
    REPO_ROOT,
    classification_metrics,
    load_protocol,
    save_oof_artifact,
    seed_everything,
)


EXTERNAL_ROOT = REPO_ROOT.parent / "external_data" / "MotionBERT"
MOTIONBERT_ROOT = EXTERNAL_ROOT
HF_ROOT = MOTIONBERT_ROOT / "hf_motionbert" / "checkpoint"
RAW_CACHE = HERE / "cache" / "skeleton_raw"
DEFAULT_RUN = REPO_ROOT / "runs" / "p90_motionbert_teacher_v1"
DEFAULT_POSE_CACHE = HERE / "cache" / "p90_motionbert_t81_v1.npz"


@dataclass(frozen=True)
class InitSpec:
    name: str
    checkpoint: Path
    checkpoint_key: str
    prefix: str
    dim_feat: int
    mlp_ratio: float
    dim_rep: int = 512
    depth: int = 5
    heads: int = 8


INIT_SPECS = {
    "action": InitSpec(
        name="action",
        checkpoint=HF_ROOT / "action" / "FT_MB_release_MB_ft_NTU60_xsub" / "best_epoch.bin",
        checkpoint_key="model",
        prefix="module.backbone.",
        dim_feat=512,
        mlp_ratio=2.0,
    ),
    "pretrain": InitSpec(
        name="pretrain",
        checkpoint=HF_ROOT / "pretrain" / "MB_release" / "latest_epoch.bin",
        checkpoint_key="model_pos",
        prefix="module.",
        dim_feat=512,
        mlp_ratio=2.0,
    ),
    "lite": InitSpec(
        name="lite",
        checkpoint=HF_ROOT / "pretrain" / "MB_lite" / "latest_epoch.bin",
        checkpoint_key="model_pos",
        prefix="module.",
        dim_feat=256,
        mlp_ratio=4.0,
    ),
}

VIEW_AXES = {
    # Kinect skeleton cache uses z as the vertical body axis.
    "front": (0, 2),
    "side": (1, 2),
    "top": (0, 1),
}


def interpolate_pose(raw: np.ndarray, target_frames: int) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(raw[..., :3], dtype=np.float32)
    confidence = np.clip(np.nan_to_num(raw[..., 3], nan=0.0), 0.0, 1.0).astype(np.float32)
    source_x = np.linspace(0.0, 1.0, len(raw), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    output = np.zeros((target_frames, 17, 3), dtype=np.float32)
    output_confidence = np.zeros((target_frames, 17), dtype=np.float32)
    for joint in range(17):
        valid = np.isfinite(xyz[:, joint]).all(axis=1) & (confidence[:, joint] > 0.0)
        if not valid.any():
            continue
        valid_x = source_x[valid]
        for axis in range(3):
            output[:, joint, axis] = np.interp(target_x, valid_x, xyz[valid, joint, axis])
        output_confidence[:, joint] = np.interp(
            target_x, valid_x, confidence[valid, joint], left=0.0, right=0.0
        )
    return output, output_confidence


def normalize_projection(
    xyz: np.ndarray, confidence: np.ndarray, axes: tuple[int, int]
) -> np.ndarray:
    xy = xyz[..., list(axes)].copy()
    valid = confidence > 0.0
    valid_values = xy[valid]
    if len(valid_values) < 4:
        return np.zeros((*xyz.shape[:2], 3), dtype=np.float32)
    lo = np.percentile(valid_values, 1.0, axis=0)
    hi = np.percentile(valid_values, 99.0, axis=0)
    center = (lo + hi) * 0.5
    scale = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-4))
    xy = (xy - center) * (2.0 / scale)
    xy = np.clip(xy, -2.0, 2.0)
    xy[~valid] = 0.0
    return np.concatenate((xy, confidence[..., None]), axis=-1).astype(np.float32)


def prepare_pose_cache(path: Path, target_frames: int) -> None:
    protocol = load_protocol()
    metadata = json.loads((RAW_CACHE / "metadata.json").read_text(encoding="utf-8"))
    if metadata["sample_ids"] != protocol.sample_ids.tolist():
        raise ValueError("raw skeleton cache is not in master manifest order")
    raw = np.load(RAW_CACHE / "skeleton_raw_float32.npy", mmap_mode="r")
    views = {
        name: np.zeros((len(protocol.labels), target_frames, 17, 3), dtype=np.float32)
        for name in VIEW_AXES
    }
    started = time.time()
    for row, (start, length) in enumerate(metadata["offsets"]):
        xyz, confidence = interpolate_pose(np.asarray(raw[start : start + length]), target_frames)
        for name, axes in VIEW_AXES.items():
            views[name][row] = normalize_projection(xyz, confidence, axes)
        if (row + 1) % 500 == 0:
            print(f"prepared {row + 1}/{len(protocol.labels)} trials", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.asarray(protocol.sample_ids, dtype=str),
        labels=protocol.labels,
        target_frames=np.int32(target_frames),
        **views,
    )
    print(f"wrote {path} in {(time.time() - started) / 60:.1f} min")


def import_dstformer():
    root = str(MOTIONBERT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from lib.model.DSTformer import DSTformer

    return DSTformer


def build_backbone(spec: InitSpec, maxlen: int, device: torch.device) -> nn.Module:
    DSTformer = import_dstformer()
    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=spec.dim_feat,
        dim_rep=spec.dim_rep,
        depth=spec.depth,
        num_heads=spec.heads,
        mlp_ratio=spec.mlp_ratio,
        num_joints=17,
        maxlen=max(243, maxlen),
        att_fuse=True,
    )
    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint[spec.checkpoint_key]
    state = {
        key[len(spec.prefix) :]: value
        for key, value in source.items()
        if key.startswith(spec.prefix)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    real_missing = set(missing) - allowed_missing
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={real_missing}, unexpected={unexpected}")
    model.head = nn.Identity()
    model.eval().to(device)
    return model


@torch.inference_mode()
def extract_features(
    spec: InitSpec,
    pose_cache: Path,
    view_names: list[str],
    output_path: Path,
    batch_size: int,
) -> None:
    cache = np.load(pose_cache, allow_pickle=False)
    target_frames = int(cache["target_frames"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(spec, target_frames, device)
    all_views: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    for view_name in view_names:
        poses = torch.from_numpy(np.asarray(cache[view_name], dtype=np.float32))
        loader = DataLoader(TensorDataset(poses), batch_size=batch_size, shuffle=False, num_workers=0)
        chunks: list[np.ndarray] = []
        for batch_id, (batch,) in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                representation = model.get_representation(batch)
                # Same temporal pooling used by the official MotionBERT action head.
                pooled = representation.mean(dim=1).flatten(1)
                temporal_std = representation.float().std(dim=1).mean(dim=1)
                feature = torch.cat((pooled.float(), temporal_std), dim=1)
            chunks.append(feature.cpu().numpy().astype(np.float16))
            if (batch_id + 1) % 25 == 0:
                print(f"{spec.name}/{view_name}: {min((batch_id + 1) * batch_size, len(poses))}/{len(poses)}", flush=True)
        all_views.append(np.concatenate(chunks, axis=0))
    features = np.concatenate(all_views, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_ids=cache["sample_ids"],
        labels=cache["labels"],
        init=np.asarray(spec.name),
        views=np.asarray(view_names),
        features=features,
    )
    print(f"wrote {output_path}, shape={features.shape}, dtype={features.dtype}")


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(input_dim, NUM_CLASSES)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(values))


LEFT_JOINTS = [4, 5, 6, 11, 12, 13]
RIGHT_JOINTS = [1, 2, 3, 14, 15, 16]


def flip_pose(batch: torch.Tensor) -> torch.Tensor:
    output = batch.clone()
    output[..., 0] *= -1.0
    source = LEFT_JOINTS + RIGHT_JOINTS
    target = RIGHT_JOINTS + LEFT_JOINTS
    output[..., source, :] = output[..., target, :]
    return output


class AugmentedPoseDataset(torch.utils.data.Dataset):
    def __init__(self, poses: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        self.poses = np.asarray(poses, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pose = torch.from_numpy(self.poses[index].copy())
        label = torch.tensor(self.labels[index], dtype=torch.long)
        if not self.augment:
            return pose, label
        if torch.rand(()) < 0.5:
            pose = flip_pose(pose)
        coordinates = pose[..., :2]
        confidence = pose[..., 2:3]
        angle = (torch.rand(()) - 0.5) * math.radians(20.0)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
        coordinates = coordinates @ rotation.T
        coordinates = coordinates * (0.9 + 0.2 * torch.rand(()))
        coordinates = coordinates + (torch.rand((1, 1, 2)) - 0.5) * 0.12
        coordinates = coordinates + torch.randn_like(coordinates) * 0.006
        if torch.rand(()) < 0.35:
            mask = torch.rand((pose.shape[0], 1, 1)) < 0.06
            coordinates = coordinates.masked_fill(mask, 0.0)
            confidence = confidence.masked_fill(mask, 0.0)
        pose = torch.cat((coordinates.clamp(-2.0, 2.0), confidence), dim=-1)
        return pose, label


class MotionBERTClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int = 1024, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(17 * 512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        representation = self.backbone.get_representation(pose)
        pooled = representation.mean(dim=1).flatten(1)
        return self.head(pooled)


def set_finetune_mode(model: MotionBERTClassifier, mode: str) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = mode == "full"
    if mode == "last2":
        modules = [
            model.backbone.blocks_st[-2:],
            model.backbone.blocks_ts[-2:],
            model.backbone.ts_attn[-2:],
            model.backbone.norm,
            model.backbone.pre_logits,
        ]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif mode not in {"full", "head"}:
        raise ValueError(mode)
    for parameter in model.head.parameters():
        parameter.requires_grad = True


@torch.inference_mode()
def predict_classifier(
    model: MotionBERTClassifier,
    poses: np.ndarray,
    batch_size: int,
    device: torch.device,
    flip_tta: bool,
) -> np.ndarray:
    loader = DataLoader(
        AugmentedPoseDataset(poses, np.zeros(len(poses), dtype=np.int64), augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for batch, _ in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch)
            if flip_tta:
                logits = (logits + model(flip_pose(batch))) * 0.5
        chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def finetune_motionbert(
    pose_cache: Path,
    output_dir: Path,
    init_name: str,
    view_name: str,
    mode: str,
    epochs: int,
    batch_size: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    fold_only: int | None,
) -> None:
    protocol = load_protocol()
    cache = np.load(pose_cache, allow_pickle=False)
    if cache["sample_ids"].astype(str).tolist() != protocol.sample_ids.tolist():
        raise ValueError("pose cache is not aligned to P90 protocol")
    poses = np.asarray(cache[view_name], dtype=np.float32)
    target_frames = int(cache["target_frames"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [fold_only] if fold_only is not None else list(range(3))
    oof_logits = np.zeros((len(protocol.labels), NUM_CLASSES), dtype=np.float32)
    completed = np.zeros(len(protocol.labels), dtype=bool)
    teacher_name = f"motionbert_{init_name}_{view_name}_{mode}_finetune"
    checkpoint_dir = output_dir / "checkpoints" / teacher_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        seed_everything(seed + fold)
        train_indices = protocol.train_indices(fold)
        val_indices = protocol.val_indices(fold)
        backbone = build_backbone(INIT_SPECS[init_name], target_frames, device)
        model = MotionBERTClassifier(backbone, hidden_dim=1024, dropout=dropout).to(device)
        set_finetune_mode(model, mode)
        trainable_backbone = [
            parameter for parameter in model.backbone.parameters() if parameter.requires_grad
        ]
        optimizer_groups = [{"params": model.head.parameters(), "lr": head_lr}]
        if trainable_backbone:
            optimizer_groups.append({"params": trainable_backbone, "lr": backbone_lr})
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
        dataset = AugmentedPoseDataset(
            poses[train_indices], protocol.labels[train_indices], augment=True
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + fold),
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        print(
            f"{teacher_name}: fold={fold}, trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
            flush=True,
        )
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            seen = 0
            for batch, labels in loader:
                batch = batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    logits = model(batch)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(labels)
                correct += int((logits.argmax(dim=1) == labels).sum())
                seen += len(labels)
            scheduler.step()
            print(
                f"  fold {fold} epoch {epoch + 1:02d}/{epochs}: loss={total_loss / seen:.4f}, train_acc={correct / seen:.4f}",
                flush=True,
            )
        logits = predict_classifier(
            model, poses[val_indices], batch_size, device, flip_tta=True
        )
        oof_logits[val_indices] = logits
        completed[val_indices] = True
        metrics = classification_metrics(logits, protocol.labels[val_indices])
        print(f"fold {fold} final accuracy={metrics['accuracy']:.6f}", flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "fold": fold,
                "init": init_name,
                "view": view_name,
                "mode": mode,
                "metrics": metrics,
            },
            checkpoint_dir / f"fold_{fold}.pt",
        )
    if fold_only is not None:
        output_path = output_dir / f"{teacher_name}_fold{fold_only}.npz"
        temporary_path = output_path.with_suffix(".tmp.npz")
        selected_logits = oof_logits[completed]
        selected_labels = protocol.labels[completed]
        np.savez_compressed(
            temporary_path,
            sample_ids=protocol.sample_ids[completed],
            labels=selected_labels,
            logits=selected_logits,
            fold_id=protocol.fold_id[completed],
        )
        temporary_path.replace(output_path)
        persisted = np.load(output_path, allow_pickle=False)
        persisted_metrics = classification_metrics(persisted["logits"], persisted["labels"])
        expected_metrics = classification_metrics(selected_logits, selected_labels)
        if not np.isclose(
            persisted_metrics["accuracy"], expected_metrics["accuracy"], atol=1e-12
        ):
            raise RuntimeError(
                f"persisted fold artifact changed accuracy: "
                f"{expected_metrics['accuracy']:.6f} -> {persisted_metrics['accuracy']:.6f}"
            )
        (output_dir / f"{teacher_name}_fold{fold_only}_metrics.json").write_text(
            json.dumps(
                {
                    "teacher": teacher_name,
                    "fold": fold_only,
                    "samples": int(completed.sum()),
                    "metrics": persisted_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    payload = save_oof_artifact(
        output_dir,
        teacher_name,
        oof_logits,
        protocol,
        metadata={
            "init": init_name,
            "view": view_name,
            "mode": mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "seed": seed,
            "augmentation": "flip, rotation, scale, translation, noise, frame mask",
            "inference": "original + horizontal-flip logit average",
            "selection": "fixed epochs; outer validation is evaluated once after training",
        },
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


def train_probe_fold(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
) -> tuple[np.ndarray, dict[str, float]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features = np.asarray(features[train_indices], dtype=np.float32)
    val_features = np.asarray(features[val_indices], dtype=np.float32)
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-4] = 1.0
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    train_set = TensorDataset(
        torch.from_numpy(train_features), torch.from_numpy(labels[train_indices])
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    model = LinearProbe(features.shape[1], dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_labels)
        scheduler.step()
        if epoch in {0, epochs - 1} or (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:03d}/{epochs}: train_loss={total_loss / len(train_set):.4f}", flush=True)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(val_features).to(device)).cpu().numpy()
    metrics = classification_metrics(logits, labels[val_indices])
    return logits.astype(np.float32), metrics


def screen_feature_file(
    feature_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
) -> None:
    protocol = load_protocol()
    data = np.load(feature_path, allow_pickle=False)
    if data["sample_ids"].astype(str).tolist() != protocol.sample_ids.tolist():
        raise ValueError("feature cache is not aligned to P90 protocol")
    features = np.asarray(data["features"])
    if not np.isfinite(features).all():
        raise ValueError("non-finite MotionBERT features")
    init_name = str(data["init"])
    views = [str(item) for item in data["views"]]
    teacher_name = f"motionbert_{init_name}_{'-'.join(views)}_linear"
    oof_logits = np.zeros((len(protocol.labels), NUM_CLASSES), dtype=np.float32)
    fold_results = []
    for fold in range(3):
        print(f"{teacher_name}: fold {fold}", flush=True)
        val_indices = protocol.val_indices(fold)
        logits, metrics = train_probe_fold(
            features,
            protocol.labels,
            protocol.train_indices(fold),
            val_indices,
            seed=seed + fold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
        )
        oof_logits[val_indices] = logits
        metrics["fold"] = fold
        fold_results.append(metrics)
        print(f"  fold {fold} accuracy={metrics['accuracy']:.6f}", flush=True)
    payload = save_oof_artifact(
        output_dir,
        teacher_name,
        oof_logits,
        protocol,
        metadata={
            "source_features": str(feature_path.resolve()),
            "init": init_name,
            "views": views,
            "feature_shape": list(features.shape),
            "probe": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "seed": seed,
                "selection": "fixed before OOF; no validation early stopping",
            },
        },
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("prepare", "extract", "screen", "finetune", "all"), default="all"
    )
    parser.add_argument("--init", choices=tuple(INIT_SPECS), default="action")
    parser.add_argument("--views", choices=("front", "multiview"), default="front")
    parser.add_argument("--pose-cache", type=Path, default=DEFAULT_POSE_CACHE)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--target-frames", type=int, default=81)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--finetune-mode", choices=("head", "last2", "full"), default="full")
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--finetune-batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-dropout", type=float, default=0.5)
    parser.add_argument("--fold-only", type=int, choices=(0, 1, 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    view_names = ["front"] if args.views == "front" else ["front", "side", "top"]
    feature_path = args.feature_cache or (
        args.output_dir / f"features_{args.init}_{args.views}_t{args.target_frames}.npz"
    )
    if args.stage == "prepare" or (args.stage == "all" and not args.pose_cache.exists()):
        prepare_pose_cache(args.pose_cache, args.target_frames)
    if args.stage == "extract" or (args.stage == "all" and not feature_path.exists()):
        extract_features(
            INIT_SPECS[args.init],
            args.pose_cache,
            view_names,
            feature_path,
            args.extract_batch_size,
        )
    if args.stage in {"screen", "all"}:
        screen_feature_file(
            feature_path,
            args.output_dir,
            args.epochs,
            args.probe_batch_size,
            args.learning_rate,
            args.weight_decay,
            args.dropout,
            args.seed,
        )
    if args.stage == "finetune":
        if args.views != "front":
            raise ValueError("P90 fine-tuning currently uses the selected front view")
        finetune_motionbert(
            args.pose_cache,
            args.output_dir,
            args.init,
            "front",
            args.finetune_mode,
            args.finetune_epochs,
            args.finetune_batch_size,
            args.backbone_lr,
            args.head_lr,
            args.weight_decay,
            args.finetune_dropout,
            args.seed,
            args.fold_only,
        )


if __name__ == "__main__":
    main()
