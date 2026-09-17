from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from aligned_data import AlignedMultimodalDataset
from aligned_model import (
    AlignedMultimodalModel,
    AttentionPool,
    TemporalBlock,
    temporal_shift,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
FINE_ACTION_CLASSES = (
    0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 14, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 37, 39,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结视觉Backbone，对layer2/3/4做统一GAP与空间感知探针")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FrozenFeatureDataset(Dataset):
    def __init__(self, features: dict[str, torch.Tensor], labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {name: value[index] for name, value in self.features.items()}, self.labels[index]


class ProbeHead(nn.Module):
    def __init__(self, input_dim: int, spatial: bool, dropout: float = 0.2) -> None:
        super().__init__()
        self.spatial = spatial
        self.channel_project = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        if spatial:
            self.position = nn.Parameter(torch.zeros(1, 1, 6, 128))
            self.spatial_score = nn.Sequential(
                nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)
            )
            self.spatial_combine = nn.Sequential(
                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
            )
        else:
            self.register_parameter("position", None)
            self.spatial_score = None
            self.spatial_combine = None
        self.temporal = TemporalBlock(128, 1, dropout * 0.5)
        self.temporal_pool = AttentionPool(128)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256), nn.Dropout(dropout), nn.Linear(256, 40)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Input is B,T,C,6. GAP and spatial probes share the same temporal head.
        if not self.spatial:
            sequence = self.channel_project(features.mean(dim=-1))
        else:
            tokens = self.channel_project(features.permute(0, 1, 3, 2))
            assert self.position is not None
            tokens = tokens + self.position
            assert self.spatial_score is not None and self.spatial_combine is not None
            weights = torch.softmax(self.spatial_score(tokens).squeeze(-1), dim=2)
            attended = torch.sum(tokens * weights.unsqueeze(-1), dim=2)
            maximum = tokens.amax(dim=2)
            sequence = self.spatial_combine(torch.cat([attended, maximum], dim=-1))
        sequence = self.temporal(sequence.transpose(1, 2)).transpose(1, 2)
        return self.classifier(self.temporal_pool(sequence))


def build_model(checkpoint: dict, device: torch.device) -> AlignedMultimodalModel:
    config = checkpoint["config"]
    modalities = list(config["modalities"])
    model = AlignedMultimodalModel(
        modalities,
        dropout=float(config["dropout"]),
        use_aux_heads=bool(config.get("use_aux_heads", False)),
        use_modality_masks=bool(config.get("use_modality_masks", False)),
        use_cross_attention=bool(config.get("use_cross_attention", False)),
        cross_attention_grid=tuple(config.get("cross_attention_grid", [2, 3])),
        cross_attention_heads=int(config.get("cross_attention_heads", 4)),
        depth_input_channels=int(config.get("depth_input_channels", 3)),
        use_layer3_spatial=bool(config.get("use_layer3_spatial", False)),
        skeleton_input_dim=int(config.get("skeleton_input_dim", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_features(
    model: AlignedMultimodalModel,
    loader: DataLoader,
    modality: str,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    assert model.visual is not None
    visual = model.visual
    collected: dict[str, list[torch.Tensor]] = {"layer2": [], "layer3": [], "layer4": []}
    labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = batch[modality].to(device, non_blocking=True)
            batch_size, time_steps = inputs.shape[:2]
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if modality == "depth":
                    assert visual.depth_stem is not None
                    x = visual.depth_stem(
                        inputs.reshape(
                            batch_size * time_steps,
                            visual.depth_input_channels,
                            *inputs.shape[-2:],
                        )
                    )
                else:
                    assert visual.ir_stem is not None
                    x = visual.ir_stem(
                        inputs.reshape(batch_size * time_steps, 1, *inputs.shape[-2:])
                    )
                x = visual.maxpool(visual.relu(visual.bn1(x)))
                x = visual.layer1(temporal_shift(x, batch_size, time_steps))
                x2 = visual.layer2(temporal_shift(x, batch_size, time_steps))
                x3 = visual.layer3(temporal_shift(x2, batch_size, time_steps))
                x4 = visual.layer4(temporal_shift(x3, batch_size, time_steps))
                for name, stage, channels in (
                    ("layer2", x2, 128),
                    ("layer3", x3, 256),
                    ("layer4", x4, 512),
                ):
                    pooled = nn.functional.adaptive_avg_pool2d(stage, (2, 3))
                    pooled = pooled.reshape(batch_size, time_steps, channels, 6)
                    collected[name].append(pooled.half().cpu())
            labels.append(batch["label"].long())
    return {name: torch.cat(values) for name, values in collected.items()}, torch.cat(labels)


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    fine_mask = np.isin(labels, np.asarray(FINE_ACTION_CLASSES))
    fine_labels = labels[fine_mask]
    fine_predictions = predictions[fine_mask]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "fine_action_accuracy": float(accuracy_score(fine_labels, fine_predictions)),
        "fine_action_balanced_accuracy": float(
            balanced_accuracy_score(fine_labels, fine_predictions)
        ),
    }


def evaluate(
    heads: nn.ModuleDict,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    for head in heads.values():
        head.eval()
    labels_all: list[int] = []
    predictions: dict[str, list[int]] = {name: [] for name in heads}
    with torch.inference_mode():
        for features, labels in loader:
            labels_all.extend(labels.tolist())
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                for name, head in heads.items():
                    layer = name.split("_")[0]
                    logits = head(features[layer].to(device, non_blocking=True))
                    predictions[name].extend(logits.argmax(1).cpu().tolist())
    labels_array = np.asarray(labels_all, dtype=np.int64)
    prediction_arrays = {
        name: np.asarray(values, dtype=np.int64) for name, values in predictions.items()
    }
    return (
        {name: metric_dict(labels_array, values) for name, values in prediction_arrays.items()},
        prediction_arrays,
    )


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    modalities = list(config["modalities"])
    if len(modalities) != 1 or modalities[0] not in {"depth", "ir"}:
        raise ValueError(f"当前probe要求Depth-only或IR-only，实际为{modalities}")
    modality = modalities[0]
    output = (
        args.output.resolve()
        if args.output
        else checkpoint_path.with_name("intermediate_feature_probe.json")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    model = build_model(checkpoint, device)

    common = dict(
        manifest_path=args.manifest.resolve(),
        modalities=modalities,
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=False,
        cache_dir=args.cache_dir.resolve(),
        skeleton_strategy=config.get("skeleton_strategy", "first"),
        depth_representation=config.get("depth_representation", "jet_rgb"),
        visual_normalization=config.get("visual_normalization", "legacy"),
        skeleton_representation=config.get("skeleton_representation", "frame_joint"),
        skeleton_raw_cache_dir=config.get("skeleton_raw_cache_dir"),
    )
    train_dataset = AlignedMultimodalDataset(split="train", **common)
    val_dataset = AlignedMultimodalDataset(split="val", **common)
    extraction_batch = int(config["batch_size"])
    train_loader = DataLoader(
        train_dataset, batch_size=extraction_batch, shuffle=False,
        num_workers=max(0, int(args.num_workers)), pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=extraction_batch, shuffle=False,
        num_workers=max(0, int(args.num_workers)), pin_memory=torch.cuda.is_available()
    )
    print(f"提取{modality}冻结中层特征：Train", flush=True)
    train_features, train_labels = extract_features(model, train_loader, modality, device, use_amp)
    print(f"提取{modality}冻结中层特征：Val", flush=True)
    val_features, val_labels = extract_features(model, val_loader, modality, device, use_amp)
    del model
    torch.cuda.empty_cache()

    channel_dims = {"layer2": 128, "layer3": 256, "layer4": 512}
    heads = nn.ModuleDict(
        {
            f"{layer}_{mode}": ProbeHead(channels, spatial=mode == "spatial")
            for layer, channels in channel_dims.items()
            for mode in ("gap", "spatial")
        }
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        heads.parameters(), lr=float(args.learning_rate), weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    counts = Counter(train_labels.tolist())
    weights = torch.tensor(
        [1.0 / counts[int(label)] for label in train_labels], dtype=torch.double
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    sampler = WeightedRandomSampler(
        weights, len(weights), replacement=True, generator=generator
    )
    probe_train_loader = DataLoader(
        FrozenFeatureDataset(train_features, train_labels),
        batch_size=int(args.batch_size),
        sampler=sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    probe_val_loader = DataLoader(
        FrozenFeatureDataset(val_features, val_labels),
        batch_size=int(args.batch_size),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    best: dict[str, dict[str, object]] = {
        name: {"epoch": 0, "metrics": {"accuracy": -1.0}, "predictions": None}
        for name in heads
    }
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        for head in heads.values():
            head.train()
        running_loss = 0.0
        batches = 0
        for features, labels in probe_train_loader:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                losses = []
                for name, head in heads.items():
                    layer = name.split("_")[0]
                    logits = head(features[layer].to(device, non_blocking=True))
                    losses.append(criterion(logits, labels))
                loss = torch.stack(losses).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss)
            batches += 1
        scheduler.step()
        val_metrics, val_predictions = evaluate(heads, probe_val_loader, device, use_amp)
        for name, values in val_metrics.items():
            if values["accuracy"] > best[name]["metrics"]["accuracy"]:
                best[name] = {
                    "epoch": epoch,
                    "metrics": values,
                    "predictions": val_predictions[name].tolist(),
                }
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
            "val": val_metrics,
        }
        history.append(row)
        compact = ", ".join(
            f"{name}={values['accuracy'] * 100:.2f}%" for name, values in val_metrics.items()
        )
        print(f"Epoch {epoch:02d}: {compact}", flush=True)

    labels_array = val_labels.numpy()
    per_class: dict[str, list[dict[str, object]]] = {}
    for name, result in best.items():
        prediction = np.asarray(result.pop("predictions"), dtype=np.int64)
        rows: list[dict[str, object]] = []
        for class_id in range(40):
            mask = labels_array == class_id
            rows.append(
                {
                    "class_id": class_id,
                    "samples": int(mask.sum()),
                    "accuracy": float(np.mean(prediction[mask] == labels_array[mask])),
                }
            )
        per_class[name] = rows

    summary = {
        "checkpoint": str(checkpoint_path),
        "modality": modality,
        "device": str(device),
        "spatial_grid": [2, 3],
        "fine_action_classes": list(FINE_ACTION_CLASSES),
        "probe_design": (
            "所有层先投影到128维并使用同一个TemporalBlock+AttentionPool；"
            "spatial probe额外保留2x3位置token，GAP probe先做空间平均。"
        ),
        "best": best,
        "per_class": per_class,
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
