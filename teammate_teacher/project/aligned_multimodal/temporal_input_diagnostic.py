from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断视觉模型对帧顺序和多帧信息的依赖")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve() if args.output else checkpoint_path.with_name("temporal_diagnostic.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    modalities = list(config["modalities"])
    if len(modalities) != 1 or modalities[0] not in {"depth", "ir"}:
        raise ValueError(f"当前诊断要求Depth-only或IR-only，实际为{modalities}")
    modality = modalities[0]

    dataset = AlignedMultimodalDataset(
        manifest_path=args.manifest.resolve(),
        split="val",
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
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
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

    prediction_lists: dict[str, list[int]] = {
        "normal": [],
        "reverse": [],
        "repeat_middle": [],
    }
    shuffle_lists: list[list[int]] = [[] for _ in range(int(args.shuffle_repeats))]
    labels_all: list[int] = []
    generator = torch.Generator(device=device).manual_seed(int(args.seed))

    with torch.inference_mode():
        for batch in loader:
            visual = batch[modality].to(device, non_blocking=True)
            labels_all.extend(batch["label"].tolist())
            time_steps = visual.shape[1]
            middle = (time_steps - 1) // 2
            variants = {
                "normal": visual,
                "reverse": visual.flip(1),
                "repeat_middle": visual[:, middle : middle + 1].expand_as(visual),
            }
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                for name, value in variants.items():
                    logits = model({modality: value})
                    prediction_lists[name].extend(logits.argmax(1).cpu().tolist())
                for repeat in range(int(args.shuffle_repeats)):
                    order = torch.rand(
                        visual.shape[0], time_steps, device=device, generator=generator
                    ).argsort(dim=1)
                    gather_index = order[:, :, None, None, None].expand_as(visual)
                    shuffled = torch.gather(visual, 1, gather_index)
                    logits = model({modality: shuffled})
                    shuffle_lists[repeat].extend(logits.argmax(1).cpu().tolist())

    labels = np.asarray(labels_all, dtype=np.int64)
    predictions = {name: np.asarray(values, dtype=np.int64) for name, values in prediction_lists.items()}
    shuffled = np.asarray(shuffle_lists, dtype=np.int64)
    shuffle_values = [metrics(labels, row) for row in shuffled]
    shuffle_summary = {
        key: {
            "mean": float(np.mean([value[key] for value in shuffle_values])),
            "std": float(np.std([value[key] for value in shuffle_values])),
        }
        for key in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    baseline = predictions["normal"]
    manifest = pd.read_csv(args.manifest.resolve())
    class_names = (
        manifest[["class_id", "class_name"]]
        .drop_duplicates("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )
    per_class: list[dict[str, object]] = []
    for class_id in sorted(np.unique(labels)):
        mask = labels == class_id
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
                "samples": int(mask.sum()),
                "normal_accuracy": float(np.mean(baseline[mask] == labels[mask])),
                "reverse_accuracy": float(np.mean(predictions["reverse"][mask] == labels[mask])),
                "repeat_middle_accuracy": float(
                    np.mean(predictions["repeat_middle"][mask] == labels[mask])
                ),
                "shuffle_accuracy_mean": float(
                    np.mean([np.mean(row[mask] == labels[mask]) for row in shuffled])
                ),
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "modality": modality,
        "device": str(device),
        "samples": len(labels),
        "shuffle_repeats": int(args.shuffle_repeats),
        "normal": metrics(labels, predictions["normal"]),
        "reverse": {
            **metrics(labels, predictions["reverse"]),
            "prediction_change_rate": float(np.mean(predictions["reverse"] != baseline)),
        },
        "repeat_middle": {
            **metrics(labels, predictions["repeat_middle"]),
            "prediction_change_rate": float(np.mean(predictions["repeat_middle"] != baseline)),
        },
        "shuffle": {
            **shuffle_summary,
            "prediction_change_rate_mean": float(
                np.mean([np.mean(row != baseline) for row in shuffled])
            ),
        },
        "interpretation_limit": (
            "repeat-middle与shuffle是推理时扰动，下降可能同时包含分布外影响；"
            "它们用于诊断，不等价于重新训练的single-frame基线。"
        ),
        "per_class": per_class,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_class"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
