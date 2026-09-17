from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset
from residual_logit_fusion import build_model, metric_dict


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一次推理评估冻结的 Skeleton/Depth/IR 专家固定融合")
    parser.add_argument("--skeleton-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--ir-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def write_predictions(
    path: Path, sample_ids: list[str], labels: np.ndarray, predictions: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prediction"])
        writer.writerows(zip(sample_ids, labels.tolist(), predictions.tolist()))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "skeleton": args.skeleton_checkpoint.resolve(),
        "depth": args.depth_checkpoint.resolve(),
        "ir": args.ir_checkpoint.resolve(),
    }
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    configs = {name: checkpoint["config"] for name, checkpoint in checkpoints.items()}
    for name in paths:
        if list(configs[name]["modalities"]) != [name]:
            raise ValueError(f"{name} checkpoint 的 modalities 不匹配")
    for key in ("num_frames", "image_height", "image_width"):
        values = {int(config[key]) for config in configs.values()}
        if len(values) != 1:
            raise ValueError(f"三个专家的 {key} 不一致：{values}")
    if configs["depth"].get("visual_normalization", "legacy") != configs["ir"].get(
        "visual_normalization", "legacy"
    ):
        raise ValueError("当前 Dataset 需要 Depth 与 IR 使用相同 visual_normalization")

    dataset = AlignedMultimodalDataset(
        manifest_path=args.manifest.resolve(),
        split="val",
        modalities=["depth", "ir", "skeleton"],
        num_frames=int(configs["depth"]["num_frames"]),
        image_height=int(configs["depth"]["image_height"]),
        image_width=int(configs["depth"]["image_width"]),
        augment=False,
        cache_dir=configs["depth"].get("cache_dir"),
        skeleton_strategy=configs["skeleton"].get("skeleton_strategy", "first"),
        depth_representation=configs["depth"].get("depth_representation", "jet_rgb"),
        visual_normalization=configs["depth"].get("visual_normalization", "legacy"),
        skeleton_representation=configs["skeleton"].get(
            "skeleton_representation", "frame_joint"
        ),
        skeleton_raw_cache_dir=configs["skeleton"].get("skeleton_raw_cache_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(configs["depth"]["batch_size"]),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: build_model(checkpoint, device) for name, checkpoint in checkpoints.items()}
    use_amp = bool(configs["depth"].get("use_amp", True) and device.type == "cuda")
    sample_ids: list[str] = []
    labels_all: list[int] = []
    logits_all: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                for name, model in models.items():
                    logits = model({name: batch[name].to(device, non_blocking=True)})
                    logits_all[name].append(logits.float().cpu())
            sample_ids.extend(batch["sample_id"])
            labels_all.extend(batch["label"].tolist())

    labels = np.asarray(labels_all, dtype=np.int64)
    logits = {name: torch.cat(parts).numpy() for name, parts in logits_all.items()}
    combinations = {
        "skeleton": {"skeleton": 1.0},
        "depth": {"depth": 1.0},
        "ir": {"ir": 1.0},
        "si_equal": {"skeleton": 0.5, "ir": 0.5},
        "di_equal": {"depth": 0.5, "ir": 0.5},
        "sd_w040": {"skeleton": 0.6, "depth": 0.4},
        "sdi_ir010": {"skeleton": 0.54, "depth": 0.36, "ir": 0.10},
        "sdi_equal": {"skeleton": 1.0 / 3, "depth": 1.0 / 3, "ir": 1.0 / 3},
    }
    summary = {
        "protocol": "All checkpoints frozen; all weights fixed before three-fold results.",
        "samples": len(labels),
        "methods": {},
        "checkpoints": {name: str(path) for name, path in paths.items()},
        "device": str(device),
    }
    for name, weights in combinations.items():
        fused = sum(weight * logits[modality] for modality, weight in weights.items())
        predictions = fused.argmax(axis=1)
        path = output_dir / f"{name}.csv"
        write_predictions(path, sample_ids, labels, predictions)
        summary["methods"][name] = {
            "weights": weights,
            "path": str(path),
            **metric_dict(labels, predictions),
        }
    np.savez_compressed(
        output_dir / "logits.npz",
        sample_ids=np.asarray(sample_ids),
        labels=labels,
        skeleton_logits=logits["skeleton"],
        depth_logits=logits["depth"],
        ir_logits=logits["ir"],
    )
    summary["seconds"] = round(time.time() - started, 2)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
