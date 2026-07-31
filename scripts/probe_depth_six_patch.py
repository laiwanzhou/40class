from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import VisualSixPatchDataset, load_modality_frames
from src.models import VisualSixPatch
from src.train_unimodal import seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_six_patch.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/depth_six_patch_fold0_14train_4val/memory_probe.json",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    train_frame, _ = load_modality_frames(
        resolve(config["manifest"]),
        resolve(config["fold"]),
        resolve(config["data_root"]),
        config["path_column"],
    )
    dataset = VisualSixPatchDataset(
        train_frame,
        config["modality"],
        int(config["num_frames"]),
        int(config["image_size"]),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0,
        prefetch_factor=2,
        multiprocessing_context="spawn",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    device = torch.device(config["device"])
    model = VisualSixPatch(
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]))
    criterion = nn.CrossEntropyLoss()
    batch = next(iter(loader))
    inputs = batch["input"].to(device)
    labels = batch["label"].to(device)
    mask = batch["temporal_mask"].to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=bool(config["amp"])):
        output = model(inputs, temporal_mask=mask)
        loss = criterion(output["logits"], labels)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    result = {
        "status": "passed",
        "batch_size": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "logits_shape": list(output["logits"].shape),
        "patch_attention_shape": list(output["patch_attention"].shape),
        "loss": float(loss.detach()),
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PROBE_JSON={json.dumps(result)}")


if __name__ == "__main__":
    main()
