from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import yaml
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_unimodal import build_datasets, build_model, loader_for, set_seed


CONFIGS = (
    PROJECT_ROOT / "configs/experiments/depth_hard_global_expert.yaml",
    PROJECT_ROOT / "configs/experiments/depth_pose_roi_expert.yaml",
)


def resolve_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("manifest", "fold", "output_root", "pose_cache"):
        if config.get(key):
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else (PROJECT_ROOT / value).resolve())
    config.update(
        {
            "config_path": str(path.resolve()),
            "smoke_test": False,
            "max_train_batches": None,
            "max_val_batches": None,
        }
    )
    return config


def probe(config: dict[str, object]) -> dict[str, object]:
    set_seed(int(config["seed"]))
    train_dataset, _ = build_datasets(config)
    model = build_model(config, train_dataset[0]).cuda()
    loader = loader_for(train_dataset, config, training=True)
    batch = next(iter(loader))
    inputs = batch["input"].cuda(non_blocking=True)
    mask = batch["temporal_mask"].cuda(non_blocking=True)
    labels = batch["label"].cuda(non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=bool(config["amp"])):
        output = model(inputs, temporal_mask=mask)
        loss = nn.functional.cross_entropy(output["logits"], labels)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    return {
        "model_name": config["model_name"],
        "status": "passed",
        "batch_size": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "logits_shape": list(output["logits"].shape),
        "roi_attention_shape": list(output["roi_attention"].shape),
        "loss": float(loss.detach()),
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


def main() -> None:
    results = [probe(resolve_config(path)) for path in CONFIGS]
    output = PROJECT_ROOT / "outputs/depth_pose_roi_probe/training_memory_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"TRAINING_PROBE_JSON={json.dumps(results)}")


if __name__ == "__main__":
    main()
