from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from thermal_oof_data import ThermalOOFDataset
from thermal_tsm_model import ThermalResNetTSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Thermal logits for a manifest split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset = ThermalOOFDataset(
        manifest_path=args.manifest,
        split=args.split,
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=False,
        normalization=str(config.get("normalization", "legacy")),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    model = ThermalResNetTSM(
        num_classes=40,
        dropout=float(config["dropout"]),
        imagenet_pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sample_ids: list[str] = []
    logits: list[torch.Tensor] = []
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                batch_logits = model(batch["clip"].to(device, non_blocking=True))
            sample_ids.extend(batch["sample_id"])
            logits.append(batch_logits.float().cpu())
    logits_array = torch.cat(logits).numpy().astype(np.float32)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=np.asarray(sample_ids),
        logits=logits_array,
    )
    with output.with_suffix(".csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "thermal_prediction"])
        writer.writerows(zip(sample_ids, logits_array.argmax(axis=1).tolist()))
    summary = {
        "checkpoint": str(checkpoint_path),
        "storage_dtype": checkpoint.get("storage_dtype", "original"),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "samples": len(sample_ids),
        "predicted_classes": int(len(np.unique(logits_array.argmax(axis=1)))),
        "device": str(device),
        "seconds": time.time() - started,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
