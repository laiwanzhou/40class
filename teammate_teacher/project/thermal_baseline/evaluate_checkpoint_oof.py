from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from thermal_oof_data import ThermalOOFDataset
from thermal_tsm_model import ThermalResNetTSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Thermal checkpoint on the validation rows in its manifest."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset = ThermalOOFDataset(
        manifest_path=Path(config["manifest"]),
        split="val",
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
    labels: list[int] = []
    logits: list[torch.Tensor] = []
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                batch_logits = model(batch["clip"].to(device, non_blocking=True))
            sample_ids.extend(batch["sample_id"])
            labels.extend(batch["label"].tolist())
            logits.append(batch_logits.float().cpu())
    logits_array = torch.cat(logits).numpy()
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = logits_array.argmax(axis=1)
    np.savez_compressed(
        output,
        sample_ids=np.asarray(sample_ids),
        labels=labels_array,
        logits=logits_array.astype(np.float32),
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "storage_dtype": checkpoint.get("storage_dtype", "original"),
        "samples": len(dataset),
        "device": str(device),
        "seconds": time.time() - started,
        **metric_dict(labels_array, predictions),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
