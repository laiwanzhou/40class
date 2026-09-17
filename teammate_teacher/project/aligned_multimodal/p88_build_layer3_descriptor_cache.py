from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build_p86_mc3_sequence_cache import (
    DEFAULT_FEATURES,
    DEFAULT_LOGITS,
    DEFAULT_PIXELS,
    load_model,
)
from p86_mc3_visual_model import KINETICS_MEAN, KINETICS_STD
from p86_visual_pixel_data import P86VisualPixelDataset, collate_p86_pixels


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "runs/p87s_visual_holdout1_v1/visual_student.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_ir_layer3_descriptors_holdout1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache fixed layer3 2x2 spatial and temporal descriptors from frozen P87 IR MC3."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def layer3_descriptors(model, images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 6:
        raise ValueError("P88 layer3 input must be [B,2,T,3,H,W]")
    batch, windows, steps, views, height, width = images.shape
    clips = images.permute(0, 1, 3, 2, 4, 5).reshape(
        batch * windows * views, steps, 1, height, width
    )
    unit = clips.float().div_(255.0)
    rgb = unit.repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
    mean = rgb.new_tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
    std = rgb.new_tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
    value = model.stem((rgb - mean) / std)
    value = model.layer1(value)
    value = model.layer2(value)
    value = model.layer3(value)
    regions = F.adaptive_avg_pool3d(value, output_size=(steps, 2, 2))
    regions = regions.flatten(-2).permute(0, 2, 3, 1)
    global_region = regions.mean(dim=2)
    top = 0.5 * (regions[:, :, 0] + regions[:, :, 1])
    bottom = 0.5 * (regions[:, :, 2] + regions[:, :, 3])
    left = 0.5 * (regions[:, :, 0] + regions[:, :, 2])
    right = 0.5 * (regions[:, :, 1] + regions[:, :, 3])
    diagonal = 0.5 * (
        regions[:, :, 0] + regions[:, :, 3]
        - regions[:, :, 1] - regions[:, :, 2]
    )
    spatial = torch.stack(
        (global_region, bottom - top, right - left, diagonal), dim=2
    )
    temporal = torch.stack(
        (
            spatial.mean(dim=1),
            spatial.std(dim=1, unbiased=False),
            spatial[:, -1] - spatial[:, 0],
        ),
        dim=1,
    )
    return temporal.reshape(batch, windows, views, 3, 4, 256)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "source_id", "user_id", "class_id")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint, device)
    reference = P86VisualPixelDataset(
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        augment=False,
    )
    loader = DataLoader(
        reference,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_p86_pixels,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    descriptor_path = output / "layer3_descriptors_fp16.npy"
    quality_path = output / "clip_quality_fp16.npy"
    completed_path = output / "completed.npy"
    descriptor = np.lib.format.open_memmap(
        descriptor_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(reference), 2, 3, 3, 4, 256),
    )
    quality = np.lib.format.open_memmap(
        quality_path, mode="w+", dtype=np.float16, shape=(len(reference), 2, 3)
    )
    completed = np.lib.format.open_memmap(
        completed_path, mode="w+", dtype=np.uint8, shape=(len(reference),)
    )
    completed[:] = 0
    started = time.perf_counter()
    emitted = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                values = layer3_descriptors(model, images)
            indices = np.asarray(batch["cache_index"], dtype=np.int64)
            descriptor[indices] = values.to(dtype=torch.float16).cpu().numpy()
            valid = batch["view_valid"].numpy().astype(np.float32)
            source_quality = batch["view_quality"].numpy().astype(np.float32)
            quality[indices] = (valid * source_quality).mean(axis=2).astype(np.float16)
            completed[indices] = 1
            emitted += len(indices)
            if emitted % 128 < len(indices):
                descriptor.flush(); quality.flush(); completed.flush()
                print(json.dumps({"completed": emitted, "total": len(reference), "elapsed_seconds": round(time.perf_counter()-started, 1)}), flush=True)
    descriptor.flush(); quality.flush(); completed.flush()
    if emitted != len(reference):
        raise RuntimeError("P88 layer3 descriptor cache is incomplete")
    write_rows(output / "rows.csv", reference.rows)
    summary = {
        "stage": "P88_frozen_P87_IR_layer3_spatial_descriptors",
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "p87_checkpoint_modified": False,
        "rows": len(reference),
        "shape": list(descriptor.shape),
        "descriptor_order": ["temporal_mean", "temporal_std", "last_minus_first"],
        "spatial_order": ["global", "bottom_minus_top", "right_minus_left", "diagonal"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
