from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_p86_mc3_sequence_cache import load_model
from p87s_test_data import P87STestPixelDataset, collate_p87s_test


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXELS = PROJECT_DIR / "runs/p87s_test_pixel_cache_t16_r160_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87s_test_mc3_sequence_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache final all-2914 Student MC3 sequences for 405 label-free Test rows."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
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


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint_path, device)
    if not bool(config.get("temporal_modeling", True)):
        raise RuntimeError("P87-S requires temporal MC3")
    reference = P87STestPixelDataset(args.pixel_cache)
    frames = int(config["frames"])
    expected_shape = (len(reference), 2, 3, frames, 512)
    sequence_path = output / "backbone_sequence_fp16.npy"
    completed_path = output / "completed.npy"
    if sequence_path.exists() or completed_path.exists():
        if not sequence_path.exists() or not completed_path.exists():
            raise RuntimeError("partial P87-S sequence cache")
        sequence = np.lib.format.open_memmap(sequence_path, mode="r+")
        completed = np.lib.format.open_memmap(completed_path, mode="r+")
        if sequence.shape != expected_shape or completed.shape != (len(reference),):
            raise RuntimeError("existing P87-S sequence cache shape differs")
    else:
        sequence = np.lib.format.open_memmap(
            sequence_path, mode="w+", dtype=np.float16, shape=expected_shape
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.uint8, shape=(len(reference),)
        )
        sequence[:] = 0
        completed[:] = 0
        sequence.flush()
        completed.flush()

    pending = np.flatnonzero(~np.asarray(completed, dtype=bool))
    dataset = P87STestPixelDataset(args.pixel_cache, indices=pending)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_p87s_test,
    )
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
                values = model.encode_backbone_sequence(images)
            values_np = values.to(dtype=torch.float16).cpu().numpy()
            indices = np.asarray(batch["cache_index"], dtype=np.int64)
            sequence[indices] = values_np
            completed[indices] = 1
            emitted += len(indices)
            if emitted % 64 < len(indices):
                sequence.flush()
                completed.flush()
                print(
                    json.dumps(
                        {
                            "cached": int(np.asarray(completed).sum()),
                            "total": len(reference),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    sequence.flush()
    completed.flush()
    with (output / "rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ("sample_id", "source_id", "user_id", "class_id")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in reference.rows)
    summary = {
        "stage": "P87S_label_free_test_MC3_sequence_cache",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "shape": list(expected_shape),
        "completed": int(np.asarray(completed).sum()),
        "total": len(reference),
        "labels_read": False,
        "large_videomae_required": False,
        "purpose": "final Student acceleration only; cached activations are not deployed",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
