from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, Dinov2Model

from build_p46_videomae_cache import (
    VIEW_NAMES,
    atomic_npz,
    prepare_trial,
    read_rows,
    safe_relative,
)
from p46_protocol import DEFAULT_MANIFEST
from p90_videomae_lora_teacher import read_aligned_rows


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_P29 = PROJECT_DIR / "runs/p29_dir_multiscale_roi_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p292_full40_dinov2_cache_v1"
MODEL_NAME = "facebook/dinov2-base"
FRAME_COUNT = 8
HIDDEN_SIZE = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen DINOv2 appearance tokens for 3 P46 IR views x 8 times."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trial-batch", type=int, default=4)
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_rgb(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    elif values.ndim == 3 and values.shape[2] == 1:
        values = np.repeat(values, 3, axis=2)
    if values.ndim != 3 or values.shape[2] != 3:
        raise RuntimeError(f"Unsupported IR image shape for DINOv2: {values.shape}")
    if values.dtype != np.uint8:
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise RuntimeError("IR image has no finite pixels")
        low, high = np.percentile(finite, (1.0, 99.0))
        values = np.clip((values - low) * (255.0 / max(high - low, 1e-6)), 0, 255).astype(np.uint8)
    return values


def valid_cache(path: Path, row: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["sample_id"].item()) == row["sample_id"]
                and int(data["label"].item()) == int(row["class_id"])
                and tuple(data["features"].shape) == (len(VIEW_NAMES), FRAME_COUNT, HIDDEN_SIZE)
                and np.isfinite(data["features"]).all()
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


@torch.inference_mode()
def encode_images(
    model: Dinov2Model,
    processor: AutoImageProcessor,
    images: list[np.ndarray],
    image_batch: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    outputs: list[np.ndarray] = []
    peak = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(images), image_batch):
        pixel_values = processor(
            images=images[start : start + image_batch], return_tensors="pt"
        ).pixel_values.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            hidden = model(pixel_values=pixel_values).last_hidden_state[:, 0]
        outputs.append(hidden.float().cpu().numpy())
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device) / 1024**3)
    return np.concatenate(outputs, axis=0), peak


def aggregate(rows: list[dict[str, str]], output: Path) -> Path:
    records: dict[str, list] = {
        "sample_ids": [],
        "source_ids": [],
        "users": [],
        "labels": [],
        "features": [],
        "person_valid_rate": [],
        "workspace_valid_rate": [],
    }
    for row in rows:
        cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
        if not valid_cache(cache, row):
            raise RuntimeError(f"Missing or invalid DINOv2 cache: {row['source_id']}")
        with np.load(cache, allow_pickle=False) as data:
            records["sample_ids"].append(row["sample_id"])
            records["source_ids"].append(row["source_id"])
            records["users"].append(row["user_id"])
            records["labels"].append(int(row["class_id"]))
            records["features"].append(np.asarray(data["features"], dtype=np.float16))
            records["person_valid_rate"].append(float(np.asarray(data["person_valid"]).mean()))
            records["workspace_valid_rate"].append(float(np.asarray(data["workspace_valid"]).mean()))
    path = output / "complete_features.npz"
    atomic_npz(
        path,
        sample_ids=np.asarray(records["sample_ids"]),
        source_ids=np.asarray(records["source_ids"]),
        users=np.asarray(records["users"]),
        labels=np.asarray(records["labels"], dtype=np.int64),
        view_names=np.asarray(VIEW_NAMES),
        frame_count=np.asarray(FRAME_COUNT, dtype=np.int64),
        features=np.stack(records["features"]),
        person_valid_rate=np.asarray(records["person_valid_rate"], dtype=np.float32),
        workspace_valid_rate=np.asarray(records["workspace_valid_rate"], dtype=np.float32),
    )
    return path


def main() -> None:
    args = parse_args()
    if args.trial_batch < 1 or args.image_batch < 1:
        raise ValueError("Batch sizes must be positive")
    rows = read_aligned_rows()
    if args.max_trials > 0:
        rows = rows[: args.max_trials]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            args.model,
            allow_patterns=("*.json", "*.safetensors", "*.txt"),
            local_files_only=True,
        )
    )
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = Dinov2Model.from_pretrained(snapshot, local_files_only=True)
    if int(model.config.hidden_size) != HIDDEN_SIZE:
        raise RuntimeError(f"DINOv2 hidden size changed: {model.config.hidden_size}")
    device = torch.device(args.device)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    completed = skipped = 0
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start : batch_start + args.trial_batch]
        work_rows: list[dict[str, str]] = []
        flat_images: list[np.ndarray] = []
        metadata: list[dict[str, np.ndarray]] = []
        for row in batch_rows:
            cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
            if not args.overwrite and valid_cache(cache, row):
                skipped += 1
                continue
            videos, trial_metadata = prepare_trial(
                row, args.p29_run.resolve(), frame_count=FRAME_COUNT
            )
            work_rows.append(row)
            flat_images.extend(to_rgb(image) for view in videos for image in view)
            metadata.append(trial_metadata)
        if work_rows:
            features, peak = encode_images(
                model, processor, flat_images, args.image_batch, device
            )
            peak_cuda_gib = max(peak_cuda_gib, peak)
            features = features.reshape(len(work_rows), len(VIEW_NAMES), FRAME_COUNT, HIDDEN_SIZE)
            for index, row in enumerate(work_rows):
                cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
                atomic_npz(
                    cache,
                    sample_id=np.asarray(row["sample_id"]),
                    source_id=np.asarray(row["source_id"]),
                    user=np.asarray(row["user_id"]),
                    label=np.asarray(int(row["class_id"]), dtype=np.int64),
                    view_names=np.asarray(VIEW_NAMES),
                    features=features[index].astype(np.float16),
                    **metadata[index],
                )
                completed += 1
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % max(20, args.trial_batch) == 0:
            print(
                json.dumps(
                    {
                        "stage": "cache_progress",
                        "processed": processed,
                        "total": len(rows),
                        "new": completed,
                        "skipped": skipped,
                        "elapsed_sec": round(time.perf_counter() - started, 1),
                        "peak_cuda_gib": round(peak_cuda_gib, 3),
                    }
                ),
                flush=True,
            )
    if len(rows) in {401, 1384, 2914}:
        complete = aggregate(rows, output)
    else:
        complete = None
    summary = {
        "model": args.model,
        "snapshot": str(snapshot),
        "frame_count": FRAME_COUNT,
        "views": list(VIEW_NAMES),
        "samples": len(rows),
        "new": completed,
        "skipped": skipped,
        "elapsed_sec": time.perf_counter() - started,
        "peak_cuda_gib": peak_cuda_gib,
        "complete_features": str(complete) if complete is not None else None,
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


