from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from build_p46_videomae_cache import (
    DEFAULT_P29,
    MODEL_NAME,
    PROJECT_DIR,
    VIEW_NAMES,
    atomic_npz,
    prepare_trial,
    read_rows,
    restore_legacy_attention_biases,
    safe_relative,
)
from p46_protocol import DEFAULT_MANIFEST


DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_temporal_v2"
TEMPORAL_TOKENS = 8
SPATIAL_TOKENS = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen VideoMAE features while preserving its eight temporal "
            "tubelet positions and a 2x2 spatial grid for each P46 IR view."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trial-batch", type=int, default=4)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def encode(
    model: VideoMAEForVideoClassification,
    processor: VideoMAEImageProcessor,
    videos: list[list[np.ndarray]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    pixel_values = processor(videos, return_tensors="pt").pixel_values.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        output = model.videomae(pixel_values).last_hidden_state
        expected_tokens = TEMPORAL_TOKENS * SPATIAL_TOKENS * SPATIAL_TOKENS
        if output.shape[1:] != (expected_tokens, 768):
            raise RuntimeError(f"unexpected VideoMAE token geometry: {tuple(output.shape)}")
        grid = output.reshape(
            len(videos), TEMPORAL_TOKENS, SPATIAL_TOKENS, SPATIAL_TOKENS, 768
        )
        pooled = output.mean(dim=1)
        temporal = grid.mean(dim=(2, 3))
        quadrants = grid.reshape(
            len(videos), TEMPORAL_TOKENS, 2, 7, 2, 7, 768
        ).mean(dim=(3, 5))
        if model.fc_norm is not None:
            pooled = model.fc_norm(pooled)
            temporal = model.fc_norm(temporal)
            quadrants = model.fc_norm(quadrants)
        logits = model.classifier(pooled)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    return (
        pooled.float().cpu().numpy(),
        logits.float().cpu().numpy(),
        temporal.float().cpu().numpy(),
        quadrants.float().cpu().numpy(),
        peak,
    )


def valid_cache(path: Path, row: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["sample_id"].item()) == row["sample_id"]
                and int(data["label"].item()) == int(row["class_id"])
                and data["features"].shape == (3, 768)
                and data["kinetics_logits"].shape == (3, 400)
                and data["temporal_features"].shape == (3, 8, 768)
                and data["quadrant_features"].shape == (3, 8, 2, 2, 768)
                and all(
                    np.isfinite(data[key]).all()
                    for key in (
                        "features",
                        "kinetics_logits",
                        "temporal_features",
                        "quadrant_features",
                    )
                )
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def aggregate(rows: list[dict[str, str]], output: Path) -> Path:
    values: dict[str, list[np.ndarray | str | int]] = {
        "sample_ids": [],
        "source_ids": [],
        "users": [],
        "labels": [],
        "features": [],
        "kinetics_logits": [],
        "temporal_features": [],
        "quadrant_features": [],
    }
    for row in rows:
        path = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
        if not valid_cache(path, row):
            raise RuntimeError(f"missing or invalid temporal VideoMAE cache: {row['source_id']}")
        with np.load(path, allow_pickle=False) as data:
            values["sample_ids"].append(row["sample_id"])
            values["source_ids"].append(row["source_id"])
            values["users"].append(row["user_id"])
            values["labels"].append(int(row["class_id"]))
            for key in (
                "features",
                "kinetics_logits",
                "temporal_features",
                "quadrant_features",
            ):
                values[key].append(np.asarray(data[key], dtype=np.float16))
    path = output / "complete_features.npz"
    atomic_npz(
        path,
        sample_ids=np.asarray(values["sample_ids"]),
        source_ids=np.asarray(values["source_ids"]),
        users=np.asarray(values["users"]),
        labels=np.asarray(values["labels"], dtype=np.int64),
        view_names=np.asarray(VIEW_NAMES),
        features=np.stack(values["features"]),
        kinetics_logits=np.stack(values["kinetics_logits"]),
        temporal_features=np.stack(values["temporal_features"]),
        quadrant_features=np.stack(values["quadrant_features"]),
    )
    return path


def main() -> None:
    args = parse_args()
    if args.trial_batch < 1:
        raise ValueError("--trial-batch must be positive")
    rows = read_rows(args.manifest.resolve())
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
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(snapshot, local_files_only=True)
    bias_report = restore_legacy_attention_biases(model, snapshot)
    device = torch.device(args.device)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    completed = 0
    skipped = 0
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start : batch_start + args.trial_batch]
        work_rows: list[dict[str, str]] = []
        videos: list[list[np.ndarray]] = []
        metadata: list[dict[str, np.ndarray]] = []
        for row in batch_rows:
            cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
            if not args.overwrite and valid_cache(cache, row):
                skipped += 1
                continue
            trial_videos, trial_metadata = prepare_trial(row, args.p29_run.resolve())
            work_rows.append(row)
            videos.extend(trial_videos)
            metadata.append(trial_metadata)
        if work_rows:
            pooled, logits, temporal, quadrants, peak = encode(
                model, processor, videos, device
            )
            peak_cuda_gib = max(peak_cuda_gib, peak)
            shapes = (
                (pooled, (len(work_rows), 3, 768)),
                (logits, (len(work_rows), 3, 400)),
                (temporal, (len(work_rows), 3, 8, 768)),
                (quadrants, (len(work_rows), 3, 8, 2, 2, 768)),
            )
            reshaped = []
            for value, shape in shapes:
                reshaped.append(value.reshape(shape))
            pooled, logits, temporal, quadrants = reshaped
            for index, row in enumerate(work_rows):
                cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
                atomic_npz(
                    cache,
                    sample_id=np.asarray(row["sample_id"]),
                    source_id=np.asarray(row["source_id"]),
                    user=np.asarray(row["user_id"]),
                    label=np.asarray(int(row["class_id"]), dtype=np.int64),
                    view_names=np.asarray(VIEW_NAMES),
                    features=pooled[index].astype(np.float16),
                    kinetics_logits=logits[index].astype(np.float16),
                    temporal_features=temporal[index].astype(np.float16),
                    quadrant_features=quadrants[index].astype(np.float16),
                    **metadata[index],
                )
                completed += 1
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % max(20, args.trial_batch) == 0:
            print(
                json.dumps(
                    {
                        "stage": "temporal_cache_progress",
                        "processed": processed,
                        "total": len(rows),
                        "built": completed,
                        "skipped": skipped,
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_cuda_gib": peak_cuda_gib,
                    }
                ),
                flush=True,
            )
    complete = aggregate(rows, output)
    summary: dict[str, Any] = {
        "protocol": "frozen VideoMAE with preserved temporal and 2x2 spatial tokens",
        "model": args.model,
        "model_snapshot": str(snapshot),
        "attention_bias_compatibility": bias_report,
        "views": list(VIEW_NAMES),
        "frames_per_view": 16,
        "temporal_tokens": 8,
        "spatial_grid": [2, 2],
        "trials": len(rows),
        "built": completed,
        "skipped": skipped,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": peak_cuda_gib,
        "complete_features": str(complete),
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
