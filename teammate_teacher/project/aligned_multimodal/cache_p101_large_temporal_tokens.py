"""Cache 12-subject-only unpooled temporal tokens from P101 visual teachers."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEImageProcessor

from p90_videomae_lora_teacher import FULL_MANIFEST, prepare_clips


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = HERE / "runs/p101_large_temporal_cache_v1"
VMAE_HISTORY = PROJECT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IV2_HISTORY = PROJECT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
DEV_USERS = frozenset(
    (
        "user1",
        "user2",
        "user21",
        "user17",
        "user23",
        "user6",
        "user8",
        "user16",
        "user18",
        "user19",
        "user5",
        "user7",
    )
)
H3_USERS = frozenset(("user3", "user4", "user9", "user20", "user22", "user24"))
USER_PATTERN = re.compile(r"__(user\d+)__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("videomaev2", "internvideo2"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def user_from_sample_id(sample_id: str) -> str:
    match = USER_PATTERN.search(str(sample_id))
    if match is None:
        raise ValueError(f"sample id has no subject: {sample_id}")
    return match.group(1)


def history_path(backbone: str) -> Path:
    return VMAE_HISTORY if backbone == "videomaev2" else IV2_HISTORY


def load_dev_rows(backbone: str, limit: int | None) -> list[dict[str, str]]:
    with np.load(history_path(backbone), allow_pickle=False) as archive:
        source_ids = archive["sample_ids"].astype(str)
    sample_ids = [
        sample_id
        for sample_id in source_ids.tolist()
        if user_from_sample_id(sample_id) in DEV_USERS
    ]
    if len(sample_ids) != 1941:
        raise RuntimeError(f"P101 development allow-list has {len(sample_ids)} rows")
    if any(user_from_sample_id(value) in H3_USERS for value in sample_ids):
        raise RuntimeError("H3 reached P101 temporal cache")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[:limit]
    with FULL_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        lookup = {row["sample_id"]: row for row in csv.DictReader(handle)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise KeyError(f"manifest is missing P101 rows: {missing[:3]}")
    return [lookup[sample_id] for sample_id in sample_ids]


class P101TemporalClipDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        clips = prepare_clips(row, "ir")
        if len(clips) != 6 or any(len(clip) != 16 for clip in clips):
            raise RuntimeError("P101 expects early/late x three 16-frame IR clips")
        return {"sample_id": row["sample_id"], "clips": clips}


class P101TemporalCollator:
    def __init__(self, processor: VideoMAEImageProcessor, frames: int) -> None:
        self.processor = processor
        self.frames = int(frames)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        videos: list[list[np.ndarray]] = []
        for item in items:
            for clip in item["clips"]:
                if self.frames == len(clip):
                    videos.append(clip)
                else:
                    positions = np.linspace(0, len(clip) - 1, self.frames).round().astype(int)
                    videos.append([clip[position] for position in positions])
        return {
            "pixel_values": self.processor(videos, return_tensors="pt").pixel_values,
            "sample_ids": [item["sample_id"] for item in items],
        }


def build_videomaev2(device: torch.device) -> tuple[torch.nn.Module, VideoMAEImageProcessor]:
    from huggingface_hub import snapshot_download
    from p90_videomaev2_distilled_teacher import PROCESSOR_REPO, build_model

    model, _ = build_model(device)
    snapshot = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    return model, processor


def build_internvideo2(device: torch.device) -> tuple[Any, torch.nn.Module, VideoMAEImageProcessor]:
    from p90_internvideo2_l_teacher import PROCESSOR_SNAPSHOT, build_model

    model, head = build_model(device)
    processor = VideoMAEImageProcessor.from_pretrained(
        PROCESSOR_SNAPSHOT, local_files_only=True
    )
    return model, head, processor


@torch.inference_mode()
def forward_videomaev2(
    model: torch.nn.Module, pixels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = model.patch_embed(pixels)
    if model.pos_embed is not None:
        values = values + model.pos_embed.expand(
            values.shape[0], -1, -1
        ).type_as(values).to(values.device).clone().detach()
    values = model.pos_drop(values)
    for block in model.blocks:
        values = block(values)
    temporal_steps = pixels.shape[2] // int(model.patch_embed.tubelet_size)
    spatial_patches = values.shape[1] // temporal_steps
    temporal_raw = values.reshape(
        values.shape[0], temporal_steps, spatial_patches, values.shape[-1]
    ).mean(dim=2)
    temporal = model.fc_norm(temporal_raw)
    pooled = model.fc_norm(values.mean(dim=1))
    action = model.head(pooled)
    return temporal, pooled, action


@torch.inference_mode()
def forward_internvideo2(
    model: Any, head: torch.nn.Module, pixels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = model.patch_embed(pixels.type(model.dtype))
    batch, temporal_steps, spatial_patches, width = values.shape
    values = values.reshape(batch, temporal_steps * spatial_patches, width)
    cls = model.cls_token.expand(batch, -1, -1)
    values = torch.cat((cls, values), dim=1) + model.pos_embed
    residual = None
    for block in model.blocks:
        if isinstance(values, tuple) and len(values) == 2:
            values, residual = values
        values = block(values, residual=residual)
    if isinstance(values, tuple) and len(values) == 2:
        values, residual = values
        if residual is not None:
            values = values + residual
    patch_values = values[:, 1:].reshape(
        batch, temporal_steps, spatial_patches, width
    )
    local = [
        model.fc_norm(model.clip_projector(patch_values[:, step]))
        for step in range(temporal_steps)
    ]
    temporal = torch.stack(local, dim=1)
    pooled = model.fc_norm(model.clip_projector(values))
    action = head(pooled)
    return temporal, pooled, action


def initialise_arrays(
    output: Path,
    rows: int,
    action_width: int,
    resume: bool,
) -> dict[str, np.memmap]:
    output.mkdir(parents=True, exist_ok=True)
    specifications = {
        "temporal_tokens": ((rows, 2, 3, 8, 768), np.float16),
        "pooled_features": ((rows, 2, 3, 768), np.float16),
        "action_logits": ((rows, 2, 3, action_width), np.float16),
        "completed": ((rows,), np.bool_),
    }
    arrays: dict[str, np.memmap] = {}
    for name, (shape, dtype) in specifications.items():
        path = output / f"{name}.npy"
        if resume and path.exists():
            array = np.load(path, mmap_mode="r+")
            if array.shape != shape or array.dtype != dtype:
                raise RuntimeError(f"existing {name} cache contract changed")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            array[...] = False if name == "completed" else 0
        arrays[name] = array
    return arrays


def historical_reference(
    backbone: str, sample_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(history_path(backbone), allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        lookup = {sample_id: index for index, sample_id in enumerate(ids)}
        order = np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
        return (
            np.asarray(archive["features"][order], dtype=np.float32),
            np.asarray(archive["action_logits"][order], dtype=np.float32),
        )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve() / args.backbone
    rows = load_dev_rows(args.backbone, args.limit)
    sample_ids = [row["sample_id"] for row in rows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P101 large temporal extraction requires CUDA")
    if args.backbone == "videomaev2":
        model, processor = build_videomaev2(device)
        head = None
        frames = 16
        action_width = 710
    else:
        model, head, processor = build_internvideo2(device)
        frames = 8
        action_width = 400
    arrays = initialise_arrays(output, len(rows), action_width, args.resume)
    with (output / "rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("cache_index", "sample_id", "user_id"))
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            writer.writerow(
                {
                    "cache_index": index,
                    "sample_id": sample_id,
                    "user_id": user_from_sample_id(sample_id),
                }
            )
    pending = np.flatnonzero(~np.asarray(arrays["completed"], dtype=bool)).astype(np.int64)
    dataset = P101TemporalClipDataset([rows[index] for index in pending])
    loader = DataLoader(
        dataset,
        batch_size=args.trial_batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=P101TemporalCollator(processor, frames),
        pin_memory=True,
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    cursor = 0
    for batch_number, batch in enumerate(loader):
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4)
        dtype = torch.float16 if args.backbone == "videomaev2" else torch.bfloat16
        pixels = pixels.to(device=device, dtype=dtype, non_blocking=True)
        if args.backbone == "videomaev2":
            temporal, pooled, action = forward_videomaev2(model, pixels)
        else:
            assert head is not None
            temporal, pooled, action = forward_internvideo2(model, head, pixels)
        trials = len(batch["sample_ids"])
        cache_rows = pending[cursor : cursor + trials]
        cursor += trials
        arrays["temporal_tokens"][cache_rows] = (
            temporal.reshape(trials, 2, 3, 8, 768).float().cpu().numpy()
        )
        arrays["pooled_features"][cache_rows] = (
            pooled.reshape(trials, 2, 3, 768).float().cpu().numpy()
        )
        arrays["action_logits"][cache_rows] = (
            action.reshape(trials, 2, 3, action_width).float().cpu().numpy()
        )
        arrays["completed"][cache_rows] = True
        for array in arrays.values():
            array.flush()
        if (batch_number + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "backbone": args.backbone,
                        "completed": int(np.asarray(arrays["completed"]).sum()),
                        "total": len(rows),
                        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
                    }
                ),
                flush=True,
            )
    if not np.asarray(arrays["completed"]).all():
        raise RuntimeError("P101 temporal cache is incomplete")
    history_features, history_actions = historical_reference(args.backbone, sample_ids)
    feature_error = float(
        np.abs(np.asarray(arrays["pooled_features"], dtype=np.float32) - history_features).max()
    )
    action_error = float(
        np.abs(np.asarray(arrays["action_logits"], dtype=np.float32) - history_actions).max()
    )
    feature_tolerance = 0.02 if args.backbone == "videomaev2" else 0.04
    action_tolerance = 0.05 if args.backbone == "videomaev2" else 0.08
    if feature_error > feature_tolerance or action_error > action_tolerance:
        raise RuntimeError(
            f"P101 pooled equivalence failed: feature={feature_error}, action={action_error}"
        )
    summary = {
        "stage": "P101_12subject_large_temporal_token_cache",
        "backbone": args.backbone,
        "rows": len(rows),
        "subjects": sorted({user_from_sample_id(value) for value in sample_ids}),
        "h3_rows": 0,
        "labels_cached": False,
        "historical_40class_logits_cached": False,
        "temporal_shape": list(arrays["temporal_tokens"].shape),
        "pooled_shape": list(arrays["pooled_features"].shape),
        "action_shape": list(arrays["action_logits"].shape),
        "dtype": "float16",
        "clips": "early/late x scene/person/workspace",
        "temporal_slots": 8,
        "pooled_feature_max_abs_error_vs_p90_fp16": feature_error,
        "action_logit_max_abs_error_vs_p90_fp16": action_error,
        "pooled_feature_tolerance": feature_tolerance,
        "action_logit_tolerance": action_tolerance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 2**30),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
