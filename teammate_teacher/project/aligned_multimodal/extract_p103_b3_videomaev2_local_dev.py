"""Extract a dev-only, label-free VideoMAEv2 hand-local cache for P103-B3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEImageProcessor

from p100a_global_teacher_data import H3_USERS
from p90_videomae_lora_teacher import FULL_MANIFEST
from p90_videomaev2_distilled_teacher import PROCESSOR_REPO, build_model
from p91_videomaev2_hand_teacher import prepare_hand_clips


HERE = Path(__file__).resolve().parent
DEFAULT_ALLOWLIST = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_OUTPUT = HERE / "runs/p103_b3_videomaev2_local_dev_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-batch", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_dev_rows(allowlist: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with np.load(allowlist.resolve(), allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        users = archive["users"].astype(str)
    if len(sample_ids) != 1941 or len(set(sample_ids.tolist())) != 1941:
        raise RuntimeError("B3 extractor requires the frozen 1941-row dev allowlist")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 user reached B3 VideoMAEv2 extraction")

    # Only label-free path/identity columns are read from the full manifest.
    columns = ["sample_id", "user_id", "source_id", "ir_dir"]
    frame = pd.read_csv(FULL_MANIFEST, usecols=columns, dtype=str)
    lookup = {
        str(row.sample_id): {
            "sample_id": str(row.sample_id),
            "user_id": str(row.user_id),
            "source_id": str(row.source_id),
            "ir_dir": str(row.ir_dir),
        }
        for row in frame.itertuples(index=False)
    }
    rows = [lookup[str(sample_id)] for sample_id in sample_ids]
    if [row["user_id"] for row in rows] != users.tolist():
        raise RuntimeError("B3 allowlist/manifest user alignment changed")
    return rows, sample_ids, users


class LocalDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"videos": prepare_hand_clips(row), "sample_id": row["sample_id"]}


class LabelFreeCollator:
    def __init__(self, processor: VideoMAEImageProcessor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if any(len(item["videos"]) != 6 for item in items):
            raise RuntimeError("B3 requires six VideoMAEv2 local views per trial")
        videos = [video for item in items for video in item["videos"]]
        return {
            "pixel_values": self.processor(videos, return_tensors="pt").pixel_values,
            "sample_ids": [str(item["sample_id"]) for item in items],
        }


@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    rows, sample_ids, users = load_dev_rows(args.allowlist)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model(device)
    snapshot = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    loader = DataLoader(
        LocalDataset(rows),
        batch_size=args.trial_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=LabelFreeCollator(processor),
        pin_memory=device.type == "cuda",
    )
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    observed_ids: list[str] = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch_number, batch in enumerate(loader):
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.float16, non_blocking=True)
        features = model.forward_features(pixels)
        actions = model.head(features)
        trials = len(batch["sample_ids"])
        feature_parts.append(features.reshape(trials, 6, 768).cpu().numpy())
        action_parts.append(actions.reshape(trials, 6, 710).cpu().numpy())
        observed_ids.extend(batch["sample_ids"])
        if (batch_number + 1) % 20 == 0:
            print(
                f"B3 VideoMAEv2 extracted={len(observed_ids)}/{len(rows)} "
                f"elapsed_min={(time.time() - started) / 60:.1f}",
                flush=True,
            )
    if observed_ids != sample_ids.tolist():
        raise RuntimeError("B3 VideoMAEv2 extraction order changed")
    payload = {
        "sample_ids": sample_ids,
        "users": users,
        "features": np.concatenate(feature_parts).astype(np.float16),
        "action_logits": np.concatenate(action_parts).astype(np.float16),
    }
    np.savez_compressed(cache, **payload)
    summary = {
        "stage": "P103-B3",
        "label_free_extraction": True,
        "manifest_columns_loaded": ["sample_id", "user_id", "source_id", "ir_dir"],
        "labels_loaded": False,
        "rows": int(len(sample_ids)),
        "subjects": sorted(set(users.tolist())),
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
        "checkpoint": str(checkpoint),
        "clips": "full/peak-motion x left-hand/right-hand/interaction",
        "elapsed_seconds": time.time() - started,
        "peak_cuda_gib": (
            float(torch.cuda.max_memory_allocated() / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    (cache.parent / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "complete_features.npz"
    if args.reuse_cache and cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            if "labels" in archive.files or "class_id" in archive.files:
                raise RuntimeError("B3 local cache unexpectedly contains labels")
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        if len(payload["sample_ids"]) != 1941:
            raise RuntimeError("cached B3 local feature rows changed")
        print(f"reuse complete B3 local cache: {cache}", flush=True)
    else:
        extract(args, cache)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
