"""Extract the two selected P90 visual teachers on the 401 readable Test rows."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import VideoMAEImageProcessor

from build_p46_videomae_multiclip_cache import prepare_trial
from p90_internvideo2_l_teacher import (
    EightFrameVideoCollator,
    PROCESSOR_SNAPSHOT,
    build_model as build_internvideo,
)
from p90_videomae_lora_teacher import VideoCollator
from p90_videomaev2_distilled_teacher import (
    PROCESSOR_REPO,
    build_model as build_videomaev2,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MANIFEST = HERE / "data/p46_test_union_manifest.csv"
P29_TEST = HERE / "runs/p29_dir_multiscale_roi_test"
BASE_OUTPUT = REPO_ROOT / "runs/p90_videomaev2_distilled_test_v1"
IV2_OUTPUT = REPO_ROOT / "runs/p90_internvideo2_l_k400_test_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-batch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        master = list(csv.DictReader(handle))
    if len(master) != 405:
        raise RuntimeError(f"Official Test count changed: {len(master)}")
    rows = [row for row in master if row["p46_ir_readable"] == "1"]
    if len(rows) != 401:
        raise RuntimeError(f"Expected 401 readable Test rows, got {len(rows)}")
    return rows


class TestTrialDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prepared = dict(row)
        prepared["source_id"] = row["sample_id"]
        prepared["ir_dir"] = row["ir_path"]
        clips, _ = prepare_trial(prepared, P29_TEST)
        return {
            "videos": clips,
            "clip_indices": list(range(len(clips))),
            "label": 0,
            "sample_id": row["official_sample_id"],
        }


def save_partial(
    path: Path,
    sample_ids: list[str],
    feature_parts: list[np.ndarray],
    action_parts: list[np.ndarray],
    action_classes: int,
) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        temporary,
        sample_ids=np.asarray(sample_ids),
        features=np.concatenate(feature_parts).reshape(-1, 2, 3, 768),
        action_logits=np.concatenate(action_parts).reshape(
            -1, 2, 3, action_classes
        ),
    )
    temporary.replace(path)


def extract(
    name: str,
    output: Path,
    rows: list[dict[str, str]],
    batch_size: int,
    workers: int,
    checkpoint_every: int,
    action_classes: int,
    build_model: Callable[[torch.device], Any],
    collator: Any,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "complete_features.npz"
    if complete.exists():
        print(json.dumps({"skip_existing": str(complete)}), flush=True)
        return
    partial = output / "partial_features.npz"
    expected_ids = [row["official_sample_id"] for row in rows]
    sample_ids: list[str] = []
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    if partial.exists():
        with np.load(partial, allow_pickle=False) as source:
            sample_ids = source["sample_ids"].astype(str).tolist()
            feature_parts = [
                np.asarray(source["features"], dtype=np.float16).reshape(-1, 6, 768)
            ]
            action_parts = [
                np.asarray(source["action_logits"], dtype=np.float16).reshape(
                    -1, 6, action_classes
                )
            ]
        if sample_ids != expected_ids[: len(sample_ids)]:
            raise RuntimeError(f"{name} partial Test order changed")
        print(f"resuming {name} rows={len(sample_ids)}/{len(rows)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P90 Test teacher extraction requires CUDA")
    built = build_model(device)
    if name == "videomaev2_base":
        model, _checkpoint = built
        head = model.head
    else:
        model, head = built
    dataset = TestTrialDataset(rows)
    loader = DataLoader(
        Subset(dataset, range(len(sample_ids), len(dataset))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=True,
    )
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4)
            dtype = torch.float16 if name == "videomaev2_base" else torch.bfloat16
            pixels = pixels.to(device=device, dtype=dtype, non_blocking=True)
            if name == "videomaev2_base":
                features = model.forward_features(pixels)
            else:
                features = model(pixels)
            action_logits = head(features)
            trials = len(batch["sample_ids"])
            if features.shape != (trials * 6, 768):
                raise RuntimeError(f"Unexpected {name} feature shape {features.shape}")
            feature_parts.append(features.reshape(trials, 6, 768).half().cpu().numpy())
            action_parts.append(
                action_logits.reshape(trials, 6, action_classes).half().cpu().numpy()
            )
            sample_ids.extend(batch["sample_ids"])
            if (batch_number + 1) % checkpoint_every == 0:
                save_partial(
                    partial, sample_ids, feature_parts, action_parts, action_classes
                )
                with np.load(partial, allow_pickle=False) as source:
                    feature_parts = [source["features"].copy().reshape(-1, 6, 768)]
                    action_parts = [
                        source["action_logits"].copy().reshape(
                            -1, 6, action_classes
                        )
                    ]
                print(
                    json.dumps(
                        {
                            "teacher": name,
                            "rows": len(sample_ids),
                            "total": len(rows),
                            "elapsed_seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
    if sample_ids != expected_ids:
        raise RuntimeError(f"{name} Test extraction order changed")
    payload = {
        "sample_ids": np.asarray(sample_ids),
        "proxy_ids": np.asarray([row["sample_id"] for row in rows]),
        "features": np.concatenate(feature_parts).reshape(-1, 2, 3, 768),
        "action_logits": np.concatenate(action_parts).reshape(
            -1, 2, 3, action_classes
        ),
    }
    np.savez_compressed(complete, **payload)
    summary = {
        "stage": "P90 selected visual teacher Test extraction",
        "teacher": name,
        "rows": len(rows),
        "clips_per_row": 6,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "complete_features": str(complete),
        "large_teacher_required_at_final_inference": False,
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    del model, head, built
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.trial_batch < 1 or args.checkpoint_every < 1:
        raise ValueError("batch/checkpoint values must be positive")
    rows = read_rows()
    base_snapshot = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    base_processor = VideoMAEImageProcessor.from_pretrained(
        base_snapshot, local_files_only=True
    )
    extract(
        "videomaev2_base",
        BASE_OUTPUT,
        rows,
        args.trial_batch,
        args.num_workers,
        args.checkpoint_every,
        710,
        build_videomaev2,
        VideoCollator(base_processor),
    )
    iv2_processor = VideoMAEImageProcessor.from_pretrained(
        PROCESSOR_SNAPSHOT, local_files_only=True
    )
    extract(
        "internvideo2_l",
        IV2_OUTPUT,
        rows,
        args.trial_batch,
        args.num_workers,
        args.checkpoint_every,
        400,
        build_internvideo,
        EightFrameVideoCollator(iv2_processor),
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
