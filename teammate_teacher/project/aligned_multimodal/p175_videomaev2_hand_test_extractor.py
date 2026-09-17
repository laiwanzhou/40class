"""Extract the exact P91 six-view VideoMAEv2 hand stream for readable Test rows."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor

import p91_videomaev2_hand_teacher as hand
from p90_videomae_lora_teacher import VideoCollator
from p90_videomaev2_distilled_teacher import PROCESSOR_REPO, build_model


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MANIFEST = HERE / "data/p46_test_union_manifest.csv"
DEFAULT_ROI = HERE / "runs/p29_dir_multiscale_roi_test"
DEFAULT_OUTPUT = REPO / "runs/p175_videomaev2_hand_test_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-batch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    output = [
        {
            **row,
            "sample_id": row["official_sample_id"],
            "source_id": row["sample_id"],
            "ir_dir": row["ir_path"],
        }
        for row in source
        if row["p46_ir_readable"] == "1"
    ]
    if len(output) != 401:
        raise RuntimeError(f"P175 readable Test universe changed: {len(output)}")
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    test_rows = rows(args.manifest)
    hand.P29_RUN = args.roi_run.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model(device)
    snapshot = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    dataset = hand.HandDataset(test_rows, np.zeros(len(test_rows), dtype=np.int64))
    loader = DataLoader(
        dataset,
        batch_size=args.trial_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VideoCollator(processor),
        pin_memory=True,
    )
    feature_parts = []
    action_parts = []
    sample_ids = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for batch_number, batch in enumerate(loader):
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.float16, non_blocking=True)
        feature = model.forward_features(pixels)
        action = model.head(feature)
        trials = len(batch["sample_ids"])
        feature_parts.append(feature.reshape(trials, 2, 3, 768).cpu().numpy())
        action_parts.append(action.reshape(trials, 2, 3, 710).cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
        if (batch_number + 1) % 10 == 0:
            print(
                f"P175 extracted={len(sample_ids)}/{len(dataset)} "
                f"elapsed_min={(time.time()-started)/60:.1f}",
                flush=True,
            )
    if sample_ids != [row["sample_id"] for row in test_rows]:
        raise RuntimeError("P175 Test order changed")
    path = output / "complete_features.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(sample_ids),
        features=np.concatenate(feature_parts).astype(np.float16),
        action_logits=np.concatenate(action_parts).astype(np.float16),
    )
    report = {
        "stage": "P175_P91_hand_stream_Test_extraction",
        "status": "complete",
        "rows": len(sample_ids),
        "shape": [len(sample_ids), 2, 3, 768],
        "checkpoint": str(checkpoint),
        "test_labels_read": False,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "output": str(path),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
