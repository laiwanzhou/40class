"""Extract the P96 dense24 V-JEPA2 feature contract for 401 readable Test rows."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoVideoProcessor, VJEPA2ForVideoClassification

import p96_vjepa2_dense24_extractor as p96


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MANIFEST = HERE / "data/p46_test_union_manifest.csv"
DEFAULT_ROI = HERE / "runs/p29_dir_multiscale_roi_test"
DEFAULT_OUTPUT = REPO / "runs/p171_vjepa2_dense24_test_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-batch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = [
        {
            **row,
            "sample_id": row["official_sample_id"],
            "source_id": row["sample_id"],
            "ir_dir": row["ir_path"],
        }
        for row in source
        if row["p46_ir_readable"] == "1"
    ]
    if len(rows) != 401 or len({row["sample_id"] for row in rows}) != 401:
        raise RuntimeError(f"readable Test universe changed: {len(rows)}")
    return rows


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    # The shared dense24 crop functions resolve ROI geometry through this frozen
    # module-level root.  Test uses the identically generated Test ROI cache.
    p96.P29_RUN = args.roi_run.resolve()
    snapshot = Path(
        snapshot_download(
            p96.MODEL_REPO,
            allow_patterns=("*.json", "*.txt", "*.safetensors"),
            local_files_only=True,
        )
    )
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if "VJEPA2ForVideoClassification" not in set(config.architectures or ()):
        raise RuntimeError("cached V-JEPA2 checkpoint contract changed")
    processor = AutoVideoProcessor.from_pretrained(snapshot, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = VJEPA2ForVideoClassification.from_pretrained(
        snapshot, local_files_only=True, dtype=dtype
    ).to(device)
    model.eval()
    frames_per_clip = int(model.config.frames_per_clip)
    crop_size = int(model.config.crop_size)
    hidden = int(model.config.hidden_size)
    labels = int(model.config.num_labels)
    features, action_logits, done = p96.open_cache(
        args.output_dir, len(rows), hidden, labels
    )
    pending = np.flatnonzero(~np.asarray(done, dtype=bool))
    if args.max_samples:
        pending = pending[: args.max_samples]
    if not len(pending):
        print(f"P171 already complete={bool(np.asarray(done).all())}", flush=True)
        return
    loader = DataLoader(
        p96.Dense24Dataset(
            rows, pending, frames_per_clip, roi_run=args.roi_run.resolve()
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=p96.one_trial,
        persistent_workers=args.num_workers > 0,
    )
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    completed = 0
    for item in loader:
        row = int(item["row"])
        feature_parts = []
        logit_parts = []
        for low in range(0, p96.VIEW_COUNT, args.clip_batch):
            prepared = [
                p96.model_ready_video(video, frames_per_clip, crop_size)
                for video in item["videos"][low : low + args.clip_batch]
            ]
            encoded = processor(
                prepared, return_tensors="pt", do_resize=False, do_center_crop=False
            )
            pixels = encoded.pixel_values_videos.to(device=device, dtype=dtype)
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
            ):
                backbone = model.vjepa2(
                    pixel_values_videos=pixels, skip_predictor=True
                )
                pooled = model.pooler(backbone.last_hidden_state)
                logits = model.classifier(pooled)
            feature_parts.append(pooled.float().cpu().numpy())
            logit_parts.append(logits.float().cpu().numpy())
        feature_value = np.concatenate(feature_parts)
        logit_value = np.concatenate(logit_parts)
        if feature_value.shape != (p96.VIEW_COUNT, hidden):
            raise RuntimeError(f"unexpected P171 feature shape: {feature_value.shape}")
        features[row] = feature_value.astype(np.float16)
        action_logits[row] = logit_value.astype(np.float16)
        done[row] = True
        completed += 1
        if completed % args.flush_every == 0:
            features.flush(); action_logits.flush(); done.flush()
            peak = float(torch.cuda.max_memory_allocated() / 2**30)
            p96.write_progress(args.output_dir, snapshot, done, started, peak, False)
            print(
                f"P171 extracted={int(np.asarray(done).sum())}/{len(done)} "
                f"elapsed_min={(time.time()-started)/60:.1f} peak_gib={peak:.2f}",
                flush=True,
            )
    features.flush(); action_logits.flush(); done.flush()
    peak = float(torch.cuda.max_memory_allocated() / 2**30)
    p96.write_progress(
        args.output_dir, snapshot, done, started, peak, bool(np.asarray(done).all())
    )
    np.save(args.output_dir / "sample_ids.npy", np.asarray([row["sample_id"] for row in rows]))
    print(
        f"P171 finished={bool(np.asarray(done).all())} rows={int(np.asarray(done).sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
