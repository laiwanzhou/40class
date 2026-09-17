from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from build_p46_videomae_cache import atomic_npz, encode, restore_legacy_attention_biases, safe_relative
from build_p46_videomae_multiclip_cache import (
    CLASSIFIER_CLASSES,
    HIDDEN_SIZE,
    VIEW_NAMES,
    WINDOW_NAMES,
    prepare_trial,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_test_union_manifest.csv"
DEFAULT_P29 = PROJECT_DIR / "runs/p29_dir_multiscale_roi_test"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1"
MODEL_NAME = "MCG-NJU/videomae-large-finetuned-kinetics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract early/late Large VideoMAE features for readable official Test IR trials."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trial-batch", type=int, default=16)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        master = list(csv.DictReader(handle))
    if len(master) != 405:
        raise RuntimeError(f"Official Test count changed: {len(master)}")
    selected = [row for row in master if row["p46_ir_readable"] == "1"]
    if len(selected) != 401:
        raise RuntimeError(f"Expected 401 readable Test IR trials, got {len(selected)}")
    return master, selected


def row_for_encoder(row: dict[str, str]) -> dict[str, str]:
    prepared = dict(row)
    prepared["source_id"] = row["sample_id"]
    prepared["ir_dir"] = row["ir_path"]
    return prepared


def cache_path(output: Path, row: dict[str, str]) -> Path:
    return output / "trial_feature_cache" / safe_relative(row["sample_id"]).with_suffix(".npz")


def valid_cache(path: Path, row: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["sample_id"].item()) == row["official_sample_id"]
                and data["features"].shape == (2, 3, HIDDEN_SIZE)
                and data["kinetics_logits"].shape == (2, 3, CLASSIFIER_CLASSES)
                and np.isfinite(data["features"]).all()
                and np.isfinite(data["kinetics_logits"]).all()
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def aggregate(rows: list[dict[str, str]], output: Path) -> Path:
    features = []
    logits = []
    for row in rows:
        path = cache_path(output, row)
        if not valid_cache(path, row):
            raise RuntimeError(f"Missing or invalid Test multi-clip cache: {row['official_sample_id']}")
        with np.load(path, allow_pickle=False) as data:
            features.append(np.asarray(data["features"], dtype=np.float16))
            logits.append(np.asarray(data["kinetics_logits"], dtype=np.float16))
    complete = output / "complete_features.npz"
    atomic_npz(
        complete,
        sample_ids=np.asarray([row["official_sample_id"] for row in rows]),
        proxy_ids=np.asarray([row["sample_id"] for row in rows]),
        official_paths=np.asarray([row["official_path"] for row in rows]),
        window_names=np.asarray(WINDOW_NAMES),
        view_names=np.asarray(VIEW_NAMES),
        features=np.stack(features),
        kinetics_logits=np.stack(logits),
    )
    return complete


def main() -> None:
    args = parse_args()
    if args.trial_batch < 1:
        raise ValueError("--trial-batch must be positive")
    master, rows = read_rows(args.manifest.resolve())
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
    if int(model.config.hidden_size) != HIDDEN_SIZE or int(model.config.num_labels) != CLASSIFIER_CLASSES:
        raise RuntimeError("Large VideoMAE Test architecture changed")
    device = torch.device(args.device)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    completed = skipped = 0
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start : batch_start + args.trial_batch]
        work_rows = []
        videos = []
        metadata = []
        for row in batch_rows:
            path = cache_path(output, row)
            if not args.overwrite and valid_cache(path, row):
                skipped += 1
                continue
            row_videos, row_metadata = prepare_trial(
                row_for_encoder(row), args.p29_run.resolve()
            )
            work_rows.append(row)
            videos.extend(row_videos)
            metadata.append(row_metadata)
        if work_rows:
            feature, kinetics, peak = encode(model, processor, videos, device)
            peak_cuda_gib = max(peak_cuda_gib, peak)
            feature = feature.reshape(len(work_rows), 2, 3, HIDDEN_SIZE)
            kinetics = kinetics.reshape(len(work_rows), 2, 3, CLASSIFIER_CLASSES)
            for index, row in enumerate(work_rows):
                atomic_npz(
                    cache_path(output, row),
                    sample_id=np.asarray(row["official_sample_id"]),
                    proxy_id=np.asarray(row["sample_id"]),
                    official_path=np.asarray(row["official_path"]),
                    window_names=np.asarray(WINDOW_NAMES),
                    view_names=np.asarray(VIEW_NAMES),
                    features=feature[index].astype(np.float16),
                    kinetics_logits=kinetics[index].astype(np.float16),
                    **metadata[index],
                )
                completed += 1
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % 80 == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "total": len(rows),
                        "built": completed,
                        "skipped": skipped,
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                        "peak_cuda_gib": round(peak_cuda_gib, 3),
                    }
                ),
                flush=True,
            )
    complete = aggregate(rows, output) if len(rows) == 401 else None
    unavailable = [row["official_sample_id"] for row in master if row["p46_ir_readable"] != "1"]
    summary = {
        "protocol": "anonymous official Test, Large VideoMAE early/late x 3 views",
        "model": args.model,
        "snapshot": str(snapshot),
        "attention_bias_compatibility": bias_report,
        "master_samples": len(master),
        "p46_readable_samples": len(rows),
        "p46_unavailable_samples": unavailable,
        "built": completed,
        "skipped": skipped,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": peak_cuda_gib,
        "complete_features": str(complete) if complete is not None else None,
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
