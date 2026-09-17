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

from build_p46_videomae_cache import (
    VIEW_NAMES,
    atomic_npz,
    encode,
    prepare_trial,
    restore_legacy_attention_biases,
    safe_relative,
    valid_cache as valid_train_cache,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "MCG-NJU/videomae-large-finetuned-kinetics"
HIDDEN_SIZE = 1024
CLASSIFIER_CLASSES = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-window Large VideoMAE features for full-40 train or anonymous Test."
    )
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--p29-run", type=Path)
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trial-batch", type=int, default=16)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def defaults(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    if args.mode == "train":
        return (
            args.manifest or PROJECT_DIR / "data/p46_single_split.csv",
            args.p29_run or PROJECT_DIR / "runs/p29_dir_multiscale_roi_full",
            args.reuse_run or PROJECT_DIR / "runs/p46_videomae_large_ir_v1",
            args.output_dir or PROJECT_DIR / "runs/p85_videomae_large_fullwindow_full40_v1",
        )
    return (
        args.manifest or PROJECT_DIR / "data/p46_test_union_manifest.csv",
        args.p29_run or PROJECT_DIR / "runs/p29_dir_multiscale_roi_test",
        None,
        args.output_dir or PROJECT_DIR / "runs/p85_videomae_large_fullwindow_test_v1",
    )


def read_rows(path: Path, mode: str) -> tuple[list[dict[str, str]], int]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        master = list(csv.DictReader(handle))
    if mode == "train":
        rows = master
        expected = 2914
        if len(rows) != expected or {int(row["class_id"]) for row in rows} != set(range(40)):
            raise RuntimeError("Frozen full-40 train universe changed")
    else:
        if len(master) != 405:
            raise RuntimeError("Official Test universe changed")
        rows = [row for row in master if row["p46_ir_readable"] == "1"]
        expected = 401
        if len(rows) != expected:
            raise RuntimeError(f"Expected 401 readable Test IR rows, got {len(rows)}")
    rows.sort(key=lambda row: row["sample_id"])
    return rows, expected


def encoder_row(row: dict[str, str], mode: str) -> dict[str, str]:
    if mode == "train":
        return row
    prepared = dict(row)
    prepared["source_id"] = row["sample_id"]
    prepared["ir_dir"] = row["ir_path"]
    return prepared


def own_cache(output: Path, row: dict[str, str]) -> Path:
    relative_id = row["source_id"] if "source_id" in row else row["sample_id"]
    return output / "trial_feature_cache" / safe_relative(relative_id).with_suffix(".npz")


def legacy_cache(reuse: Path, row: dict[str, str]) -> Path:
    return reuse / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")


def valid_test_cache(path: Path, row: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["sample_id"].item()) == row["official_sample_id"]
                and data["features"].shape == (3, HIDDEN_SIZE)
                and data["kinetics_logits"].shape == (3, CLASSIFIER_CLASSES)
                and np.isfinite(data["features"]).all()
                and np.isfinite(data["kinetics_logits"]).all()
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def is_valid(path: Path, row: dict[str, str], mode: str) -> bool:
    if mode == "test":
        return valid_test_cache(path, row)
    return valid_train_cache(path, row, CLASSIFIER_CLASSES, HIDDEN_SIZE)


def resolve_cache(
    row: dict[str, str], mode: str, output: Path, reuse: Path | None
) -> tuple[Path | None, str]:
    own = own_cache(output, row)
    if is_valid(own, row, mode):
        return own, "output"
    if mode == "train" and reuse is not None:
        legacy = legacy_cache(reuse, row)
        if is_valid(legacy, row, mode):
            return legacy, "detail21_reuse"
    return None, "missing"


def aggregate(
    rows: list[dict[str, str]], mode: str, output: Path, reuse: Path | None
) -> Path:
    features: list[np.ndarray] = []
    kinetics: list[np.ndarray] = []
    sources: list[str] = []
    for row in rows:
        path, source = resolve_cache(row, mode, output, reuse)
        if path is None:
            raise RuntimeError(f"Missing full-window cache: {row['sample_id']}")
        with np.load(path, allow_pickle=False) as data:
            features.append(np.asarray(data["features"], dtype=np.float16))
            kinetics.append(np.asarray(data["kinetics_logits"], dtype=np.float16))
        sources.append(source)
    complete = output / "complete_features.npz"
    values: dict[str, np.ndarray] = {
        "sample_ids": np.asarray(
            [row["sample_id"] if mode == "train" else row["official_sample_id"] for row in rows]
        ),
        "view_names": np.asarray(VIEW_NAMES),
        "cache_sources": np.asarray(sources),
        "features": np.stack(features),
        "kinetics_logits": np.stack(kinetics),
    }
    if mode == "train":
        values.update(
            source_ids=np.asarray([row["source_id"] for row in rows]),
            users=np.asarray([row["user_id"] for row in rows]),
            labels=np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64),
        )
    else:
        values.update(
            proxy_ids=np.asarray([row["sample_id"] for row in rows]),
            official_paths=np.asarray([row["official_path"] for row in rows]),
        )
    atomic_npz(complete, **values)
    return complete


def main() -> None:
    args = parse_args()
    if args.trial_batch < 1:
        raise ValueError("--trial-batch must be positive")
    manifest, p29, reuse, output = defaults(args)
    output = output.resolve()
    reuse = reuse.resolve() if reuse is not None else None
    rows, expected = read_rows(manifest, args.mode)
    if args.max_trials > 0:
        rows = rows[: args.max_trials]
    output.mkdir(parents=True, exist_ok=True)
    existing = int(sum(resolve_cache(row, args.mode, output, reuse)[0] is not None for row in rows))
    print(json.dumps({"mode": args.mode, "rows": len(rows), "existing_or_reusable": existing}), flush=True)

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
        raise RuntimeError("Large VideoMAE architecture changed")
    device = torch.device(args.device)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    started = time.perf_counter()
    built = reused_count = skipped = 0
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start : batch_start + args.trial_batch]
        work_rows: list[dict[str, str]] = []
        videos: list[list[np.ndarray]] = []
        metadata: list[dict[str, np.ndarray]] = []
        for row in batch_rows:
            own = own_cache(output, row)
            if not args.overwrite and is_valid(own, row, args.mode):
                skipped += 1
                continue
            if args.mode == "train" and reuse is not None and not args.overwrite:
                legacy = legacy_cache(reuse, row)
                if is_valid(legacy, row, args.mode):
                    reused_count += 1
                    continue
            clips, trial_metadata = prepare_trial(encoder_row(row, args.mode), p29.resolve())
            work_rows.append(row)
            videos.extend(clips)
            metadata.append(trial_metadata)
        if work_rows:
            feature, logits, peak = encode(model, processor, videos, device)
            peak_cuda_gib = max(peak_cuda_gib, peak)
            feature = feature.reshape(len(work_rows), 3, HIDDEN_SIZE)
            logits = logits.reshape(len(work_rows), 3, CLASSIFIER_CLASSES)
            for index, row in enumerate(work_rows):
                values: dict[str, np.ndarray] = {
                    "sample_id": np.asarray(
                        row["sample_id"] if args.mode == "train" else row["official_sample_id"]
                    ),
                    "view_names": np.asarray(VIEW_NAMES),
                    "features": feature[index].astype(np.float16),
                    "kinetics_logits": logits[index].astype(np.float16),
                    **metadata[index],
                }
                if args.mode == "train":
                    values.update(
                        source_id=np.asarray(row["source_id"]),
                        user=np.asarray(row["user_id"]),
                        label=np.asarray(int(row["class_id"]), dtype=np.int64),
                    )
                else:
                    values.update(
                        proxy_id=np.asarray(row["sample_id"]),
                        official_path=np.asarray(row["official_path"]),
                    )
                atomic_npz(own_cache(output, row), **values)
                built += 1
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % 160 == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "total": len(rows),
                        "built": built,
                        "reused": reused_count,
                        "skipped": skipped,
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                        "peak_cuda_gib": round(peak_cuda_gib, 3),
                    }
                ),
                flush=True,
            )
    complete = aggregate(rows, args.mode, output, reuse) if len(rows) == expected else None
    summary = {
        "mode": args.mode,
        "protocol": "full-window 16-frame Large VideoMAE x scene/person/workspace IR",
        "deployment_rule_note": (
            "This public Large VideoMAE cache may be used for knowledge distillation. "
            "The teacher itself must not be required by final inference; all deployed "
            "weights, including ensembles, must be packaged below 100 MB."
        ),
        "model": args.model,
        "attention_bias_compatibility": bias_report,
        "rows": len(rows),
        "built": built,
        "reused": reused_count,
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
